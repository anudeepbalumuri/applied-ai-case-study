"""Video and photo tiers: no depth sensor, no poses.

Video tier
  frames  : sharpest frame per time window (Laplacian variance), dark frames flagged
  poses   : COLMAP incremental SfM (pycolmap, SIFT, sequential + quadratic overlap matching)
  depth   : Depth-Anything-V2 Metric-Indoor per frame, re-scaled per frame to the sparse SfM points
            (removes the monocular per-frame scale wobble; keeps the network's dense shape)
  gravity : mean camera "down" axis (phones are held near-level in roll), refined by the floor plane
  scale   : SfM is scale-free. Metric scale fuses two independent cues by inverse variance:
              (a) network metric depth / SfM depth, median over all frames, divided by the bias
                  measured against LiDAR on our benchmark (MONO_SCALE_BIAS)
              (b) handheld camera height above the floor (CAMERA_HEIGHT prior)
            and the fused relative sigma is propagated into every interval.
Photo tier: see reconstruct_photos.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import cv2
import numpy as np

from .fusion import FusedCloud, normals_from_pointmap, voxel_reduce

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
# Network metric depth / LiDAR depth, capture-level medians on our three captures: 1.051, 1.051, 1.104
MONO_SCALE_BIAS = 1.069
MONO_SCALE_REL_SIGMA = 0.035      # spread of the capture-level bias (conservative)
# Handheld camera height above floor (our operator: 1.41, 1.41, 1.46 m); widened for unknown operators
CAMERA_HEIGHT = 1.42
CAMERA_HEIGHT_REL_SIGMA = 0.06
IPHONE_FOCAL_FACTOR = 0.83        # fx / image width for the iPhone 15 main camera (1598 / 1920)


# ----------------------------------------------------------------------------- depth network
class DepthNet:
    _inst = None

    def __init__(self):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.float16 if self.dev == "cuda" else torch.float32
        self.proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
        self.model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL, dtype=dt).to(self.dev).eval()
        self.dtype = dt

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = DepthNet()
        return cls._inst

    def __call__(self, bgr):
        from PIL import Image
        h, w = bgr.shape[:2]
        with self.torch.no_grad():
            inp = self.proc(images=Image.fromarray(bgr[:, :, ::-1]), return_tensors="pt").to(self.dev, self.dtype)
            out = self.model(**inp)
        d = self.proc.post_process_depth_estimation(out, target_sizes=[(h, w)])[0]["predicted_depth"]
        return d.float().cpu().numpy()


# ----------------------------------------------------------------------------- frame selection
def select_video_frames(video_path, target_fps=2.5, max_frames=320, long_side=960):
    vc = cv2.VideoCapture(video_path)
    fps = vc.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = n / fps
    rate = min(target_fps, max_frames / max(dur, 1e-6))
    win = max(1, int(round(fps / rate)))
    best, frames, i = None, [], 0
    while True:
        ok = vc.grab()
        if not ok:
            break
        if i % 2 == 0:            # score every other frame (speed)
            ok, img = vc.retrieve()
            if ok:
                s = long_side / max(img.shape[:2])
                small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                sharp = cv2.Laplacian(g, cv2.CV_32F).var()
                if best is None or sharp > best[0]:
                    best = (sharp, i, small, float(g.mean()))
        if (i + 1) % win == 0 and best is not None:
            frames.append(best)
            best = None
        i += 1
    if best is not None:
        frames.append(best)
    vc.release()
    return [{"index": b[1], "img": b[2], "sharpness": float(b[0]), "brightness": b[3]} for b in frames], fps


# ----------------------------------------------------------------------------- SfM
LOCAL_WINDOW = 12          # selected frames (~2 s of video): always matched
LOOP_MIN_MATCHES = 150     # long-range pairs must be this strong (repetitive tiles/white walls
                           # otherwise glue rigid segments together at wrong angles)


def _learned_matches(images, db_path, work, exhaustive, log, feats=None):
    """Replace SIFT keypoints/matches in the COLMAP database with DISK + LightGlue, then verify."""
    import pycolmap
    lm = LearnedMatcher.get()
    feats = feats if feats is not None else [lm.features(im) for im in images]
    db = pycolmap.Database.open(db_path) if hasattr(pycolmap.Database, "open") and not isinstance(
        getattr(pycolmap.Database, "open"), property) else pycolmap.Database(db_path)
    ids = {im.name: im.image_id for im in db.read_all_images()}
    for k, f in enumerate(feats):
        iid = ids[f"{k:05d}.jpg"]
        db.update_keypoints(iid, f["kp"].cpu().numpy().astype(np.float32))
    db.clear_descriptors()
    db.clear_matches()
    db.clear_two_view_geometries()
    n = len(images)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)] if exhaustive else _pairs(n)
    lines = []
    tot = 0
    for i, j in pairs:
        m = lm.match(feats[i], feats[j])
        need = 15 if (exhaustive or j - i <= LOCAL_WINDOW) else LOOP_MIN_MATCHES
        if len(m) >= need:
            db.write_matches(ids[f"{i:05d}.jpg"], ids[f"{j:05d}.jpg"], m)
            lines.append(f"{i:05d}.jpg {j:05d}.jpg")
            tot += len(m)
    db.close()
    pairs_path = os.path.join(work, "pairs.txt")
    with open(pairs_path, "w") as fh:
        fh.write("\n".join(lines))
    pycolmap.verify_matches(db_path, pairs_path)
    log(f"matching: DISK+LightGlue, {len(lines)}/{len(pairs)} pairs with >=15 matches, {tot} matches")


def _cam_from_world(img):
    cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
    M = cfw.matrix()
    T = np.eye(4)
    T[:3, :4] = M
    return T


class LearnedMatcher:
    """DISK keypoints + LightGlue matching (kornia). Far more robust than SIFT on white walls."""
    _inst = None

    def __init__(self, max_kp=2048):
        import torch
        import kornia.feature as KF
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.disk = KF.DISK.from_pretrained("depth").to(self.dev).eval()
        self.lg = KF.LightGlue("disk").to(self.dev).eval()
        self.max_kp = max_kp

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = LearnedMatcher()
        return cls._inst

    def features(self, bgr):
        t = self.torch
        h, w = bgr.shape[:2]
        x = t.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1).float()[None].to(self.dev) / 255.0
        H, W = (h // 16) * 16, (w // 16) * 16                        # DISK needs multiples of 16
        x = x[:, :, :H, :W]
        with t.no_grad():
            f = self.disk(x, self.max_kp, pad_if_not_divisible=True)[0]
        return {"kp": f.keypoints, "desc": f.descriptors, "size": (W, H)}

    def match(self, fa, fb):
        t = self.torch
        def lafs(f):
            import kornia.feature as KF
            return KF.laf_from_center_scale_ori(f["kp"][None], t.ones(1, len(f["kp"]), 1, 1, device=self.dev))
        inp = {"image0": {"keypoints": fa["kp"][None], "descriptors": fa["desc"][None],
                          "image_size": t.tensor([fa["size"]], device=self.dev)},
               "image1": {"keypoints": fb["kp"][None], "descriptors": fb["desc"][None],
                          "image_size": t.tensor([fb["size"]], device=self.dev)}}
        with t.no_grad():
            out = self.lg(inp)
        m = out["matches"][0].cpu().numpy()
        return m.astype(np.uint32)


def _pairs(n, window=LOCAL_WINDOW, quad=(16, 24, 32, 48, 64, 96, 128)):
    P = set()
    for i in range(n):
        for d in list(range(1, window + 1)) + list(quad):
            if i + d < n:
                P.add((i, i + d))
    return sorted(P)


def run_sfm(images, work, focal_factor=IPHONE_FOCAL_FACTOR, log=print, exhaustive=False, matcher="disk_lightglue",
            feats=None):
    import pycolmap
    img_dir = os.path.join(work, "images")
    os.makedirs(img_dir, exist_ok=True)
    names = []
    for k, im in enumerate(images):
        nm = f"{k:05d}.jpg"
        cv2.imwrite(os.path.join(img_dir, nm), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        names.append(nm)
    h, w = images[0].shape[:2]
    db = os.path.join(work, "db.db")
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    f0 = focal_factor * w
    reader.camera_params = f"{f0},{w / 2},{h / 2},0"
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = max(w, h)
    ext.sift.max_num_features = 8192
    ext.sift.peak_threshold = 0.004          # low-texture interiors: accept weaker blobs
    ext.sift.darkness_adaptivity = True      # low light
    pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                              extraction_options=ext)
    if matcher == "disk_lightglue":
        _learned_matches(images, db, work, exhaustive or len(images) <= 40, log, feats=feats)
    elif exhaustive or len(images) <= 40:
        pycolmap.match_exhaustive(db)
    else:
        po = pycolmap.SequentialPairingOptions()
        po.overlap = 20
        po.quadratic_overlap = True
        pycolmap.match_sequential(db, pairing_options=po)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = True
    opts.min_model_size = 5
    opts.random_seed = 0
    opts.mapper.init_min_num_inliers = 50
    opts.mapper.abs_pose_min_num_inliers = 15
    opts.mapper.abs_pose_min_inlier_ratio = 0.15
    opts.mapper.init_min_tri_angle = 4.0
    opts.min_num_matches = 12
    opts.num_threads = os.cpu_count() or 8
    out = os.path.join(work, "sparse")
    os.makedirs(out, exist_ok=True)
    recs = pycolmap.incremental_mapping(db, img_dir, out, options=opts)
    if not recs:
        raise RuntimeError("SfM failed: no reconstruction")
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    log(f"sfm: {rec.num_reg_images()}/{len(images)} images registered in largest model "
        f"({len(recs)} model(s)), {rec.num_points3D()} points")
    cam = next(iter(rec.cameras.values()))
    K = cam.calibration_matrix()
    poses = {}
    sparse = {}
    for iid, img in rec.images.items():
        k = int(img.name.split(".")[0])
        T_cw = _cam_from_world(img)
        poses[k] = np.linalg.inv(T_cw)
        uv, xyz = [], []
        for p2 in img.points2D:
            if p2.has_point3D():
                uv.append(p2.xy)
                xyz.append(rec.points3D[p2.point3D_id].xyz)
        if uv:
            uv, xyz = np.array(uv), np.array(xyz)
            zc = (xyz @ T_cw[:3, :3].T + T_cw[:3, 3])[:, 2]
            sparse[k] = (uv, zc)
    return poses, K, sparse, {"registered": rec.num_reg_images(), "selected": len(images),
                              "models": len(recs), "points": rec.num_points3D(),
                              "focal_px": float(K[0, 0]), "focal_prior_px": float(f0)}


# ----------------------------------------------------------------------------- geometry helpers
def gravity_from_cameras(poses):
    """World 'down' direction: mean of camera +y axes (OpenCV: y points down in the image)."""
    ys = np.array([T[:3, 1] for T in poses.values()])
    g = ys.mean(0)
    return g / np.linalg.norm(g)


def manhattan_axes(normals, iters=3):
    """Three orthogonal dominant directions of surface normals (rooms are boxes)."""
    n = normals[np.random.default_rng(0).choice(len(normals), min(len(normals), 60000), replace=False)]
    # seed with the strongest direction via mean of the largest |n.a| cluster (power iteration on n n^T)
    C = (n.T @ n) / len(n)
    w, V = np.linalg.eigh(C)
    R = V[:, ::-1].copy()
    for _ in range(iters):
        # assign each normal to its closest axis (sign-agnostic), re-estimate, re-orthogonalise
        dots = np.abs(n @ R)
        a = dots.argmax(1)
        good = dots.max(1) > np.cos(np.deg2rad(20))
        M = np.zeros((3, 3))
        for k in range(3):
            sel = n[good & (a == k)]
            if len(sel) > 50:
                sgn = np.sign(sel @ R[:, k])[:, None]
                M[:, k] = (sel * sgn).mean(0)
            else:
                M[:, k] = R[:, k]
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1                      # proper rotation (right-handed axes)
    return R


def estimate_gravity(pts, normals, cam_centres, cam_forward=None):
    """World 'down' vector, independent of how the phone was held.

    Axis: the Manhattan axis that (a) the camera trajectory varies least along (handheld walkthroughs
    stay at near-constant height) and (b) the optical axes are most perpendicular to (people film
    walls, not the floor). Sign: up-facing surfaces (floor, table/bed/sofa tops) outnumber
    down-facing ones (ceiling). A geometric below/above test cannot decide the sign: flipping the
    world swaps floor and ceiling and stays consistent.
    """
    axes = manhattan_axes(normals)
    score = np.zeros(3)
    if len(cam_centres) >= 5:
        c = cam_centres - cam_centres.mean(0)
        w, V = np.linalg.eigh(c.T @ c)
        flat = 1.0 - np.sqrt(max(w[0], 0) / max(w[1], 1e-12))          # 1 = trajectory is planar
        score += flat * np.abs(axes.T @ V[:, 0])
    if cam_forward is not None and len(cam_forward):
        score += 1.0 - np.mean(np.abs(cam_forward @ axes), 0)
    if not score.any():
        # no trajectory / orientation cue: the axis with the most normal support (floor + ceiling)
        score = np.array([np.sum(np.abs(normals @ axes[:, i]) > 0.95) for i in range(3)], float)
    a = axes[:, int(np.argmax(score))]
    n_pos = np.sum(normals @ a > 0.9)
    n_neg = np.sum(normals @ a < -0.9)
    up = a if n_pos >= n_neg else -a
    return -up


def align_gravity(down):
    """Rotation mapping world so that `down` -> (0, -1, 0)  (y up, like ARKit)."""
    a = down / np.linalg.norm(down)
    b = np.array([0.0, -1.0, 0.0])
    v = np.cross(a, b)
    c = a @ b
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def refine_gravity(pts, normals, R):
    """Refine with the dominant floor-ish normal cluster (normals within 25 deg of up)."""
    n = normals @ R.T
    up = n[:, 1] > np.cos(np.deg2rad(25))
    p = pts @ R.T
    if up.sum() < 500:
        return R
    # lowest dense horizontal layer = floor
    y = p[up, 1]
    lo = np.percentile(y, 3)
    sel = up.copy()
    sel[up] = y < lo + 0.15 * (np.percentile(y, 97) - lo)
    m = n[sel].mean(0)
    m /= np.linalg.norm(m)
    return align_gravity(-m) @ R


# ----------------------------------------------------------------------------- dense fusion
def fuse_mono(frames, poses, K, sparse, net, stride=3, max_depth=6.0, log=print):
    """Back-project per-frame network depth, rescaled per frame to SfM units. Returns world pts/normals,
    camera centres, metric-to-sfm ratios (for the scale estimate)."""
    keys = sorted(poses)
    chunks_p, chunks_n, ratios, per_frame = [], [], [], {}
    for k in keys:
        img = frames[k]["img"]
        d = net(img)                                                     # metres (network)
        h, w = d.shape
        s_k = None
        if k in sparse and len(sparse[k][0]) >= 15:
            uv, zc = sparse[k]
            ui = np.clip(uv[:, 0].astype(int), 0, w - 1)
            vi = np.clip(uv[:, 1].astype(int), 0, h - 1)
            dn = d[vi, ui]
            ok = (zc > 0) & (dn > 0.2) & (dn < max_depth)
            if ok.sum() >= 15:
                r = zc[ok] / dn[ok]                                     # sfm units per network metre
                s_k = float(np.median(r))
                ratios.extend((dn[ok] / zc[ok]).tolist())
                per_frame[k] = s_k
        if s_k is None:
            continue
        ds = d[::stride, ::stride] * s_k
        fx, fy, cx, cy = K[0, 0] / stride, K[1, 1] / stride, K[0, 2] / stride, K[1, 2] / stride
        vv, uu = np.mgrid[0:ds.shape[0], 0:ds.shape[1]].astype(np.float32)
        pm = np.stack([(uu + 0.5 - cx) / fx * ds, (vv + 0.5 - cy) / fy * ds, ds], -1)
        valid = (d[::stride, ::stride] > 0.2) & (d[::stride, ::stride] < max_depth)
        nrm, ok = normals_from_pointmap(pm, valid)
        T = poses[k]
        chunks_p.append(pm[ok] @ T[:3, :3].T + T[:3, 3])
        chunks_n.append(nrm[ok] @ T[:3, :3].T)
    if not chunks_p:
        raise RuntimeError("no frames could be scaled to SfM")
    return np.concatenate(chunks_p), np.concatenate(chunks_n), np.array(ratios), per_frame


def metric_scale(ratios, cam_heights_sfm, log=print):
    """Fuse the two metric-scale cues. Returns (metres per SfM unit, relative sigma, details)."""
    cues = []
    if len(ratios) > 100:
        s_net = float(np.median(ratios)) / MONO_SCALE_BIAS          # metres per sfm unit
        cues.append(("depth_network", s_net, MONO_SCALE_REL_SIGMA))
    if cam_heights_sfm is not None and np.isfinite(cam_heights_sfm) and cam_heights_sfm > 0:
        s_h = CAMERA_HEIGHT / cam_heights_sfm
        cues.append(("camera_height", s_h, CAMERA_HEIGHT_REL_SIGMA))
    if not cues:
        raise RuntimeError("no metric scale cue")
    # inverse-variance fusion in log space
    w = np.array([1 / c[2] ** 2 for c in cues])
    ls = np.array([np.log(c[1]) for c in cues])
    s = float(np.exp((w * ls).sum() / w.sum()))
    sig = float(1 / np.sqrt(w.sum()))
    # inflate if the cues disagree beyond their sigmas (chi2 > 1)
    if len(cues) > 1:
        chi2 = float((w * (ls - np.log(s)) ** 2).sum() / (len(cues) - 1))
        sig *= max(1.0, np.sqrt(chi2))
    det = {c[0]: {"metres_per_unit": round(c[1], 5), "rel_sigma": c[2]} for c in cues}
    log(f"scale: {det} -> {s:.5f} m/unit (rel sigma {sig:.3f})")
    return s, sig, det


def _to_cloud(pts, nrm, cams, voxel=0.03):
    p, c, n = voxel_reduce(pts, voxel, nrm)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    keep = c >= 2                         # drop single-observation noise
    return FusedCloud(p[keep], n[keep], c[keep], np.zeros(keep.sum()), cams, voxel)


# ----------------------------------------------------------------------------- video tier
VIDEO_FPS = 6.0
MAX_FRAMES = 360
SEQ_WINDOW = 3                 # odometry edges to the next 1..3 frames (bridges blurred frames)
LOOP_OFFSETS = (12, 18, 24, 36, 48, 72, 96, 144)
LOOP_MIN_INLIERS = 150
BRIDGE_PER_GAP = 8
DAMAGE_VIEWS = 60              # same view budget as the LiDAR tier
FUSE_STRIDE = 8                # mono depth is too coarse to justify denser fusion; bounds memory on long clips


def _frame_normals_cam(depth, K, stride=8):
    ds = depth[::stride, ::stride]
    fx, fy, cx, cy = K[0, 0] / stride, K[1, 1] / stride, K[0, 2] / stride, K[1, 2] / stride
    vv, uu = np.mgrid[0:ds.shape[0], 0:ds.shape[1]].astype(np.float32)
    pm = np.stack([(uu + 0.5 - cx) / fx * ds, (vv + 0.5 - cy) / fy * ds, ds], -1)
    n, ok = normals_from_pointmap(pm, (ds > 0.2) & (ds < 6))
    return n[ok]


def densify_by_matches(video_path, frames, lm, min_matches=100, long_side=960, log=print):
    """Insert raw frames wherever consecutive selected frames share < min_matches LightGlue matches.

    Fast pans blur every frame of a selection window and leave too little overlap between the chosen
    frames; consecutive raw frames (1/46 s apart) still overlap. Repeats once on the new gaps.
    """
    from .capture import iter_video_frames
    feats = {f["index"]: lm.features(f["img"]) for f in frames}
    for rnd in range(2):
        order = sorted(frames, key=lambda f: f["index"])
        gaps = []
        for a, b in zip(order[:-1], order[1:]):
            if b["index"] - a["index"] > 1 and len(lm.match(feats[a["index"]], feats[b["index"]])) < min_matches:
                gaps.append((a["index"], b["index"]))
        if not gaps:
            break
        want = set()
        for a, b in gaps:
            step = max(1, (b - a) // 6)                  # up to ~6 bridge frames per gap
            want.update(range(a + step, b, step))
        for idx, img in iter_video_frames(video_path, indices=sorted(want)):
            sc = long_side / max(img.shape[:2])
            small = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            f = {"index": idx, "img": small, "sharpness": float(cv2.Laplacian(g, cv2.CV_32F).var()),
                 "brightness": float(g.mean()), "bridge": True}
            frames.append(f)
            feats[idx] = lm.features(small)
        log(f"video: densify round {rnd + 1}: {len(gaps)} weak gaps, +{len(want)} raw frames")
    frames.sort(key=lambda f: f["index"])
    return frames, [feats[f["index"]] for f in frames]


def _odometry_edges(frames, D, F, K, lm, pair_pose, cache=None):
    """Odometry edges to the next 1..SEQ_WINDOW frames. `cache` (keyed by video frame indices) avoids
    recomputing pairs when bridge frames are inserted and edges rebuilt."""
    n = len(frames)
    raw = {}
    cache = {} if cache is None else cache
    for i in range(n):
        for d in range(1, SEQ_WINDOW + 1):
            if i + d < n:
                key = (frames[i]["index"], frames[i + d]["index"])
                if key not in cache:
                    cache[key] = pair_pose(F[i], F[i + d], D[i], D[i + d], K, K, lm)
                e = cache[key]
                if e is not None:
                    raw[(i, i + d)] = dict(e, loop=False)
    return raw


def _breaks(n, raw):
    """Consecutive frame pairs (i, i+1) not joined by any path through odometry edges."""
    import networkx as nx
    G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(raw)
    comp = {}
    for ci, c in enumerate(nx.connected_components(G)):
        for k in c:
            comp[k] = ci
    return [i for i in range(n - 1) if comp[i] != comp[i + 1]]


def _densify(video_path, frames, D, F, K, gaps, net, lm, pair_pose, log, per_gap=None, cache=None, long_side=960):
    from .capture import iter_video_frames
    want = set()
    for i in gaps:
        a, b = frames[max(i - 1, 0)]["index"], frames[min(i + 2, len(frames) - 1)]["index"]
        if per_gap is None:
            want.update(range(a + 1, b))                 # every raw frame across the gap
        else:
            want.update(np.linspace(a + 1, b - 1, min(per_gap, max(b - a - 1, 0))).round().astype(int).tolist())
    want -= {f["index"] for f in frames}
    extra = []
    for idx, img in iter_video_frames(video_path, indices=sorted(want)):
        s = long_side / max(img.shape[:2])
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        extra.append({"index": idx, "img": small, "sharpness": float(cv2.Laplacian(g, cv2.CV_32F).var()),
                      "brightness": float(g.mean()), "bridge": True})
    pos = {f["index"]: k for k, f in enumerate(frames)}
    allf = sorted(frames + extra, key=lambda f: f["index"])
    D2, F2 = [], []
    for f in allf:
        if not f.get("bridge"):
            D2.append(D[pos[f["index"]]]); F2.append(F[pos[f["index"]]])
        else:
            D2.append((net(f["img"]) / MONO_SCALE_BIAS).astype(np.float16)); F2.append(lm.features(f["img"]))
    for f in allf:
        if f.get("bridge"):
            f["img"] = None          # bridge nodes only link the graph; never fused or shown to damage
    raw = _odometry_edges(allf, D2, F2, K, lm, pair_pose, cache)
    left = _breaks(len(allf), raw)
    log(f"video: densified {len(gaps)} breaks with {len(extra)} raw frames"
        f" ({'all' if per_gap is None else per_gap} per gap); {len(left)} breaks remain")
    return allf, D2, F2, raw


def reconstruct_video(cap, log=print, timer=None, target_fps=VIDEO_FPS, max_frames=MAX_FRAMES):
    """Video tier: PnP-on-metric-depth pose graph with Manhattan rotation anchors (see posegraph.py)."""
    from .photo import pair_pose
    from .posegraph import chain_init, frame_axis_obs, optimise
    frames, fps = select_video_frames(cap.video_path, target_fps, max_frames)
    dark = sum(f["brightness"] < 45 for f in frames)
    log(f"video: {len(frames)} frames selected ({dark} low-light)")
    if timer:
        timer.lap("frame_selection")
    net, lm = DepthNet.get(), LearnedMatcher.get()
    h, w = frames[0]["img"].shape[:2]
    K = np.array([[IPHONE_FOCAL_FACTOR * max(w, h), 0, w / 2], [0, IPHONE_FOCAL_FACTOR * max(w, h), h / 2], [0, 0, 1.0]])
    # float16 depth: hundreds of 960x720 maps in float32 pushed an 11 GB machine into swap
    D = [(net(f["img"]) / MONO_SCALE_BIAS).astype(np.float16) for f in frames]
    F = [lm.features(f["img"]) for f in frames]
    if timer:
        timer.lap("depth_and_features")
    cache = {}
    raw = _odometry_edges(frames, D, F, K, lm, pair_pose, cache)
    # densify across breaks: fast pans blur every frame of a selection window; consecutive *raw* frames
    # (1/46 s apart) still overlap, so decode them and chain through as bridge nodes. Two stages: a few
    # evenly spaced raw frames per gap, then every raw frame across the breaks that remain.
    for per_gap in (BRIDGE_PER_GAP, None):
        gaps = _breaks(len(frames), raw)
        if not gaps:
            break
        frames, D, F, raw = _densify(cap.video_path, frames, D, F, K, gaps, net, lm, pair_pose, log,
                                     per_gap=per_gap, cache=cache)
    n = len(frames)
    for i in range(0, n, 2):
        if frames[i].get("bridge"):
            continue
        for d in LOOP_OFFSETS:
            if i + d < n and (i, i + d) not in raw and not frames[i + d].get("bridge"):
                e = pair_pose(F[i], F[i + d], D[i], D[i + d], K, K, lm, min_inliers=LOOP_MIN_INLIERS)
                if e is not None:
                    raw[(i, i + d)] = dict(e, loop=True)
    log(f"video: {sum(not e['loop'] for e in raw.values())} odometry edges, {sum(e['loop'] for e in raw.values())} loop edges")
    # consistent per-frame depth scales (log-ratio least squares), then metric edge transforms
    from .photo import solve_graph
    _, scales, comps = solve_graph(n, {k: {"inliers": e["inliers"], "ratio": e["ratio"], "T_i_j": e["T_i_j"]} for k, e in raw.items()})
    main = set(comps[0])
    edges = {}
    for (i, j), e in raw.items():
        if i in main and j in main:
            T = e["T_i_j"].copy()
            T[:3, 3] *= scales[i]
            edges[(i, j)] = {"T": T, "w": e["inliers"], "loop": e["loop"]}
    init, _ = chain_init(n, edges)
    init = {k: v for k, v in init.items() if k in main}
    if timer:
        timer.lap("pairwise_poses")
    # Manhattan anchors
    ncam = {k: _frame_normals_cam(D[k] * scales[k], K) for k in init}
    poses = init
    W = None
    for it in range(2):
        allw = np.concatenate([ncam[k] @ poses[k][:3, :3].T for k in poses])
        W = manhattan_axes(allw)
        anchors = {}
        for k in poses:
            ob = frame_axis_obs(ncam[k], poses[k][:3, :3].T @ W)
            if ob:
                anchors[k] = ob
        poses, info = optimise(n, edges, poses, anchors, W)
        log(f"video: pose graph pass {it + 1}: {info}")
    if timer:
        timer.lap("pose_graph")
    keys = [k for k in sorted(poses) if not frames[k].get("bridge")]
    pts, nrm = [], []
    for k in keys:
        st = FUSE_STRIDE
        ds = D[k][::st, ::st].astype(np.float32) * np.float32(scales[k])
        vv, uu = np.mgrid[0:ds.shape[0], 0:ds.shape[1]].astype(np.float32)
        pm = np.stack([(uu * st + st / 2 - K[0, 2]) / K[0, 0] * ds, (vv * st + st / 2 - K[1, 2]) / K[1, 1] * ds, ds], -1)
        nn, ok = normals_from_pointmap(pm, (ds > 0.2) & (ds < 5.0))
        T = poses[k].astype(np.float32)
        pts.append(pm[ok] @ T[:3, :3].T + T[:3, 3])
        nrm.append(nn[ok] @ T[:3, :3].T)
    pts, nrm = np.concatenate(pts), np.concatenate(nrm)
    cams = np.array([poses[k][:3, 3] for k in keys])
    fwd = np.array([poses[k][:3, 2] for k in keys])
    R = align_gravity(estimate_gravity(pts, nrm, cams, fwd))
    R = refine_gravity(pts, nrm, R)
    pts, nrm, cams = pts @ R.T, nrm @ R.T, cams @ R.T
    # metric scale: depth network (already bias-corrected => factor 1) fused with camera height
    from .photo import pick_floor
    fy = pick_floor(pts, nrm, cams)
    cam_h = float(np.median(cams[:, 1]) - fy) if fy is not None else None
    s, sig, det = metric_scale(np.full(200, MONO_SCALE_BIAS), cam_h, log=log)
    cloud = _to_cloud(pts * s, nrm, cams * s)
    if timer:
        timer.lap("scale_and_fusion")
    info = {"method": "PnP pose graph on metric mono depth + Manhattan rotation anchors",
            "frames_selected": n, "frames_posed": len(keys), "low_light_frames": int(dark),
            "odometry_edges": sum(not e["loop"] for e in edges.values()), "loop_edges": sum(e["loop"] for e in edges.values()),
            "anchors": len(anchors), "scale": {"factor": s, "cues": det}, "scale_rel_sigma": sig,
            "depth_model": DEPTH_MODEL, "frame_rate": fps}
    vkeys = [keys[i] for i in np.linspace(0, len(keys) - 1, min(DAMAGE_VIEWS, len(keys))).astype(int)]
    views = {"images": {k: frames[k]["img"] for k in vkeys}, "views": [
        {"frame_index": frames[k]["index"], "key": k, "T_wc": _scaled_pose(poses[k], R, s), "K": K,
         "rgb_size": frames[k]["img"].shape[1::-1]} for k in vkeys]}
    info["_frame_map"] = {k: frames[k]["index"] for k in keys}
    info["_poses"] = {k: _scaled_pose(poses[k], R, s) for k in keys}
    return cloud, info, views


def _scaled_pose(T, R, s):
    out = np.eye(4)
    out[:3, :3] = R @ T[:3, :3]
    out[:3, 3] = (R @ T[:3, 3]) * s
    return out
