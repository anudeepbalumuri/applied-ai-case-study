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
def _learned_matches(images, db_path, work, exhaustive, log):
    """Replace SIFT keypoints/matches in the COLMAP database with DISK + LightGlue, then verify."""
    import pycolmap
    lm = LearnedMatcher.get()
    feats = [lm.features(im) for im in images]
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
        if len(m) >= 15:
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


def _pairs(n, window=10, quad=(16, 24, 32, 48, 64, 96, 128)):
    P = set()
    for i in range(n):
        for d in list(range(1, window + 1)) + list(quad):
            if i + d < n:
                P.add((i, i + d))
    return sorted(P)


def run_sfm(images, work, focal_factor=IPHONE_FOCAL_FACTOR, log=print, exhaustive=False, matcher="disk_lightglue"):
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
        _learned_matches(images, db, work, exhaustive or len(images) <= 40, log)
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
    return R


def estimate_gravity(pts, normals, cam_centres):
    """World 'down' vector.

    Gravity is the Manhattan axis best aligned with the camera trajectory's smallest-variance
    direction (handheld walkthroughs move at near-constant height). The sign is chosen so that
    up-facing surfaces (floor, tables) lie below the cameras. Independent of how the phone was
    held (portrait video stored landscape breaks the 'camera y is down' heuristic: 93 deg error).
    """
    axes = manhattan_axes(normals)
    if len(cam_centres) >= 5:
        c = cam_centres - cam_centres.mean(0)
        w, V = np.linalg.eigh(c.T @ c)
        thin = V[:, 0]
        k = int(np.argmax(np.abs(axes.T @ thin)))
    else:
        # few views (photo tier): the axis with the most normal support is usually the floor/ceiling
        k = int(np.argmax([np.sum(np.abs(normals @ axes[:, i]) > 0.95) for i in range(3)]))
    a = axes[:, k]
    ref = cam_centres.mean(0)
    best = None
    for sgn in (1.0, -1.0):
        up = sgn * a
        facing = normals @ up > 0.9
        if facing.sum() < 50:
            continue
        below = np.mean((pts[facing] - ref) @ up < 0)
        if best is None or below > best[0]:
            best = (below, up)
    up = best[1] if best else a
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
def reconstruct_video(cap, log=print, timer=None, target_fps=6.0, max_frames=360, keep_work=False):
    frames, fps = select_video_frames(cap.video_path, target_fps, max_frames)
    dark = sum(f["brightness"] < 45 for f in frames)
    log(f"video: {len(frames)} frames selected ({dark} low-light)")
    if timer:
        timer.lap("frame_selection")
    work = tempfile.mkdtemp(prefix="roomscan_sfm_")
    try:
        poses, K, sparse, sfm = run_sfm([f["img"] for f in frames], work, log=log)
    finally:
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)
    if timer:
        timer.lap("sfm")
    net = DepthNet.get()
    pts, nrm, ratios, per_frame = fuse_mono(frames, poses, K, sparse, net, log=log)
    if timer:
        timer.lap("mono_depth")
    # gravity (see estimate_gravity: robust to portrait/landscape handling)
    cam_c = np.array([poses[k][:3, 3] for k in sorted(poses)])
    R = align_gravity(estimate_gravity(pts, nrm, cam_c))
    R = refine_gravity(pts, nrm, R)
    pts_g, nrm_g = pts @ R.T, nrm @ R.T
    cams = np.array([poses[k][:3, 3] for k in sorted(poses)]) @ R.T
    # camera height (sfm units): camera y minus floor y (lowest dense up-facing layer)
    up = nrm_g[:, 1] > 0.9
    if up.sum() > 500:
        yf = pts_g[up, 1]
        h, e = np.histogram(yf, bins=200)
        cand = np.where(h >= 0.3 * h.max())[0]
        floor_y = e[cand[0]]
        cam_h = float(np.median(cams[:, 1]) - floor_y)
    else:
        cam_h = None
    s, sig, det = metric_scale(ratios, cam_h, log=log)
    cloud = _to_cloud(pts_g * s, nrm_g, cams * s)
    if timer:
        timer.lap("scale_and_fusion")
    info = {"sfm": sfm, "frames_selected": len(frames), "low_light_frames": int(dark),
            "scale": {"metres_per_sfm_unit": s, "cues": det}, "scale_rel_sigma": sig,
            "depth_model": DEPTH_MODEL, "frame_rate": fps}
    views = {"images": {k: frames[k]["img"] for k in poses}, "views": [
        {"frame_index": frames[k]["index"], "key": k,
         "T_wc": _scaled_pose(poses[k], R, s), "K": K, "rgb_size": frames[k]["img"].shape[1::-1]}
        for k in sorted(poses)]}
    return cloud, info, views


def _scaled_pose(T, R, s):
    out = np.eye(4)
    out[:3, :3] = R @ T[:3, :3]
    out[:3, 3] = (R @ T[:3, 3]) * s
    return out
