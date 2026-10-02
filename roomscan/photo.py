"""Photo tier: 2-8 stills per room, one folder per room, no depth, no poses -> one stitched plan.

Per room
  1. intrinsics from EXIF (35 mm-equivalent focal) or the iPhone main-camera prior
  2. metric depth per image (Depth-Anything-V2 Metric-Indoor), corrected by the bias measured vs LiDAR
  3. DISK + LightGlue matches between every pair; matched pixels of image i lifted to 3D with i's depth;
     cv2.solvePnPRansac gives pose of j relative to i  -> pairwise pose graph
  4. per-image depth scale made mutually consistent (log-scale least squares on pair ratios), poses
     chained along the maximum-inlier spanning tree, then refined
  5. fused room cloud -> gravity (Manhattan + floor) -> room layout with the shared plan extractor
Across rooms (stitching)
  6. photos that see through a doorway match photos of the neighbouring room: the same PnP machinery
     gives the room-to-room transform  ("via": "image_match")
  7. rooms with no image link are placed by door matching: openings with compatible widths are made to
     coincide, 4 Manhattan rotations x candidate door pairs, the placement with no polygon overlap and the
     most wall contact wins  ("via": "door_match"); unplaced rooms are reported, never silently overlapped
Intervals use the photo-tier error model (scale + surface terms) and widen with the per-room scale spread.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .fusion import FusedCloud, normals_from_pointmap, voxel_reduce
from .mono import (CAMERA_HEIGHT, IPHONE_FOCAL_FACTOR, MONO_SCALE_BIAS, MONO_SCALE_REL_SIGMA, DepthNet,
                   LearnedMatcher, align_gravity, estimate_gravity, refine_gravity)

MAX_SIDE = 1024


# ----------------------------------------------------------------------------- image IO + intrinsics
def load_photo(path):
    from PIL import Image, ImageOps
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    im = Image.open(path)
    exif = im.getexif()
    im = ImageOps.exif_transpose(im).convert("RGB")
    f35 = None
    try:
        sub = exif.get_ifd(0x8769)
        f35 = sub.get(0xA405)          # FocalLengthIn35mmFilm
    except Exception:
        pass
    bgr = np.array(im)[:, :, ::-1]
    s = MAX_SIDE / max(bgr.shape[:2])
    if s < 1:
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    h, w = bgr.shape[:2]
    if f35:
        f = float(f35) * np.hypot(w, h) / 43.27          # 35 mm full-frame diagonal
        src = "exif_35mm"
    else:
        f = IPHONE_FOCAL_FACTOR * max(w, h)
        src = "iphone_prior"
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    return np.ascontiguousarray(bgr), K, src


# ----------------------------------------------------------------------------- pairwise geometry
def _lift(kp, depth, K):
    u, v = kp[:, 0], kp[:, 1]
    h, w = depth.shape
    z = depth[np.clip(v.astype(int), 0, h - 1), np.clip(u.astype(int), 0, w - 1)]
    X = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
    return X, z


def pair_pose(fi, fj, di, dj, Ki, Kj, lm, min_inliers=20):
    """Pose of camera j in camera i's frame (T_i_j) from matches + i's metric depth. None if weak."""
    m = lm.match(fi, fj)
    if len(m) < min_inliers:
        return None
    ki = fi["kp"].cpu().numpy()[m[:, 0]]
    kj = fj["kp"].cpu().numpy()[m[:, 1]]
    Xi, zi = _lift(ki, di, Ki)
    ok = (zi > 0.2) & (zi < 8)
    if ok.sum() < min_inliers:
        return None
    succ, rvec, tvec, inl = cv2.solvePnPRansac(Xi[ok].astype(np.float64), kj[ok].astype(np.float64), Kj, None,
                                               iterationsCount=2000, reprojectionError=6.0, confidence=0.999,
                                               flags=cv2.SOLVEPNP_SQPNP)
    if not succ or inl is None or len(inl) < min_inliers:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T_j_i = np.eye(4)
    T_j_i[:3, :3], T_j_i[:3, 3] = R, tvec[:, 0]
    inl = inl[:, 0]
    # depth-scale ratio: depth of the inlier points seen from j (via pose) vs j's own network depth
    Xj = Xi[ok][inl] @ R.T + tvec[:, 0]
    _, zj_net = _lift(kj[ok][inl], dj, Kj)
    good = (zj_net > 0.2) & (Xj[:, 2] > 0.2)
    ratio = float(np.median(Xj[good, 2] / zj_net[good])) if good.sum() > 10 else 1.0   # s_j / s_i
    # baseline sanity: pure-rotation pairs give no scale information but pose is still fine
    return {"T_i_j": np.linalg.inv(T_j_i), "inliers": int(len(inl)), "ratio": ratio}


SCALE_PRIOR_SIGMA = 0.15     # per-image metric-depth scale noise (log); measured 0.84-1.39 IQR per frame


def solve_graph(n, edges, prior_sigma=SCALE_PRIOR_SIGMA):
    """edges: {(i,j): pair} -> (poses T_0_k dict, scales dict, components list).

    Depth scales: pair ratios fix *relative* scale; a per-image prior (log s_k ~ N(0, prior_sigma))
    ties each image to the network's own metric scale. Without the prior, small per-edge ratio
    biases compound along long chains (video: +43% scale drift over ~480 edges).
    """
    # depth scales: log s_j - log s_i = log ratio_ij, weighted by inliers; anchor s_root = 1
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for (i, j), e in edges.items():
        G.add_edge(i, j, w=e["inliers"], weight=-e["inliers"])
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    poses, scales = {}, {}
    for comp in comps:
        comp = sorted(comp)
        root = max(comp, key=lambda k: G.degree(k, weight="w"))
        idx = {k: a for a, k in enumerate(comp)}
        rows, b, wts = [], [], []
        for (i, j), e in edges.items():
            if i in idx and j in idx:
                r = np.zeros(len(comp))
                r[idx[j]], r[idx[i]] = 1, -1
                rows.append(r); b.append(np.log(max(e["ratio"], 1e-3))); wts.append(np.sqrt(e["inliers"]))
        for k in comp:
            r = np.zeros(len(comp)); r[idx[k]] = 1
            rows.append(r); b.append(0.0); wts.append(1.0 / prior_sigma)
        A = np.array(rows) * np.array(wts)[:, None]
        ls = np.linalg.lstsq(A, np.array(b) * np.array(wts), rcond=None)[0]
        for k in comp:
            scales[k] = float(np.exp(ls[idx[k]]))
        # chain poses along the maximum spanning tree (translations rescaled to the anchored scale)
        T = nx.maximum_spanning_tree(G.subgraph(comp), weight="w")
        poses[root] = np.eye(4)
        for a, c in nx.bfs_edges(T, root):
            if (a, c) in edges:
                Tac = edges[(a, c)]["T_i_j"].copy()
            else:
                Tac = np.linalg.inv(edges[(c, a)]["T_i_j"])
            # edge translation was measured in units of a's depth: convert to anchored metric units
            Tac[:3, 3] *= scales[a]
            poses[c] = poses[a] @ Tac
    return poses, scales, comps


# ----------------------------------------------------------------------------- room reconstruction
def spread_of(scales, main):
    return float(np.std([np.log(scales[k]) for k in main])) if len(main) > 1 else 0.0


def reconstruct_room(paths, net, lm, log=print):
    imgs, Ks, srcs = zip(*[load_photo(p) for p in paths])
    depths = [net(im) / MONO_SCALE_BIAS for im in imgs]
    feats = [lm.features(im) for im in imgs]
    n = len(imgs)
    edges = {}
    for i in range(n):
        for j in range(i + 1, n):
            e = pair_pose(feats[i], feats[j], depths[i], depths[j], Ks[i], Ks[j], lm)
            if e is not None:
                edges[(i, j)] = e
    poses, scales, comps = solve_graph(n, edges)
    main = sorted(comps[0])
    log(f"photo room: {len(main)}/{n} photos connected, {len(edges)} pair poses, "
        f"scale spread {np.std([np.log(scales[k]) for k in main]):.3f}")
    pts, nrm = [], []
    for k in main:
        d = depths[k] * scales[k]
        st = 4
        ds = d[::st, ::st]
        K = Ks[k]
        vv, uu = np.mgrid[0:ds.shape[0], 0:ds.shape[1]].astype(np.float32)
        pm = np.stack([(uu * st + st / 2 - K[0, 2]) / K[0, 0] * ds, (vv * st + st / 2 - K[1, 2]) / K[1, 1] * ds, ds], -1)
        valid = (ds > 0.2) & (ds < 7.0)
        nn, ok = normals_from_pointmap(pm, valid)
        T = poses[k]
        pts.append(pm[ok] @ T[:3, :3].T + T[:3, 3])
        nrm.append(nn[ok] @ T[:3, :3].T)
    pts, nrm = np.concatenate(pts), np.concatenate(nrm)
    cams = np.array([poses[k][:3, 3] for k in main])
    fwd = np.array([poses[k][:3, 2] for k in main])
    R = align_gravity(estimate_gravity(pts, nrm, cams, fwd))
    R = refine_gravity(pts, nrm, R)
    G = np.eye(4); G[:3, :3] = R
    pts, nrm, cams = pts @ R.T, nrm @ R.T, cams @ R.T
    # metric scale: depth network (bias-corrected) fused with the handheld camera-height prior
    floor_y = pick_floor(pts, nrm, cams)
    cam_h = float(np.median(cams[:, 1]) - floor_y) if floor_y is not None else None
    rel_net = float(np.hypot(MONO_SCALE_REL_SIGMA, spread_of(scales, main)))
    if cam_h is not None and cam_h > 0.5:
        w_net, w_h = 1 / rel_net ** 2, 1 / 0.06 ** 2
        ls = (w_h * np.log(CAMERA_HEIGHT / cam_h)) / (w_net + w_h)      # net cue = log(1) = 0
        sfac = float(np.exp(ls))
        scale_rel = float(1 / np.sqrt(w_net + w_h))
        chi = abs(np.log(CAMERA_HEIGHT / cam_h)) / np.hypot(rel_net, 0.06)
        scale_rel *= max(1.0, chi)
    else:
        sfac, scale_rel = 1.0, rel_net
    pts, cams = pts * sfac, cams * sfac
    floor_y = floor_y * sfac if floor_y is not None else None
    p, c, nm = voxel_reduce(pts, 0.03, nrm)
    nm /= np.maximum(np.linalg.norm(nm, axis=1, keepdims=True), 1e-9)
    cloud = FusedCloud(p, nm, c, np.zeros(len(p)), cams, 0.03)
    room_poses = {}
    for k in main:
        T = G @ poses[k]
        T[:3, 3] *= sfac
        room_poses[k] = T
    spread = spread_of(scales, main)
    return {"cloud": cloud, "poses": room_poses, "images": {k: imgs[k] for k in range(n)}, "K": list(Ks),
            "feats": feats, "depths": depths, "scales": scales, "main": main, "n": n, "edges": len(edges),
            "cam_height": cam_h, "scale_spread": spread, "floor_y": floor_y, "scale_rel": scale_rel,
            "scale_factor_from_height": sfac, "focal_source": srcs[0], "paths": list(paths)}


# ----------------------------------------------------------------------------- photo-tier layout
UNMEASURED_WALL_SIGMA = 0.35     # metres: wall position when no photo saw that wall (floor-extent fallback)


def pick_floor(pts, normals, cams):
    """Floor = the up-facing layer below the cameras that gives the most plausible handheld height.

    Candidates are histogram peaks of up-facing points; each is scored by support x prior on the
    implied camera height (N(1.42, 0.25) m, metric depth units). Avoids table tops and mono outliers.
    """
    up = normals[:, 1] > 0.9
    y = pts[up, 1]
    cam_y = float(np.median(cams[:, 1]))
    y = y[y < cam_y - 0.4]
    if len(y) < 100:
        return None
    h, e = np.histogram(y, bins=np.arange(y.min(), y.max() + 0.04, 0.03))
    hs = np.convolve(h, [0.25, 0.5, 0.25], mode="same")
    best, best_s = None, -1
    for i in range(1, len(hs) - 1):
        if hs[i] >= hs[i - 1] and hs[i] >= hs[i + 1] and hs[i] > 0.1 * hs.max():
            yy = e[i] + 0.015
            camh = cam_y - yy
            sc = hs[i] * np.exp(-0.5 * ((camh - CAMERA_HEIGHT) / 0.25) ** 2)
            if sc > best_s:
                best, best_s = yy, sc
    if best is None:
        return None
    sel = y[np.abs(y - best) < 0.05]
    return float(np.median(sel))


def box_layout(cloud, floor_y):
    """Manhattan box room from partial observations. Returns (PlanFrame, Room) or None."""
    from . import layout as L
    from .layout import Edge, PlanFrame, Room
    p, n = cloud.pts, cloud.normals
    yaw, strength = L.manhattan_yaw(p, n, floor_y)
    fr = PlanFrame(yaw, floor_y)
    uv = fr.to_plan(p[:, [0, 2]])
    nuv = fr.to_plan(n[:, [0, 2]])
    rel = p[:, 1] - floor_y
    cam = fr.to_plan(cloud.cam_positions[:, [0, 2]]).mean(0)
    flo = (n[:, 1] > 0.9) & (np.abs(rel) < 0.1)
    if flo.sum() < 50:
        return None
    fext_lo, fext_hi = np.percentile(uv[flo], 2, axis=0), np.percentile(uv[flo], 98, axis=0)
    band = (np.abs(n[:, 1]) < 0.3) & (rel > 0.3) & (rel < 2.3)
    coords = {}
    for k in (0, 1):
        for inward in (+1, -1):
            # wall on the far side in direction -inward (its normal points back into the room: +inward)
            m = band & (nuv[:, k] * inward > 0.85)
            vals = uv[m, k]
            side = vals[(vals - cam[k]) * (-inward) > 0.3]
            measured = False
            if len(side) > 150:
                h, e = np.histogram(side, bins=np.arange(side.min(), side.max() + 0.03, 0.02))
                hs = np.convolve(h, [0.25, 0.5, 0.25], mode="same")
                strong = np.where(hs >= max(0.3 * hs.max(), 60))[0]
                if len(strong):
                    i = strong.max() if inward < 0 else strong.min()     # farthest strong layer
                    c0 = e[i] + 0.01
                    sel = side[np.abs(side - c0) < 0.04]
                    med = float(np.median(sel))
                    mad = float(np.median(np.abs(sel - med))) * 1.4826
                    coords[(k, inward)] = (med, float(np.hypot(mad / np.sqrt(max(len(sel) / 25, 1)), 0.005)), len(sel), True)
                    measured = True
            if not measured:
                fb = fext_hi[k] if inward < 0 else fext_lo[k]
                coords[(k, inward)] = (float(fb), UNMEASURED_WALL_SIGMA, 0, False)
    x0, x1 = coords[(0, +1)][0], coords[(0, -1)][0]
    z0, z1 = coords[(1, +1)][0], coords[(1, -1)][0]
    if x1 - x0 < 0.8 or z1 - z0 < 0.8:
        return None
    # CCW rectangle edges: bottom (z=z0, inward +z), right (x=x1, inward -x), top (z=z1, inward -z), left (x=x0, +x)
    def E(axis, key, lo, hi, inward):
        c, sg, sup, meas = coords[key]
        return Edge(axis, c, lo, hi, inward, sg, sup, meas)
    edges = [E("z", (1, +1), x0, x1, +1), E("x", (0, -1), z0, z1, -1), E("z", (1, -1), x0, x1, -1), E("x", (0, +1), z0, z1, +1)]
    room = Room(0, None, edges)
    room.verts = L.edges_to_verts(edges)
    room.floor_y, room.floor_sigma = floor_y, 0.02
    ce = (n[:, 1] < -0.9) & (rel > 1.9) & (rel < 4.0)
    if ce.sum() > 100:
        room.ceil_y, room.ceil_sigma = L.plane_height(p[ce, 1])
        room.ceiling_observed = True
    else:
        room.ceil_y = float(np.percentile(p[band, 1], 99.5)) if band.any() else None
        room.ceiling_observed = False
    room.entered = True
    room.measured_wall_frac = sum(e.hi - e.lo for e in edges if e.measured) / sum(e.hi - e.lo for e in edges)
    room.partial = room.measured_wall_frac < 0.75
    from matplotlib.path import Path
    from scipy.spatial import cKDTree
    from .openings import detect_openings
    inside = Path(room.verts).contains_points(uv)
    tree = cKDTree(np.c_[uv[inside], rel[inside]]) if inside.sum() > 100 else None
    ceil_rel = (room.ceil_y - floor_y) if room.ceiling_observed else None
    room.openings = [o for o in detect_openings(edges, uv, rel, ceil_rel, tree) if edges[o.wall_index].measured]
    return fr, room, float(strength)


# ----------------------------------------------------------------------------- stitching
def _room_link(A, B, lm, min_inliers=30):
    """Best image-match transform taking room B's frame into room A's frame (2D: yaw + translation)."""
    best = None
    for i in A["main"]:
        for j in B["main"]:
            # depths in each room's final metric units (pairwise scales x camera-height correction)
            e = pair_pose(A["feats"][i], B["feats"][j],
                          A["depths"][i] * A["scales"][i] * A["scale_factor_from_height"],
                          B["depths"][j] * B["scales"][j] * B["scale_factor_from_height"],
                          A["K"][i], B["K"][j], lm, min_inliers)
            if e is None:
                continue
            if best is None or e["inliers"] > best[0]["inliers"]:
                best = (e, i, j)
    if best is None:
        return None
    e, i, j = best
    # camera j in A's world: T_A_i @ T_i_j ; B's world -> camera j: inv(T_B_j)
    T_A_B = A["poses"][i] @ e["T_i_j"] @ np.linalg.inv(B["poses"][j])
    yaw = np.arctan2(T_A_B[0, 2], T_A_B[0, 0])
    return {"yaw": float(yaw), "t": T_A_B[[0, 2], 3].copy(), "dy": float(T_A_B[1, 3]), "inliers": e["inliers"]}


def _poly_overlap(pa, pb):
    from shapely.geometry import Polygon
    a, b = Polygon(pa).buffer(-0.05), Polygon(pb).buffer(-0.05)
    if not a.is_valid or not b.is_valid:
        return 0.0
    return a.intersection(b).area


def _xform(poly, R, t):
    return np.asarray(poly) @ R.T + t


def _rot2(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _door_candidates(room_json):
    out = []
    for o in room_json["openings"]:
        if o["kind"] in ("door", "passage"):
            s, e = np.array(o["start"]), np.array(o["end"])
            out.append({"mid": (s + e) / 2, "w": o["width_m"]["value"], "n": np.array(o["inward_normal"]), "id": o["id"]})
    return out


def place_by_doors(placed, cand, wall_gap=0.12):
    """Place room `cand` (its own plan frame) against already placed rooms by matching a door.

    placed: list of (room_json, R, t). Returns (R, t, link) or None.
    """
    best = None
    dc = _door_candidates(cand)
    for pj, (pr, PR, Pt) in enumerate(placed):
        for da in _door_candidates(pr):
            mid_a = da["mid"] @ PR.T + Pt
            n_a = da["n"] @ PR.T
            for db in dc:
                if abs(da["w"] - db["w"]) > max(0.12, 0.15 * da["w"]):
                    continue
                # rotate cand so its door normal points opposite to a's (the rooms face each other)
                for k in range(4):
                    R = _rot2(np.pi / 2 * k)
                    if (db["n"] @ R.T) @ n_a > -0.9:
                        continue
                    # door midpoints coincide, separated by a wall thickness along the normal
                    t = mid_a - db["mid"] @ R.T - n_a * wall_gap
                    poly = _xform(cand["polygon"], R, t)
                    ov = sum(_poly_overlap(poly, _xform(q["polygon"], QR, Qt)) for q, QR, Qt in placed)
                    score = -ov * 10 - abs(da["w"] - db["w"])
                    if best is None or score > best[0]:
                        best = (score, R, t, {"to": pr["id"], "door_a": da["id"], "door_b": db["id"], "overlap_m2": ov})
    if best is None or best[3]["overlap_m2"] > 0.05:
        return None
    return best[1], best[2], best[3]


def reconstruct_photos(cap, capture_id="photos", log=print, timer=None):
    from .output import plan_to_json
    from .plan import extract_plan
    from .uncertainty import ErrorModel
    net = DepthNet.get()
    lm = LearnedMatcher.get()
    rooms_in = list(cap.rooms.items())
    recon, plans, jsons = [], [], []
    for name, paths in rooms_in:
        R = reconstruct_room(paths, net, lm, log=log)
        bl = box_layout(R["cloud"], R["floor_y"]) if R["floor_y"] is not None else None
        if bl is None:
            log(f"photo room {name}: no layout recovered (no floor / walls seen)")
            recon.append(None); plans.append(None); jsons.append(None)
            continue
        fr, room, strength = bl
        P = {"frame": fr, "rooms": [room], "adjacency": {}, "manhattan_strength": strength}
        err = ErrorModel("photo", scale_rel=R["scale_rel"])
        J = plan_to_json(P, "photo", name, err)
        J["rooms"][0]["name"] = name
        recon.append(R); plans.append(P); jsons.append(J)
    if timer:
        timer.lap("photo_rooms")
    # ---- stitching: image links first, then doors
    idx = [i for i, j in enumerate(jsons) if j is not None]
    placed = {}          # i -> (R2, t2) mapping room-i plan coords into the global plan
    links = []
    if idx:
        placed[idx[0]] = (np.eye(2), np.zeros(2))
    pending = set(idx[1:])
    progress = True
    while pending and progress:
        progress = False
        for i in sorted(pending):
            for a in list(placed):
                lk = _room_link(recon[a], recon[i], lm)
                if lk is None:
                    continue
                # world(B) -> world(A) is (yaw, t) in x/z; plan(B) -> world(B) and world(A) -> plan(A) via frames
                fa, fb = plans[a]["frame"], plans[i]["frame"]
                Rw = _rot2(-lk["yaw"])  # x/z rotation in plan-convention coordinates
                M = fa.R @ Rw @ fb.R.T
                # snap to Manhattan
                ang = np.arctan2(M[1, 0], M[0, 0])
                ang = np.round(ang / (np.pi / 2)) * (np.pi / 2)
                M = _rot2(ang)
                t = fa.R @ lk["t"]
                RA, tA = placed[a]
                R2, t2 = RA @ M, RA @ t + tA
                poly = _xform(jsons[i]["rooms"][0]["polygon"], R2, t2)
                ov = sum(_poly_overlap(poly, _xform(jsons[q]["rooms"][0]["polygon"], *placed[q])) for q in placed)
                if ov > 0.3:
                    continue
                placed[i] = (R2, t2)
                links.append({"a": jsons[a]["rooms"][0]["name"], "b": jsons[i]["rooms"][0]["name"],
                              "via": ["image_match"], "inliers": lk["inliers"], "overlap_m2": round(ov, 3)})
                pending.discard(i)
                progress = True
                break
        if not progress:
            for i in sorted(pending):
                pl = [(jsons[q]["rooms"][0], *placed[q]) for q in placed]
                res = place_by_doors(pl, jsons[i]["rooms"][0])
                if res is None:
                    continue
                R2, t2, info = res
                placed[i] = (R2, t2)
                to = next(jsons[q]["rooms"][0]["name"] for q in placed if jsons[q]["rooms"][0]["id"] == info["to"])
                links.append({"a": to, "b": jsons[i]["rooms"][0]["name"], "via": ["door_match"], **info})
                pending.discard(i)
                progress = True
                break
    unplaced = [jsons[i]["rooms"][0]["name"] for i in pending] + [rooms_in[i][0] for i, j in enumerate(jsons) if j is None]
    if timer:
        timer.lap("stitch")
    return _assemble(jsons, placed, links, unplaced, recon, plans, capture_id)


def L_area(v):
    v = np.asarray(v)
    return 0.5 * (np.dot(v[:, 0], np.roll(v[:, 1], -1)) - np.dot(np.roll(v[:, 0], -1), v[:, 1]))


def _assemble(jsons, placed, links, unplaced, recon, plans, capture_id):
    from .uncertainty import ErrorModel
    rooms = []
    views = {"images": {}, "views": []}
    name_to_id = {}
    for i, J in enumerate(jsons):
        if J is None or i not in placed:
            continue
        R2, t2 = placed[i]
        r = J["rooms"][0]
        rid = f"R{len(rooms)}"
        name_to_id[r["name"]] = rid
        rr = _xform_room_json(r, R2, t2, rid)
        rr["photo_stats"] = {"photos": recon[i]["n"], "connected": len(recon[i]["main"]),
                             "pair_poses": recon[i]["edges"], "scale_spread_log": round(recon[i]["scale_spread"], 4),
                             "camera_height_m": (round(recon[i]["cam_height"], 3) if recon[i]["cam_height"] else None),
                             "focal_source": recon[i]["focal_source"]}
        rooms.append(rr)
    adjacency = []
    for lk in links:
        if lk["a"] in name_to_id and lk["b"] in name_to_id:
            adjacency.append({"a": name_to_id[lk["a"]], "b": name_to_id[lk["b"]], "via": lk["via"],
                              **{k: v for k, v in lk.items() if k not in ("a", "b", "via", "to")}})
    err = ErrorModel("photo", scale_rel=MONO_SCALE_REL_SIGMA)
    tot = sum(r["floor_area_m2"]["value"] for r in rooms)
    tot_s = float(np.sqrt(sum(r["floor_area_m2"]["sigma"] ** 2 for r in rooms))) if rooms else 0.0
    allv = np.concatenate([np.array(r["polygon"]) for r in rooms]) if rooms else np.zeros((1, 2))
    bb = allv.max(0) - allv.min(0)
    # global overlap audit
    ov = 0.0
    for a in range(len(rooms)):
        for b in range(a + 1, len(rooms)):
            ov += _poly_overlap(rooms[a]["polygon"], rooms[b]["polygon"])
    out = {
        "schema_version": "1.0.0", "capture_id": capture_id, "tier": "photo",
        "units": "metres; plan frame: gravity-aligned, axes aligned with the dominant walls",
        "property": {"room_count": len(rooms), "total_floor_area_m2": err.measure(tot, tot_s, "area", "m2", 3),
                     "footprint_bbox_m": {"x": err.generic(bb[0], 0.05, "footprint"), "y": err.generic(bb[1], 0.05, "footprint")},
                     "adjacency": adjacency, "unplaced_rooms": unplaced,
                     "room_overlap_m2": round(ov, 3)},
        "rooms": rooms, "scope_line_items": [],
        "reconstruction": {"method": "pairwise PnP pose graph on metric mono depth (DISK+LightGlue)",
                           "depth_model_bias_correction": MONO_SCALE_BIAS},
    }
    # views for damage: every connected photo with its pose in the global plan's world convention
    # (each room keeps its own world frame; damage casts per room, so poses are expressed in room frames
    # and rooms are given their own plan transforms)
    out["_frame"] = None
    out["_per_room"] = [(name_to_id.get(jsons[i]["rooms"][0]["name"]), plans[i]["frame"], recon[i], placed[i])
                        for i in placed]
    return out


def _xform_room_json(r, R, t, rid):
    old = r["id"]
    def P(p):
        return [round(float(x), 4) for x in (np.asarray(p) @ R.T + t)]
    def N(n):
        return [round(float(x), 4) for x in (np.asarray(n) @ R.T)]
    rr = dict(r)
    rr["id"] = rid
    rr["polygon"] = [P(p) for p in r["polygon"]]
    rr["walls"] = [dict(w, id=w["id"].replace(old, rid, 1), surface_id=w["surface_id"].replace(old, rid, 1),
                        start=P(w["start"]), end=P(w["end"]), inward_normal=N(w["inward_normal"])) for w in r["walls"]]
    rr["openings"] = [dict(o, id=o["id"].replace(old, rid, 1), wall_id=o["wall_id"].replace(old, rid, 1),
                           start=P(o["start"]), end=P(o["end"]), inward_normal=N(o["inward_normal"]),
                           connects_to_room=None) for o in r["openings"]]
    rr["surfaces"] = [dict(s, id=s["id"].replace(old, rid, 1)) for s in r["surfaces"]]
    return rr
