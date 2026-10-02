"""Drift correction for long multi-room captures.

ARKit VIO poses drift (we measure ~5-9 cm vertical and several cm / sub-degree yaw over a
100 m walk). Gravity is reliable, so drift is modelled as a 4-DoF (x, y, z, yaw) correction per
submap and solved as a pose graph:

  * odometry edges   : consecutive submaps keep their relative pose (VIO is locally good)
  * loop edges       : 2D point-to-line ICP between non-consecutive submaps that observe the
                       same walls (revisits) gives a measured relative correction
  * Manhattan anchor : each submap's dominant wall direction should agree with the global one
                       (plane-anchored yaw correction)
  * floor anchor     : each submap's floor plane height should be the same (plane-anchored
                       vertical correction; solved separately, smoothed)

Corrections are interpolated per keyframe and applied as world-frame rigid transforms.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .fusion import fuse, select_keyframes


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _xz(p):
    return p[:, [0, 2]]


def correction_matrix(dx, dy, dz, dyaw, pivot):
    """World-frame rigid transform: rotate by dyaw about vertical axis through pivot, then translate."""
    R = _rot_y(dyaw)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pivot - R @ pivot + np.array([dx, dy, dz])
    return T


def dominant_yaw(xz_normals, weights=None):
    """Dominant Manhattan direction (radians in [-pi/4, pi/4)) from 2D wall normals."""
    ang = np.arctan2(xz_normals[:, 1], xz_normals[:, 0])
    w = np.ones(len(ang)) if weights is None else weights
    z = (w * np.exp(4j * ang)).sum()
    return np.angle(z) / 4.0, np.abs(z) / max(w.sum(), 1e-9)


def icp_2d(src, src_n, dst, dst_n, iters=30, max_dist=0.25):
    """Point-to-line ICP aligning src -> dst in 2D. Returns (theta, t(2), inlier_frac, rmse)."""
    tree = cKDTree(dst)
    th, t = 0.0, np.zeros(2)
    c0 = src.mean(0)
    inl, rmse = 0.0, np.inf
    for it in range(iters):
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        p = (src - c0) @ R.T + c0 + t
        pn = src_n @ R.T
        md = max_dist if it > 5 else max_dist * 2
        d, j = tree.query(p, distance_upper_bound=md)
        m = np.isfinite(d)
        m[m] &= (pn[m] * dst_n[j[m]]).sum(1) > 0.8
        if m.sum() < 50:
            return th, t, 0.0, np.inf
        q, n = dst[j[m]], dst_n[j[m]]
        pp = p[m]
        r = ((pp - q) * n).sum(1)
        # linearised: r + n.(dt + dth * perp(pp - c0 - t)) = 0
        perp = np.stack([-(pp - c0 - t)[:, 1], (pp - c0 - t)[:, 0]], 1)
        A = np.stack([n[:, 0], n[:, 1], (n * perp).sum(1)], 1)
        hub = 1.0 / np.maximum(1.0, np.abs(r) / 0.03)          # Huber weights
        x = np.linalg.lstsq(A * hub[:, None], -r * hub, rcond=None)[0]
        t = t + x[:2]
        th = th + x[2]
        inl = m.mean()
        rmse = np.sqrt(np.mean(r ** 2))
        if np.abs(x).max() < 1e-5:
            break
    return th, t, inl, rmse


class Submap:
    def __init__(self, kf_ids, cloud, floor_y):
        self.kf_ids = kf_ids
        self.cloud = cloud
        wall = np.abs(cloud.normals[:, 1]) < 0.3
        band = (cloud.pts[:, 1] > floor_y + 0.4) & (cloud.pts[:, 1] < floor_y + 2.3)
        m = wall & band
        n2 = _xz(cloud.normals[m])
        n2 /= np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-9)
        self.wall_xz = _xz(cloud.pts[m])
        self.wall_n = n2
        self.center = cloud.cam_positions.mean(0)
        up = (cloud.normals[:, 1] > 0.95) & (np.abs(cloud.pts[:, 1] - floor_y) < 0.25)
        self.floor_obs = float(np.median(cloud.pts[up, 1])) if up.sum() > 200 else None
        self.yaw, self.yaw_strength = dominant_yaw(n2) if len(n2) > 100 else (0.0, 0.0)


def build_submaps(capture, keyframes, floor_y, size=40):
    subs = []
    for s0 in range(0, len(keyframes), size):
        ids = keyframes[s0:s0 + size]
        if len(ids) < 5 and subs:
            subs[-1].kf_ids = subs[-1].kf_ids + ids
            continue
        cl = fuse(capture, keyframes=ids, voxel=0.04, stride=2)
        subs.append(Submap(ids, cl, floor_y))
    return subs


def _wrap4(a):
    """Wrap angle difference to [-pi/4, pi/4) (Manhattan ambiguity)."""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def correct_drift(capture, keyframes=None, floor_y=None, submap_size=40, log=print):
    """Return (poses dict frame.index -> corrected T_wc, report dict)."""
    frames = capture.frames
    if keyframes is None:
        keyframes = select_keyframes(frames)
    if floor_y is None:
        cl = fuse(capture, keyframes=keyframes[::4], voxel=0.05, stride=4)
        up = cl.normals[:, 1] > 0.95
        h, e = np.histogram(cl.pts[up, 1], bins=np.arange(cl.pts[:, 1].min(), cl.pts[:, 1].max() + 0.02, 0.02))
        floor_y = e[h.argmax()] + 0.01
    subs = build_submaps(capture, keyframes, floor_y, submap_size)
    S = len(subs)
    log(f"drift: {S} submaps")

    # ---- loop closures (non-consecutive submaps with overlapping wall evidence) ----
    loops = []
    for i in range(S):
        for j in range(i + 2, S):
            if np.linalg.norm(_xz(subs[i].center[None]) - _xz(subs[j].center[None])) > 4.0:
                continue
            a, b = subs[i], subs[j]
            if len(a.wall_xz) < 300 or len(b.wall_xz) < 300:
                continue
            th, t, inl, rmse = icp_2d(b.wall_xz, b.wall_n, a.wall_xz, a.wall_n)
            if inl > 0.35 and rmse < 0.035 and abs(th) < np.deg2rad(5) and np.linalg.norm(t) < 0.4:
                loops.append((i, j, th, t, b.wall_xz.mean(0), inl))
    log(f"drift: {len(loops)} loop closures accepted")

    # ---- global Manhattan direction from the strongest submaps ----
    ys = np.array([s.yaw for s in subs])
    ws = np.array([s.yaw_strength * len(s.wall_n) for s in subs])
    g_yaw = np.angle((ws * np.exp(4j * ys)).sum()) / 4

    centers = np.array([_xz(s.center[None])[0] for s in subs])

    # unknowns per submap: dx, dz, dyaw (pivot = submap center)
    def apply(c, k, pts):
        th = c[3 * k + 2]
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return (pts - centers[k]) @ R.T + centers[k] + c[3 * k:3 * k + 2]

    w_odo, w_loop, w_man = 1.0 / 0.01, 1.0 / 0.01, 1.0 / np.deg2rad(0.5)

    def resid(c):
        r = []
        for k in range(S - 1):
            # consecutive submaps: relative correction should be ~0 (evaluate on the shared boundary point)
            pb = (centers[k] + centers[k + 1])[None] / 2
            r.extend(((apply(c, k, pb) - apply(c, k + 1, pb)) * w_odo).ravel())
            r.append((c[3 * k + 2] - c[3 * (k + 1) + 2]) * w_odo * 0.5)
        for i, j, th, t, pc, inl in loops:
            # measurement: points of j moved by (th,t) about pc coincide with points of i
            Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            probe = pc[None] + np.array([[1.0, 0], [-1.0, 0], [0, 1.0], [0, -1.0]])
            target = (probe - pc) @ Rm.T + pc + t
            r.extend(((apply(c, j, probe) - apply(c, i, target)) * w_loop * inl).ravel())
        for k, s in enumerate(subs):
            if s.yaw_strength > 0.3:
                r.append(_wrap4(s.yaw + c[3 * k + 2] - g_yaw) * w_man * min(1.0, s.yaw_strength))
        r.extend(c[:3] * 100)     # gauge: first submap fixed
        return np.array(r)

    sol = least_squares(resid, np.zeros(3 * S), loss="huber", f_scale=3.0)
    c = sol.x

    # ---- vertical: floor-plane anchoring, smoothed ----
    fy = np.array([s.floor_obs if s.floor_obs is not None else np.nan for s in subs])
    dy = np.where(np.isfinite(fy), floor_y - fy, np.nan)
    idx = np.arange(S)
    ok = np.isfinite(dy)
    dy = np.interp(idx, idx[ok], dy[ok]) if ok.any() else np.zeros(S)
    dy = np.clip(dy, -0.15, 0.15)
    if S >= 3:
        k = np.array([0.25, 0.5, 0.25])
        dy = np.convolve(np.pad(dy, 1, mode="edge"), k, mode="valid")

    # ---- interpolate per keyframe and per frame ----
    kf_sub = np.concatenate([[k] * len(s.kf_ids) for k, s in enumerate(subs)])
    kf_list = np.concatenate([s.kf_ids for s in subs])
    sub_t = np.array([np.mean(s.kf_ids) for s in subs])
    par = np.stack([c[0::3], c[1::3], c[2::3], dy], 1)
    poses = {}
    for fi, f in enumerate(frames):
        if S == 1:
            p = par[0]
            cen = centers[0]
        else:
            p = np.array([np.interp(fi, sub_t, par[:, q]) for q in range(4)])
            cen = np.array([np.interp(fi, sub_t, centers[:, q]) for q in range(2)])
        C = correction_matrix(p[0], p[3], p[1], -p[2], np.array([cen[0], 0, cen[1]]))
        poses[f.index] = C @ f.T_wc
    report = {
        "method": "submap pose graph (2D point-to-line ICP loop closures) + Manhattan yaw anchor + floor-plane vertical anchor",
        "submaps": S,
        "loop_closures": len(loops),
        "max_translation_correction_m": float(np.abs(par[:, :2]).max()) if S else 0.0,
        "max_yaw_correction_deg": float(np.rad2deg(np.abs(par[:, 2]).max())) if S else 0.0,
        "max_vertical_correction_m": float(np.abs(par[:, 3]).max()) if S else 0.0,
        "floor_y": float(floor_y),
    }
    return poses, report
