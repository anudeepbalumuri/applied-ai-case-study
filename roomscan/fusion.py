"""Depth back-projection, keyframe selection and point-cloud fusion (numpy only)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np


def depth_intrinsics(K_rgb, rgb_size, depth_shape):
    h, w = depth_shape
    sx, sy = w / rgb_size[0], h / rgb_size[1]
    return K_rgb[0, 0] * sx, K_rgb[1, 1] * sy, K_rgb[0, 2] * sx, K_rgb[1, 2] * sy


def backproject_cam(depth, K_rgb, rgb_size, max_depth=5.0):
    """Camera-frame point map (H,W,3) and validity mask."""
    h, w = depth.shape
    fx, fy, cx, cy = depth_intrinsics(K_rgb, rgb_size, depth.shape)
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    z = depth
    pm = np.stack([(u + 0.5 - cx) / fx * z, (v + 0.5 - cy) / fy * z, z], -1)
    valid = (z > 0.1) & (z < max_depth)
    return pm, valid


def normals_from_pointmap(pm, valid):
    """Per-pixel normals from central differences, oriented towards the camera."""
    dx = np.zeros_like(pm)
    dy = np.zeros_like(pm)
    dx[:, 1:-1] = pm[:, 2:] - pm[:, :-2]
    dy[1:-1] = pm[2:] - pm[:-2]
    n = np.cross(dx, dy)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(nn, 1e-9)
    flip = (n * pm).sum(-1) > 0
    n[flip] *= -1
    ok = valid.copy()
    ok[:, [0, -1]] = False
    ok[[0, -1]] = False
    ok[1:-1, 1:-1] &= valid[2:, 1:-1] & valid[:-2, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2]
    # reject depth discontinuities (flying pixels): neighbour depth jumps > 8% of depth
    z = pm[..., 2]
    jump = np.zeros_like(z, dtype=bool)
    jump[1:-1, 1:-1] = (np.maximum.reduce([np.abs(z[2:, 1:-1] - z[1:-1, 1:-1]), np.abs(z[:-2, 1:-1] - z[1:-1, 1:-1]),
                                           np.abs(z[1:-1, 2:] - z[1:-1, 1:-1]), np.abs(z[1:-1, :-2] - z[1:-1, 1:-1])])
                        > 0.08 * z[1:-1, 1:-1])
    ok &= ~jump
    return n, ok


def select_keyframes(frames, min_trans=0.05, min_rot_deg=4.0, poses=None):
    """Indices (into frames) of keyframes chosen by camera motion."""
    keep = [0]
    last = poses[frames[0].index] if poses else frames[0].T_wc
    cos_thr = np.cos(np.deg2rad(min_rot_deg))
    for i, f in enumerate(frames[1:], 1):
        T = poses[f.index] if poses else f.T_wc
        dt = np.linalg.norm(T[:3, 3] - last[:3, 3])
        cosr = (np.trace(last[:3, :3].T @ T[:3, :3]) - 1) / 2
        if dt > min_trans or cosr < cos_thr:
            keep.append(i)
            last = T
    return keep


def _hash(keys):
    keys = keys - keys.min(0)
    dims = keys.max(0) + 1
    return (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]


def voxel_reduce(pts, voxel, *extras):
    """Average points (and extra per-point arrays) per voxel. Returns (pts, counts, *extras)."""
    keys = np.floor(pts / voxel).astype(np.int64)
    h = _hash(keys)
    uniq, inv, counts = np.unique(h, return_inverse=True, return_counts=True)
    out = [np.zeros((uniq.size, 3))]
    np.add.at(out[0], inv, pts)
    out[0] /= counts[:, None]
    for e in extras:
        e2 = e.reshape(len(e), -1).astype(np.float64)
        acc = np.zeros((uniq.size, e2.shape[1]))
        np.add.at(acc, inv, e2)
        out.append(acc / counts[:, None])
    return (out[0], counts, *out[1:])


class FusedCloud:
    """World-frame fused cloud with normals and observation metadata."""

    def __init__(self, pts, normals, counts, first_t, cam_positions, voxel):
        self.pts = pts
        self.normals = normals
        self.counts = counts
        self.first_t = first_t
        self.cam_positions = cam_positions
        self.voxel = voxel


def _frame_points(capture, f, T, min_conf, max_depth, stride):
    d = capture.load_depth(f, min_conf=min_conf)
    pm, valid = backproject_cam(d, f.K, f.rgb_size, max_depth)
    n, ok = normals_from_pointmap(pm, valid)
    ok = ok[::stride, ::stride]
    p = pm[::stride, ::stride][ok]
    nc = n[::stride, ::stride][ok]
    R, t = T[:3, :3], T[:3, 3]
    return p @ R.T + t, nc @ R.T


_OFF = 1 << 20


def _vkey(pts, voxel):
    k = np.floor(pts / voxel).astype(np.int64) + _OFF
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


def _key_center(keys, voxel):
    x = (keys >> 42) - _OFF
    y = ((keys >> 21) & ((1 << 21) - 1)) - _OFF
    z = (keys & ((1 << 21) - 1)) - _OFF
    return (np.stack([x, y, z], 1) + 0.5) * voxel


class _Accumulator:
    """Running per-voxel sums; merged chunk by chunk to bound memory."""

    def __init__(self):
        self.keys = np.zeros(0, np.int64)
        self.sums = np.zeros((0, 7))      # xyz, normal xyz, first-keyframe (min)
        self.cnt = np.zeros(0, np.int64)

    def add(self, keys, vals, cnt):
        keys = np.concatenate([self.keys, keys])
        vals = np.concatenate([self.sums, vals])
        cnt = np.concatenate([self.cnt, cnt])
        u, inv = np.unique(keys, return_inverse=True)
        s = np.zeros((len(u), 6))
        np.add.at(s, inv, vals[:, :6])
        t = np.full(len(u), np.inf)
        np.minimum.at(t, inv, vals[:, 6])
        c = np.zeros(len(u), np.int64)
        np.add.at(c, inv, cnt)
        self.keys, self.sums, self.cnt = u, np.concatenate([s, t[:, None]], 1), c


def fuse(capture, poses=None, keyframes=None, voxel=0.02, min_conf=2, max_depth=4.0, stride=1, workers=8,
         chunk=48):
    """Fuse keyframes into a FusedCloud. poses: optional {frame.index: T_wc} override.

    Memory is bounded by reducing every `chunk` keyframes into a running voxel accumulator.
    """
    frames = capture.frames
    if keyframes is None:
        keyframes = select_keyframes(frames, poses=poses)
    sel = [frames[i] for i in keyframes]

    def job(f):
        T = poses[f.index] if poses is not None else f.T_wc
        return _frame_points(capture, f, T, min_conf, max_depth, stride)

    acc = _Accumulator()
    with ThreadPoolExecutor(workers) as ex:
        for c0 in range(0, len(sel), chunk):
            res = list(ex.map(job, sel[c0:c0 + chunk]))
            pts = np.concatenate([r[0] for r in res])
            nrm = np.concatenate([r[1] for r in res])
            tt = np.concatenate([np.full(len(r[0]), c0 + k, np.float64) for k, r in enumerate(res)])
            keys = _vkey(pts, voxel)
            u, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
            vals = np.zeros((len(u), 7))
            np.add.at(vals[:, :3], inv, pts)
            np.add.at(vals[:, 3:6], inv, nrm)
            vals[:, 6] = np.inf
            np.minimum.at(vals[:, 6], inv, tt)
            acc.add(u, vals, cnt)
    p = acc.sums[:, :3] / acc.cnt[:, None]
    n = acc.sums[:, 3:6]
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    cams = np.array([(poses[f.index] if poses is not None else f.T_wc)[:3, 3] for f in sel])
    return FusedCloud(p, n, acc.cnt, acc.sums[:, 6], cams, voxel)


def carve_free_space(capture, poses, keyframes, grid_origin, grid_shape, res, min_conf=1,
                     max_depth=4.0, pix_stride=6, workers=8):
    """3D free-space / occupancy evidence grid via ray marching.

    Returns int16 arrays (free, hit) with shape grid_shape (x, y, z cells) where `free` counts rays
    that passed through a cell and `hit` counts ray endpoints in it.
    """
    frames = capture.frames
    free = np.zeros(grid_shape, np.int32)
    hit = np.zeros(grid_shape, np.int32)
    origin = np.asarray(grid_origin)
    shape = np.asarray(grid_shape)

    def job(i):
        f = frames[i]
        T = poses[f.index] if poses is not None else f.T_wc
        d = capture.load_depth(f, min_conf=min_conf)[::pix_stride, ::pix_stride]
        h, w = d.shape
        fx, fy, cx, cy = depth_intrinsics(f.K, f.rgb_size, (h * pix_stride, w * pix_stride))
        v, u = np.mgrid[0:h, 0:w]
        u = u * pix_stride + 0.5 * pix_stride
        v = v * pix_stride + 0.5 * pix_stride
        m = (d > 0.1) & (d < max_depth)
        z = d[m]
        dirs = np.stack([(u[m] - cx) / fx, (v[m] - cy) / fy, np.ones_like(z)], 1)
        ends_c = dirs * z[:, None]
        R, t = T[:3, :3], T[:3, 3]
        ends = ends_c @ R.T + t
        steps = np.arange(0.0, 1.0, res / max_depth * 0.7)
        # free samples along each ray up to (depth - 1.5 res)
        L = np.linalg.norm(ends - t, axis=1)
        frac_stop = np.clip(1.0 - 1.5 * res / np.maximum(L, 1e-6), 0, 1)
        S = steps[None, :] * np.ones((len(L), 1))
        keep = S < frac_stop[:, None]
        samp = t + (ends - t)[:, None, :] * S[..., None]
        fs = samp[keep]
        fi = np.floor((fs - origin) / res).astype(np.int64)
        fi = fi[np.all((fi >= 0) & (fi < shape), 1)]
        hi = np.floor((ends - origin) / res).astype(np.int64)
        hi = hi[np.all((hi >= 0) & (hi < shape), 1)]
        return np.unique(np.ravel_multi_index(fi.T, grid_shape)), np.ravel_multi_index(hi.T, grid_shape)

    with ThreadPoolExecutor(workers) as ex:
        for fidx, hidx in ex.map(job, keyframes):
            np.add.at(free.reshape(-1), fidx, 1)
            np.add.at(hit.reshape(-1), hidx, 1)
    return free, hit
