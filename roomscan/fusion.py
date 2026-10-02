"""Depth back-projection and point-cloud fusion (numpy only, no open3d dependency)."""
from __future__ import annotations

import numpy as np


def backproject(depth: np.ndarray, K_rgb: np.ndarray, rgb_size, T_wc: np.ndarray,
                max_depth: float = 5.0, stride: int = 1):
    """Back-project a (possibly low-res) depth map to world points.

    K_rgb is at RGB resolution; it is rescaled to the depth resolution here.
    Returns (N,3) world points and (N,3) unit view rays (world) for normal orientation.
    """
    h, w = depth.shape
    sx, sy = w / rgb_size[0], h / rgb_size[1]
    fx, fy, cx, cy = K_rgb[0, 0] * sx, K_rgb[1, 1] * sy, K_rgb[0, 2] * sx, K_rgb[1, 2] * sy
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    m = (z > 0.1) & (z < max_depth)
    u, v, z = u[m] + 0.5, v[m] + 0.5, z[m]
    pc = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], 1)
    pw = pc @ T_wc[:3, :3].T + T_wc[:3, 3]
    return pw


def voxel_downsample(pts: np.ndarray, voxel: float, extra: np.ndarray | None = None):
    """Average points per voxel. Returns (points, counts[, extra averaged])."""
    keys = np.floor(pts / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = np.zeros((counts.size, 3))
    np.add.at(out, inv, pts)
    out /= counts[:, None]
    if extra is None:
        return out, counts
    ex = np.zeros((counts.size, extra.shape[1]))
    np.add.at(ex, inv, extra)
    return out, counts, ex / counts[:, None]


def fuse(capture, frame_ids=None, poses=None, stride_frames: int = 5, voxel: float = 0.02,
         min_conf: int = 2, max_depth: float = 4.5, return_frame_tags: bool = False):
    """Fuse LiDAR depth frames into a voxelised world point cloud.

    poses: optional dict frame.index -> 4x4 overriding capture poses (drift-corrected).
    """
    frames = capture.frames if frame_ids is None else [capture.frames[i] for i in frame_ids]
    frames = frames[::stride_frames] if frame_ids is None else frames
    chunks, tags = [], []
    for f in frames:
        T = poses[f.index] if poses is not None else f.T_wc
        d = capture.load_depth(f, min_conf=min_conf)
        p = backproject(d, f.K, f.rgb_size, T, max_depth=max_depth)
        chunks.append(p)
        tags.append(np.full(len(p), f.index))
    pts = np.concatenate(chunks)
    tg = np.concatenate(tags).astype(float)[:, None]
    p, c, t = voxel_downsample(pts, voxel, tg)
    if return_frame_tags:
        return p, c, t[:, 0]
    return p, c
