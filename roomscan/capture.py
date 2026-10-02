"""Loading of captures into a tier-agnostic in-memory representation.

Supported inputs
  * Stray Scanner export folder (LiDAR tier): rgb.mp4, depth/*.png (uint16 mm, 256x192),
    confidence/*.png (0/1/2), odometry.csv (ARKit camera-to-world poses), camera_matrix.csv.
  * A bare video file (video tier).
  * A folder of photo folders, one per room (photo tier).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

IMG_EXT = (".jpg", ".jpeg", ".png", ".heic", ".JPG", ".JPEG", ".PNG", ".HEIC")


@dataclass
class Frame:
    index: int
    timestamp: float
    T_wc: np.ndarray | None          # 4x4 camera-to-world (OpenCV camera convention: x right, y down, z fwd)
    K: np.ndarray                    # 3x3 intrinsics at RGB resolution
    rgb_size: tuple[int, int]        # (w, h)
    depth_path: str | None = None
    conf_path: str | None = None


@dataclass
class Capture:
    path: str
    tier: str                        # "lidar" | "video" | "photo"
    frames: list[Frame] = field(default_factory=list)
    video_path: str | None = None
    rooms: dict[str, list[str]] = field(default_factory=dict)   # photo tier: room -> image paths

    def load_depth(self, f: Frame, min_conf: int = 2):
        d = cv2.imread(f.depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        if f.conf_path and os.path.exists(f.conf_path):
            c = cv2.imread(f.conf_path, cv2.IMREAD_UNCHANGED)
            d[c < min_conf] = 0.0
        return d


# Stray Scanner already stores camera-to-world poses in the OpenCV camera convention
# (x right, y down, z forward) in an ARKit gravity-aligned world (y up). Verified empirically:
# multi-frame fusion is ~2.2x more compact with the identity than with the ARKit->CV flip.
_ARKIT_TO_CV = np.eye(4)


def detect_tier(path: str) -> str:
    if os.path.isfile(path):
        return "video"
    if os.path.exists(os.path.join(path, "odometry.csv")) and os.path.isdir(os.path.join(path, "depth")):
        return "lidar"
    subdirs = [d for d in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, d))]
    if subdirs and all(_list_images(os.path.join(path, d)) for d in subdirs):
        return "photo"
    if _list_images(path):
        return "photo"
    vids = [f for f in os.listdir(path) if f.lower().endswith((".mp4", ".mov"))]
    if vids:
        return "video"
    raise ValueError(f"cannot determine capture tier for {path}")


def _list_images(d: str) -> list[str]:
    return sorted(p for p in glob.glob(os.path.join(d, "*")) if p.endswith(IMG_EXT))


def load_stray(path: str) -> Capture:
    K = np.loadtxt(os.path.join(path, "camera_matrix.csv"), delimiter=",")
    od = np.genfromtxt(os.path.join(path, "odometry.csv"), delimiter=",", skip_header=1)
    cap = Capture(path=path, tier="lidar", video_path=os.path.join(path, "rgb.mp4"))
    vc = cv2.VideoCapture(cap.video_path)
    w, h = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vc.release()
    for row in od:
        ts, idx, x, y, z, qx, qy, qz, qw = row[:9]
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = [x, y, z]
        T = T @ _ARKIT_TO_CV
        Kf = K.copy()
        if row.shape[0] >= 13 and not np.isnan(row[9]):
            Kf = np.array([[row[9], 0, row[11]], [0, row[10], row[12]], [0, 0, 1.0]])
        i = int(idx)
        cap.frames.append(Frame(i, ts, T, Kf, (w, h),
                                os.path.join(path, "depth", f"{i:06d}.png"),
                                os.path.join(path, "confidence", f"{i:06d}.png")))
    return cap


def load_video(path: str) -> Capture:
    if os.path.isdir(path):
        vids = sorted(f for f in os.listdir(path) if f.lower().endswith((".mp4", ".mov")))
        path = os.path.join(path, vids[0])
    return Capture(path=path, tier="video", video_path=path)


def load_photos(path: str) -> Capture:
    cap = Capture(path=path, tier="photo")
    subdirs = [d for d in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, d))]
    if subdirs:
        for d in subdirs:
            imgs = _list_images(os.path.join(path, d))
            if imgs:
                cap.rooms[d] = imgs
    else:
        cap.rooms[os.path.basename(os.path.normpath(path))] = _list_images(path)
    return cap


def load_capture(path: str, tier: str | None = None) -> Capture:
    tier = tier or detect_tier(path)
    if tier == "lidar":
        return load_stray(path)
    if tier == "video":
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "rgb.mp4")):
            return load_video(os.path.join(path, "rgb.mp4"))
        return load_video(path)
    if tier == "photo":
        return load_photos(path)
    raise ValueError(tier)


def iter_video_frames(video_path: str, indices: list[int] | None = None, stride: int = 1):
    """Yield (index, BGR image). If indices is given, only those frames are decoded to arrays."""
    vc = cv2.VideoCapture(video_path)
    want = set(indices) if indices is not None else None
    i = 0
    while True:
        ok = vc.grab()
        if not ok:
            break
        if (want is None and i % stride == 0) or (want is not None and i in want):
            ok, img = vc.retrieve()
            if ok:
                yield i, img
        i += 1
        if want is not None and i > max(want):
            break
    vc.release()
