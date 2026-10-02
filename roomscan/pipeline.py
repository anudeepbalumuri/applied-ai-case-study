"""End-to-end pipeline: one capture in, plan JSON + rendered plan out."""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .capture import load_capture
from .output import plan_to_json
from .plan import extract_plan
from .render import render_plan
from .uncertainty import ErrorModel

PIPELINE_VERSION = "0.3.0"


class Timer:
    def __init__(self):
        self.t = {}
        self._t0 = time.time()

    def lap(self, name):
        now = time.time()
        self.t[name] = round(now - self._t0, 2)
        self._t0 = now


def reconstruct(cap, tier, opts, log, timer):
    """Tier-specific reconstruction -> (FusedCloud, extra-info dict, keyframe images for damage)."""
    if tier == "lidar":
        from .drift import correct_drift
        from .fusion import fuse, select_keyframes
        kf = select_keyframes(cap.frames)
        info = {"keyframes": len(kf), "frames": len(cap.frames)}
        if opts.get("drift", True):
            poses, rep = correct_drift(cap, kf, log=log)
        else:
            poses, rep = None, {"method": "disabled (ablation): ARKit poses used as-is"}
        timer.lap("drift_correction")
        cloud = fuse(cap, poses=poses, keyframes=kf, voxel=0.02)
        timer.lap("fusion")
        info["drift"] = rep
        views = _lidar_views(cap, kf, poses)
        return cloud, info, views, None
    if tier == "video":
        from .mono import reconstruct_video
        cloud, info, views = reconstruct_video(cap, log=log, timer=timer, **opts.get("video", {}))
        return cloud, info, views, info.get("scale_rel_sigma")
    if tier == "photo":
        from .mono import reconstruct_photos
        return reconstruct_photos(cap, log=log, timer=timer, **opts.get("photo", {}))
    raise ValueError(tier)


def _lidar_views(cap, kf, poses, n=60):
    """Evenly spaced keyframes with poses + depth for damage projection."""
    sel = [kf[i] for i in np.linspace(0, len(kf) - 1, min(n, len(kf))).astype(int)]
    out = []
    for i in sel:
        f = cap.frames[i]
        out.append({"frame_index": f.index, "T_wc": poses[f.index] if poses is not None else f.T_wc,
                    "K": f.K, "rgb_size": f.rgb_size, "depth": cap.load_depth(f, min_conf=1)})
    return {"video_path": cap.video_path, "views": out}


def run(path, out_dir, tier=None, capture_id=None, opts=None, log=print, damage=True):
    opts = opts or {}
    timer = Timer()
    cap = load_capture(path, tier)
    tier = cap.tier
    capture_id = capture_id or os.path.basename(os.path.normpath(path))
    os.makedirs(out_dir, exist_ok=True)
    log(f"[{capture_id}] tier={tier}")
    timer.lap("load")
    if tier == "photo":
        from .mono import reconstruct_photos
        result = reconstruct_photos(cap, capture_id=capture_id, log=log, timer=timer, **opts.get("photo", {}))
        views = result.pop("_views", None)
    else:
        cloud, info, views, scale_rel = reconstruct(cap, tier, opts, log, timer)
        plan = extract_plan(cloud, cloud.cam_positions, log=log)
        timer.lap("layout")
        err = ErrorModel(tier, scale_rel=scale_rel)
        result = plan_to_json(plan, tier, capture_id, err, extra={"reconstruction": info})
        result["_plan"] = plan
    if damage and views is not None and opts.get("damage", True):
        from .damage import analyse_damage
        analyse_damage(result, views, log=log, **opts.get("damage_opts", {}))
        timer.lap("damage")
    result.pop("_plan", None)
    timer.lap("serialise")
    result["pipeline_version"] = PIPELINE_VERSION
    result["timing_s"] = dict(timer.t, total=round(sum(timer.t.values()), 2))
    with open(os.path.join(out_dir, "plan.json"), "w") as f:
        json.dump(result, f, indent=1, default=_json_default)
    render_plan(result, os.path.join(out_dir, "plan.png"), os.path.join(out_dir, "plan.svg"))
    log(f"[{capture_id}] {len(result['rooms'])} rooms, "
        f"{result['property']['total_floor_area_m2']['value']:.2f} m2, {result['timing_s']['total']} s -> {out_dir}")
    return result


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(type(o))
