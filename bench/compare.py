"""Compare two plan JSONs of the same space (repeatability, cross-tier agreement).

Plans live in arbitrary frames (each capture has its own world origin), so we
  1. try the 8 axis-aligned rotations/reflections-free rotations (4 x 90 deg) of plan B,
  2. find the translation maximising overlap of rasterised room masks (phase correlation),
  3. refine with ICP on wall segments,
then match rooms by IoU and walls by axis / inward side / position, and report per-wall deltas.
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np
from matplotlib.path import Path

RES = 0.02


def _load(p):
    return json.load(open(p)) if isinstance(p, str) else p


def _rot(k):
    a = np.pi / 2 * k
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _raster(polys, lo, shape):
    img = np.zeros(shape, np.uint8)
    for v in polys:
        q = np.round((np.asarray(v) - lo) / RES).astype(np.int32)
        cv2.fillPoly(img, [q[:, ::-1].reshape(-1, 1, 2)], 1)
    return img


def _wall_pts(plan, R=np.eye(2), t=np.zeros(2), step=0.05):
    pts = []
    for r in plan["rooms"]:
        for w in r["walls"]:
            a, b = np.array(w["start"]) @ R.T + t, np.array(w["end"]) @ R.T + t
            n = max(2, int(np.linalg.norm(b - a) / step))
            pts.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    return np.concatenate(pts)


def align(planA, planB, include_partial=False):
    """Return (R, t) mapping plan B coordinates into plan A, and the mask IoU."""
    def polys(pl):
        return [np.array(r["polygon"]) for r in pl["rooms"]
                if include_partial or not r.get("observation", {}).get("partial", False)]
    PA = polys(planA)
    best = None
    for k in range(4):
        R = _rot(k)
        PB = [v @ R.T for v in polys(planB)]
        allp = np.concatenate(PA + PB)
        lo = allp.min(0) - 2.0
        span = allp.max(0) - allp.min(0) + 4.0
        shape = tuple(int(s) for s in np.ceil(span / RES))
        A = _raster(PA, lo, shape).astype(np.float32)
        B = _raster(PB, lo, shape).astype(np.float32)
        # translation maximising mask overlap: FFT cross-correlation (2A-1 vs B penalises spill-over)
        from scipy.signal import fftconvolve
        cc = fftconvolve(2 * A - 1, B[::-1, ::-1], mode="same")
        i, j = np.unravel_index(np.argmax(cc), cc.shape)
        t = (np.array([i, j]) - np.array(shape) // 2) * RES
        PBt = [v + t for v in PB]
        Bt = _raster(PBt, lo, shape)
        iou = (A.astype(bool) & Bt.astype(bool)).sum() / max((A.astype(bool) | Bt.astype(bool)).sum(), 1)
        if best is None or iou > best[2]:
            best = (R, t, iou)
    R, t, iou = best
    # both plans are Manhattan-snapped, so the residual is a pure translation: refine it by
    # iterated median nearest-wall offsets
    from scipy.spatial import cKDTree
    WA = _wall_pts(planA)
    tree = cKDTree(WA)
    Rf = R
    # recompute translation with fixed rotation (closed form on matched points)
    for _ in range(15):
        WB = _wall_pts(planB, Rf, t)
        d, j = tree.query(WB, distance_upper_bound=0.3)
        m = np.isfinite(d)
        if m.sum() < 20:
            break
        t = t + np.median(WA[j[m]] - WB[m], axis=0)
    return Rf, t, float(iou)


def _xform_room(r, R, t):
    out = {"polygon": [list(np.array(p) @ R.T + t) for p in r["polygon"]], "walls": []}
    for w in r["walls"]:
        a, b = np.array(w["start"]) @ R.T + t, np.array(w["end"]) @ R.T + t
        out["walls"].append(dict(w, start=list(a), end=list(b), inward_normal=list(np.array(w["inward_normal"]) @ R.T)))
    return out


def _iou(pa, pb):
    allp = np.concatenate([pa, pb])
    lo = allp.min(0) - 0.1
    shape = tuple(int(s) for s in np.ceil((allp.max(0) - lo + 0.1) / RES))
    A, B = _raster([pa], lo, shape).astype(bool), _raster([pb], lo, shape).astype(bool)
    return (A & B).sum() / max((A | B).sum(), 1)


def compare(planA, planB, room_iou=0.5, wall_tol=0.15):
    planA, planB = _load(planA), _load(planB)
    R, t, iou = align(planA, planB)
    rooms = []
    usedB = set()
    for ia, ra in enumerate(planA["rooms"]):
        pa = np.array(ra["polygon"])
        best = (None, 0)
        for ib, rb in enumerate(planB["rooms"]):
            if ib in usedB:
                continue
            pb = np.array(rb["polygon"]) @ R.T + t
            v = _iou(pa, pb)
            if v > best[1]:
                best = (ib, v)
        if best[0] is None or best[1] < room_iou:
            continue
        ib = best[0]
        usedB.add(ib)
        rb = _xform_room(planB["rooms"][ib], R, t)
        walls = []
        for wa in ra["walls"]:
            na = np.array(wa["inward_normal"])
            a0, a1 = np.array(wa["start"]), np.array(wa["end"])
            for wb in rb["walls"]:
                nb = np.array(wb["inward_normal"])
                if na @ nb < 0.9:
                    continue
                b0, b1 = np.array(wb["start"]), np.array(wb["end"])
                ax = int(np.argmax(np.abs(na)))
                if abs(a0[ax] - b0[ax]) > wall_tol:
                    continue
                o = 1 - ax
                ov = min(max(a0[o], a1[o]), max(b0[o], b1[o])) - max(min(a0[o], a1[o]), min(b0[o], b1[o]))
                la, lb = wa["length_m"]["value"], wb["length_m"]["value"]
                if ov < 0.5 * min(la, lb) or abs(la - lb) > max(0.25, 0.2 * la):
                    continue
                walls.append({"wall_a": wa["id"], "wall_b": wb["id"], "len_a": la, "len_b": lb,
                              "delta_m": round(lb - la, 4), "delta_pct": round(100 * (lb - la) / la, 3),
                              "pass_1cm_or_0.5pct": bool(abs(lb - la) <= max(0.01, 0.005 * la))})
                break
        ha = ra["ceiling_height_m"]["value"]
        hb = planB["rooms"][ib]["ceiling_height_m"]["value"]
        rooms.append({"room_a": ra["id"], "room_b": planB["rooms"][ib]["id"], "iou": round(best[1], 3),
                      "area_a": ra["floor_area_m2"]["value"], "area_b": planB["rooms"][ib]["floor_area_m2"]["value"],
                      "ceiling_a": ha, "ceiling_b": hb,
                      "ceiling_spread_m": (round(abs(ha - hb), 4) if ha is not None and hb is not None else None),
                      "walls": walls})
    allw = [w for r in rooms for w in r["walls"]]
    summary = {
        "alignment_iou": round(iou, 3),
        "rooms_a": len(planA["rooms"]), "rooms_b": len(planB["rooms"]), "rooms_matched": len(rooms),
        "walls_matched": len(allw),
        "walls_pass": sum(w["pass_1cm_or_0.5pct"] for w in allw),
        "median_abs_delta_m": round(float(np.median([abs(w["delta_m"]) for w in allw])), 4) if allw else None,
        "p90_abs_delta_m": round(float(np.percentile([abs(w["delta_m"]) for w in allw], 90)), 4) if allw else None,
        "total_area_a": planA["property"]["total_floor_area_m2"]["value"],
        "total_area_b": planB["property"]["total_floor_area_m2"]["value"],
    }
    return {"summary": summary, "rooms": rooms, "transform_b_to_a": {"R": R.tolist(), "t": t.tolist()}}


if __name__ == "__main__":
    res = compare(sys.argv[1], sys.argv[2])
    print(json.dumps(res["summary"], indent=1))
    for r in res["rooms"]:
        print(r["room_a"], r["room_b"], "iou", r["iou"], "area", r["area_a"], r["area_b"], "ceil", r["ceiling_a"], r["ceiling_b"])
        for w in r["walls"]:
            print("   ", w["wall_a"], w["len_a"], w["len_b"], w["delta_m"], "PASS" if w["pass_1cm_or_0.5pct"] else "fail")
