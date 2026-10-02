"""Opening (door / window / passage) detection on walls.

For each refined wall we build an elevation occupancy image (along-wall u x height h) from points
within a slab around the wall plane. A column run with no wall surface in the 0.9-1.8 m band is an
opening *candidate*. Candidates must then pass two evidence tests that distinguish a real opening
from a hole in the data:

  see-through : points exist beyond the wall plane inside the gap (rays went through it).
                Unobserved wall patches have nothing behind them -> rejected (no phantom).
  mirror test : points behind a mirror are reflections of the room; reflecting them back about the
                wall plane lands them on in-room geometry. High agreement -> mirror, not opening.

Width is measured jamb-to-jamb from the point distribution at sub-cell precision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

CELL = 0.02


@dataclass
class Opening:
    kind: str            # door | window | passage
    wall_index: int
    u0: float            # along-wall start (plan coordinate along the wall)
    u1: float
    bottom: float        # height above floor
    top: float
    width_sigma: float
    evidence: dict
    connects: int | None = None

    @property
    def width(self):
        return self.u1 - self.u0

    @property
    def height(self):
        return self.top - self.bottom


def _runs(mask):
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def detect_openings(edges, uv, rel_h, ceil_rel, room_tree=None, slab=0.10, behind=(0.15, 3.5),
                    min_width=0.45, max_width=2.6):
    """Detect openings on each wall edge of one room.

    edges   : refined Edge list (plan frame); uv: (N,2) plan coords of all points; rel_h: heights above floor
    ceil_rel: ceiling height above floor (or None)
    room_tree: cKDTree over in-room points (u, v, h) for the mirror test
    """
    out = []
    top_h = (ceil_rel - 0.05) if ceil_rel else 2.5
    for wi, e in enumerate(edges):
        k = 0 if e.axis == "x" else 1
        o = 1 - k
        L = e.hi - e.lo
        if L < min_width + 0.1:
            continue
        perp = (uv[:, k] - e.coord) * (-e.inward)          # >0 : outside the room (behind wall)
        along = uv[:, o]
        inext = (along > e.lo + 0.03) & (along < e.hi - 0.03)
        on = inext & (np.abs(perp) < slab) & (rel_h > 0.02) & (rel_h < top_h)
        nb = int(np.ceil(L / CELL))
        nh = int(np.ceil(top_h / CELL))
        iu = np.clip(((along[on] - e.lo) / CELL).astype(int), 0, nb - 1)
        ih = np.clip((rel_h[on] / CELL).astype(int), 0, nh - 1)
        occ = np.zeros((nb, nh), bool)
        occ[iu, ih] = True
        b0, b1 = int(0.9 / CELL), int(1.8 / CELL)
        frac = occ[:, b0:b1].mean(1)
        empty = frac < 0.08
        # bridge single-column noise
        for i in range(1, nb - 1):
            if not empty[i] and empty[i - 1] and empty[i + 1]:
                empty[i] = True
        beh = inext & (perp > behind[0]) & (perp < behind[1]) & (rel_h > 0.1) & (rel_h < top_h)
        for s, t in _runs(empty):
            w = (t - s) * CELL
            if w < min_width or w > max_width:
                continue
            ua, ub = e.lo + s * CELL, e.lo + t * CELL
            # vertical extent of the gap: lowest/highest empty rows over the run
            col = occ[s:t].mean(0)
            rows_empty = col < 0.1
            # bottom: first occupied row band below 0.9 m (sill), else floor
            bottom = 0.0
            for r in range(b0, -1, -1):
                if not rows_empty[r]:
                    bottom = (r + 1) * CELL
                    break
            top = top_h
            for r in range(b1, nh):
                if not rows_empty[r]:
                    top = r * CELL
                    break
            # evidence: points seen through the gap
            in_gap = beh & (along > ua + 0.05) & (along < ub - 0.05) & (rel_h > bottom + 0.05) & (rel_h < top - 0.05)
            n_through = int(in_gap.sum())
            area = max((ub - ua - 0.1) * (top - bottom - 0.1), 0.05)
            density = n_through / area
            mirror_frac = 0.0
            if room_tree is not None and n_through > 30:
                q = np.c_[uv[in_gap], rel_h[in_gap]]
                # reflect about the wall plane
                q[:, k] = e.coord - (q[:, k] - e.coord)
                d, _ = room_tree.query(q[:: max(1, len(q) // 400)], distance_upper_bound=0.05)
                mirror_frac = float(np.isfinite(d).mean())
            ev = {"see_through_points": n_through, "see_through_density_per_m2": round(density, 1),
                  "mirror_reflection_agreement": round(mirror_frac, 2), "gap_columns": t - s}
            if density < 40:
                continue                      # unobserved hole, not an opening
            if mirror_frac > 0.5:
                ev["rejected_as"] = "mirror"
                continue
            # jamb refinement from wall-surface points next to the gap
            ua_r, sa = _refine_jamb(along[on], rel_h[on], ua, side=-1, bottom=bottom, top=top)
            ub_r, sb = _refine_jamb(along[on], rel_h[on], ub, side=+1, bottom=bottom, top=top)
            kind = "door" if bottom < 0.12 and top > 1.85 else ("window" if bottom >= 0.3 else "passage")
            if kind == "door" and top >= top_h - 0.02:
                kind = "passage"              # full-height opening with no header
            out.append(Opening(kind, wi, ua_r, ub_r, bottom, top, float(np.hypot(sa, sb)), ev))
    return out


def _refine_jamb(along, h, edge, side, bottom, top, win=0.08):
    """Jamb position: the extreme wall-surface point adjacent to the gap (robust percentile)."""
    band = (h > max(bottom, 0.1) + 0.1) & (h < min(top, 2.0) - 0.1)
    if side < 0:
        m = band & (along > edge - win) & (along <= edge + 0.02)
        if m.sum() < 8:
            return edge, CELL / 2
        a = along[m]
        v = np.percentile(a, 97)
    else:
        m = band & (along < edge + win) & (along >= edge - 0.02)
        if m.sum() < 8:
            return edge, CELL / 2
        a = along[m]
        v = np.percentile(a, 3)
    # spread of the extreme points ~ edge sharpness (flying pixels, frame mouldings)
    s = float(np.clip(np.std(a[np.abs(a - v) < 0.02]), 0.002, 0.02)) if m.sum() > 8 else 0.01
    return float(v), s
