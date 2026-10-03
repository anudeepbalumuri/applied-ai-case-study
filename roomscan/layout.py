"""Room layout extraction from a gravity-aligned fused cloud.

Pipeline
  1. floor height + Manhattan yaw            -> "plan frame" (x, z horizontal, walls axis-aligned)
  2. 2D evidence maps (floor, walls, high walls/door headers)
  3. room segmentation: free space split by walls *including door headers* (band above 2.1 m),
     so doors separate rooms while header-less passages (open plan) do not
  4. per-room rectilinear polygon from the mask, then every edge snapped to the wall surface
     measured by LiDAR points (sub-cm: median over thousands of points)
  5. per-room floor / ceiling planes -> ceiling height
All measurements carry a standard deviation used later for confidence intervals.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

from .drift import dominant_yaw

RES = 0.025


@dataclass
class Edge:
    axis: str            # "x": wall at constant x (runs along z); "z": wall at constant z
    coord: float         # plan-frame coordinate of the wall surface
    lo: float            # extent along the wall
    hi: float
    inward: int          # +1/-1: direction of the room interior along the normal axis
    sigma: float = 0.02  # std of coord
    support: int = 0     # number of LiDAR points supporting the surface fit
    measured: bool = False


@dataclass
class Room:
    rid: int
    mask: np.ndarray
    edges: list = field(default_factory=list)
    floor_y: float = 0.0
    floor_sigma: float = 0.01
    ceil_y: float | None = None
    ceil_sigma: float | None = None
    ceiling_observed: bool = False
    openings: list = field(default_factory=list)
    name: str = ""


class PlanFrame:
    """World (x, y up, z) <-> plan (u, v) rotation by the dominant Manhattan yaw."""

    def __init__(self, yaw, floor_y):
        self.yaw = yaw
        self.floor_y = floor_y
        c, s = np.cos(-yaw), np.sin(-yaw)
        self.R = np.array([[c, -s], [s, c]])

    def to_plan(self, xz):
        return xz @ self.R.T

    def to_world(self, uv):
        return uv @ self.R


def floor_height(pts, normals):
    up = normals[:, 1] > 0.95
    y = pts[up, 1]
    h, e = np.histogram(y, bins=np.arange(y.min(), y.max() + 0.02, 0.01))
    # lowest strong peak (avoid table tops): first bin with >= 30% of the max mass
    hs = np.convolve(h, np.ones(3) / 3, mode="same")
    cand = np.where(hs >= 0.3 * hs.max())[0]
    i0 = cand[0]
    win = slice(max(i0 - 3, 0), i0 + 6)
    i = win.start + np.argmax(hs[win])
    c = e[i] + 0.005
    sel = y[np.abs(y - c) < 0.04]
    return float(np.median(sel))


def manhattan_yaw(pts, normals, floor_y):
    w = (np.abs(normals[:, 1]) < 0.2) & (pts[:, 1] > floor_y + 0.3)
    n2 = normals[w][:, [0, 2]]
    n2 /= np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-9)
    yaw, strength = dominant_yaw(n2)
    return yaw, strength


class Grid:
    def __init__(self, uv_min, uv_max, res=RES):
        self.o = np.floor(uv_min / res) * res - 0.5
        self.res = res
        self.shape = tuple((np.ceil((uv_max - self.o) / res) + 20).astype(int))   # (nu, nv)

    def idx(self, uv):
        return np.floor((uv - self.o) / self.res).astype(int)

    def center(self, ij):
        return self.o + (np.asarray(ij) + 0.5) * self.res

    def count(self, uv, w=None):
        ij = self.idx(uv)
        ok = np.all((ij >= 0) & (ij < self.shape), 1)
        m = np.zeros(self.shape, np.float32)
        np.add.at(m, (ij[ok, 0], ij[ok, 1]), 1.0 if w is None else w[ok])
        return m


def evidence_maps(cloud, frame: PlanFrame, grid: Grid | None = None, ceil_rel=None):
    p, n = cloud.pts, cloud.normals
    fy = frame.floor_y
    uv = frame.to_plan(p[:, [0, 2]])
    if grid is None:
        grid = Grid(uv.min(0), uv.max(0))
    rel = p[:, 1] - fy
    horiz = np.abs(n[:, 1]) < 0.3
    floor_pts = (n[:, 1] > 0.9) & (np.abs(rel) < 0.06)
    wall_full = horiz & (rel > 0.25) & (rel < 2.4)
    top = 2.75 if ceil_rel is None else min(ceil_rel - 0.08, 3.2)
    wall_hi = horiz & (rel > 2.12) & (rel < top)
    any_low = (rel > 0.08) & (rel < 1.9)
    maps = {
        "floor": grid.count(uv[floor_pts]),
        "wall": grid.count(uv[wall_full]),
        "wall_hi": grid.count(uv[wall_hi]),
        "obstacle": grid.count(uv[any_low]),
    }
    return grid, uv, maps


def _k(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def structural_walls(maps, res, min_run=1.2):
    """Wall-band cells that belong to long axis-aligned runs (walls), not furniture clutter.

    Full-band vertical surfaces include sofa backs, desks and shelves; walls are the ones that
    form straight runs of >= `min_run` metres (doors may cut a wall, so runs are measured on
    a lightly closed map), or that also appear in the >2.1 m band.
    """
    wall = (maps["wall"] >= 2).astype(np.uint8)
    wall = cv2.dilate(wall, np.ones((3, 3), np.uint8))
    n = max(3, int(min_run / res))
    run_u = cv2.morphologyEx(wall, cv2.MORPH_OPEN, np.ones((n, 1), np.uint8))
    run_v = cv2.morphologyEx(wall, cv2.MORPH_OPEN, np.ones((1, n), np.uint8))
    hi = cv2.dilate((maps["wall_hi"] >= 1).astype(np.uint8), np.ones((3, 3), np.uint8))
    keep = (run_u | run_v | (wall & hi)) > 0
    return keep & (maps["wall"] >= 1)


def close_door_gaps(wall, res, max_gap=1.05):
    """Bridge door-sized gaps between collinear wall runs.

    Captures that never look above ~2.1 m have no door-header evidence, so a doorway is just a gap
    in a wall line. Closing with 1-D line kernels bridges gaps only *along* a wall direction.
    """
    n = int(max_gap / res) + 1
    w = wall.astype(np.uint8)
    cu = cv2.morphologyEx(w, cv2.MORPH_CLOSE, np.ones((n, 1), np.uint8))
    cv_ = cv2.morphologyEx(w, cv2.MORPH_CLOSE, np.ones((1, n), np.uint8))
    return (cu | cv_) > 0


def segment_rooms(grid, maps, cam_uv, min_area=1.2, max_neck=1.3, neck_ratio=0.6):
    """Label map of rooms (0 = outside).

    Doors with headers are closed by the high-wall band. Header-less constrictions (open
    archways, corridor mouths) are split by a distance-transform watershed when the neck is
    narrower than both `max_neck` and `neck_ratio` x the inscribed diameter on *both* sides.
    """
    from skimage.segmentation import watershed
    from skimage.morphology import h_maxima

    res = grid.res
    floor = maps["floor"] > 0
    cam = np.zeros(grid.shape, bool)
    ij = grid.idx(cam_uv)
    ok = np.all((ij >= 0) & (ij < grid.shape), 1)
    cam[ij[ok, 0], ij[ok, 1]] = True
    interior0 = cv2.morphologyEx(floor.astype(np.uint8), cv2.MORPH_CLOSE, _k(int(0.3 / res))) > 0
    interior0 |= cv2.dilate(cam.astype(np.uint8), _k(int(0.2 / res))) > 0
    wall = structural_walls(maps, res)
    # door headers: wall points above 2.1 m close a doorway even though its low band is empty
    hi = maps["wall_hi"] >= 1
    barrier = cv2.dilate((close_door_gaps(wall, res) | hi).astype(np.uint8), _k(1)) > 0
    # furniture holes: anything enclosed by floor and walls is interior
    filled = ndimage.binary_fill_holes(interior0 | barrier) & ~barrier
    filled &= cv2.dilate(interior0.astype(np.uint8), _k(int(0.6 / res))) > 0
    space = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_OPEN, _k(int(0.12 / res))) > 0
    lab, nl = ndimage.label(space)

    out = np.zeros_like(lab)
    nxt = 1
    for l in range(1, nl + 1):
        comp = lab == l
        if comp.sum() * res * res < min_area:
            continue
        for sub in _split_necks(comp, res, max_neck, neck_ratio, watershed, h_maxima):
            if sub.sum() * res * res >= min_area:
                out[sub] = nxt
                nxt += 1
    # grow rooms into remaining interior cells (doorways, wall-adjacent furniture) without crossing walls
    grow_mask = (filled | interior0) & ~wall
    out = _geodesic_grow(out, grow_mask, iters=int(1.0 / res))
    return out, filled


def _split_necks(comp, res, max_neck, neck_ratio, watershed, h_maxima):
    dt = ndimage.distance_transform_edt(comp) * res
    q = np.round(dt / 0.01).astype(np.int32)
    mk = h_maxima(q, int(0.15 / 0.01))
    markers, nm = ndimage.label(mk)
    if nm <= 1:
        return [comp]
    ws = watershed(-dt, markers, mask=comp)
    regions = {i: ws == i for i in range(1, nm + 1)}
    diam = {i: 2 * dt[r].max() for i, r in regions.items()}
    # merge across boundaries that are not necks
    parent = {i: i for i in regions}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    kern = np.ones((3, 3), np.uint8)
    pairs = {}
    for i, r in regions.items():
        ring = (cv2.dilate(r.astype(np.uint8), kern) > 0) & ~r & comp
        for j in np.unique(ws[ring]):
            if j > i:
                pairs[(i, j)] = None
    for i, j in pairs:
        ri = cv2.dilate(regions[i].astype(np.uint8), kern) > 0
        border = ri & regions[j]
        # neck width ~ 2 x max DT along the shared border
        neck = 2 * dt[border].max() if border.any() else 0
        small = min(diam[i], diam[j])
        if not (neck < max_neck and neck < neck_ratio * small):
            parent[find(j)] = find(i)
    groups = {}
    for i in regions:
        groups.setdefault(find(i), []).append(i)
    return [np.logical_or.reduce([regions[i] for i in g]) for g in groups.values()]


def _geodesic_grow(lab, mask, iters):
    lab = lab.copy()
    kern = np.ones((3, 3), np.uint8)
    for _ in range(iters):
        grown = cv2.dilate(lab.astype(np.float32), kern).astype(lab.dtype)
        new = (lab == 0) & mask & (grown > 0)
        if not new.any():
            break
        lab[new] = grown[new]
    return lab


def rectilinear_polygon(mask, grid, eps=0.08):
    """Axis-aligned polygon (list of (u, v) vertices, CCW) approximating a binary room mask."""
    res = grid.res
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((int(0.3 / res),) * 2, np.uint8))
    m = ndimage.binary_fill_holes(m).astype(np.uint8)
    lab, nl = ndimage.label(m)
    if nl > 1:
        sizes = ndimage.sum(m, lab, range(1, nl + 1))
        m = (lab == 1 + int(np.argmax(sizes))).astype(np.uint8)
    cs, _ = cv2.findContours(m.T.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)   # (x=col=u idx, y=row=v idx) because of .T
    approx = cv2.approxPolyDP(c.astype(np.float32), eps / res, True)[:, 0, :].astype(float)
    pts = grid.o + (approx + 0.5) * res
    return _orthogonalize(pts, min_len=eps)


def _orthogonalize(pts, min_len=0.08):
    """Turn a simple polygon into an axis-aligned one by classifying edges as horizontal/vertical."""
    n = len(pts)
    edges = []
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        d = b - a
        L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        ax = "x" if abs(d[1]) >= abs(d[0]) else "z"     # x-wall: runs along v (constant u)
        coord = (a[0] + b[0]) / 2 if ax == "x" else (a[1] + b[1]) / 2
        edges.append([ax, coord, L])
    # merge consecutive edges with same axis (length-weighted coordinate)
    merged = []
    for e in edges:
        if merged and merged[-1][0] == e[0]:
            m = merged[-1]
            m[1] = (m[1] * m[2] + e[1] * e[2]) / (m[2] + e[2])
            m[2] += e[2]
        else:
            merged.append(list(e))
    if len(merged) > 1 and merged[0][0] == merged[-1][0]:
        m = merged.pop()
        f = merged[0]
        f[1] = (f[1] * f[2] + m[1] * m[2]) / (f[2] + m[2])
        f[2] += m[2]
    # drop tiny jogs repeatedly: an edge shorter than min_len between two parallel edges
    changed = True
    while changed and len(merged) > 4:
        changed = False
        for i in range(len(merged)):
            if merged[i][2] < min_len:
                prv, nxt = merged[i - 1], merged[(i + 1) % len(merged)]
                if prv[0] == nxt[0]:
                    tot = prv[2] + nxt[2]
                    prv[1] = (prv[1] * prv[2] + nxt[1] * nxt[2]) / tot
                    prv[2] = tot
                    j = (i + 1) % len(merged)
                    for idx in sorted([i, j], reverse=True):
                        merged.pop(idx)
                    changed = True
                    break
    # alternate axes now; vertices are intersections of consecutive edges
    verts = []
    for i in range(len(merged)):
        a, b = merged[i], merged[(i + 1) % len(merged)]
        if a[0] == b[0]:
            continue
        u = a[1] if a[0] == "x" else b[1]
        v = a[1] if a[0] == "z" else b[1]
        verts.append((u, v))
    verts = np.array(verts)
    if len(verts) >= 3 and _signed_area(verts) < 0:
        verts = verts[::-1]
    return verts


def _signed_area(v):
    x, y = v[:, 0], v[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def polygon_edges(verts):
    """Edges of a CCW axis-aligned polygon with inward direction."""
    edges = []
    n = len(verts)
    for i in range(n):
        a, b = verts[i], verts[(i + 1) % n]
        d = b - a
        if abs(d[0]) < 1e-9 and abs(d[1]) < 1e-9:
            continue
        if abs(d[0]) < abs(d[1]):            # vertical in plan: constant u, runs along v
            # CCW: interior on the left of direction d. left of (0, dv) is (-dv, 0)
            inward = -int(np.sign(d[1]))
            edges.append(Edge("x", a[0], min(a[1], b[1]), max(a[1], b[1]), inward))
        else:
            inward = int(np.sign(d[0]))       # left of (du, 0) is (0, du)
            edges.append(Edge("z", a[1], min(a[0], b[0]), max(a[0], b[0]), inward))
    return edges


def refine_edges(edges, uv, pts_y, normals_uv, rel_h, search=0.25, margin=0.12):
    """Snap each edge to the wall surface measured by LiDAR points.

    Uses wall points whose normal faces into the room, inside the edge's extent, high band first
    (furniture-free), picking the outermost dense layer of a 5 mm histogram (furniture sits inside
    the wall surface, never outside it).
    """
    for e in edges:
        k = 0 if e.axis == "x" else 1
        o = 1 - k
        n_al = normals_uv[:, k] * e.inward               # normal component pointing into the room
        along = uv[:, o]
        perp = uv[:, k]
        L = e.hi - e.lo
        mg = min(margin, 0.25 * L)
        base = (n_al > 0.85) & (along > e.lo + mg) & (along < e.hi - mg) & (np.abs(perp - e.coord) < search)
        for band in [(1.2, 2.6), (0.3, 2.6)]:
            m = base & (rel_h > band[0]) & (rel_h < band[1])
            if m.sum() >= 40:
                break
        if m.sum() < 40:
            e.sigma = 0.05
            e.measured = False
            continue
        # signed offset measured outward from the room
        off = (perp[m] - e.coord) * (-e.inward)
        h, b = np.histogram(off, bins=np.arange(-search, search + 0.005, 0.005))
        hs = np.convolve(h, [0.25, 0.5, 0.25], mode="same")
        strong = np.where(hs >= 0.35 * hs.max())[0]
        i_out = strong.max()                              # outermost dense layer
        lo_i = max(i_out - 3, 0)
        while lo_i > 0 and hs[lo_i - 1] >= 0.35 * hs[i_out] and i_out - lo_i < 6:
            lo_i -= 1
        c0 = b[i_out] + 0.0025
        sel = off[np.abs(off - c0) < 0.02]
        if len(sel) < 20:
            sel = off[np.abs(off - c0) < 0.04]
        cen = float(np.median(sel))
        mad = float(np.median(np.abs(sel - cen))) * 1.4826
        e.coord = e.coord + cen * (-e.inward)
        e.support = int(len(sel))
        # std of the surface estimate: point scatter / sqrt(effective n) with correlation inflation
        e.sigma = float(np.hypot(mad / np.sqrt(max(len(sel) / 25.0, 1.0)), 0.002))
        e.measured = True
    return edges


def merge_jogs(edges, tol=0.10):
    """Remove short connector edges whose two neighbours are parallel and nearly coplanar.

    After snapping, two parallel walls separated by a 1-4 cm 'jog' are almost always one wall
    measured with slightly different furniture occlusion; merge them (support-weighted).
    """
    changed = True
    while changed and len(edges) > 4:
        changed = False
        n = len(edges)
        for i in range(n):
            e = edges[i]
            a, b = edges[i - 1], edges[(i + 1) % n]
            if a.axis == b.axis and (e.hi - e.lo) < tol and abs(a.coord - b.coord) < tol and a.inward == b.inward:
                wa, wb = max(a.support, 1) * (a.hi - a.lo), max(b.support, 1) * (b.hi - b.lo)
                a.coord = (a.coord * wa + b.coord * wb) / (wa + wb)
                a.sigma = float(np.hypot(min(a.sigma, b.sigma), abs(a.coord - b.coord) / 4))
                a.support += b.support
                a.measured = a.measured or b.measured
                for j in sorted([i, (i + 1) % n], reverse=True):
                    edges.pop(j)
                changed = True
                break
    return edges


MIN_EDGE_SUPPORT = 40      # LiDAR points on an edge's own face for it to count as a measured wall
MAX_TONGUE_DEPTH = 0.6     # doorway notches / spurs deeper than this are kept (could be real alcoves)
MAX_JOG_DEPTH = 0.30


def collapse_unsupported(edges, min_support=MIN_EDGE_SUPPORT, max_tongue=MAX_TONGUE_DEPTH, max_jog=None):
    """Evidence-based polygon regularisation: an edge survives only if LiDAR points support it.

    Mask-contour polygons carry notches and jogs that are not walls (doorway cells grown into the
    room, furniture-shadowed floor, unobserved strips). They split real walls and move their
    endpoints differently in every capture. Two rules, iterated to a fixed point:

      jog    : unsupported edge e between parallel, same-facing neighbours a, b  ->  a and b are one
               wall: merge at the coordinate of the better-supported one (never averaged toward an
               unsupported side).
      tongue : unsupported edge e between parallel, opposite-facing neighbours that are both short
               (a doorway notch / spur) -> remove the tongue; the walls on either side of it merge.
    Real steps (niches, columns) keep their jog edge because that edge's own face is measured.
    """
    def unsupported(e):
        return (not e.measured) or e.support < min_support

    def weight(e):
        return max(e.support, 0) * max(e.hi - e.lo, 1e-3)

    def merge_into(a, b):
        """Merge parallel same-facing b into a (a keeps the better-supported coordinate)."""
        if weight(b) > weight(a):
            a.coord, a.sigma = b.coord, b.sigma
        a.support += b.support
        a.measured = a.measured or b.measured

    changed = True
    while changed and len(edges) > 4:
        changed = False
        n = len(edges)
        # most-unsupported first: shortest unsupported edge
        order = sorted(range(n), key=lambda i: (not unsupported(edges[i]), edges[i].hi - edges[i].lo))
        for i in order:
            e = edges[i]
            if not unsupported(e):
                break
            a, b = edges[i - 1], edges[(i + 1) % n]
            if a.axis != b.axis:
                continue
            if a.inward == b.inward and abs(a.coord - b.coord) <= (MAX_JOG_DEPTH if max_jog is None else max_jog):  # jog
                merge_into(a, b)
                for j in sorted({i, (i + 1) % n}, reverse=True):
                    edges.pop(j)
                changed = True
                break
            # tongue: a and b opposite-facing and short; their outer neighbours must be parallel
            if len(edges) >= 8 and (a.hi - a.lo) < max_tongue and (b.hi - b.lo) < max_tongue:
                pa, nb = edges[i - 2], edges[(i + 2) % n]
                if pa.axis == nb.axis and pa.inward == nb.inward and pa is not nb:
                    merge_into(pa, nb)
                    for j in sorted({(i - 1) % n, i, (i + 1) % n, (i + 2) % n}, reverse=True):
                        edges.pop(j)
                    changed = True
                    break
    return edges


def edges_to_verts(edges):
    verts = []
    n = len(edges)
    for i in range(n):
        a, b = edges[i], edges[(i + 1) % n]
        if a.axis == b.axis:
            continue
        u = a.coord if a.axis == "x" else b.coord
        v = a.coord if a.axis == "z" else b.coord
        verts.append((u, v))
    return np.array(verts)


def update_extents(edges):
    """Recompute each edge's [lo, hi] from neighbouring (refined) edges."""
    n = len(edges)
    for i, e in enumerate(edges):
        p, q = edges[i - 1], edges[(i + 1) % n]
        a, b = p.coord, q.coord
        e.lo, e.hi = min(a, b), max(a, b)
    return edges


def plane_height(y, sigma_floor=0.003):
    """Robust plane height from a set of point heights. Returns (height, std)."""
    h, e = np.histogram(y, bins=np.arange(y.min() - 0.01, y.max() + 0.02, 0.005))
    hs = np.convolve(h, [0.25, 0.5, 0.25], mode="same")
    c = e[np.argmax(hs)] + 0.0025
    sel = y[np.abs(y - c) < 0.025]
    med = float(np.median(sel))
    mad = float(np.median(np.abs(sel - med))) * 1.4826
    return med, float(np.hypot(mad / np.sqrt(max(len(sel) / 25.0, 1.0)), sigma_floor))


def room_planes(room_poly, frame, cloud, uv):
    """Per-room floor and ceiling heights from points inside the (shrunk) polygon."""
    from matplotlib.path import Path
    path = Path(room_poly)
    cen = room_poly.mean(0)
    shrunk = cen + (room_poly - cen) * 0.85
    inside = Path(shrunk).contains_points(uv)
    p, n = cloud.pts, cloud.normals
    rel = p[:, 1] - frame.floor_y
    fl = inside & (n[:, 1] > 0.95) & (np.abs(rel) < 0.12)
    ce = inside & (n[:, 1] < -0.95) & (rel > 1.9) & (rel < 4.5)
    floor = plane_height(p[fl, 1]) if fl.sum() > 100 else (frame.floor_y, 0.01)
    ceil = None
    if ce.sum() > 150:
        # ceiling: the highest strong horizontal layer (avoid lamps, soffits, cabinet bottoms)
        y = p[ce, 1]
        h, e = np.histogram(y, bins=np.arange(y.min() - 0.01, y.max() + 0.02, 0.01))
        strong = np.where(h >= 0.3 * h.max())[0]
        c = e[strong.max()] + 0.005
        sel = y[np.abs(y - c) < 0.06]
        if len(sel) > 80:
            ceil = plane_height(sel)
    wall_top = None
    wm = inside | (~inside)   # any wall points near this room (used as lower bound when no ceiling)
    near = Path(cen + (room_poly - cen) * 1.05).contains_points(uv) & (np.abs(n[:, 1]) < 0.3)
    if near.sum() > 50:
        wall_top = float(np.percentile(p[near, 1], 99.5))
    return floor, ceil, wall_top
