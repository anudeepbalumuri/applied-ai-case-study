"""Tier-agnostic plan extraction: fused cloud -> rooms, walls, ceiling, openings, adjacency."""
from __future__ import annotations

import numpy as np
from matplotlib.path import Path
from scipy.spatial import cKDTree

from . import layout as L
from .openings import detect_openings


def _poly_area(v):
    return abs(L._signed_area(v))


def extract_plan(cloud, cam_positions, log=print, min_room_area=1.2):
    p, n = cloud.pts, cloud.normals
    fy = L.floor_height(p, n)
    yaw, strength = L.manhattan_yaw(p, n, fy)
    frame = L.PlanFrame(yaw, fy)
    grid, uv, maps = L.evidence_maps(cloud, frame)
    cam_uv = frame.to_plan(cam_positions[:, [0, 2]])
    lab, interior = L.segment_rooms(grid, maps, cam_uv, min_area=min_room_area)
    rel = p[:, 1] - fy
    n_uv = frame.to_plan(n[:, [0, 2]])
    rooms = []
    for r in range(1, lab.max() + 1):
        mask = lab == r
        verts = L.rectilinear_polygon(mask, grid)
        if len(verts) < 4 or _poly_area(verts) < min_room_area:
            continue
        edges = L.polygon_edges(verts)
        # refine twice: coordinates first, then extents from refined neighbours
        L.refine_edges(edges, uv, p[:, 1], n_uv, rel)
        L.update_extents(edges)
        L.refine_edges(edges, uv, p[:, 1], n_uv, rel)
        L.update_extents(edges)
        L.merge_jogs(edges)
        L.update_extents(edges)
        verts = L.edges_to_verts(edges)
        if len(verts) < 4 or _poly_area(verts) < min_room_area:
            continue
        room = L.Room(len(rooms), mask, edges)
        (room.floor_y, room.floor_sigma), ceil, wall_top = L.room_planes(verts, frame, cloud, uv)
        if ceil is not None:
            room.ceil_y, room.ceil_sigma = ceil
            room.ceiling_observed = True
        else:
            room.ceil_y = wall_top
            room.ceiling_observed = False
        room.verts = verts
        # observation completeness: did the operator enter the room, how much wall was measured
        inside_cam = Path(verts).contains_points(cam_uv)
        perim = sum(e.hi - e.lo for e in edges)
        room.measured_wall_frac = sum(e.hi - e.lo for e in edges if e.measured) / max(perim, 1e-6)
        room.entered = bool(inside_cam.any())
        room.partial = (not room.entered) and room.measured_wall_frac < 0.75
        rooms.append(room)
    log(f"plan: {len(rooms)} rooms")

    # openings (need in-room point trees for the mirror test)
    for room in rooms:
        path = Path(room.verts)
        inside = path.contains_points(uv)
        tree = cKDTree(np.c_[uv[inside], rel[inside]][::3]) if inside.sum() > 100 else None
        ceil_rel = (room.ceil_y - room.floor_y) if room.ceiling_observed else None
        room.openings = detect_openings(room.edges, uv, rel - (room.floor_y - fy), ceil_rel, tree)
    _link_openings(rooms)
    adjacency = _adjacency(rooms)
    return {"frame": frame, "grid": grid, "rooms": rooms, "adjacency": adjacency,
            "manhattan_strength": float(strength), "labels": lab}


def _opening_mid(room, o):
    e = room.edges[o.wall_index]
    m = (o.u0 + o.u1) / 2
    return np.array([e.coord, m]) if e.axis == "x" else np.array([m, e.coord]), e


def _link_openings(rooms):
    """An opening connects to the room whose polygon contains the point just beyond it."""
    paths = [Path(r.verts) for r in rooms]
    for i, r in enumerate(rooms):
        for o in r.openings:
            mid, e = _opening_mid(r, o)
            k = 0 if e.axis == "x" else 1
            probe = mid.copy()
            for d in (0.35, 0.6, 0.9):
                probe[k] = e.coord - e.inward * d
                hits = [j for j, pa in enumerate(paths) if j != i and pa.contains_point(probe)]
                if hits:
                    o.connects = hits[0]
                    break


def _adjacency(rooms, max_gap=0.45, min_overlap=0.3):
    adj = {}
    for i, a in enumerate(rooms):
        for o in a.openings:
            if o.connects is not None:
                key = tuple(sorted((i, o.connects)))
                adj.setdefault(key, set()).add(f"{o.kind}")
    for i, a in enumerate(rooms):
        for j in range(i + 1, len(rooms)):
            b = rooms[j]
            for ea in a.edges:
                for eb in b.edges:
                    if ea.axis != eb.axis or ea.inward == eb.inward:
                        continue
                    gap = (ea.coord - eb.coord) * (-ea.inward)     # b's wall beyond a's wall
                    ov = min(ea.hi, eb.hi) - max(ea.lo, eb.lo)
                    if 0 < gap < max_gap and ov > min_overlap:
                        adj.setdefault((i, j), set()).add("shared_wall")
    return {k: sorted(v) for k, v in adj.items()}
