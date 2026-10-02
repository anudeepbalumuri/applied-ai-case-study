"""Plan objects -> output JSON (schema/plan.schema.json)."""
from __future__ import annotations

import numpy as np

from .uncertainty import ErrorModel

SCHEMA_VERSION = "1.0.0"


def _r(x, d=4):
    return [round(float(v), d) for v in x]


def room_name(room_dict):
    """Heuristic name from shape (connectors are long and narrow)."""
    w = room_dict["_bbox"]
    short, long_ = min(w), max(w)
    if short < 1.6 and long_ / max(short, 1e-6) > 2.2:
        return "Hallway"
    if room_dict["floor_area_m2"]["value"] < 3.0:
        return "Small room"
    return "Room"


def plan_to_json(plan, tier, capture_id, err: ErrorModel, extra=None):
    rooms_out = []
    for ri, room in enumerate(plan["rooms"]):
        v = room.verts
        edges = room.edges
        n = len(edges)
        walls = []
        edge_terms = []
        for i, e in enumerate(edges):
            prv, nxt = edges[i - 1], edges[(i + 1) % n]
            if e.axis == "x":
                a, b = np.array([e.coord, prv.coord]), np.array([e.coord, nxt.coord])
                inward = [float(e.inward), 0.0]
            else:
                a, b = np.array([prv.coord, e.coord]), np.array([nxt.coord, e.coord])
                inward = [0.0, float(e.inward)]
            L = float(np.linalg.norm(b - a))
            edge_terms.append((L, e.sigma))
            walls.append({
                "id": f"R{ri}-W{i}", "start": _r(a), "end": _r(b), "inward_normal": inward,
                "length_m": err.wall_length(L, prv.sigma, nxt.sigma),
                "measured_from_points": bool(e.measured), "support_points": int(e.support),
                "surface_id": f"R{ri}-W{i}",
            })
        A = abs(0.5 * (np.dot(v[:, 0], np.roll(v[:, 1], -1)) - np.dot(np.roll(v[:, 0], -1), v[:, 1])))
        H = (room.ceil_y - room.floor_y) if room.ceil_y is not None else None
        openings = []
        for k, o in enumerate(room.openings):
            e = edges[o.wall_index]
            if e.axis == "x":
                s, t = [e.coord, o.u0], [e.coord, o.u1]
                inward = [float(e.inward), 0.0]
            else:
                s, t = [o.u0, e.coord], [o.u1, e.coord]
                inward = [0.0, float(e.inward)]
            openings.append({
                "id": f"R{ri}-O{k}", "kind": o.kind, "wall_id": f"R{ri}-W{o.wall_index}",
                "start": _r(s), "end": _r(t), "inward_normal": inward,
                "width_m": err.opening_width(o.width, o.width_sigma),
                "height_m": err.generic(o.top - o.bottom, 0.02, "opening_height"),
                "sill_height_m": round(float(o.bottom), 3),
                "connects_to_room": (f"R{o.connects}" if o.connects is not None else None),
                "evidence": o.evidence,
            })
        perim = sum(l for l, _ in edge_terms)
        rd = {
            "id": f"R{ri}", "name": "", "polygon": [_r(p) for p in v],
            "floor_area_m2": err.area(A, edge_terms),
            "perimeter_m": err.generic(perim, np.sqrt(sum(s ** 2 for _, s in edge_terms)), "perimeter"),
            "ceiling_height_m": err.height(H, room.floor_sigma, room.ceil_sigma or 0.0, room.ceiling_observed)
            if H is not None else err.height(None, 0, 0),
            "walls": walls, "openings": openings,
            "surfaces": (
                [{"id": w["id"], "type": "wall", "area_m2": round(w["length_m"]["value"] * (H or 2.4), 3)} for w in walls]
                + [{"id": f"R{ri}-F", "type": "floor", "area_m2": round(A, 3)},
                   {"id": f"R{ri}-C", "type": "ceiling", "area_m2": round(A, 3)}]),
            "observation": {"operator_entered": bool(getattr(room, "entered", True)),
                            "measured_wall_fraction": round(float(getattr(room, "measured_wall_frac", 1.0)), 3),
                            "partial": bool(getattr(room, "partial", False))},
            "damage": [], "concealed_damage_flags": [], "scope": [],
            "_bbox": (v.max(0) - v.min(0)).tolist(),
        }
        rooms_out.append(rd)
    # names
    counts = {}
    for rd in rooms_out:
        base = room_name(rd)
        counts[base] = counts.get(base, 0) + 1
        rd["name"] = f"{base} {counts[base]}"
        del rd["_bbox"]
    adjacency = [{"a": f"R{i}", "b": f"R{j}", "via": via} for (i, j), via in sorted(plan["adjacency"].items())]
    tot_A = sum(r["floor_area_m2"]["value"] for r in rooms_out)
    tot_s = np.sqrt(sum(r["floor_area_m2"]["sigma"] ** 2 for r in rooms_out)) if rooms_out else 0.0
    allv = np.concatenate([np.array(r["polygon"]) for r in rooms_out]) if rooms_out else np.zeros((1, 2))
    bb = allv.max(0) - allv.min(0)
    out = {
        "schema_version": SCHEMA_VERSION,
        "capture_id": capture_id,
        "tier": tier,
        "units": "metres; plan frame: gravity-aligned, x/y horizontal axes aligned with the dominant walls",
        "property": {
            "room_count": len(rooms_out),
            "total_floor_area_m2": err.measure(tot_A, tot_s, "area", "m2", 3),
            "footprint_bbox_m": {"x": err.generic(bb[0], 0.01, "footprint"), "y": err.generic(bb[1], 0.01, "footprint")},
            "adjacency": adjacency,
            "manhattan_strength": round(plan.get("manhattan_strength", 0.0), 3),
        },
        "rooms": rooms_out,
        "scope_line_items": [],
    }
    if extra:
        out.update(extra)
    return out
