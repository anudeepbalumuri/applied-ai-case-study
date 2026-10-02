"""Gate evaluation against tape / laser ground truth.

Ground truth is attached to the elements of a frozen *reference plan* of each physical space
(bench/reference/<space>.plan.json). Any run of any tier is aligned to the reference plan with
bench.compare, its matched elements are scored against the tape value, and gates are applied.

ground_truth/<space>.json:
{
  "space": "home_floor",
  "reference_plan": "bench/reference/home_floor.plan.json",
  "instrument": "Bosch GLM 50 laser (+-1.5 mm)",
  "measurements": [
    {"ref_id": "R3-W2", "type": "wall_length",    "tape_m": 3.412},
    {"ref_id": "R3",    "type": "ceiling_height", "tape_m": 2.468},
    {"ref_id": "R3-O1", "type": "opening_width",  "tape_m": 0.812},
    {"ref_id": null,    "type": "opening_width",  "tape_m": 0.78, "note": "door the pipeline missed (R5 north wall)"}
  ],
  "phantom_openings": ["R2-O3"]
}
Generate an empty template from a reference plan:  python -m bench.gates template <plan.json> <space>
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .compare import compare

GATES = {
    "lidar": {"wall_abs": 0.01, "wall_rel": 0.005, "ceiling": 0.015, "opening": 0.02, "opening_rate": 0.85},
    "video": {"wall_rel": 0.03, "ceiling": None, "opening": None},
    "photo": {"wall_rel": 0.08, "footprint_rel": 0.08},
}


def template(plan_path, space):
    P = json.load(open(plan_path))
    ms = []
    for r in P["rooms"]:
        ms.append({"ref_id": r["id"], "type": "ceiling_height", "tape_m": None, "room_name": r["name"]})
        for w in r["walls"]:
            if w["length_m"]["value"] >= 0.3:
                ms.append({"ref_id": w["id"], "type": "wall_length", "tape_m": None,
                           "pipeline_m": w["length_m"]["value"]})
        for o in r["openings"]:
            ms.append({"ref_id": o["id"], "type": "opening_width", "tape_m": None, "kind": o["kind"],
                       "pipeline_m": o["width_m"]["value"]})
    return {"space": space, "reference_plan": plan_path, "instrument": None, "measurements": ms,
            "phantom_openings": [], "_instructions": "fill tape_m (metres); add missed openings with ref_id null; "
            "list pipeline openings that do not exist under phantom_openings; delete rows you cannot measure"}


def _index(plan):
    idx = {}
    for r in plan["rooms"]:
        idx[r["id"]] = ("room", r)
        for w in r["walls"]:
            idx[w["id"]] = ("wall", w)
        for o in r["openings"]:
            idx[o["id"]] = ("opening", o)
    return idx


def evaluate(plan, gt, tier):
    """Score a plan of any tier against ground truth. Returns per-measurement rows and gate summary."""
    ref = json.load(open(gt["reference_plan"]))
    same = plan.get("capture_id") == ref.get("capture_id") and plan.get("tier") == ref.get("tier")
    if same:
        wall_map = {w["id"]: w["id"] for r in ref["rooms"] for w in r["walls"]}
        room_map = {r["id"]: r["id"] for r in ref["rooms"]}
        open_map = {o["id"]: o["id"] for r in ref["rooms"] for o in r["openings"]}
    else:
        c = compare(ref, plan)
        wall_map = {w["wall_a"]: w["wall_b"] for rr in c["rooms"] for w in rr["walls"]}
        room_map = {rr["room_a"]: rr["room_b"] for rr in c["rooms"]}
        open_map = {o["opening_a"]: o["opening_b"] for o in c["openings"]}
    idx = _index(plan)
    rows = []
    for m in gt["measurements"]:
        if m.get("tape_m") is None:
            continue
        t, rid = m["type"], m.get("ref_id")
        key = {"wall_length": wall_map, "ceiling_height": room_map, "opening_width": open_map}[t].get(rid) if rid else None
        val = ci = None
        if key and key in idx:
            kind, el = idx[key]
            q = {"wall_length": "length_m", "ceiling_height": "ceiling_height_m", "opening_width": "width_m"}[t]
            val, ci = el[q]["value"], el[q]["ci95"]
        row = {"ref_id": rid, "type": t, "tape_m": m["tape_m"], "pred_m": val, "ci95": ci,
               "err_m": (round(val - m["tape_m"], 4) if val is not None else None),
               "covered": (bool(ci[0] <= m["tape_m"] <= ci[1]) if ci else None), "detected": val is not None}
        rows.append(row)
    g = GATES[tier]
    out = {"tier": tier, "rows": rows}
    walls = [r for r in rows if r["type"] == "wall_length" and r["detected"]]
    if walls:
        if tier == "lidar":
            ok = [abs(r["err_m"]) <= max(g["wall_abs"], g["wall_rel"] * r["tape_m"]) for r in walls]
        else:
            ok = [abs(r["err_m"]) <= g["wall_rel"] * r["tape_m"] for r in walls]
        out["wall_pass_rate"] = round(float(np.mean(ok)), 3)
    ceil = [r for r in rows if r["type"] == "ceiling_height" and r["detected"]]
    if ceil and g.get("ceiling"):
        out["ceiling_max_abs_err_m"] = round(max(abs(r["err_m"]) for r in ceil), 4)
        out["ceiling_gate"] = out["ceiling_max_abs_err_m"] <= g["ceiling"]
    ops = [r for r in rows if r["type"] == "opening_width"]
    if ops and g.get("opening"):
        good = sum(1 for r in ops if r["detected"] and abs(r["err_m"]) <= g["opening"])
        total = len(ops) + len(gt.get("phantom_openings", []))      # misses and phantoms both count
        out["opening_rate"] = round(good / max(total, 1), 3)
        out["opening_gate"] = out["opening_rate"] >= g["opening_rate"]
    cov = [r["covered"] for r in rows if r["covered"] is not None]
    if cov:
        out["ci95_coverage"] = round(float(np.mean(cov)), 3)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "template":
        print(json.dumps(template(sys.argv[2], sys.argv[3]), indent=1))
    else:
        plan, gt = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
        print(json.dumps(evaluate(plan, gt, plan["tier"]), indent=1))
