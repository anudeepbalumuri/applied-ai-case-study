"""Benchmark orchestrator: run every configured capture x tier, then write results.json + BENCHMARK.md.

    python -m bench.run_benchmark            # run missing outputs, then report
    python -m bench.run_benchmark --force    # rerun everything
    python -m bench.run_benchmark --report   # report only (no pipeline runs)

Everything reported is regenerated from data/raw (+ data/photo_proxy built by bench.make_photo_tier).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

from .compare import compare

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "bench")
RAW = {
    "floor_only": "data/raw/floor_only/1a8384c3f6",
    "with_ceiling": "data/raw/with_ceiling/c7d28f72c6",
    "single_room": "data/raw/single_room/c00a170fe1",
}
# space -> captures of that space (same rooms); first is the reference (observes ceilings)
SPACES = {"home_floor": ["with_ceiling", "floor_only"], "single_room": ["single_room"]}
RUNS = (
    [(f"lidar_{c}", RAW[c], "lidar", []) for c in RAW]
    + [(f"lidar_nodrift_{c}", RAW[c], "lidar", ["--no-drift"]) for c in RAW]
    + [(f"video_{c}", RAW[c], "video", []) for c in RAW]
    + [(f"photo_{c}", f"data/photo_proxy/{c}", "photo", []) for c in RAW]
)


def run_all(force=False, only=None):
    env = dict(os.environ, HF_HOME=os.path.join(ROOT, "weights", "hf"), TORCH_HOME=os.path.join(ROOT, "weights", "torch"),
               GLOG_minloglevel="2")
    for name, path, tier, extra in RUNS:
        if only and not any(o in name for o in only):
            continue
        out = os.path.join(OUT, name)
        if os.path.exists(os.path.join(out, "plan.json")) and not force:
            continue
        if tier == "photo" and not os.path.isdir(os.path.join(ROOT, path)):
            src = name.replace("photo_", "")
            subprocess.run([sys.executable, "-m", "bench.make_photo_tier", RAW[src],
                            os.path.join(OUT, f"lidar_{src}", "plan.json"), path], cwd=ROOT, env=env, check=True)
        cmd = [sys.executable, "-m", "roomscan", path, "--tier", tier, "--id", name, "--out", out] + extra
        print(">>", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, env=env, check=False)


def load(name):
    p = os.path.join(OUT, name, "plan.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _wall_table(c):
    return [w for r in c["rooms"] for w in r["walls"]]


def repeatability(a, b):
    A, B = load(a), load(b)
    if not A or not B:
        return None
    c = compare(A, B)
    ws = _wall_table(c)
    ceil = [(r["room_a"], r["ceiling_a"], r["ceiling_b"], r["ceiling_spread_m"]) for r in c["rooms"]]
    return {"pair": [a, b], "summary": c["summary"], "walls": ws, "ceilings": ceil, "openings": c["openings"]}


def cross_tier(ref, other):
    A, B = load(ref), load(other)
    if not A or not B:
        return None
    c = compare(A, B)
    ws = _wall_table(c)
    rel = [abs(w["delta_m"]) / w["len_a"] for w in ws if w["len_a"] > 0.3]
    cov = [w["a_in_b_ci"] for w in ws]
    return {"pair": [ref, other], "summary": c["summary"], "walls_matched": len(ws),
            "median_rel_err": (round(float(np.median(rel)), 4) if rel else None),
            "p90_rel_err": (round(float(np.percentile(rel, 90)), 4) if rel else None),
            "ci95_coverage_of_lidar": (round(float(np.mean(cov)), 3) if cov else None),
            "area_ref": A["property"]["total_floor_area_m2"]["value"], "area_other": B["property"]["total_floor_area_m2"],
            "rooms_ref": len(A["rooms"]), "rooms_other": len(B["rooms"]),
            "footprint_rel_err": round(abs(B["property"]["total_floor_area_m2"]["value"] - A["property"]["total_floor_area_m2"]["value"])
                                       / A["property"]["total_floor_area_m2"]["value"], 4)}


def drift_ablation():
    out = {}
    for c in RAW:
        on, off = load(f"lidar_{c}"), load(f"lidar_nodrift_{c}")
        if not on or not off:
            continue
        out[c] = {"rooms_on": len(on["rooms"]), "rooms_off": len(off["rooms"]),
                  "area_on": on["property"]["total_floor_area_m2"]["value"],
                  "area_off": off["property"]["total_floor_area_m2"]["value"],
                  "bbox_on": [on["property"]["footprint_bbox_m"]["x"]["value"], on["property"]["footprint_bbox_m"]["y"]["value"]],
                  "bbox_off": [off["property"]["footprint_bbox_m"]["x"]["value"], off["property"]["footprint_bbox_m"]["y"]["value"]],
                  "drift_report": on.get("reconstruction", {}).get("drift")}
    rep_on = repeatability("lidar_with_ceiling", "lidar_floor_only")
    rep_off = repeatability("lidar_nodrift_with_ceiling", "lidar_nodrift_floor_only")
    out["repeatability_on"] = rep_on["summary"] if rep_on else None
    out["repeatability_off"] = rep_off["summary"] if rep_off else None
    return out


def gt_gates():
    from .gates import evaluate
    res = {}
    gdir = os.path.join(ROOT, "ground_truth")
    for space, caps in SPACES.items():
        p = os.path.join(gdir, f"{space}.json")
        if not os.path.exists(p):
            res[space] = "ground truth pending: ground_truth/%s.json not provided" % space
            continue
        gt = json.load(open(p))
        res[space] = {}
        for c in caps:
            for tier in ("lidar", "video", "photo"):
                P = load(f"{tier}_{c}")
                if P:
                    res[space][f"{tier}_{c}"] = evaluate(P, gt, tier)
    return res


def timing():
    rows = {}
    for name, *_ in RUNS:
        P = load(name)
        if P:
            rows[name] = P.get("timing_s", {})
    return rows


def report():
    R = {
        "repeatability": {
            "lidar": repeatability("lidar_with_ceiling", "lidar_floor_only"),
            "video": repeatability("video_with_ceiling", "video_floor_only"),
            "photo": repeatability("photo_with_ceiling", "photo_floor_only"),
        },
        "cross_tier_vs_lidar": {
            f"{t}_{c}": cross_tier(f"lidar_{c}", f"{t}_{c}") for t in ("video", "photo") for c in RAW
        },
        "drift_ablation": drift_ablation(),
        "ground_truth_gates": gt_gates(),
        "timing_s": timing(),
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
    from .report_md import write_markdown
    write_markdown(R, os.path.join(ROOT, "docs", "BENCHMARK.md"))
    print("wrote out/bench/results.json and docs/BENCHMARK.md")
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if not a.report:
        run_all(a.force, a.only)
    report()
