"""Interval calibration: per-tier multiplier k so that 95 % intervals cover 95 % of reference values.

For every wall matched between a tier's plan and the LiDAR plan of the same capture we take
z = |tier - lidar| / sigma_tier, where sigma_tier is the uncalibrated sigma (k = 1). The multiplier is
k = q95(z) / 1.96, clipped to >= 1 (we never shrink an interval on this little data).

Caveats (also in the technical report): the reference is LiDAR, not tape, so LiDAR's own error is
folded into the tier's; and the fit is in-sample on our benchmark.

    python -m bench.calibrate            # prints table, writes roomscan/calibration.json
"""
import json
import os

import numpy as np

from .compare import compare
from .run_benchmark import OUT, RAW, ROOT

CAL = os.path.join(ROOT, "roomscan", "calibration.json")


def ratios(tier):
    z = []
    for c in RAW:
        a, b = os.path.join(OUT, f"lidar_{c}", "plan.json"), os.path.join(OUT, f"{tier}_{c}", "plan.json")
        if not (os.path.exists(a) and os.path.exists(b)):
            continue
        B = json.load(open(b))
        k_used = float(B.get("calibration_k", {}).get("wall_length", 1.0))
        sig = {w["id"]: w["length_m"]["sigma"] / k_used for r in B["rooms"] for w in r["walls"]}
        res = compare(a, b)
        for r in res["rooms"]:
            for w in r["walls"]:
                s = sig.get(w["wall_b"])
                if s:
                    z.append(abs(w["delta_m"]) / s)
    return np.array(z)


def main():
    cal = json.load(open(CAL)) if os.path.exists(CAL) else {}
    for tier in ("video", "photo"):
        z = ratios(tier)
        if len(z) < 5:
            print(f"{tier}: {len(z)} matched walls - too few to calibrate, keeping k = 1")
            continue
        q95 = float(np.quantile(z, 0.95))
        k = max(1.0, q95 / 1.96)
        cov1 = float(np.mean(z <= 1.96))
        covk = float(np.mean(z <= 1.96 * k))
        print(f"{tier}: n={len(z)} walls, coverage at k=1: {cov1:.2f}, q95(z)={q95:.2f} -> k={k:.2f} "
              f"(in-sample coverage {covk:.2f})")
        cal.setdefault(tier, {}).update({"wall_length": round(k, 3), "area": round(k, 3), "opening_width": round(k, 3),
                                         "ceiling_height": round(k, 3), "footprint": round(k, 3), "perimeter": round(k, 3),
                                         "n_walls": int(len(z)), "coverage_k1": round(cov1, 3)})
    json.dump(cal, open(CAL, "w"), indent=1)
    print("wrote", CAL)


if __name__ == "__main__":
    main()
