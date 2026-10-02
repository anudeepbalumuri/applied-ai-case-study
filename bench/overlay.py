"""Overlay two plans of the same space (B aligned onto A) with per-wall deltas - repeatability diagnosis.

    python -m bench.overlay A/plan.json B/plan.json out.png
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .compare import compare


def overlay(pa, pb, out):
    A, B = json.load(open(pa)), json.load(open(pb))
    c = compare(A, B)
    R, t = np.array(c["transform_b_to_a"]["R"]), np.array(c["transform_b_to_a"]["t"])
    fig, ax = plt.subplots(figsize=(14, 14))
    for r in A["rooms"]:
        v = np.array(r["polygon"]); v = np.vstack([v, v[:1]])
        ax.plot(v[:, 0], v[:, 1], "-", color="#1f6feb", lw=2.5, alpha=0.8)
        ax.text(*np.array(r["polygon"]).mean(0), r["id"], color="#1f6feb", fontsize=12)
    for r in B["rooms"]:
        v = np.array(r["polygon"]) @ R.T + t; v = np.vstack([v, v[:1]])
        ax.plot(v[:, 0], v[:, 1], "--", color="#d1242f", lw=1.6)
    wid = {w["id"]: w for r in A["rooms"] for w in r["walls"]}
    for rr in c["rooms"]:
        for w in rr["walls"]:
            a = wid[w["wall_a"]]
            m = (np.array(a["start"]) + np.array(a["end"])) / 2 + np.array(a["inward_normal"]) * 0.18
            ax.text(*m, f"{w['delta_m'] * 100:+.1f}", fontsize=8, ha="center",
                    color="green" if w["pass_1cm_or_0.5pct"] else "black")
    s = c["summary"]
    ax.set_title(f"A (blue) vs B (red dashed)   walls pass {s['walls_pass']}/{s['walls_matched']}  "
                 f"median |d| {s['median_abs_delta_m']} m   (labels: B - A, cm)")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=90)


if __name__ == "__main__":
    overlay(*sys.argv[1:4])
