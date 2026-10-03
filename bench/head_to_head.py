"""Part 3 head-to-head: our LiDAR-tier output vs a consumer scanning app, dimension by dimension.

Input CSV (one row per shared dimension; fill `app_m` from the app's export, `tape_m` from the laser):
    room,dimension,our_id,app_m,tape_m
    Kitchen,wall north,R3-W2,3.412,3.405
    Kitchen,ceiling,R3,2.47,2.468
    Kitchen,door to hall,R3-O1,0.81,0.812
`our_id` is the wall / room / opening id in our plan.json (use bench.gates template to list them).

    python -m bench.head_to_head <our plan.json> <shared_dims.csv> [--app "Polycam 3.x (free)"]
Writes a Markdown table to stdout: our error, the app's error, and who wins (tie = within 0.5 cm).
Gate: we beat or tie on >= 70 % of shared dimensions.
"""
import argparse
import csv
import json

TIE = 0.005


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan"); ap.add_argument("csv"); ap.add_argument("--app", default="consumer app")
    a = ap.parse_args()
    P = json.load(open(a.plan))
    idx = {}
    for r in P["rooms"]:
        idx[r["id"]] = r["ceiling_height_m"]["value"]
        for w in r["walls"]:
            idx[w["id"]] = w["length_m"]["value"]
        for o in r["openings"]:
            idx[o["id"]] = o["width_m"]["value"]
    rows, wins = [], 0
    for row in csv.DictReader(open(a.csv)):
        tape, app = float(row["tape_m"]), float(row["app_m"])
        ours = idx.get(row["our_id"])
        eo = None if ours is None else abs(ours - tape)
        ea = abs(app - tape)
        res = "miss (ours)" if eo is None else ("tie" if abs(eo - ea) <= TIE else ("ours" if eo < ea else a.app))
        wins += res in ("ours", "tie")
        rows.append((row["room"], row["dimension"], tape, ours, eo, app, ea, res))
    print(f"| room | dimension | tape (m) | ours (m) | our err (cm) | {a.app} (m) | app err (cm) | better |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]:.3f} | {'—' if r[3] is None else f'{r[3]:.3f}'} | "
              f"{'—' if r[4] is None else f'{r[4]*100:.1f}'} | {r[5]:.3f} | {r[6]*100:.1f} | {r[7]} |")
    n = len(rows)
    print(f"\nbeat or tie: {wins}/{n} = {100 * wins / max(n, 1):.0f} %  (gate >= 70 %: {'PASS' if n and wins / n >= 0.7 else 'FAIL'})")


if __name__ == "__main__":
    main()
