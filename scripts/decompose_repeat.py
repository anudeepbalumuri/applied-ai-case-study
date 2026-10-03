"""Decompose wall-length disagreement between two plans into neighbour-surface shifts."""
import sys, json, numpy as np
sys.path.insert(0, '.')
from bench.compare import compare
A, B = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
c = compare(A, B); R = np.array(c["transform_b_to_a"]["R"]); t = np.array(c["transform_b_to_a"]["t"])
def room_of(P):
    return {w["id"]: r for r in P["rooms"] for w in r["walls"]}
ra, rb = room_of(A), room_of(B)
def nbrs(w, r, R=np.eye(2), t=np.zeros(2)):
    ws = r["walls"]; i = [x["id"] for x in ws].index(w["id"])
    out = []
    for nb in (ws[i - 1], ws[(i + 1) % len(ws)]):
        n = np.array(nb["inward_normal"]) @ R.T; ax = int(np.argmax(np.abs(n)))
        out.append((ax, (np.array(nb["start"]) @ R.T + t)[ax], nb["measured_from_points"]))
    return out
wa = {w["id"]: w for r in A["rooms"] for w in r["walls"]}; wb = {w["id"]: w for r in B["rooms"] for w in r["walls"]}
cls = {"surface (both nbr shifts < 5 cm)": [], "topology (a nbr shift >= 15 cm)": [], "mixed": []}
for rr in c["rooms"]:
    for m in rr["walls"]:
        na, nb = nbrs(wa[m["wall_a"]], ra[m["wall_a"]]), nbrs(wb[m["wall_b"]], rb[m["wall_b"]], R, t)
        # pair neighbours by axis and position along the wall
        shifts = []
        for ax, ca, _ in na:
            cands = [cb for axb, cb, _ in nb if axb == ax]
            shifts.append(min(abs(cb - ca) for cb in cands) if cands else 9.9)
        mx = max(shifts)
        key = "surface (both nbr shifts < 5 cm)" if mx < 0.05 else ("topology (a nbr shift >= 15 cm)" if mx >= 0.15 else "mixed")
        cls[key].append((m["wall_a"], round(m["delta_m"] * 100, 1), [round(s * 100, 1) for s in shifts]))
for k, v in cls.items():
    print(f"{k}: {len(v)} walls, |delta| median {np.median([abs(x[1]) for x in v]) if v else float('nan'):.1f} cm, pass {sum(abs(x[1]) <= 1.0 for x in v)}")
    for x in v: print("    ", x)
