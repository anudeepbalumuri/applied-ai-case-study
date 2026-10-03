"""Fast repeatability loop for layout changes: cached fused clouds -> plans -> compare (no fusion/drift rerun)."""
import sys, pickle, json, os; sys.path.insert(0, '.')
from roomscan.plan import extract_plan
from roomscan.output import plan_to_json
from roomscan.uncertainty import ErrorModel
from bench.compare import compare
os.makedirs("out/fast", exist_ok=True)
P = {}
for name in ["with_ceiling", "floor_only"]:
    fc, rep = pickle.load(open(f"data/cache/{name}_cloud.pkl", "rb"))
    plan = extract_plan(fc, fc.cam_positions, log=lambda *a: None)
    J = plan_to_json(plan, "lidar", name, ErrorModel("lidar"))
    json.dump(J, open(f"out/fast/{name}.json", "w"), default=float)
    P[name] = J
    E = [w for r in J["rooms"] for w in r["walls"]]
    print(name, "rooms", len(J["rooms"]), "edges", len(E), "short<0.6", sum(w["length_m"]["value"] < 0.6 for w in E),
          "unmeasured", sum(not w["measured_from_points"] for w in E))
c = compare(P["with_ceiling"], P["floor_only"])
print(json.dumps({k: c["summary"][k] for k in ["rooms_matched", "walls_matched", "walls_pass", "median_abs_delta_m", "p90_abs_delta_m", "alignment_iou"]}))
