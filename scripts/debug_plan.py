import sys, pickle, json; sys.path.insert(0, '.')
import numpy as np
from roomscan.plan import extract_plan
from roomscan.output import plan_to_json
from roomscan.uncertainty import ErrorModel
from roomscan.render import render_plan
name = sys.argv[1]
fc, rep = pickle.load(open(f"data/cache/{name}_cloud.pkl", "rb"))
P = extract_plan(fc, fc.cam_positions, log=lambda *a: None)
J = plan_to_json(P, "lidar", name, ErrorModel("lidar"))
g = P["grid"]; fr = P["frame"]
uv = fr.to_plan(fc.pts[:, [0, 2]]); rel = fc.pts[:, 1] - fr.floor_y
m = (np.abs(fc.normals[:, 1]) < 0.3) & (rel > 0.3) & (rel < 2.4)
img = np.log1p(g.count(uv[m])).T
ext = [g.o[0], g.o[0] + g.shape[0] * g.res, g.o[1], g.o[1] + g.shape[1] * g.res]
render_plan(J, f"out/explore/plan_{name}.png", background=(img, ext))
json.dump(J, open(f"out/explore/plan_{name}.json", "w"), indent=1)
print(name, "rooms", len(J["rooms"]), "total area", J["property"]["total_floor_area_m2"]["value"])
