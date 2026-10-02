import sys, os, pickle; sys.path.insert(0, '.')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from roomscan.capture import load_capture
from roomscan.fusion import fuse, select_keyframes
from roomscan.drift import correct_drift
from roomscan import layout as L

name, d = sys.argv[1], sys.argv[2]
cache = f"data/cache/{name}_cloud.pkl"
if os.path.exists(cache):
    fc, rep = pickle.load(open(cache, 'rb'))
else:
    cap = load_capture(d); kf = select_keyframes(cap.frames)
    poses, rep = correct_drift(cap, kf, log=lambda *a: None)
    fc = fuse(cap, poses=poses, keyframes=kf, voxel=0.02)
    pickle.dump((fc, rep), open(cache, 'wb'))
fy = L.floor_height(fc.pts, fc.normals)
yaw, st = L.manhattan_yaw(fc.pts, fc.normals, fy)
fr = L.PlanFrame(yaw, fy)
grid, uv, maps = L.evidence_maps(fc, fr)
cam_uv = fr.to_plan(fc.cam_positions[:, [0, 2]])
lab, interior = L.segment_rooms(grid, maps, cam_uv)
print(name, "floor", round(fy, 3), "yaw deg", round(np.rad2deg(yaw), 2), "strength", round(st, 2), "rooms", lab.max())
fig, ax = plt.subplots(1, 3, figsize=(27, 9))
ext = [grid.o[1], grid.o[1] + grid.shape[1] * grid.res, grid.o[0] + grid.shape[0] * grid.res, grid.o[0]]
ax[0].imshow(np.log1p(maps["wall"]), extent=ext); ax[0].set_title("wall")
ax[1].imshow(np.log1p(maps["wall_hi"]) + 0.5 * (maps["floor"] > 0), extent=ext); ax[1].set_title("hi + floor")
ax[2].imshow(lab, cmap='tab20', extent=ext, interpolation='nearest'); ax[2].set_title("rooms")
for r in range(1, lab.max() + 1):
    v = L.rectilinear_polygon(lab == r, grid)
    if len(v) >= 4:
        vv = np.vstack([v, v[:1]]); ax[2].plot(vv[:, 1], vv[:, 0], 'k-', lw=1.5)
        ax[2].text(v[:, 1].mean(), v[:, 0].mean(), str(r), fontsize=14)
plt.tight_layout(); plt.savefig(f"out/explore/layout_{name}.png", dpi=60)
