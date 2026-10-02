import sys, pickle, json; sys.path.insert(0, '.')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from roomscan.capture import load_capture
from roomscan.mono import reconstruct_video
from roomscan import layout as L
cap = load_capture(sys.argv[1]); name = sys.argv[2]
vcap = load_capture(cap.video_path, "video")
cloud, info, views = reconstruct_video(vcap, log=lambda *a: None)
pickle.dump((cloud, info), open(f"data/cache/{name}_video_cloud.pkl", "wb"))
fc, rep = pickle.load(open(f"data/cache/{name}_cloud.pkl", "rb"))
fig, ax = plt.subplots(1, 3, figsize=(27, 9))
for a, c, t in [(ax[0], fc, "LiDAR"), (ax[1], cloud, "video")]:
    fy = L.floor_height(c.pts, c.normals); rel = c.pts[:, 1] - fy
    m = (np.abs(c.normals[:, 1]) < 0.3) & (rel > 0.3) & (rel < 2.3)
    a.scatter(c.pts[m, 0], c.pts[m, 2], s=0.05, c='k'); a.plot(c.cam_positions[:, 0], c.cam_positions[:, 2], 'r-', lw=0.5)
    a.set_aspect('equal'); a.set_title(f"{t} wall band, floor {fy:.2f}")
c = cloud; fy = L.floor_height(c.pts, c.normals)
ax[2].hist(c.pts[:, 1] - fy, bins=200, range=(-1, 4)); ax[2].set_yscale('log'); ax[2].set_title("video height histogram")
plt.tight_layout(); plt.savefig(f"out/explore/video_vs_lidar_{name}.png", dpi=55)
