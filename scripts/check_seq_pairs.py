import sys; sys.path.insert(0, '.')
import numpy as np
from roomscan.capture import load_capture
from roomscan import mono as M
from roomscan.photo import pair_pose
cap = load_capture(sys.argv[1])
frames, fps = M.select_video_frames(cap.video_path, 6.0, 360)
net, lm = M.DepthNet.get(), M.LearnedMatcher.get()
D = [net(f["img"]) / M.MONO_SCALE_BIAS for f in frames]
F = [lm.features(f["img"]) for f in frames]
h, w = frames[0]["img"].shape[:2]
K = np.array([[M.IPHONE_FOCAL_FACTOR * w, 0, w / 2], [0, M.IPHONE_FOCAL_FACTOR * w, h / 2], [0, 0, 1.0]])
fails, inl, rot_err, tr_err = [], [], [], []
for i in range(1, len(frames)):
    e = pair_pose(F[i - 1], F[i], D[i - 1], D[i], K, K, lm)
    if e is None:
        fails.append(i); continue
    inl.append(e["inliers"])
    Ta, Tb = cap.frames[frames[i - 1]["index"]].T_wc, cap.frames[frames[i]["index"]].T_wc
    Tt = np.linalg.inv(Ta) @ Tb
    Rerr = e["T_i_j"][:3, :3].T @ Tt[:3, :3]
    rot_err.append(np.degrees(np.arccos(np.clip((np.trace(Rerr) - 1) / 2, -1, 1))))
    tr_err.append(np.linalg.norm(e["T_i_j"][:3, 3] - Tt[:3, 3]))
print(f"consecutive pairs: {len(fails)} fail of {len(frames)-1}; median inliers {np.median(inl):.0f}; "
      f"rot err median {np.median(rot_err):.2f} p90 {np.percentile(rot_err,90):.2f} deg; "
      f"trans err median {np.median(tr_err)*100:.1f} p90 {np.percentile(tr_err,90)*100:.1f} cm")
print("fail idx:", fails[:40], "sharpness at fails:", [round(frames[i]["sharpness"]) for i in fails[:15]])
