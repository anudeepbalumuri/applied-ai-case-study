"""How good is sequential PnP chaining on metric mono depth for the video tier? Compare to ARKit."""
import sys; sys.path.insert(0, '.')
import numpy as np
from roomscan.capture import load_capture
from roomscan import mono as M
from roomscan.photo import pair_pose
d = sys.argv[1]
cap = load_capture(d)
frames, fps = M.select_video_frames(cap.video_path, 6.0, 360)
net, lm = M.DepthNet.get(), M.LearnedMatcher.get()
D = [net(f["img"]) / M.MONO_SCALE_BIAS for f in frames]
F = [lm.features(f["img"]) for f in frames]
h, w = frames[0]["img"].shape[:2]
K = np.array([[M.IPHONE_FOCAL_FACTOR * w, 0, w / 2], [0, M.IPHONE_FOCAL_FACTOR * w, h / 2], [0, 0, 1.0]])
poses = {0: np.eye(4)}; s_cum = {0: 1.0}; fails = 0
for i in range(1, len(frames)):
    e = None
    for back in (1, 2, 3):
        if i - back in poses:
            e = pair_pose(F[i - back], F[i], D[i - back], D[i], K, K, lm)
            if e is not None:
                Tij = e["T_i_j"].copy(); Tij[:3, 3] *= s_cum[i - back]
                poses[i] = poses[i - back] @ Tij; s_cum[i] = s_cum[i - back] * e["ratio"]
                break
    if e is None:
        fails += 1
keys = sorted(poses)
A = np.array([poses[k][:3, 3] for k in keys]); B = np.array([cap.frames[frames[k]["index"]].T_wc[:3, 3] for k in keys])
ma, mb = A.mean(0), B.mean(0); Ac, Bc = A - ma, B - mb
U, S, Vt = np.linalg.svd(Bc.T @ Ac / len(A)); Dg = np.eye(3); Dg[2, 2] = np.sign(np.linalg.det(U @ Vt)); R = U @ Dg @ Vt
s = np.trace(np.diag(S) @ Dg) / (Ac ** 2).sum(1).mean()
res = np.linalg.norm(s * Ac @ R.T - Bc, axis=1)
path = np.linalg.norm(np.diff(B, axis=0), axis=1).sum()
print(f"chained {len(keys)}/{len(frames)} (breaks {fails}); best-fit scale {s:.3f} (1.0 = metric-correct); "
      f"residual median {np.median(res)*100:.1f} cm, max {res.max()*100:.1f} cm over {path:.1f} m path; "
      f"depth scale drift {min(s_cum.values()):.2f}-{max(s_cum.values()):.2f}")
