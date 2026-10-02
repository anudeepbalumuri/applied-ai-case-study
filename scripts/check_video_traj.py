"""Video tier trajectory vs ARKit (similarity fit) - the honest drift/scale check for the video path."""
import sys; sys.path.insert(0, '.')
import numpy as np
from roomscan.capture import load_capture
from roomscan.mono import reconstruct_video
cap = load_capture(sys.argv[1])
vcap = load_capture(cap.video_path, "video")
cloud, info, views = reconstruct_video(vcap)
P = info["_poses"]; fm = info["_frame_map"]
keys = sorted(P)
A = np.array([P[k][:3, 3] for k in keys]); B = np.array([cap.frames[fm[k]].T_wc[:3, 3] for k in keys])
ma, mb = A.mean(0), B.mean(0); Ac, Bc = A - ma, B - mb
U, S, Vt = np.linalg.svd(Bc.T @ Ac / len(A)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt)); R = U @ D @ Vt
s = np.trace(np.diag(S) @ D) / (Ac ** 2).sum(1).mean()
res = np.linalg.norm(s * Ac @ R.T - Bc, axis=1)
path = np.linalg.norm(np.diff(B, axis=0), axis=1).sum()
up_err = np.degrees(np.arccos(np.clip((R @ np.array([0, 1.0, 0])) @ np.array([0, 1.0, 0]), -1, 1)))
print(f"posed {len(keys)}/{info['frames_selected']}; metric scale error {100*(1/s-1):+.1f}% (pred/true); "
      f"traj residual median {np.median(res)*100:.1f} cm, p90 {np.percentile(res,90)*100:.1f} cm over {path:.1f} m; gravity err {up_err:.2f} deg")
