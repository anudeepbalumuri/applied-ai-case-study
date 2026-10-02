"""COLMAP on densified frames vs ARKit: registration, rotation and trajectory error."""
import sys, tempfile, shutil, pickle; sys.path.insert(0, '.')
import numpy as np
from roomscan.capture import load_capture
from roomscan import mono as M
cap = load_capture(sys.argv[1]); name = sys.argv[2]
frames, fps = M.select_video_frames(cap.video_path, M.VIDEO_FPS, M.MAX_FRAMES)
lm = M.LearnedMatcher.get()
frames, feats = M.densify_by_matches(cap.video_path, frames, lm)
work = tempfile.mkdtemp()
poses, K, sparse, sfm = M.run_sfm([f["img"] for f in frames], work, log=print, feats=feats)
shutil.rmtree(work)
pickle.dump((frames, poses, K, sparse), open(f"data/cache/sfm_dense_{name}.pkl", "wb"))
keys = sorted(poses)
Rs = [poses[k][:3, :3] for k in keys]; Rt = [cap.frames[frames[k]["index"]].T_wc[:3, :3] for k in keys]
Mm = sum(t @ s.T for s, t in zip(Rs, Rt)); U, _, Vt = np.linalg.svd(Mm); R0 = U @ Vt
e = np.array([np.degrees(np.arccos(np.clip((np.trace((R0 @ s).T @ t) - 1) / 2, -1, 1))) for s, t in zip(Rs, Rt)])
A = np.array([poses[k][:3, 3] for k in keys]); B = np.array([cap.frames[frames[k]["index"]].T_wc[:3, 3] for k in keys])
ma, mb = A.mean(0), B.mean(0); Ac, Bc = A - ma, B - mb
U, S, Vt = np.linalg.svd(Bc.T @ Ac / len(A)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt)); R = U @ D @ Vt
sc = np.trace(np.diag(S) @ D) / (Ac ** 2).sum(1).mean()
res = np.linalg.norm(sc * Ac @ R.T - Bc, axis=1)
nreal = sum(not f.get("bridge") for f in frames)
print(f"registered {len(keys)}/{len(frames)} (real {sum(not frames[k].get('bridge') for k in keys)}/{nreal}); "
      f"rot err median {np.median(e):.1f} p90 {np.percentile(e, 90):.1f}; traj residual median {np.median(res)*100:.1f} cm p90 {np.percentile(res,90)*100:.1f} cm")
