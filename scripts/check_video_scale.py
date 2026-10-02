"""Diagnose video-tier scale/gravity against the ARKit trajectory of the same capture."""
import sys, tempfile, shutil; sys.path.insert(0, '.')
import numpy as np
from roomscan.capture import load_capture
from roomscan import mono as M

d = sys.argv[1]
cap = load_capture(d)
frames, fps = M.select_video_frames(cap.video_path, 6.0, 360)
work = tempfile.mkdtemp()
poses, K, sparse, sfm = M.run_sfm([f["img"] for f in frames], work, log=print)
shutil.rmtree(work)
keys = sorted(poses)
A = np.array([poses[k][:3, 3] for k in keys])                       # sfm
B = np.array([cap.frames[frames[k]["index"]].T_wc[:3, 3] for k in keys])   # arkit (metres)
# Umeyama similarity A -> B
ma, mb = A.mean(0), B.mean(0)
Ac, Bc = A - ma, B - mb
U, S, Vt = np.linalg.svd(Bc.T @ Ac / len(A))
D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
R = U @ D @ Vt
s = np.trace(np.diag(S) @ D) / (Ac ** 2).sum(1).mean()
res = np.linalg.norm((s * Ac @ R.T) - Bc, axis=1)
print(f"true scale {s:.4f} m/unit, trajectory residual median {np.median(res)*100:.1f} cm")
# gravity check: sfm down from cameras vs truth (arkit -y mapped into sfm frame)
g_est = M.gravity_from_cameras(poses)
g_true = R.T @ np.array([0, -1.0, 0]); g_true /= np.linalg.norm(g_true)
print("gravity error from cameras: %.2f deg" % np.degrees(np.arccos(np.clip(g_est @ g_true, -1, 1))))
# camera height in sfm units under true gravity
net = M.DepthNet.get()
pts, nrm, ratios, per = M.fuse_mono(frames, poses, K, sparse, net)
print(f"depth-network cue: median(dn/zc)={np.median(ratios):.4f} -> /bias {np.median(ratios)/M.MONO_SCALE_BIAS:.4f} m/unit")
Rg = M.align_gravity(g_true)
p = pts @ Rg.T; n = nrm @ Rg.T
cams = A @ Rg.T
up = n[:, 1] > 0.9
h, e = np.histogram(p[up, 1], bins=200)
cand = np.where(h >= 0.3 * h.max())[0]
fy = e[cand[0]]
print(f"cam height (true gravity) {np.median(cams[:,1]) - fy:.4f} units -> {M.CAMERA_HEIGHT/(np.median(cams[:,1]) - fy):.4f} m/unit")
Rr = M.refine_gravity(pts, nrm, M.align_gravity(g_est))
g_ref = Rr.T @ np.array([0, -1.0, 0])
print("gravity error after floor refine: %.2f deg" % np.degrees(np.arccos(np.clip(g_ref @ g_true, -1, 1))))
