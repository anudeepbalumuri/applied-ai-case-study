import sys, tempfile, shutil, pickle; sys.path.insert(0, '.')
import numpy as np
from roomscan.capture import load_capture
from roomscan import mono as M
d = sys.argv[1]
cap = load_capture(d)
frames, fps = M.select_video_frames(cap.video_path, 6.0, 360)
work = tempfile.mkdtemp()
poses, K, sparse, sfm = M.run_sfm([f["img"] for f in frames], work, log=print)
shutil.rmtree(work)
pickle.dump((frames, poses, K, sparse), open("data/cache/sfm_single_room.pkl", "wb"))
keys = sorted(poses)
print("K", K.round(1), "img", frames[0]["img"].shape)
errs = []
for a, b in zip(keys[:-5], keys[5:]):
    Rs = poses[a][:3, :3].T @ poses[b][:3, :3]
    Ta, Tb = cap.frames[frames[a]["index"]].T_wc, cap.frames[frames[b]["index"]].T_wc
    Rt = Ta[:3, :3].T @ Tb[:3, :3]
    ang_s = np.degrees(np.arccos(np.clip((np.trace(Rs) - 1) / 2, -1, 1)))
    ang_t = np.degrees(np.arccos(np.clip((np.trace(Rt) - 1) / 2, -1, 1)))
    e = np.degrees(np.arccos(np.clip((np.trace(Rs.T @ Rt) - 1) / 2, -1, 1)))
    errs.append((ang_s, ang_t, e))
errs = np.array(errs)
print("relative rotation: sfm median %.1f deg, arkit median %.1f deg, disagreement median %.2f p90 %.2f" %
      (np.median(errs[:, 0]), np.median(errs[:, 1]), np.median(errs[:, 2]), np.percentile(errs[:, 2], 90)))
Rs=[poses[k][:3,:3] for k in keys]; Rt=[cap.frames[frames[k]["index"]].T_wc[:3,:3] for k in keys]
Mm=sum(t@s.T for s,t in zip(Rs,Rt)); U,_,Vt=np.linalg.svd(Mm); R0=U@Vt
e=np.array([np.degrees(np.arccos(np.clip((np.trace((R0@s).T@t)-1)/2,-1,1))) for s,t in zip(Rs,Rt)])
print("absolute rotation error: median %.1f p90 %.1f max %.1f deg; registered %d/%d" % (np.median(e), np.percentile(e,90), e.max(), len(keys), len(frames)))
