"""Build a photo-tier proxy benchmark (one folder of stills per room) from a LiDAR capture.

For every room of the capture's LiDAR plan, choose up to N sharp frames taken from inside the room
with maximally different viewing directions, rotate them upright (as a phone's EXIF orientation
would) and save as JPEG without depth or poses. Folder names are room ids of the source plan so the
photo-tier result can be scored against it.

This is a *proxy* (video frames, not stills) - the brief's photo benchmark needs real iPhone stills.
    python -m bench.make_photo_tier <stray_capture> <lidar_plan.json> <out_dir> [--per-room 6]
"""
import argparse, json, os, sys
import cv2
import numpy as np
from matplotlib.path import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from roomscan.capture import load_capture, iter_video_frames
from roomscan.damage import upright_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture"); ap.add_argument("plan"); ap.add_argument("out")
    ap.add_argument("--per-room", type=int, default=8)
    ap.add_argument("--min-area", type=float, default=2.0)
    a = ap.parse_args()
    cap = load_capture(a.capture)
    J = json.load(open(a.plan))
    yaw, fy = J["plan_frame"]["yaw_rad"], J["plan_frame"]["floor_y_world"]
    c, s = np.cos(-yaw), np.sin(-yaw); R = np.array([[c, -s], [s, c]])
    cams = np.array([f.T_wc[:3, 3] for f in cap.frames])
    uv = cams[:, [0, 2]] @ R.T
    fwd = np.array([f.T_wc[:3, 2] for f in cap.frames])[:, [0, 2]] @ R.T
    heading = np.arctan2(fwd[:, 1], fwd[:, 0])
    pitch = np.array([f.T_wc[1, 2] for f in cap.frames])          # forward . world up
    chosen = {}
    for r in J["rooms"]:
        if r["floor_area_m2"]["value"] < a.min_area or r.get("observation", {}).get("partial"):
            continue
        poly = Path(np.array(r["polygon"]))
        # inside and >= 0.3 m from the boundary (not standing in a doorway)
        inside = poly.contains_points(uv) & Path(np.array(r["polygon"])).contains_points(uv, radius=-0.6)
        cand = np.where(inside & (np.abs(pitch) < 0.5))[0]
        if len(cand) < 2:
            continue
        # protocol-like sweep (PHOTO_PROTOCOL.md): consecutive shots while turning, so each photo
        # overlaps the previous one. Walk the longest time-contiguous stay in the room and take a frame
        # every >= 22 deg of turn or >= 0.6 m of movement; keep the 8-shot window with most coverage.
        runs, start = [], cand[0]
        for p_, q_ in zip(cand[:-1], cand[1:]):
            if q_ - p_ > 15:
                runs.append((start, p_)); start = q_
        runs.append((start, cand[-1]))
        s0, s1 = max(runs, key=lambda r_: r_[1] - r_[0])
        seq, last = [s0], s0
        for i in range(s0 + 1, s1 + 1):
            if not inside[i] or abs(pitch[i]) >= 0.5:
                continue
            turn = abs(np.angle(np.exp(1j * (heading[i] - heading[last]))))
            if turn >= np.deg2rad(22) or np.linalg.norm(uv[i] - uv[last]) >= 0.6:
                seq.append(i); last = i
        def coverage(win):
            h = np.sort(np.mod(heading[win], 2 * np.pi))
            gaps = np.diff(np.r_[h, h[0] + 2 * np.pi])
            return 2 * np.pi - gaps.max()
        n = min(a.per_room, len(seq))
        sel = max((seq[k:k + n] for k in range(len(seq) - n + 1)), key=coverage)
        chosen[r["id"]] = sorted(sel)
    # protocol step 1/3: entry photo (just inside the doorway, facing in) and a through-door photo
    # (last spot in the room before leaving, facing the next room). Taken from the trajectory's
    # room-to-room transitions; they replace sweep shots so a folder stays <= per-room photos.
    polys = {r["id"]: Path(np.array(r["polygon"])) for r in J["rooms"] if r["id"] in chosen}
    lab = np.full(len(uv), None, dtype=object)
    for rid, P in polys.items():
        lab[P.contains_points(uv)] = rid
    fpsr = len(cap.frames) / max(cap.frames[-1].timestamp - cap.frames[0].timestamp, 1e-6)
    off = int(0.6 * fpsr)
    extra = {rid: [] for rid in chosen}
    last = None
    for i in range(len(lab)):
        if lab[i] is None:
            continue
        if last is not None and lab[last] != lab[i] and i - last < fpsr * 2:
            a_, b_ = lab[last], lab[i]
            pre, post = max(last - off, 0), min(i + off, len(lab) - 1)
            # doorway photo (protocol): taken from ~2 m inside room A, facing the doorway, so it shows
            # A's wall around the door *and* B through it
            door = (uv[last] + uv[i]) / 2
            best = None
            for q in range(last, max(last - int(8 * fpsr), 0), -1):
                if lab[q] != a_ or abs(pitch[q]) >= 0.5:
                    continue
                dv = door - uv[q]
                dist = np.linalg.norm(dv)
                ang = abs(np.angle(np.exp(1j * (np.arctan2(dv[1], dv[0]) - heading[q]))))
                if 1.2 <= dist <= 2.5 and ang < np.deg2rad(25):
                    if best is None or ang < best[0]:
                        best = (ang, q)
            if best is not None:
                extra[a_].append(best[1])
            elif lab[pre] == a_ and abs(pitch[pre]) < 0.5:
                extra[a_].append(pre)
            if lab[post] == b_ and abs(pitch[post]) < 0.5:
                extra[b_].append(post)
        last = i
    for rid in chosen:
        ex = sorted(set(extra[rid]))[:3]
        keep = [i for i in chosen[rid] if i not in ex][: max(a.per_room - len(ex), 2)]
        chosen[rid] = sorted(keep + ex)
    want = sorted({i for v in chosen.values() for i in v})
    imgs = {}
    for i, img in iter_video_frames(cap.video_path, indices=want):
        imgs[i] = np.ascontiguousarray(np.rot90(img, upright_k(cap.frames[i].T_wc)))
    for rid in list(chosen):
        chosen[rid] = [i for i in chosen[rid] if i in imgs]     # odometry can outlast the video by a frame
    for rid, sel in chosen.items():
        d = os.path.join(a.out, rid)
        os.makedirs(d, exist_ok=True)
        for k, i in enumerate(sel):
            cv2.imwrite(os.path.join(d, f"IMG_{k:02d}.jpg"), imgs[i], [cv2.IMWRITE_JPEG_QUALITY, 93])
    meta = {"source_capture": a.capture, "source_plan": a.plan, "frames": {k: [int(x) for x in v] for k, v in chosen.items()},
            "note": "photo-tier PROXY built from LiDAR-capture video frames; not real stills"}
    json.dump(meta, open(os.path.join(a.out, "_proxy_meta.json"), "w"), indent=1)
    print({k: len(v) for k, v in chosen.items()})


if __name__ == "__main__":
    main()
