"""Recall (synthetic damage pasted on real wall frames) vs phantom rate (clean frames) for the damage chain.

Synthetic damage only checks that the chain *can* fire; real staged damage is still required (see report).
"""
import sys, json; sys.path.insert(0, '.')
import numpy as np, cv2
from roomscan.capture import load_capture, iter_video_frames
from roomscan.damage import Models, upright_k, _thin

box_thr, verify_thr = float(sys.argv[1]), float(sys.argv[2])
M = Models.get(); rng = np.random.default_rng(1)


def fractal(h, w, seed):
    r = np.random.default_rng(seed); out = np.zeros((h, w))
    for o in range(1, 6):
        s = 2 ** o; n = r.normal(size=(h // (64 // s) + 2, w // (64 // s) + 2))
        out += cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC) / s
    return out


def stain(img, cx, cy, r, seed):
    h, w = img.shape[:2]; yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - cx, yy - cy) / r + 0.35 * fractal(h, w, seed)
    body = np.clip(1 - d, 0, 1) ** 0.5 * 0.35
    tide = np.exp(-((d - 1) / 0.05) ** 2) * 0.6
    a = np.clip(body + tide, 0, 0.8)
    col = np.array([70, 115, 160], float)
    return (img * (1 - a[..., None]) + col * a[..., None]).astype(np.uint8)


def crack(img, x, y, L):
    pts = [(x, y)]
    for _ in range(40):
        x += rng.normal(0, 4); y += L / 40; pts.append((x, y))
    cv2.polylines(img, [np.array(pts, np.int32)], False, (70, 70, 75), 2, cv2.LINE_AA)
    return img


def fires(rgb, kind):
    for d in M.detect(rgb, box_thr=box_thr, text_thr=box_thr):
        if kind is not None and d["class"] != kind:
            continue
        x0, y0, x1, y1 = [int(v) for v in d["box"]]
        crop = rgb[max(y0, 0):y1, max(x0, 0):x1]
        if crop.size == 0 or min(crop.shape[:2]) < 12:
            continue
        if M.verify(crop, d["class"]) < verify_thr:
            continue
        if d["class"] == "crack" and not _thin(M.segment(rgb, d["box"])):
            continue
        return True
    return False


res = {"water_stain": 0, "crack": 0, "phantom_frames": 0, "n": 0}
for cid, path in [("floor_only", "data/raw/floor_only/1a8384c3f6"), ("with_ceiling", "data/raw/with_ceiling/c7d28f72c6")]:
    cap = load_capture(path)
    idx = list(np.linspace(200, len(cap.frames) - 200, 10).astype(int))
    for i, img in iter_video_frames(cap.video_path, indices=idx):
        up = np.ascontiguousarray(np.rot90(cv2.resize(img, None, fx=0.5, fy=0.5), upright_k(cap.frames[i].T_wc)))
        h, w = up.shape[:2]; res["n"] += 1
        res["phantom_frames"] += fires(up[:, :, ::-1].copy(), None)
        res["water_stain"] += fires(stain(up, w * 0.5, h * 0.25, min(h, w) * 0.1, i)[:, :, ::-1].copy(), "water_stain")
        res["crack"] += fires(crack(up.copy(), int(w * 0.5), int(h * 0.1), h * 0.3)[:, :, ::-1].copy(), "crack")
print(json.dumps({"box_thr": box_thr, "verify_thr": verify_thr, **res}))
