"""Damage detection, metric extent on plan surfaces, concealed-damage rules, scope line items.

Detection   : Grounding-DINO (open vocabulary) proposes boxes for damage phrases; CLIP verifies each
              crop against damage vs. look-alike negatives (shadows, picture frames, reflections,
              wet-look glossy surfaces); SAM turns the box into a mask.
Metric      : every mask pixel is ray-cast against the *room's own plan surfaces* (walls, floor,
              ceiling) -> surface id + surface (s, h) coordinates in metres. This needs no depth
              sensor, so it runs on every tier; on LiDAR the sensor depth rejects pixels whose surface
              hit is occluded by furniture (|depth - plane hit| > 15 cm).
Fusion      : detections are merged across views per (surface, class) by overlap in surface coords;
              a region must be seen in >= 2 views on LiDAR/video tiers (phantom control).
Rules       : concealed-damage flags with the rule id that fired (RULES below).
Scope       : line items keyed to surface ids, quantities with intervals.
All model weights are disclosed in MODELS and fetched by scripts/fetch_weights.py.
"""
from __future__ import annotations

import numpy as np

MODELS = {
    "detector": "IDEA-Research/grounding-dino-base",
    "segmenter": "facebook/sam-vit-base",
    "verifier": "openai/clip-vit-large-patch14",
}

DAMAGE_CLASSES = {
    "water_stain": ["water stain", "water damage stain", "brown stain on ceiling"],
    "mold": ["mold", "black mold spots", "mildew"],
    "crack": ["crack in wall", "plaster crack", "ceiling crack"],
    "hole": ["hole in wall", "damaged drywall"],
    "peeling_paint": ["peeling paint", "bubbling paint", "flaking paint"],
}
NEGATIVES = ["a clean painted wall", "a shadow on a wall", "a picture frame", "a light reflection on a glossy surface",
             "a light switch or outlet", "a wet-look glossy tiled floor", "furniture", "a window", "a mirror",
             "tile grout lines", "a door", "the seam between two cabinet doors", "the edge of a countertop",
             "a corner where two walls meet", "a skirting board", "a drain or plug hole", "a towel hook",
             "a glass shower screen", "a clean ceiling", "a door handle"]
ROOM_TYPES = ["bathroom", "kitchen", "bedroom", "living room", "hallway", "laundry room", "closet", "dining room",
              "home office"]
WET_ROOMS = {"bathroom", "kitchen", "laundry room"}


class Models:
    _inst = None

    def __init__(self):
        import torch
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, CLIPModel, CLIPProcessor,
                                  SamModel, SamProcessor)
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.det_proc = AutoProcessor.from_pretrained(MODELS["detector"])
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(MODELS["detector"]).to(self.dev).eval()
        self.sam_proc = SamProcessor.from_pretrained(MODELS["segmenter"])
        self.sam = SamModel.from_pretrained(MODELS["segmenter"]).to(self.dev).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(MODELS["verifier"])
        self.clip = CLIPModel.from_pretrained(MODELS["verifier"]).to(self.dev).eval()
        self._text_cache = {}

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = Models()
        return cls._inst

    # ---- detection
    def detect(self, rgb, box_thr=0.20, text_thr=0.20):   # tuned: scripts/damage_recall.py
        from PIL import Image
        phrases = [p for v in DAMAGE_CLASSES.values() for p in v]
        text = ". ".join(phrases) + "."
        img = Image.fromarray(rgb)
        with self.torch.no_grad():
            inp = self.det_proc(images=img, text=text, return_tensors="pt").to(self.dev)
            out = self.det(**inp)
        res = self.det_proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=box_thr, text_threshold=text_thr, target_sizes=[img.size[::-1]])[0]
        dets = []
        labels = res.get("text_labels", res.get("labels"))
        for box, score, lab in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), labels):
            cls = _phrase_to_class(str(lab))
            if cls is None:
                continue
            x0, y0, x1, y1 = box
            # very large boxes are almost always whole walls/scenes, not damage regions
            if (x1 - x0) * (y1 - y0) > 0.6 * rgb.shape[0] * rgb.shape[1]:
                continue
            dets.append({"box": box, "score": float(score), "class": cls})
        return dets

    # ---- CLIP verification and room typing
    def _text_emb(self, texts):
        key = tuple(texts)
        if key not in self._text_cache:
            with self.torch.no_grad():
                t = self.clip_proc(text=list(texts), return_tensors="pt", padding=True).to(self.dev)
                e = _feat(self.clip.get_text_features(**t))
            self._text_cache[key] = e / e.norm(dim=-1, keepdim=True)
        return self._text_cache[key]

    def _img_emb(self, crops):
        from PIL import Image
        with self.torch.no_grad():
            i = self.clip_proc(images=[Image.fromarray(c) for c in crops], return_tensors="pt").to(self.dev)
            e = _feat(self.clip.get_image_features(**i))
        return e / e.norm(dim=-1, keepdim=True)

    def verify(self, crop, cls):
        """Probability-like score that the crop shows `cls` rather than its best look-alike.

        Softmax over {best positive phrase, every negative}; summing positives (the earlier version)
        let three near-synonyms outvote a single correct negative such as tile grout.
        """
        pos = [f"a photo of {p}" for p in DAMAGE_CLASSES[cls]]
        texts = pos + [f"a photo of {n}" for n in NEGATIVES]
        ie = self._img_emb([crop])
        te = self._text_emb(texts)
        logit = (100 * ie @ te.T)[0].cpu().numpy()
        z = np.concatenate([[logit[:len(pos)].max()], logit[len(pos):]])
        p = np.exp(z - z.max())
        return float(p[0] / p.sum())

    def room_type(self, images):
        if not images:
            return None, 0.0
        ie = self._img_emb(images)
        te = self._text_emb([f"a photo of a {r}" for r in ROOM_TYPES])
        p = (100 * ie @ te.T).softmax(-1).mean(0).cpu().numpy()
        k = int(np.argmax(p))
        return ROOM_TYPES[k], float(p[k])

    # ---- segmentation
    def segment(self, rgb, box):
        from PIL import Image
        with self.torch.no_grad():
            inp = self.sam_proc(Image.fromarray(rgb), input_boxes=[[list(map(float, box))]], return_tensors="pt").to(self.dev)
            out = self.sam(**inp, multimask_output=False)
        m = self.sam_proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]
        return m[0, 0].numpy().astype(bool)


def _feat(out):
    """transformers >= 5 returns a ModelOutput from get_*_features; older versions a tensor."""
    if hasattr(out, "norm"):
        return out
    for k in ("text_embeds", "image_embeds", "pooler_output"):
        v = getattr(out, k, None)
        if v is not None:
            return v
    return out[0]


def _phrase_to_class(lab):
    lab = lab.lower()
    for cls, phrases in DAMAGE_CLASSES.items():
        for p in phrases:
            if p in lab or lab in p:
                return cls
    for cls in DAMAGE_CLASSES:
        if cls.split("_")[0] in lab:
            return cls
    return None


# ----------------------------------------------------------------------------- geometry
class SurfaceCaster:
    """Ray casting against one room's plan surfaces (plan frame: u, v horizontal; h height above floor)."""

    def __init__(self, room, frame_yaw, floor_y_world):
        c, s = np.cos(-frame_yaw), np.sin(-frame_yaw)
        self.R = np.array([[c, -s], [s, c]])
        self.fy = floor_y_world
        self.room = room
        self.H = room["ceiling_height_m"]["value"] or room["ceiling_height_m"].get("lower_bound") or 2.5
        self.walls = [(w["id"], np.array(w["start"]), np.array(w["end"]), np.array(w["inward_normal"])) for w in room["walls"]]

    def to_plan(self, p_world):
        uv = p_world[..., [0, 2]] @ self.R.T
        return np.concatenate([uv, (p_world[..., 1] - self.fy)[..., None]], -1)

    def cast(self, origin_w, dirs_w):
        """origin (3,), dirs (N,3) world. Returns (surface index, t, surface coords (s,h) or (u,v))."""
        o = self.to_plan(origin_w[None])[0]
        d2 = dirs_w[:, [0, 2]] @ self.R.T
        dh = dirs_w[:, 1]
        N = len(dirs_w)
        best_t = np.full(N, np.inf)
        best_k = np.full(N, -1)
        best_c = np.zeros((N, 2))
        # floor (h = 0) and ceiling (h = H)
        for k, hp in ((len(self.walls), 0.0), (len(self.walls) + 1, self.H)):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (hp - o[2]) / dh
            p = o[:2] + d2 * t[:, None]
            ok = (t > 0.05) & (t < best_t) & self._inside(p)
            best_t[ok], best_k[ok], best_c[ok] = t[ok], k, p[ok]
        for k, (_, a, b, n) in enumerate(self.walls):
            e = b - a
            L = np.linalg.norm(e)
            if L < 1e-6:
                continue
            ed = e / L
            # solve o + d t = a + ed s
            den = d2[:, 0] * (-ed[1]) - d2[:, 1] * (-ed[0])
            with np.errstate(divide="ignore", invalid="ignore"):
                rx, ry = a[0] - o[0], a[1] - o[1]
                t = (rx * (-ed[1]) - ry * (-ed[0])) / den
                sp = (d2[:, 0] * ry - d2[:, 1] * rx) / den
            h = o[2] + dh * t
            ok = (t > 0.05) & (t < best_t) & (sp >= 0) & (sp <= L) & (h >= 0) & (h <= self.H)
            best_t[ok], best_k[ok] = t[ok], k
            best_c[ok] = np.stack([sp[ok], h[ok]], 1)
        return best_k, best_t, best_c

    def _inside(self, p):
        from matplotlib.path import Path
        return Path(np.array(self.room["polygon"])).contains_points(p)

    def surface_id(self, k):
        if k < len(self.walls):
            return self.walls[k][0], "wall"
        rid = self.room["id"]
        return (f"{rid}-F", "floor") if k == len(self.walls) else (f"{rid}-C", "ceiling")

    def contains_camera(self, origin_w):
        o = self.to_plan(origin_w[None])[0]
        return bool(self._inside(o[None, :2])[0])


def _rays(K, rgb_size, img_shape, T_wc, pix):
    """World ray directions for pixel coords `pix` (N,2) of an image of shape img_shape."""
    sx, sy = rgb_size[0] / img_shape[1], rgb_size[1] / img_shape[0]
    u, v = pix[:, 0] * sx, pix[:, 1] * sy
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(len(u))], 1)
    d = d @ T_wc[:3, :3].T
    return d / np.linalg.norm(d, axis=1, keepdims=True)


# ----------------------------------------------------------------------------- main entry
def _view_images(views, max_side=1024):
    import cv2
    from .capture import iter_video_frames
    vlist = views["views"]
    if "images" in views:
        for v in vlist:
            yield v, views["images"][v["key"]][:, :, ::-1]
        return
    idx = [v["frame_index"] for v in vlist]
    lookup = {v["frame_index"]: v for v in vlist}
    for i, img in iter_video_frames(views["video_path"], indices=idx):
        s = max_side / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        yield lookup[i], img[:, :, ::-1].copy()


def upright_k(T_wc):
    """Number of 90-degree CCW rotations (np.rot90) that make the image upright, from the pose.

    Phones held in portrait store landscape frames; detectors and CLIP are trained on upright photos.
    """
    c = T_wc[:3, :3].T @ np.array([0.0, 1.0, 0.0])        # world up in camera coords
    x, y = c[0], c[1]
    if abs(y) >= abs(x):
        return 0 if y < 0 else 2
    return 1 if x > 0 else 3


def analyse_damage(result, views, log=print, min_views=None, verify_thr=0.6, max_pix=4000):
    plan = result.get("_plan")
    frame = plan["frame"] if plan else result["_frame"]
    tier = result["tier"]
    if min_views is None:
        min_views = 1 if tier == "photo" else 2
    casters = [SurfaceCaster(r, frame.yaw, frame.floor_y) for r in result["rooms"]]
    M = Models.get()
    raw = []
    room_imgs = {i: [] for i in range(len(casters))}
    n_views = 0
    for v, rgb in _view_images(views):
        n_views += 1
        T = np.asarray(v["T_wc"])
        cam = T[:3, 3]
        rooms_here = [i for i, c in enumerate(casters) if c.contains_camera(cam)]
        if not rooms_here:
            continue
        ri = rooms_here[0]
        if len(room_imgs[ri]) < 6:
            import cv2
            room_imgs[ri].append(cv2.resize(rgb, (336, 252)))
        kr = upright_k(T)
        up = np.ascontiguousarray(np.rot90(rgb, kr))
        for d in M.detect(up):
            x0, y0, x1, y1 = [int(round(x)) for x in d["box"]]
            crop = up[max(y0, 0):y1, max(x0, 0):x1]
            if crop.size == 0 or min(crop.shape[:2]) < 12:
                continue
            pv = M.verify(crop, d["class"])
            if pv < verify_thr:
                continue
            mask_up = M.segment(up, d["box"])
            if d["class"] == "crack" and not _thin(mask_up):
                continue
            mask = np.rot90(mask_up, -kr)
            ys, xs = np.nonzero(mask)
            if len(xs) < 30:
                continue
            sel = np.random.default_rng(0).choice(len(xs), min(len(xs), max_pix), replace=False)
            pix = np.stack([xs[sel], ys[sel]], 1).astype(float) + 0.5
            dirs = _rays(np.asarray(v["K"]), v["rgb_size"], rgb.shape, T, pix)
            k, t, c = casters[ri].cast(cam, dirs)
            ok = k >= 0
            if "depth" in v and v["depth"] is not None:
                dep = v["depth"]
                dh, dw = dep.shape
                zi = dep[np.clip((pix[:, 1] / rgb.shape[0] * dh).astype(int), 0, dh - 1),
                         np.clip((pix[:, 0] / rgb.shape[1] * dw).astype(int), 0, dw - 1)]
                zray = t * (dirs @ T[:3, 2])                    # plane-hit depth along the optical axis
                ok &= (zi <= 0) | (np.abs(zi - zray) < 0.15)
            if ok.sum() < 20:
                continue
            ks, cnt = np.unique(k[ok], return_counts=True)
            ks_main = ks[np.argmax(cnt)]
            m2 = ok & (k == ks_main)
            pix_area = _pixel_area(t[m2], dirs[m2], T, v, rgb.shape)
            # metric area: mask pixels x fraction landing on the main surface x area of one pixel there
            area = len(xs) * (m2.sum() / len(sel)) * pix_area
            raw.append({"room": ri, "surface_k": int(ks_main), "class": d["class"], "score": d["score"],
                        "verify": pv, "coords": c[m2], "area_m2": float(area), "view": int(v["frame_index"])})
    log(f"damage: {len(raw)} verified detections from {n_views} views")
    regions = _merge(raw, casters, min_views, tier)
    # room semantics (used by rules and naming)
    for i, imgs in room_imgs.items():
        rt, p = M.room_type(imgs)
        if rt:
            result["rooms"][i]["room_type"] = {"label": rt, "confidence": round(p, 3), "source": "CLIP zero-shot"}
    for reg in regions:
        result["rooms"][reg.pop("_room")]["damage"].append(reg)
    apply_rules(result)
    build_scope(result)
    result["damage_models"] = MODELS
    return result


def _thin(mask, min_elong=4.0):
    """Cracks are long and thin: elongation of the mask's principal axes."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return False
    c = np.cov(np.stack([xs, ys]))
    w = np.sort(np.linalg.eigvalsh(c))
    return bool(np.sqrt(w[1] / max(w[0], 1e-6)) >= min_elong)


def _pixel_area(t, dirs, T, v, shape):
    """Approximate metric area of one image pixel at the hit distance (mean over hits)."""
    K = np.asarray(v["K"])
    sx = v["rgb_size"][0] / shape[1]
    fx = K[0, 0] / sx
    cosang = np.abs(dirs @ T[:3, 2])
    return float(np.mean((t * cosang / fx) ** 2 / np.maximum(cosang, 0.3)))


def _merge(raw, casters, min_views, tier):
    from .uncertainty import ErrorModel
    err = ErrorModel(tier)
    groups = []
    for d in raw:
        lo, hi = np.percentile(d["coords"], 5, axis=0), np.percentile(d["coords"], 95, axis=0)
        placed = False
        for g in groups:
            if g["room"] == d["room"] and g["surface_k"] == d["surface_k"] and g["class"] == d["class"]:
                glo, ghi = g["lo"], g["hi"]
                inter = np.maximum(0, np.minimum(hi, ghi) - np.maximum(lo, glo))
                if inter.prod() > 0 or np.linalg.norm((lo + hi) / 2 - (glo + ghi) / 2) < 0.3:
                    g["members"].append((lo, hi, d))
                    g["lo"], g["hi"] = np.minimum(lo, glo), np.maximum(hi, ghi)
                    placed = True
                    break
        if not placed:
            groups.append({"room": d["room"], "surface_k": d["surface_k"], "class": d["class"], "lo": lo, "hi": hi,
                           "members": [(lo, hi, d)]})
    out = []
    for g in groups:
        views = {m[2]["view"] for m in g["members"]}
        if len(views) < min_views:
            continue
        c = casters[g["room"]]
        sid, stype = c.surface_id(g["surface_k"])
        L = np.array([m[0] for m in g["members"]])
        Hh = np.array([m[1] for m in g["members"]])
        lo, hi = np.median(L, 0), np.median(Hh, 0)
        ext = hi - lo
        spread = np.std(Hh - L, 0) if len(L) > 1 else ext * 0.25
        area = float(ext.prod())
        area_s = float(np.hypot(spread[0] * ext[1], spread[1] * ext[0]))
        reg = {
            "id": f"{sid}-D{len(out)}", "_room": g["room"], "class": g["class"], "surface_id": sid, "surface_type": stype,
            "extent_m": {"along_m" if stype == "wall" else "x_m": err.generic(float(ext[0]), float(spread[0]), "damage_extent"),
                         "height_m" if stype == "wall" else "y_m": err.generic(float(ext[1]), float(spread[1]), "damage_extent")},
            "bbox_surface_m": [round(float(x), 3) for x in (*lo, *hi)],
            "area_m2": err.measure(area, max(area_s, 0.1 * area), "damage_area", "m2", 3),
            "bottom_height_m": round(float(lo[1]), 3) if stype == "wall" else None,
            "views": len(views),
            "detector_score": round(float(np.mean([m[2]["score"] for m in g["members"]])), 3),
            "verifier_prob": round(float(np.mean([m[2]["verify"] for m in g["members"]])), 3),
        }
        if stype == "wall":
            w = next(w for w in c.room["walls"] if w["id"] == sid)
            a, b = np.array(w["start"]), np.array(w["end"])
            dvec = (b - a) / max(np.linalg.norm(b - a), 1e-9)
            reg["footprint_plan"] = [list(a + dvec * lo[0]), list(a + dvec * hi[0])]
        out.append(reg)
    return out


# ----------------------------------------------------------------------------- rules
RULES = [
    ("R1-ceiling-stain", "water_stain on a ceiling", "Possible active leak or saturated insulation above the ceiling "
     "(plumbing, roof or HVAC condensate). Moisture-meter and thermal check of the ceiling cavity."),
    ("R2-wall-base-moisture", "water_stain or mold on a wall with its bottom edge < 0.30 m above the floor",
     "Possible wicking into the wall cavity / bottom plate and subfloor. Check baseboard and drywall base with a "
     "moisture meter; inspect behind baseboard."),
    ("R3-mold-extent", "mold area > 0.25 m2 on any surface",
     "Visible mold this size usually extends behind drywall (IICRC S520). Cavity inspection required; "
     "> 0.93 m2 (10 sq ft) calls for professional remediation."),
    ("R4-wet-room-shared-wall", "water_stain or mold on a wall shared with a wet room (bathroom/kitchen/laundry)",
     "Possible supply/drain leak inside the shared wall cavity."),
    ("R5-opening-corner-crack", "crack within 0.30 m of an opening's top corner",
     "Diagonal cracks from door/window corners indicate header deflection or settlement; structural check."),
    ("R6-floor-water", "water_stain or mold on a floor", "Possible subfloor moisture / delamination under the finish."),
    ("R7-under-window", "water_stain on a wall directly below a window", "Possible window flashing or sill failure; "
     "check the cavity below the sill."),
]


def apply_rules(result):
    rooms = result["rooms"]
    by_id = {r["id"]: r for r in rooms}
    wet = {r["id"] for r in rooms if r.get("room_type", {}).get("label") in WET_ROOMS}
    shared = {}
    for a in result["property"]["adjacency"]:
        if "shared_wall" in a["via"] or "door" in a["via"]:
            shared.setdefault(a["a"], set()).add(a["b"])
            shared.setdefault(a["b"], set()).add(a["a"])
    for r in rooms:
        flags = []
        for d in r["damage"]:
            cls, st = d["class"], d["surface_type"]
            fired = []
            if cls == "water_stain" and st == "ceiling":
                fired.append("R1-ceiling-stain")
            if cls in ("water_stain", "mold") and st == "wall" and d.get("bottom_height_m") is not None \
                    and d["bottom_height_m"] < 0.30:
                fired.append("R2-wall-base-moisture")
            if cls == "mold" and d["area_m2"]["value"] > 0.25:
                fired.append("R3-mold-extent")
            if cls in ("water_stain", "mold") and st == "wall" and (shared.get(r["id"], set()) & wet) and r["id"] not in wet:
                fired.append("R4-wet-room-shared-wall")
            if cls in ("water_stain", "mold") and st == "floor":
                fired.append("R6-floor-water")
            if st == "wall":
                for o in r["openings"]:
                    if o["wall_id"] != d["surface_id"]:
                        continue
                    w = next(w for w in r["walls"] if w["id"] == d["surface_id"])
                    a = np.array(w["start"])
                    s0 = np.linalg.norm(np.array(o["start"]) - a)
                    s1 = np.linalg.norm(np.array(o["end"]) - a)
                    top = o["sill_height_m"] + o["height_m"]["value"]
                    b = d["bbox_surface_m"]
                    if cls == "crack":
                        for sc in (s0, s1):
                            if min(abs(b[0] - sc), abs(b[2] - sc)) < 0.3 and abs(b[1] - top) < 0.4:
                                fired.append("R5-opening-corner-crack")
                    if cls == "water_stain" and o["kind"] == "window" and b[0] < max(s0, s1) and b[2] > min(s0, s1) \
                            and b[3] <= o["sill_height_m"] + 0.1:
                        fired.append("R7-under-window")
            for rid in sorted(set(fired)):
                rule = next(x for x in RULES if x[0] == rid)
                flags.append({"rule": rid, "condition": rule[1], "finding": rule[2], "damage_id": d["id"],
                              "surface_id": d["surface_id"]})
        r["concealed_damage_flags"] = flags
    result["rules"] = [{"id": a, "condition": b, "finding": c} for a, b, c in RULES]


# ----------------------------------------------------------------------------- scope
def _q(v, lo, hi, unit):
    return {"value": round(v, 3), "ci95": [round(lo, 3), round(hi, 3)], "unit": unit}


def build_scope(result):
    """Scope line items keyed to the surface each one repairs."""
    def paint(sid, stype, S):
        return ("PNT-2C", f"Paint {stype} (full surface, 2 coats)", _q(S, S * 0.97, S * 1.03, "m2"), sid)

    def cutback(x):
        return (np.sqrt(max(x, 0)) + 0.6) ** 2          # 0.3 m cut-back margin on each side

    result["scope_line_items"] = []
    for r in result["rooms"]:
        surf = {s["id"]: s for s in r["surfaces"]}
        items = []
        for d in r["damage"]:
            sid, st = d["surface_id"], d["surface_type"]
            a, (alo, ahi) = d["area_m2"]["value"], d["area_m2"]["ci95"]
            S = surf.get(sid, {"area_m2": a})["area_m2"]
            cls = d["class"]
            if cls == "water_stain":
                items.append(("PNT-SEAL", "Seal stain with stain-blocking primer", _q(a, alo, ahi, "m2"), sid))
            elif cls == "mold":
                items.append(("MLD-REM", "Mold remediation incl. containment (IICRC S520)",
                              _q(cutback(a), cutback(alo), cutback(ahi), "m2"), sid))
                items.append(("DRY-RR", "Remove and replace drywall (cut-back area)",
                              _q(cutback(a), cutback(alo), cutback(ahi), "m2"), sid))
            elif cls == "crack":
                L = max(d["extent_m"].values(), key=lambda e: e["value"])
                items.append(("DRY-TAPE", "Tape, mud and sand crack", _q(L["value"], *L["ci95"], "m"), sid))
            elif cls == "hole":
                items.append(("DRY-PATCH", "Drywall patch", _q(a, alo, ahi, "m2"), sid))
            elif cls == "peeling_paint":
                items.append(("PNT-PREP", "Scrape, sand and prime", _q(a, alo, ahi, "m2"), sid))
            items.append(paint(sid, st, S))
        for f in r["concealed_damage_flags"]:
            items.append(("INSP-MOIST", f"Moisture-meter / cavity inspection ({f['rule']})", _q(1, 1, 1, "ea"),
                          f["surface_id"]))
        seen, out = set(), []
        for code, desc, qty, sid in items:
            if code in ("PNT-2C", "INSP-MOIST") and (code, sid, desc) in seen:
                continue                              # one repaint / inspection per surface
            seen.add((code, sid, desc))
            out.append({"id": f"{r['id']}-S{len(out)}", "code": code, "description": desc, "quantity": qty,
                        "room_id": r["id"], "surface_id": sid})
        r["scope"] = out
        result["scope_line_items"].extend(out)
