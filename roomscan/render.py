"""Rendered floor plan (PNG + SVG): rooms, dimensioned walls, openings, labels, damage overlays."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Polygon

ROOM_FILL = "#f4efe6"
WALL = "#2b2b2b"
DOOR = "#b5532a"
WINDOW = "#3a7bbf"
PASSAGE = "#7a9a3a"
DAMAGE = "#d1342f"


def _fmt_len(m, s=None):
    t = f"{m:.2f} m"
    if s is not None:
        t += f" ±{s * 100:.1f}cm" if s < 0.1 else f" ±{s:.2f}m"
    return t


def render_plan(result, path_png, path_svg=None, background=None, title=None, show_dims=True):
    """result: the output JSON dict (plan coordinates in metres)."""
    rooms = result["rooms"]
    allv = np.concatenate([np.array(r["polygon"]) for r in rooms]) if rooms else np.zeros((1, 2))
    lo, hi = allv.min(0) - 1.0, allv.max(0) + 1.0
    span = hi - lo
    fig_w = 12
    fig_h = max(4, fig_w * span[1] / max(span[0], 1e-6))
    fig, ax = plt.subplots(figsize=(fig_w, min(fig_h, 18)))
    if background is not None:
        img, ext = background
        ax.imshow(img, extent=ext, origin="lower", cmap="Greys", alpha=0.35, interpolation="nearest")
    for r in rooms:
        v = np.array(r["polygon"])
        ax.add_patch(Polygon(v, closed=True, facecolor=ROOM_FILL, edgecolor=WALL, lw=2.2, zorder=2))
        c = v.mean(0) if "label_point" not in r else np.array(r["label_point"])
        a = r["floor_area_m2"]
        ch = r["ceiling_height_m"]
        txt = f"{r['name']}\n{a['value']:.2f} m² (±{a['ci95'][1] - a['value']:.2f})"
        if ch["value"] is not None:
            txt += f"\nH {ch['value']:.2f} m" + ("" if ch.get("observed", True) else " (est.)")
        ax.text(c[0], c[1], txt, ha="center", va="center", fontsize=8, zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8))
        if show_dims:
            for w in r["walls"]:
                p0, p1 = np.array(w["start"]), np.array(w["end"])
                L = w["length_m"]["value"]
                if L < 0.4:
                    continue
                mid = (p0 + p1) / 2
                d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
                nrm = np.array(w["inward_normal"])
                tp = mid + nrm * 0.22
                rot = np.degrees(np.arctan2(d[1], d[0]))
                if rot > 90 or rot < -90:
                    rot += 180
                ax.text(tp[0], tp[1], f"{L:.2f}", rotation=rot, ha="center", va="center",
                        fontsize=6.5, color="#444", zorder=5)
        for o in r["openings"]:
            p0, p1 = np.array(o["start"]), np.array(o["end"])
            col = {"door": DOOR, "window": WINDOW}.get(o["kind"], PASSAGE)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="white", lw=4.0, zorder=3, solid_capstyle="butt")
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=col, lw=1.6, zorder=4,
                    ls="-" if o["kind"] != "window" else (0, (1, 1)))
            if o["kind"] == "door":
                wdt = np.linalg.norm(p1 - p0)
                nrm = np.array(o["inward_normal"])
                ang0 = np.degrees(np.arctan2(*(p1 - p0)[::-1]))
                sweep = 90 if np.cross(p1 - p0, nrm) > 0 else -90
                a0, a1 = (ang0, ang0 + sweep) if sweep > 0 else (ang0 + sweep, ang0)
                ax.add_patch(Arc(p0, 2 * wdt, 2 * wdt, theta1=a0, theta2=a1, color=col, lw=0.8, zorder=4))
        for dmg in r.get("damage", []):
            if "footprint_plan" in dmg:
                fp = np.array(dmg["footprint_plan"])
                ax.plot(fp[:, 0], fp[:, 1], color=DAMAGE, lw=3, zorder=7)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.axis("off")
    # scale bar
    sb = 1.0
    ax.plot([lo[0] + 0.3, lo[0] + 0.3 + sb], [lo[1] + 0.3] * 2, color="k", lw=3)
    ax.text(lo[0] + 0.3 + sb / 2, lo[1] + 0.45, "1 m", ha="center", fontsize=8)
    t = title or f"{result.get('capture_id', '')} — {result.get('tier', '')} tier"
    tot = result.get("property", {}).get("total_floor_area_m2", {})
    if tot:
        t += f" — {tot['value']:.1f} m² total"
    ax.set_title(t, fontsize=11)
    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    if path_svg:
        fig.savefig(path_svg)
    plt.close(fig)
