"""Error budget -> per-measurement standard deviations and calibrated 95% intervals.

sigma^2 = random (from the fit: point scatter / sqrt(n_eff))
        + systematic surface term (sensor bias, per surface)
        + scale term (relative scale error x length)    <- dominates on video / photo tiers
The tier constants are priors; `calibration.json` holds the multiplier k per tier and quantity
learned from the benchmark (empirical coverage analysis, see bench/calibrate.py).
"""
from __future__ import annotations

import json
import os

import numpy as np

TIER_MODEL = {
    # scale_rel: relative std of metric scale; surface: per-surface position bias std (m);
    # opening: jamb localisation bias std (m)
    "lidar": {"scale_rel": 0.002, "surface": 0.004, "opening": 0.008, "height": 0.004},
    "video": {"scale_rel": 0.020, "surface": 0.015, "opening": 0.03, "height": 0.02},
    "photo": {"scale_rel": 0.045, "surface": 0.03, "opening": 0.05, "height": 0.04},
}

_CAL_PATH = os.path.join(os.path.dirname(__file__), "calibration.json")


def _calib():
    if os.path.exists(_CAL_PATH):
        with open(_CAL_PATH) as f:
            return json.load(f)
    return {}


class ErrorModel:
    def __init__(self, tier, scale_rel=None):
        self.tier = tier
        self.m = dict(TIER_MODEL[tier])
        if scale_rel is not None:
            self.m["scale_rel"] = scale_rel
        self.k = _calib().get(tier, {})

    def _k(self, q):
        return float(self.k.get(q, 1.0))

    def measure(self, value, sigma, q, unit="m", digits=4):
        if value is None:
            return {"value": None, "sigma": None, "ci95": None, "unit": unit}
        half = 1.96 * self._k(q) * sigma
        return {"value": round(float(value), digits), "sigma": round(float(sigma), digits + 1),
                "ci95": [round(float(value - half), digits), round(float(value + half), digits)], "unit": unit}

    def wall_length(self, L, s_a, s_b):
        s = np.sqrt(s_a ** 2 + s_b ** 2 + 2 * self.m["surface"] ** 2 + (L * self.m["scale_rel"]) ** 2)
        return self.measure(L, s, "wall_length")

    def area(self, A, edge_terms):
        """edge_terms: list of (edge_length, coord_sigma)."""
        s2 = sum((l * (s ** 2 + self.m["surface"] ** 2) ** 0.5) ** 2 for l, s in edge_terms)
        s2 += (2 * A * self.m["scale_rel"]) ** 2
        return self.measure(A, np.sqrt(s2), "area", unit="m2", digits=3)

    def height(self, H, s_f, s_c, observed=True):
        if H is None:
            return {"value": None, "sigma": None, "ci95": None, "unit": "m", "observed": False}
        s = np.sqrt(s_f ** 2 + s_c ** 2 + 2 * self.m["height"] ** 2 + (H * self.m["scale_rel"]) ** 2)
        if not observed:
            # no ceiling evidence: refuse a point estimate rather than emit confident garbage
            return {"value": None, "sigma": None, "ci95": None, "unit": "m", "observed": False,
                    "lower_bound": round(float(H), 3),
                    "note": "ceiling not observed in this capture; lower_bound is the highest wall point seen"}
        out = self.measure(H, s, "ceiling_height")
        out["observed"] = True
        return out

    def opening_width(self, W, s):
        s = np.sqrt(s ** 2 + self.m["opening"] ** 2 + (W * self.m["scale_rel"]) ** 2)
        return self.measure(W, s, "opening_width")

    def generic(self, v, s, q="generic", unit="m"):
        return self.measure(v, np.sqrt(s ** 2 + (abs(v) * self.m["scale_rel"]) ** 2), q, unit)
