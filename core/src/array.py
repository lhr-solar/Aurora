from .string import PVString
from typing import List, Tuple
import numpy as np


class Array:
    def __init__(self, string_list: List[PVString]):
        self.string_list = list(string_list)
        self.cached_iv = None

    def set_conditions(self, irradiance, temperature):
        for s in self.string_list:
            s.set_conditions(irradiance, temperature)
        self.cached_iv = None

    def isc(self) -> float:
        """Parallel array short-circuit current = sum of string Isc."""
        total = 0.0
        for s in self.string_list:
            try:
                total += float(s.isc())
            except Exception:
                # fallback via IV at V≈0
                try:
                    Vs, Is = s.iv_curve(points=32)
                    idx = int(np.nanargmin(np.abs(np.asarray(Vs))))
                    total += float(Is[idx])
                except Exception:
                    pass
        return float(total)

    def voc(self) -> float:
        """Parallel array Voc = min of string Voc."""
        if not self.string_list:
            return 0.0
        return float(min(s.voc() for s in self.string_list))

    def i_at_v(self, v: float) -> float:
        """Sum string currents at a common voltage (parallel rule)."""
        total = 0.0
        for s in self.string_list:
            try:
                total += float(s.i_at_v(float(v)))
            except Exception:
                # fallback: interpolate that string's IV
                try:
                    Vs, Is = s.iv_curve(points=200)
                    total += float(np.interp(float(v), Vs, Is))
                except Exception:
                    pass
        return float(total)

    def iv_curve(self, points: int) -> Tuple[np.ndarray, np.ndarray]:
        """Array IV with strings in parallel: V grid up to min Voc; I sums."""
        voc_val = float(self.voc())
        V = np.linspace(0.0, voc_val, int(points))
        I_total = np.zeros_like(V, dtype=float)

        # coarse sampling per string, then interpolate onto common V grid
        coarse_pts = max(64, int(max(8, points // 4)))
        per_string = []
        for s in self.string_list:
            try:
                Vs, Is = s.iv_curve(points=coarse_pts)
                Vs = np.asarray(Vs, dtype=float)
                Is = np.asarray(Is, dtype=float)
                order = np.argsort(Vs)
                Vs = Vs[order]; Is = Is[order]
                per_string.append((Vs, Is))
                I_total += np.interp(V, Vs, Is)
            except Exception:
                per_string.append((None, None))

        # refine near MPP and near Voc using exact i_at_v where available
        P = V * I_total
        mpp_idx = int(np.nanargmax(P)) if np.any(np.isfinite(P)) else len(V) - 1
        window = max(3, int(points * 0.03))
        voc_window = max(2, int(points * 0.02))
        refine_idxs = set(range(max(0, mpp_idx - window), min(len(V), mpp_idx + window + 1)))
        refine_idxs.update(range(max(0, len(V) - voc_window), len(V)))

        max_refine = max(20, int(points * 0.05))
        if len(refine_idxs) > max_refine:
            idxs = sorted(refine_idxs, key=lambda x: abs(x - mpp_idx))[:max_refine]
            refine_idxs = set(idxs)

        for idx in sorted(refine_idxs):
            v = float(V[idx])
            acc = 0.0
            for s in self.string_list:
                try:
                    acc += float(s.i_at_v(v))
                except Exception:
                    pass
            if acc != 0.0:
                I_total[idx] = acc

        return V, I_total

    def mpp(self) -> Tuple[float, float, float]:
        """Return (Vmpp, Impp, Pmpp) for the array."""
        V, I = self.iv_curve(points=600)
        P = V * I
        k = int(np.nanargmax(P))
        return float(V[k]), float(I[k]), float(P[k])
