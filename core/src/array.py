# array.py
from typing import List, Tuple
import numpy as np
from .string import PVString

class Array:
    def __init__(self, string_list: List[PVString], topology: str = "series"):
        """
        topology: "series" (sum voltages at same I) or "parallel" (sum currents at same V)
        """
        assert topology in ("series", "parallel")
        self.string_list = list(string_list)
        self.topology = topology
        self.cached_iv = None

    def set_conditions(self, irradiance, temperature_c):
        for s in self.string_list:
            s.set_conditions(irradiance, temperature_c)
        self.cached_iv = None

    def isc(self) -> float:
        if self.topology == "parallel":
            return float(sum(s.isc() for s in self.string_list))
        else:  # series
            return float(min(s.isc() for s in self.string_list))

    def voc(self) -> float:
        if self.topology == "parallel":
            return float(max(s.voc() for s in self.string_list))
        else:  # series
            return float(sum(s.voc() for s in self.string_list))

    def iv_curve(self, points: int = 600) -> Tuple[np.ndarray, np.ndarray]:
        # array.py (inside Array.iv_curve)
        if self.topology == "parallel":
            # 1) Get dense (V,I) for every string
            per = []
            vocs = []
            for s in self.string_list:
                V_s, I_s = s.iv_curve(points=max(1600, points))  # dense helps the knee
                # Ensure strictly increasing V for interpolation
                order = np.argsort(V_s)
                V_s = V_s[order]
                I_s = I_s[order]
                # Clamp tiny negative currents (numeric noise) to 0 near Voc
                I_s = np.maximum(I_s, 0.0)
                per.append((V_s, I_s))
                vocs.append(float(V_s[-1]))

            # 2) Shared voltage grid across all strings
            Vmin = 0.0
            Vmax = float(max(vocs))
            V = np.linspace(Vmin, Vmax, int(points))

            # 3) Interpolate I(V) for each string on the shared grid and sum
            I = np.zeros_like(V)

            eps = 1e-12  # tiny floor to avoid log(0)
            for (V_s, I_s) in per:
                # ensure strictly increasing V for safe interp (you already sorted)
                # clamp tiny negatives to zero
                I_s = np.maximum(I_s, 0.0)

                # Interpolate in log-space so the tail bends instead of forming a chord
                I_pos = np.maximum(I_s, eps)
                y = np.log(I_pos)

                # left: log Isc; right: treat beyond Voc as -inf (=> 0 current)
                y_left = float(np.log(I_pos[0]))
                y_right = -np.inf

                y_interp = np.interp(V, V_s, y, left=y_left, right=y_right)
                # exp(-inf) -> 0
                I += np.exp(y_interp, where=np.isfinite(y_interp), out=np.zeros_like(y_interp))

            return V, I

        else:
            # series: sweep I, sum V(I)
            iscs = [s.isc() for s in self.string_list]
            I = np.linspace(0.0, min(iscs), points)
            V = np.zeros_like(I)
            for k, i in enumerate(I):
                V[k] = sum(s.v_at_i(i) for s in self.string_list)
            return V, I

    def mpp(self) -> Tuple[float, float, float]:
        V, I = self.iv_curve(points=600)
        P = V * I
        idx = int(np.nanargmax(P))
        return float(V[idx]), float(I[idx]), float(P[idx])
