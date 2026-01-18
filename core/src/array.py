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
    
    def v_at_i(self, current: float) -> float:
        """
        Array voltage at a given current (only meaningful for series topology).
        For series topology: array voltage is sum of string voltages at that current.
        """
        if self.topology != "series":
            raise ValueError("v_at_i is only valid for series topology")
        return float(sum(s.v_at_i(current) for s in self.string_list))

    def i_at_v(self, v_target: float, tol_v: float = 1e-6, maxit: int = 60) -> float:
        """
        Array current at a given terminal voltage.

        - parallel topology: I(V) = sum_i string_i_at_v(V)
        - series topology:   V(I) = sum_i string_v_at_i(I) -> invert with bisection
        """
        v_target = float(v_target)

        if self.topology == "parallel":
            # Same terminal voltage across strings, currents add
            return float(sum(s.i_at_v(v_target, tol_v=tol_v, maxit=maxit) for s in self.string_list))

        # series topology: same current through strings, voltages add; invert V(I)
        V0 = self.v_at_i(0.0)  # ~ Voc_total
        if v_target >= V0 - 1e-12:
            return 0.0

        Isc_est = float(max(1e-12, self.isc()))
        Visc = self.v_at_i(Isc_est)
        if v_target <= Visc + 1e-12:
            return Isc_est

        lo, hi = 0.0, Isc_est

        for _ in range(maxit):
            mid = 0.5 * (lo + hi)
            Vmid = self.v_at_i(mid)
            if abs(Vmid - v_target) <= tol_v:
                return float(mid)
            # V(I) is decreasing in I
            if Vmid < v_target:
                hi = mid
            else:
                lo = mid

        return float(0.5 * (lo + hi))