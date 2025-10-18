from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

from .substring import Substring
from .cell import Cell


class PVString:
    """
    A PV string made of substrings in series.
    Each Substring handles its own physics (cells + bypass).
    The string's total voltage at a given current is the sum of the substrings' voltages.
    """

    def __init__(self, substrings: List[Substring]):
        assert len(substrings) >= 1, "Need at least one Substring"
        self.substrings: List[Substring] = substrings

    def set_conditions(self, irradiance: float, temperature_c: float) -> None:
        """Forward conditions to each substring."""
        for sub in self.substrings:
            sub.set_conditions(irradiance, temperature_c)

    def v_at_i(self, current: float) -> float:
        """
        String voltage at a given current (A).
        Relies on each Substring implementing v_at_i(current).
        """
        v_sum = 0.0
        for sub in self.substrings:
            v_sum += sub.v_at_i(current)
        return float(v_sum)

    def i_at_v(self, v_target: float, tol: float = 1e-7, max_iter: int = 80) -> float:
        """
        String current given voltage.
        """
        lo = 0.0
        hi = max(self._estimate_imax(), 0.1)

        def f(i: float) -> float:
            # positive when V(I) is above the target, negative when below
            return self.v_at_i(i) - v_target

        # essentially the IV curve is current at 0 is the max V
        # if the voltage at current=0 is less than our target voltage, return 0 since that is the max possible
        flo = f(lo)
        if flo<=0:
            return 0.0

        tries = 0
        fhi = f(hi)
        # while the v at hi current is greater than the v target, increase current to decrease voltage
        # this shifts the current range to capture more values of V
        while fhi > 0.0 and tries<8:
            hi *= 1.5
            fhi = f(hi)
            tries+=1

        # if we can't include the voltage from increasing current
        # we just return which ever current was higher

        if fhi > 0.0:
            return hi if abs(fhi) < abs(flo) else lo

        # bisection search to find the correct current
        # get the middle i and solve for v and determine how far off from target
        # if its greater than target, change low to the midpoint
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            fmid = f(mid)
            if abs(fmid) < tol or abs(hi - lo) < 1e-9:
                return float(mid)
            if fmid > 0.0:
                lo, flo = mid, fmid
            else:
                hi, fhi = mid, fmid

        return float(0.5 * (lo + hi))

    def _estimate_imax(self) -> float:
        """
        Estimate a safe current sweep cap using the cells sitting inside the substrings.
        Find all cells i at v(0) and then get the median if there are some outliers
        Under no shading they r all the same
        """
        isc_list = []
        for sub in self.substrings:
            # Expect Substring to expose its cells as `cell_list`
            for c in getattr(sub, "cell_list", []):
                if isinstance(c, Cell):
                    isc_list.append(float(c.solve_i_at_v(0.0)))
        if not isc_list:
            return 1.0  # fallback
        # Median is robust to outliers (e.g., heavy shading in only one substring)
        imax = float(np.median(isc_list))
        return max(0.1, imax)

    # -------- Curve builders --------

    def iv_curve(self, points: int = 400, i_max: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a string I–V curve by sweeping current and summing substring voltages.
        Returns (V_array, I_array) with ascending I from 0..i_max.
        """
        if i_max is None:
            i_max = self._estimate_imax()

        I = np.linspace(0.0, float(i_max), int(points))
        V = np.zeros_like(I, dtype=float)
        for k, i in enumerate(I):
            V[k] = self.v_at_i(float(i))
        return V, I

    def pv_curve(self, points: int = 400, i_max: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convenience: return (V, I, P) with P = V*I over the same sweep.
        """
        V, I = self.iv_curve(points=points, i_max=i_max)
        P = V * I
        return V, I, P

    # -------- Scalar operating points --------

    def voc(self) -> float:
        """
        Open-circuit voltage of the string ≈ sum of substring voltages at ~0 A.
        We evaluate at a tiny positive current to avoid edge cases.
        """
        tiny = 1e-9
        return float(self.v_at_i(tiny))

    def isc(self, tol: float = 1e-6, max_iter: int = 60) -> float:
        """
        Short-circuit current of the string, found by solving V_total(I)=0 with bisection.
        We bracket [0, I_hi], where I_hi is the estimated max current.
        """
        def f(i: float) -> float:
            return self.v_at_i(i)

        lo, hi = 0.0, self._estimate_imax()

        f_lo = f(lo)              # should be ~Voc (>0)
        f_hi = f(hi)              # likely <= 0 (or near), else extend the bracket

        # If the voltage at I_hi is still positive, expand the bracket a bit
        tries = 0
        while f_hi > 0.0 and tries < 6:
            hi *= 1.5
            f_hi = f(hi)
            tries += 1

        if f_lo * f_hi > 0.0:
            # Could not bracket a root; return best effort
            return float(hi if abs(f_hi) < abs(f_lo) else lo)

        # Bisection on V(I)=0
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            if abs(f_mid) < tol or abs(hi - lo) < 1e-9:
                return float(mid)
            if np.sign(f_mid) == np.sign(f_lo):
                lo, f_lo = mid, f_mid
            else:
                hi = mid
        return float(0.5 * (lo + hi))

    def mpp(self, points: int = 600, i_max: Optional[float] = None):
        """
        Brute-force MPP over a sweep (good enough for tests/plots).
        Returns (Vmpp, Impp, Pmpp, Voc_est, Isc_est).
        """
        V, I, P = self.pv_curve(points=points, i_max=i_max)
        idx = int(np.argmax(P))
        Vmpp = float(V[idx])
        Impp = float(I[idx])
        Pmpp = float(P[idx])
        Voc_est = float(self.voc())
        Isc_est = float(self.isc())
        return Vmpp, Impp, Pmpp, Voc_est, Isc_est
