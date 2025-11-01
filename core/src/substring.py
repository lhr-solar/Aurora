from .cell import Cell
from .bypassdiode import Bypass_Diode
from typing import List, Tuple
import numpy as np

class Substring:
    def __init__(self, cell_list: List[Cell], bypass: Bypass_Diode = None):
        self.cell_list = cell_list
        self.bypass = bypass  # Only one bypass diode per substring?
        self.cached_iv = None

        self.voltage = None
        self.short_circuit_current = None
        self.total_voltage = None
        self.bypass_total_voltage = None
    
    def set_conditions(self, irradiance, temperature) :
        for cell in self.cell_list :
            cell.set_conditions(irradiance, temperature)
        if self.bypass:
            self.bypass.set_temperature(temperature)
        self.cached_iv = None
    
    def isc(self) -> float:
        """Short-circuit current estimate for this substring (A)."""
        # Use each cell's short-circuit current; substring Isc is the minimum
        # of its cell Isc values (cells in series limit current).
        # Cell API exposes a solver for I at a specific voltage; use V=0 to
        # query Isc for each cell and take the minimum.
        vals = []
        for cell in self.cell_list:
            try:
                vals.append(float(cell.solve_i_at_v(0.0)))
            except Exception:
                # fallback to any attribute that may exist
                v = getattr(cell, "isc_ref", None)
                if v is not None:
                    vals.append(float(v))
        if not vals:
            return 0.0
        return float(min(vals))

    def voc(self) -> float:
        # Sum the per-cell open-circuit voltages. Use v_at_i(0.0) which
        # asks the cell for the voltage at zero current (≈Voc) under current
        # operating conditions.
        total = 0.0
        for cell in self.cell_list:
            try:
                total += float(cell.v_at_i(0.0))
            except Exception:
                total += float(getattr(cell, "voc_ref", 0.0))
        return total

    def iv_curve(self, points: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (V, I) arrays for this substring with V ascending.

        We sweep current from 0..Isc and compute V(I), then sort/interpolate
        so the returned V array is ascending and I matches.
        """
        isc_val = float(self.isc())
        i_values = np.linspace(0.0, isc_val, int(points))
        v_values = np.array([self.v_at_i(float(i)) for i in i_values], dtype=float)
        I = np.asarray(i_values, dtype=float)
        V = np.asarray(v_values, dtype=float)
        # Ensure V is sorted ascending for callers: sort by V and reorder I accordingly
        if V.size >= 2:
            order = np.argsort(V)
            Vs = V[order]
            Is = I[order]
            return Vs, Is
        return V, I

    def mpp(self) -> Tuple[float, float, float, float, float]:
        # Use per-cell vmpp/impp attributes when available (Cell stores vmpp, impp)
        v_mpp_total = 0.0
        impp_vals = []
        for cell in self.cell_list:
            v_mpp_total += float(getattr(cell, "vmpp", getattr(cell, "v_mpp", 0.0)))
            impp_vals.append(float(getattr(cell, "impp", getattr(cell, "i_mpp", 0.0))))
        i_mpp = min(impp_vals) if impp_vals else 0.0
        p_mpp = v_mpp_total * i_mpp
        return v_mpp_total, i_mpp, p_mpp, self.voc(), self.voc()

    # pre: current > 0
    def v_at_i(self, current: float) -> float:
        cell_voltages = sum(cell.v_at_i(current) for cell in self.cell_list)
        if self.bypass:
            bypass_voltage = self.bypass.v_at_i(current)
            return max(cell_voltages, bypass_voltage)
        return cell_voltages
