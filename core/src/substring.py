from .cell import Cell
from .bypassdiode import Bypass_Diode
from typing import List, Tuple
import numpy as np

class Substring:
    def __init__(self, cell_list: List[Cell], bypass: Bypass_Diode = None):
        self.cell_list = cell_list
        self.bypass = bypass  # Only one bypass diode per substring?
        # cache per-cell IV info to avoid repeated expensive solves
        # maps a resolution key (int points) -> { 'i_values': np.ndarray, 'per_cell_v': [np.ndarray,...] }
        self.cached_iv = {}

        self.voltage = None
        self.short_circuit_current = None
        self.total_voltage = None
        self.bypass_total_voltage = None
    
    def set_conditions(self, irradiance, temperature) :
        for cell in self.cell_list :
            cell.set_conditions(irradiance, temperature)
        if self.bypass:
            self.bypass.set_temperature(temperature)
        # clear cached IV curves when conditions change
        self.cached_iv.clear()
    
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

        Under partial shading (and with bypass diodes), the substring operating
        current can exceed the weakest cell's Isc (because shaded cells can be
        driven into reverse bias and bypass can conduct). So we sweep up to ~1.2x
        the *maximum* cell Isc (under current conditions).
        """
        # Compute per-cell Isc under current conditions
        isc_cells: List[float] = []
        for cell in self.cell_list:
            try:
                isc_cells.append(float(cell.solve_i_at_v(0.0)))
            except Exception:
                isc_cells.append(float(getattr(cell, "isc_ref", 0.0)))

        # Use max cell Isc and also ensure we at least cover substring Isc()
        i_max = max(isc_cells) if isc_cells else float(self.isc())
        i_max = max(i_max, float(self.isc()))
        if i_max <= 0.0:
            return np.array([], dtype=float), np.array([], dtype=float)

        i_values = np.linspace(0.0, 1.2 * i_max, int(points))

        # Vectorized voltage calculation (uses cached per-cell IV curves)
        v_values = self.v_at_i_vector(i_values, cache_points=max(128, int(points)))

        I = np.asarray(i_values, dtype=float)
        V = np.asarray(v_values, dtype=float)

        # Sort so V is ascending (many callers assume this)
        if V.size >= 2:
            order = np.argsort(V)
            return V[order], I[order]
        return V, I

    def v_at_i_vector(self, i_values, cache_points: int = 256):
        """Return an array of substring voltages for each current in i_values.

        This builds per-cell IV curves once (using Cell.get_iv_curve) and
        interpolates each cell's voltage at the requested currents, then sums
        across cells. Results are cached keyed by cache_points to amortize
        the expensive per-cell solves.
        """
        i_values = np.asarray(i_values, dtype=float)
        if i_values.size == 0:
            return np.array([], dtype=float)

        key = int(cache_points)
        cache = self.cached_iv.get(key)
        if cache is None:
            # build per-cell interpolants at this resolution
            cell_curves = []   # list of (i_sorted, v_sorted)
            max_i = 0.0

            for cell in self.cell_list:
                # include reverse-bias region so shaded cells can be forced above Isc
                pts = cell.get_iv_curve(key, vmin=-0.8, vmax=None)
                v_arr = np.array([p[0] for p in pts], dtype=float)
                i_arr = np.array([p[1] for p in pts], dtype=float)

                # Ensure i_arr is ascending for interpolation
                order = np.argsort(i_arr)
                i_sorted = i_arr[order]
                v_sorted = v_arr[order]

                # Remove duplicate current samples (np.interp expects increasing x)
                i_unique, idx = np.unique(i_sorted, return_index=True)
                v_unique = v_sorted[idx]

                if i_unique.size:
                    max_i = max(max_i, float(i_unique[-1]))

                cell_curves.append((i_unique, v_unique))

            # Define a shared current grid that spans the *largest* cell current range
            if max_i <= 0.0:
                i_grid = np.array([], dtype=float)
                per_cell_v = []
            else:
                i_grid = np.linspace(0.0, 1.2 * max_i, key)  # key = cache_points
                per_cell_v = [np.interp(i_grid, i_c, v_c) for (i_c, v_c) in cell_curves]

            cache = {"i_values": i_grid, "per_cell_v": per_cell_v}
            self.cached_iv[key] = cache

        # Now interpolate per-cell cached voltages to requested i_values and sum
        i_grid = cache["i_values"]
        per_cell_v = cache["per_cell_v"]
        if i_grid.size == 0 or not per_cell_v:
            # fallback: compute scalar v_at_i for each value
            out = np.array([sum(cell.v_at_i(float(i)) for cell in self.cell_list) for i in i_values], dtype=float)
        else:
            # stack per-cell arrays (cells x len(i_grid)) then sum and interpolate to i_values
            stacked = np.vstack(per_cell_v)
            v_sum_grid = np.sum(stacked, axis=0)
            # apply bypass diode if present by computing its voltage at requested currents
            if self.bypass:
                bypass_v = np.array([self.bypass.v_at_i(float(i)) for i in i_grid], dtype=float)
                v_sum_grid = np.maximum(v_sum_grid, bypass_v)
            # interpolate summed voltage onto requested i_values
            out = np.interp(i_values, i_grid, v_sum_grid)

        return out

    def mpp(self, points: int = 600) -> Tuple[float, float, float, float, float]:
        """Return (Vmpp, Impp, Pmpp, Voc, Isc) for this substring.

        Uses the substring IV curve to locate the true MPP, which is required
        under partial shading (per-cell Vmpp sums are not valid).
        """
        V, I = self.iv_curve(points=points)
        if V.size == 0 or I.size == 0:
            return 0.0, 0.0, 0.0, float(self.voc()), float(self.isc())

        P = V * I
        idx = int(np.nanargmax(P))

        Vmpp = float(V[idx])
        Impp = float(I[idx])
        Pmpp = float(P[idx])
        Voc = float(self.voc())
        Isc = float(self.isc())
        return Vmpp, Impp, Pmpp, Voc, Isc

    # pre: current > 0
    def v_at_i(self, current: float) -> float:
        # Try fast path via cached per-cell IVs
        try:
            v = float(self.v_at_i_vector(np.array([current]), cache_points=1024)[0])
            return v
        except Exception:
            # fallback to original slow per-cell calls
            cell_voltages = sum(cell.v_at_i(current) for cell in self.cell_list)
            if self.bypass:
                bypass_voltage = self.bypass.v_at_i(current)
                return max(cell_voltages, bypass_voltage)
            return cell_voltages
