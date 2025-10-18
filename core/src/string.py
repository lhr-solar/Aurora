from typing import List, Tuple
import numpy as np
from substring import Substring

class String:
    def __init__(self, substrings: List[Substring]):
        self.substrings = substrings
        self.cached_iv = None

    def set_conditions(self, irradiance: float, temperature: float):
        """Set environmental conditions for all substrings."""
        for substring in self.substrings:
            substring.set_conditions(irradiance, temperature)
        self.cached_iv = None

    def isc(self) -> float:
        """
        The string's short-circuit current is limited by the substring with
        the *lowest Isc*, since substrings are in series.
        """
        return min(sub.isc() for sub in self.substrings)

    def voc(self) -> float:
        """
        Open-circuit voltage is the sum of all substring Voc values
        (since they are in series).
        """
        return sum(sub.voc() for sub in self.substrings)

    def v_at_i(self, current: float) -> float:
        """
        Computes total string voltage at a given current.
        For each substring, find its voltage at that current,
        considering bypass activation automatically.
        """
        total_voltage = 0.0
        for sub in self.substrings:
            total_voltage += sub.v_at_i(current)
        return total_voltage

    def iv_curve(self, points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate I–V curve for the entire string.
        The current range is determined by the limiting substring (min Isc).
        """
        isc_total = self.isc()
        i_values = np.linspace(0, isc_total, points)
        v_values = [self.v_at_i(i) for i in i_values]
        return np.array(i_values), np.array(v_values)

    def mpp(self) -> Tuple[float, float, float]:
        """
        Estimate maximum power point by scanning the I–V curve.
        Returns: (Vmpp_total, Impp_total, Pmpp_total)
        """
        i_values, v_values = self.iv_curve(points=200)
        p_values = v_values * i_values
        idx = int(np.argmax(p_values))
        return v_values[idx], i_values[idx], p_values[idx]
