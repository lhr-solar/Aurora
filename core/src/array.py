from string import String
from typing import List, Tuple
import numpy as np

class Array:
    def __init__(self, string_list: List[String]):
        self.string_list = string_list
        self.cached_iv = None

        self.total_voltage = None
    
    def set_conditions(self, irradiance, temperature) :
        for string in self.string_list :
            string.set_conditions(irradiance, temperature)
        self.cached_iv = None
    
    # pre: current > 0
    def isc(self, current) -> float:
        return sum(string.v_oc() for string in self.string_list)

    def voc(self) -> float:
        return max(string.v_oc() for string in self.cell_list) 

    def iv_curve(self, points: int) -> Tuple[np.ndarray, np.ndarray]: # fixed coordinates
        voc = self.voc()
        v_values = np.linspace(0, voc, points)
        i_values = [self.i_at_v(v) for v in v_values]
        return np.array(i_values), np.array(v_values)

    def mpp(self) -> Tuple[float, float, float, float, float]:
        i_mpp_total = 0.0
        v_mpp = max(string.v_mpp for string in self.string_list)
        for string in self.string_list:
            i_mpp_total += string.i_mpp
        p_mpp = v_mpp * i_mpp_total
        return v_mpp, i_mpp_total, p_mpp

    # pre: current > 0
    def i_at_v(self, i: float) -> float:
        string_voltages = sum(string.v_at_i(i) for string in self.string_list)
        return string_voltages