from src.cell import Cell
from src.bypassdiode import Bypass_Diode
from typing import List, Tuple
import numpy as np

class Substring:
    def __init__(self, cell_list: List[Cell], bypass: Bypass_Diode):
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
    
    # pre: current > 0
    def isc(self, current) -> float:
        return min(cell.i_sc() for cell in self.cell_list) 

    def voc(self) -> float:
        return sum(cell.v_oc() for cell in self.cell_list)

    def iv_curve(self, points: int) -> Tuple[np.ndarray, np.ndarray]: # fixed coordinates
        isc = self.isc()
        i_values = np.linspace(0, isc, points)
        v_values = [self.v_at_i(i) for i in i_values]
        return np.array(i_values), np.array(v_values)

    def mpp(self) -> Tuple[float, float, float, float, float]:
        v_mpp_total = 0.0
        i_mpp = min(cell.i_mpp for cell in self.cell_list)
        for cell in self.cell_list:
            v_mpp_total += cell.v_mpp
        p_mpp = v_mpp_total * i_mpp
        return v_mpp_total, i_mpp, p_mpp, self.voc(), self.voc()

    # pre: current > 0
    def v_at_i(self, current: float) -> float:
        cell_voltages = sum(cell.v_at_i(current) for cell in self.cell_list)
        if self.bypass:
            bypass_voltage = self.bypass.v_at_i(current)
            return max(cell_voltages, bypass_voltage)
        return cell_voltages

    # pre: current > 0
    def bypass_total_voltage(self, current) -> float:
        for diode in self.bypass_list :
            self.bypass_total_voltage += diode.v_at_i(current)
        return self.bypass_total_voltage