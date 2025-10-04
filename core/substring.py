from cell import Cell
from bypassdiode import Bypass_Diode
import math
import numpy as np
import threading

class Substring:
    def __init__ (self, cell_list: list, bypass_list: list, current, irradiance, temperature) :
        self.cell_list = cell_list
        self.bypass_list = bypass_list
        self.current = current # series wiring - current is common, voltages add
        self.irradiance = irradiance
        self.temperature = temperature

        self.voltage = None
        self.short_circuit_current = None
    
    def set_conditions(self, irradiance, temperature) :
        for cell in self.cell_list :
            cell.setConditions(irradiance, temperature);
    
    # pre: current > 0
    def isc(self, current) :
        self.voltage = max(self.cell_total_voltage(), self.bypass_total_voltage())
        return self.voltage

    def total_open_circuit_v(self) :
        for cell in self.cell_list :
            total_open_circuit += cell.v_oc

    def iv_curve(self, points) :
        # need Cell class short circuit current
        self.short_circuit_current = min(cell.i_sc for cell in self.cell_list)
        i_values = np.linspace(0, self.short_circuit_current, points)
        v_values = []

        for i in i_values :
            v_values.append(self.v_at_i(i))
        return v_values

    def mpp(self) :
        for cell in self.cell_List :
            v_mpp += cell.v_mpp
            i_mpp = min(cell.i_mpp for cell in self.cell_list)
            p_mpp = v_mpp * i_mpp
        return v_mpp, i_mpp, p_mpp, self.total_open_circuit_v(), self.short_circuit_current

    # Helper methods:
    # pre: current > 0
    def v_at_i(self, current) :
        for cell in self.cell_list :
            total_voltage += cell.voltage

    # pre: current > 0
    def bypass_total_voltage(self, current) :
        for diode in self.bypass_list :
            diode.v_at_i(current)