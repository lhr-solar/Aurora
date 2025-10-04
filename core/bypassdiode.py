import math

boltzmann_const = 1.38e-23
charge_electron = 1.6e-19
bandgap_energy_0 = 1.17 # assumption for silicon
varshni_coefficient_alpha = 4.73e-4
varshni_coefficient_beta = 636

class Bypass_Diode: 
    def __init__ (self, irradiance, temperature, ideality_factor, series_resistance, V_cells: list) :
        self.irradiance = irradiance
        self.temperature = temperature
        self.ideality_factor = ideality_factor # between 1 and 2
        self.series_resistance = series_resistance
        self.V_cells = V_cells

        self.thermal_voltage = None
        self.temp_kelvin = None
        self.reverse_saturation_current = None
        self.diode_voltage_curve_slope = None
        self.V_string = None
        self.voltage = None
        self.sub_voltage = None

    def set_temperature(self, temp_celcius: int) :
        self.temp_kelvin = temp_celcius + 273.15 # convert to kelvin
        self.thermal_voltage = boltzmann_const * self.temp_kelvin/charge_electron
        self.reverse_saturation_current = bandgap_energy_0 - (varshni_coefficient_alpha * self.temp_kelvin**2) / (self.temp_kelvin + varshni_coefficient_beta)

    def v_at_i(self, terminal_voltage_at_current) :
        if terminal_voltage_at_current > 0 :
            voltage = -(self.ideality_factor * self.thermal_voltage * math.log(1 + (terminal_voltage_at_current / (self.reverse_saturation_current)) + terminal_voltage_at_current * self.series_resistance)) #The negative sign reflects the diode’s orientation (bypass is wired in reverse to the cell)
        else :
            voltage = float('-inf') # diode is off when current is negative or zero
        return voltage
    
    def dv_dI(self, terminal_voltage_at_current) :
        if terminal_voltage_at_current > 0 :
            self.diode_voltage_curve_slope = -(self.ideality_factor * self.thermal_voltage) / (terminal_voltage_at_current + self.reverse_saturation_current) + self.series_resistance
            return self.diode_voltage_curve_slope
    
    def activation_condition(self, terminal_voltage_at_current):
        self.V_string = sum(self.V_cells)
        self.voltage = self.v_at_i(terminal_voltage_at_current)
        return self.V_string <= self.voltage

    def Clamp(self):
        self.sub_voltage = max(self.V_string, self.voltage)