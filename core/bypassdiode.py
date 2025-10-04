import math
import unittest

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
        self.bypass_voltage = None
        self.sub_voltage = None

    def set_temperature(self, temp_celcius: int) :
        self.temp_kelvin = temp_celcius + 273.15 # convert to kelvin
        self.thermal_voltage = boltzmann_const * self.temp_kelvin/charge_electron
        self.reverse_saturation_current = bandgap_energy_0 - (varshni_coefficient_alpha * self.temp_kelvin**2) / (self.temp_kelvin + varshni_coefficient_beta)

    def v_at_i(self, terminal_voltage_at_current) :
        if terminal_voltage_at_current > 0 :
            bypass_voltage = -(self.ideality_factor * self.thermal_voltage * math.log(1 + (terminal_voltage_at_current / (self.reverse_saturation_current)) + terminal_voltage_at_current * self.series_resistance)) #The negative sign reflects the diode’s orientation (bypass is wired in reverse to the cell)
        else :
            bypass_voltage = float('-inf') # diode is off when current is negative or zero
        return bypass_voltage
    
    def dv_dI(self, terminal_voltage_at_current) :
        if terminal_voltage_at_current > 0 :
            self.diode_voltage_curve_slope = -(self.ideality_factor * self.thermal_voltage) / (terminal_voltage_at_current + self.reverse_saturation_current) + self.series_resistance
            return self.diode_voltage_curve_slope
    
    def activation_condition(self, terminal_voltage_at_current):
        self.V_string = sum(self.V_cells)
        self.bypass_voltage = self.v_at_i(terminal_voltage_at_current)
        return self.V_string <= self.bypass_voltage

    def Clamp(self):
        self.sub_voltage = max(self.V_string, self.bypass_voltage)

class TestBypassDiode(unittest.TestCase):
    def setUp(self):
        self.bypass = Bypass_Diode(
            irradiance = 1000,
            temperature = 25,
            ideality_factor = 1.3,
            series_resistance = 0.01,
            V_cells = [0.5, 0.5, 0.5]
        )
        self.bypass.set_temperature(25)
    
    def test_set_temperature(self):
        self.assertAlmostEqual(self.bypass.temp_kelvin, 298.15, places=2)
        self.assertAlmostEqual(self.bypass.thermal_voltage, 0.0257, places=4)
        self.assertAlmostEqual(self.bypass.reverse_saturation_current, 1.125, places=3)
    
    def test_v_at_i(self):
        Vt = 0.5
        expected = -1.3 * 0.0257 * math.log(1 + (Vt / 1.125) + Vt * 0.01)
        actual = self.bypass.v_at_i(Vt)
        self.assertAlmostEqual(actual, expected, places=3)
        self.assertAlmostEqual(actual, -0.01235, places=3)  # freeze value
    
    def test_dv_dI(self):
        I = 0.5 # Arbitruary value, depends on conditions
        expected = - (1.3 * 0.0257) / (I + 1.125) + 0.01
        actual = self.bypass.dv_dI(I)
        self.assertAlmostEqual(actual, expected, places=4)
        self.assertAlmostEqual(actual, -0.01054, places=4)  # freeze value
    
    def test_activation_condition(self):
        I = 0.5
        condition = self.bypass.activation_condition(I)
        self.assertFalse(condition)
    
    def test_Clamp(self):
        I = 0.5
        self.bypass.activation_condition(I)  # to set bypass_voltage and V_string
        self.bypass.Clamp()
        self.assertAlmostEqual(self.bypass.sub_voltage, 1.5, places=4)

if __name__ == "__main__":
    unittest.main()