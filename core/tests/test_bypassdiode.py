from src.bypassdiode import Bypass_Diode
import math
import unittest

class Test_Bypass_Diode(unittest.TestCase):
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