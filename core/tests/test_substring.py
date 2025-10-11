from src.cell import Cell
from src.substring import Substring
from src.bypassdiode import Bypass_Diode
from unittest.mock import Mock
import math
import unittest
from typing import List, Tuple

mock_cell = Mock(spec=Cell)
mock_cell.i_sc = Mock(return_value=5.0)
mock_cell.v_oc = Mock(return_value=0.6)
mock_cell.v_at_i = Mock(return_value=0.5)
mock_cell.i_mpp = 4.5
mock_cell.v_mpp = 0.5

mock_bypass = Mock(spec=Bypass_Diode)
mock_bypass.v_at_i.return_value = 0.2

# 2 mocks
class Test_Substring(unittest.TestCase):
    
    def setUp(self):
        self.substring = Substring(
            cell_list=[mock_cell, mock_cell], bypass=mock_bypass
        )
    
    def test_isc(self):
        isc = self.substring.isc(current = 1)
        self.assertEqual(isc, 5.0)  # mocks should all return 5.0

    def test_voc(self):
        voc = self.substring.voc()
        self.assertEqual(voc, 1.2)  # 2 cells * the constant voc 0.6
    
    def test_v_at_i(self):
        voltage = self.substring.v_at_i(2)
        # sum cell voltages = 0.5 * 2 = 1.0; bypass voltage = 0.2 * 2 = 0.4; max = 2.0
        self.assertAlmostEqual(voltage, 1.0, places = 4)

    def test_mpp(self):
        v_mpp_total, i_mpp, p_mpp, voc, bypass_voc = self.substring.mpp()
        self.assertAlmostEqual(v_mpp_total, 1.0, places = 4)  # 2 cells * 0.5
        self.assertAlmostEqual(i_mpp, 4.5, places = 4)
        self.assertAlmostEqual(p_mpp, 4.5, places = 4)  # 1.0 * 4.5
        self.assertAlmostEqual(voc, 1.2, places = 4)
        self.assertAlmostEqual(bypass_voc, 1.2, places = 4)

if __name__ == "__main__":
    unittest.main()