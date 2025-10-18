from core.src.cell import Cell
from core.src.substring import Substring
from core.src.bypassdiode import Bypass_Diode
from unittest.mock import Mock
import math
import unittest
import matplotlib.pyplot as plt
import numpy as np
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

    def debug_plot_substring(self, name="debug_substring"):
        import os

        os.makedirs("plots", exist_ok=True)

        try:
            i_vals, v_vals = self.substring.iv_curve(points=100)

            if np.any(np.isnan(v_vals)) or np.any(np.isnan(i_vals)):
                print("NaNs found in IV data — check v_at_i or i_sc methods.")
                return

            if np.all(v_vals == 0):
                print("All voltages are zero. Likely an issue with v_at_i() or bypass logic.")
            
            if np.all(i_vals == 0):
                print("All currents are zero. Check i_sc() or cell initialization.")

            p_vals = v_vals * i_vals

            try:
                v_mpp, i_mpp, p_mpp, _, _ = self.substring.mpp()
                print(f"Vmpp: {v_mpp:.2f} V, Impp: {i_mpp:.2f} A, Pmpp: {p_mpp:.2f} W")
            except Exception as e:
                print("Error computing MPP:", e)
                v_mpp = None

            plt.figure(figsize=(8, 5))
            plt.plot(v_vals, i_vals, label="I-V curve")
            plt.plot(v_vals, p_vals, label="P-V curve", linestyle="--")

            if v_mpp:
                plt.axvline(v_mpp, color="r", linestyle=":", label=f"Vmpp ≈ {v_mpp:.2f} V")

            plt.xlabel("Voltage (V)")
            plt.ylabel("Current (A) / Power (W)")
            plt.title(f"I-V and P-V Curve: {name}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            out = os.path.join("plots", f"{name}_iv_debug.png")
            plt.savefig(out)
            print(f"Saved debug plot to {out}")
            plt.close()

        except Exception as e:
            print(f"Failed to plot substring '{name}':", e)

if __name__ == "__main__":
    unittest.main()