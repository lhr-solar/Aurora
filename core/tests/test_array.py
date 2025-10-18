import os
import numpy as np
import pytest

from core.src.cell import Cell
from core.src.substring import Substring
from core.src.string import PVString
from core.src.array import Array


def _mk_cells(n, G=1000.0, T=25.0):
    return [
        Cell(
            isc_ref=5.84, voc_ref=0.621, diode_ideality=1.30,
            r_s=0.02, r_sh=800.0, vmpp=0.521, impp=5.40,
            irradiance=G, temperature_c=T, autofit=False,
        )
        for _ in range(n)
    ]


def three_substrings_series():
    # 3 substrings in series; each substring = 4 series cells
    sub1 = Substring(_mk_cells(4, G=1000.0, T=25.0), bypass=None)
    sub2 = Substring(_mk_cells(4, G=1000.0,  T=25.0), bypass=None)
    sub3 = Substring(_mk_cells(4, G=1000.0, T=25.0), bypass=None)

    for s in (sub1, sub2, sub3):
        s.set_conditions(irradiance=s.cell_list[0].irradiance, temperature=25.0)

    return PVString([sub1, sub2, sub3])

@pytest.fixture
def three_strings_series():
    # 3 substrings in series; each substring = 4 series cells
    sub1 = three_substrings_series()
    sub2 = three_substrings_series()
    sub3 = three_substrings_series()

    for s in (sub1, sub2, sub3):
        s.set_conditions(irradiance=s.substrings[0].cell_list[0].irradiance, temperature_c=25.0)

    return Array([sub1, sub2, sub3])


def test_plot_array_iv_pv(three_strings_series):
    pvarr = three_strings_series

    # Build curves
    V, I = pvarr.iv_curve(points=480)   # array-level IV (sum over strings)
    P = V * I

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib not available, skipping plot:", e)
        return

    os.makedirs("plots", exist_ok=True)

    # I–V
    plt.figure(figsize=(6, 4))
    plt.plot(V, I, label="Array I–V (3 strings, middle shaded)")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title("PV String I–V")
    plt.grid(True)
    plt.legend()
    iv_out = os.path.join("plots", "pvarray_iv.png")
    plt.tight_layout()
    plt.savefig(iv_out)
    print("Saved:", iv_out)
    plt.close()

    # P–V with MPP marker
    plt.figure(figsize=(6, 4))
    plt.plot(V, P, label="String P–V")
    k = int(np.argmax(P))
    plt.scatter([V[k]], [P[k]], marker="x", label=f"MPP ~ ({V[k]:.3f} V, {I[k]:.3f} A)")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Power (W)")
    plt.title("PV Array P–V")
    plt.grid(True)
    plt.legend()
    pv_out = os.path.join("plots", "pvarray_pv.png")
    plt.tight_layout()
    plt.savefig(pv_out)
    print("Saved:", pv_out)
    plt.close()

    # quick sanity checks (non-assert so the test always saves plots)
    vmpp, impp, pmpp = pvarr.mpp()
    print("Vmpp, Impp, Pmpp, Voc_est, Isc_est =",
          vmpp, impp, pmpp)