import os
import numpy as np
import pytest

from core.src.cell import Cell
from core.src.substring import Substring
from core.src.string import PVString


def _mk_cells(n, G=1000.0, T=25.0):
    return [
        Cell(
            isc_ref=5.84, voc_ref=0.621, diode_ideality=1.30,
            r_s=0.02, r_sh=800.0, vmpp=0.521, impp=5.40,
            irradiance=G, temperature_c=T, autofit=False,
        )
        for _ in range(n)
    ]


@pytest.fixture
def three_substrings_series():
    # 3 substrings in series; each substring = 4 series cells
    sub1 = Substring(_mk_cells(4, G=1000.0, T=25.0), bypass=None)
    sub2 = Substring(_mk_cells(4, G=1000.0,  T=25.0), bypass=None)
    sub3 = Substring(_mk_cells(4, G=1000.0, T=25.0), bypass=None)

    # make sure caches are hot
    for s in (sub1, sub2, sub3):
        s.set_conditions(irradiance=s.cell_list[0].irradiance, temperature=25.0)

    return PVString([sub1, sub2, sub3])


def test_plot_string_iv_pv(three_substrings_series):
    pvstr = three_substrings_series

    # Build curves
    V, I = pvstr.iv_curve(points=480)   # string-level IV (sum over substrings)
    P = V * I

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib not available, skipping plot:", e)
        return

    os.makedirs("plots", exist_ok=True)

    # I–V
    plt.figure(figsize=(6, 4))
    plt.plot(V, I, label="String I–V (3 substrings, middle shaded)")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title("PV String I–V")
    plt.grid(True)
    plt.legend()
    iv_out = os.path.join("plots", "pvstring_iv.png")
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
    plt.title("PV String P–V")
    plt.grid(True)
    plt.legend()
    pv_out = os.path.join("plots", "pvstring_pv.png")
    plt.tight_layout()
    plt.savefig(pv_out)
    print("Saved:", pv_out)
    plt.close()

    # quick sanity checks (non-assert so the test always saves plots)
    vmpp, impp, pmpp, voc_est, isc_est = pvstr.mpp(points=480)
    print("Vmpp, Impp, Pmpp, Voc_est, Isc_est =",
          vmpp, impp, pmpp, voc_est, isc_est)
