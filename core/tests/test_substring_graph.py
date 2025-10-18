import numpy as np
import os
import pytest

from core.src.cell import Cell

# ---------- Helpers: invert I(V) per cell using bisection ----------

def cell_voltage_at_current(cell: Cell, I_target: float, *, tol=1e-7, max_iter=80):
    """
    Find v such that cell.solve_i_at_v(v) ~= I_target.
    Assumes monotonic I(V) between (v=0 -> ~Isc) and (v=Voc -> ~0).
    """
    # Reasonable brackets
    v_lo = -0.05                       # tiny negative to avoid edge/rounding issues
    v_hi_guess = cell.voc_lin if cell.voc_lin is not None else cell.voc_ref
    v_hi = max(0.1, float(v_hi_guess)) * 1.10  # pad a bit above Voc

    # Evaluate at ends
    i_lo = cell.solve_i_at_v(v_lo)     # ≈ Isc
    i_hi = cell.solve_i_at_v(v_hi)     # ≈ 0

    # If target is out of range, clamp gracefully
    if I_target >= i_lo:
        return v_lo
    if I_target <= i_hi:
        return v_hi

    # Bisection on g(v) = I(v) - I_target
    for _ in range(max_iter):
        vm = 0.5 * (v_lo + v_hi)
        im = cell.solve_i_at_v(vm)
        g_lo = i_lo - I_target
        g_m  = im   - I_target

        if abs(g_m) < tol or abs(v_hi - v_lo) < 1e-10:
            return vm
        if np.sign(g_m) == np.sign(g_lo):
            v_lo, i_lo = vm, im
        else:
            v_hi = vm
    return 0.5 * (v_lo + v_hi)


def substring_iv(cells, points=300):
    """
    Series substring: current is common; total voltage is sum of per-cell voltages at that current.
    We sweep current from 0..min(Isc_cell) and compute V_total(I).
    """
    # Compute each cell's Isc ~ I(V=0)
    iscs = [c.solve_i_at_v(0.0) for c in cells]
    isc_min = float(min(iscs))

    I = np.linspace(0.0, isc_min, points)
    V = np.zeros_like(I)

    # Warm-start: use previous v as initial bracket center (via bisection we only need brackets)
    for k, i in enumerate(I):
        v_sum = 0.0
        for cell in cells:
            v_cell = cell_voltage_at_current(cell, i)
            v_sum += v_cell
        V[k] = v_sum
    return V, I


# ---------- Test: plot substring IV & PV ----------

@pytest.fixture
def unshaded_cells():
    # 12 identical cells at STC; rs/rsh are rough but fine for a plot
    return [
        Cell(
            isc_ref=5.84,
            voc_ref=0.621,
            diode_ideality=1.30,
            r_s=0.02,
            r_sh=800.0,
            vmpp=0.521,
            impp=5.40,
            irradiance=1000.0,
            temperature_c=25.0,
            autofit=False,
        )
        for _ in range(12)
    ]


def test_substring_ivcurve_plot(unshaded_cells):
    # Ensure all cells are at same operating point
    for c in unshaded_cells:
        c.set_conditions(irradiance=1000.0, temperature_c=25.0)

    V, I = substring_iv(unshaded_cells, points=300)
    P = V * I

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib not available, skipping plot:", e)
        return

    os.makedirs("plots", exist_ok=True)

    # I–V
    plt.figure(figsize=(6, 4))
    plt.plot(V, I, label="I–V (Substring, series)")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title("Substring I–V (12 cells, unshaded)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_iv = os.path.join("plots", "substring_iv.png")
    plt.savefig(out_iv)
    print("Saved:", out_iv)
    plt.close()

    # P–V
    plt.figure(figsize=(6, 4))
    plt.plot(V, P, label="P–V (Substring)")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Power (W)")
    plt.title("Substring P–V (12 cells, unshaded)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_pv = os.path.join("plots", "substring_pv.png")
    plt.savefig(out_pv)
    print("Saved:", out_pv)
    plt.close()
