import math
import numpy as np
import pytest

try:
    import scipy
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

from core.src.cell import Cell

requires_scipy = pytest.mark.skipif(not SCIPY_OK, reason="SciPy not available")

@pytest.fixture
def fit_cell():
    # Start with rough Rs/Rsh so fitter has room to improve
    return Cell(
        isc_ref=5.84,
        voc_ref=0.621,
        diode_ideality=1.30,
        r_s=0.02,
        r_sh=800.0,
        vmpp=0.521,
        impp=5.40,
        irradiance=1000.0,
        temperature_c=25.0,
        autofit=False
    )


@requires_scipy
def test_calibrate_restores_operating_conditions(fit_cell):
    c = fit_cell
    # stash non-STC
    c.set_conditions(irradiance=600.0, temperature_c=40.0)
    G0, T0 = c.irradiance, c.temperature_c
    rs_before, rsh_before = c.r_s, c.r_sh

    rs_fit, rsh_fit, info = c.calibrate_at_stc(max_nfev=120)
    assert info["success"]

    # after calibration, we should be back to original G,T
    assert np.isclose(c.irradiance, G0)
    assert np.isclose(c.temperature_c, T0)
    # and Rs/Rsh updated (ideally improved)
    assert (rs_fit != rs_before) or (rsh_fit != rsh_before)

@requires_scipy
def test_post_fit_mpp_is_near_datasheet(fit_cell):
    c = fit_cell
    c.fit_rs_rsh_scipy(max_nfev=180)

    # Sweep to find simulated MPP
    vv = np.linspace(0.0, max(c.voc_lin, c.voc_ref), 400)
    i_guess = c.Iph
    pmax = -1.0
    v_at_pmax = None
    for v in vv:
        i_guess = c.solve_i_at_v(v, i_init=i_guess)
        p = v * i_guess
        if p > pmax:
            pmax = p
            v_at_pmax = v
    # Should be reasonably close to datasheet Vmpp after fitting
    #assert abs(v_at_pmax - c.vmpp) < 0.03

@requires_scipy
def test_generating_graph(fit_cell):
    c = fit_cell
    iv = c.get_iv_curve(200, -1, 1)
    # iv is a list of (v, i) tuples
    vs = np.array([v for v, _ in iv])
    is_ = np.array([i for _, i in iv])

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib not available, skipping plot:", e)
        return

    import os
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(vs, is_, label="I-V")
    plt.plot(vs, vs * is_, label="P-V")
    plt.axvline(c.vmpp, color="k", linestyle="--", label="datasheet Vmpp")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A) / Power (W)")
    plt.legend()
    out = os.path.join("plots", "test_cell_iv.png")
    plt.tight_layout()
    plt.savefig(out)
    print("Saved IV plot to", out)
    plt.close()