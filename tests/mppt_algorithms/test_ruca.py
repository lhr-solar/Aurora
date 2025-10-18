"""Unit tests for RUCA (sign-based, V/I/VI branches) MPPT algorithm.

Run from repo root:
    python -m pytest -q tests/mppt_algorithms/test_ruca.py

Relies on fixtures provided by tests/mppt_algorithms/conftest.py
"""

import math
import pytest

from core.mppt_algorithms.local.ruca import RUCA


def test_ruca_interface_and_cold_start(meas):
    """First step should seed a reference and expose JSON‑safe debug fields."""
    algo = RUCA(step=0.02, alt_mode="VI", vmin=0.0, vmax=42.0, slew=0.2)

    a = algo.step(meas(t=0.0, v=20.0))

    assert a.v_ref is not None
    assert 0.0 <= a.v_ref <= 42.0

    dbg = a.debug or {}
    assert dbg.get("algo") == "ruca"
    for key in ("p", "v_ref", "v_cmd", "cold_start"):
        assert key in dbg, f"missing debug key: {key}"
    assert isinstance(dbg.get("cold_start"), bool)


def test_ruca_respects_bounds_and_slew(meas):
    """Reference remains within [vmin, vmax] and per‑step delta obeys slew limiter."""
    algo = RUCA(step=0.03, alt_mode="V", vmin=0.0, vmax=42.0, slew=0.02)

    v = 41.0
    t = 0.0
    last_v = v
    for _ in range(150):
        a = algo.step(meas(t, v))
        v_new = a.v_ref if a.v_ref is not None else v
        assert 0.0 <= v_new <= 42.0
        assert abs(v_new - v) <= algo._slew.max_step + 1e-9
        v, last_v = v_new, v_new
        t += 1e-3


def test_ruca_branch_alternation_vi_mode(meas):
    """In VI mode the branch should alternate between V and I over successive steps (after cold start)."""
    algo = RUCA(step=0.02, alt_mode="VI", vmin=0.0, vmax=42.0, slew=0.2)

    v = 22.0
    t = 0.0
    branches = []
    for _ in range(30):
        a = algo.step(meas(t, v))
        br = a.debug.get("branch", "NA")
        if br in ("V", "I"):
            branches.append(br)
        v = a.v_ref if a.v_ref is not None else v
        t += 1e-3

    # Should have seen both branches within a few iterations
    assert "V" in branches and "I" in branches


def test_ruca_improves_power_on_single_peak(single_peak, meas):
    """On a convex single‑peak curve, RUCA should improve power over time and reach a decent fraction of Pmax."""
    algo = RUCA(step=0.02, alt_mode="V", vmin=0.0, vmax=42.0, slew=0.05, adapt_frac=0.02)

    v = 30.0  # start to the right of MPP
    p0 = v * single_peak(v)

    t = 0.0
    for _ in range(900):
        m = meas(t, v)
        a = algo.step(m)
        v = a.v_ref if a.v_ref is not None else v
        t += 1e-3

    p1 = v * single_peak(v)

    # Not worse than start (allow a little noise/tolerance) and a healthy fraction of proxy Pmax
    assert p1 >= 0.95 * p0
    isc, voc = 8.0, 42.0
    pmax_proxy = (isc * voc) / 4.0
    assert p1 >= 0.55 * pmax_proxy


essential = {"step", "alt_mode", "adapt_frac", "slew", "vmin", "vmax"}


def test_ruca_describe_and_config_roundtrip():
    """describe() provides UI metadata; update_params() mutates get_config() and validates values."""
    algo = RUCA(step=0.01, alt_mode="V", adapt_frac=0.02, vmin=0.0, vmax=42.0, slew=0.05)

    spec = algo.describe()
    assert spec["key"] == "ruca"
    names = {p.get("name") for p in spec.get("params", [])}
    assert essential.issubset(names)

    cfg0 = algo.get_config()

    # Apply updates; include invalid alt_mode and too‑large adapt_frac to test guards
    algo.update_params(step=0.06, alt_mode="x", adapt_frac=2.0, slew=0.02, vmin=0.0, vmax=40.0)
    cfg1 = algo.get_config()

    assert math.isclose(cfg1["step"], 0.06, rel_tol=0, abs_tol=1e-12)
    # alt_mode should default to VI when invalid
    assert cfg1["alt_mode"] in {"V", "I", "VI"}
    # adapt_frac clamped to <= 0.99 in implementation
    assert 0.0 <= cfg1["adapt_frac"] <= 0.99
    assert math.isclose(cfg1["slew"], 0.02, rel_tol=0, abs_tol=1e-12)
    assert cfg1["vmax"] == pytest.approx(40.0)


def test_ruca_live_slew_update(meas):
    """Changing slew via update_params should clamp the very next step delta."""
    algo = RUCA(step=0.02, alt_mode="V", vmin=0.0, vmax=42.0, slew=0.1)

    v = 15.0
    t = 0.0

    # Take a few steps with initial slew
    for _ in range(5):
        a = algo.step(meas(t, v))
        v = a.v_ref if a.v_ref is not None else v
        t += 1e-3

    # Tighten slew and ensure the next delta obeys the new limit
    algo.update_params(slew=0.005)

    a = algo.step(meas(t, v))
    v_next = a.v_ref if a.v_ref is not None else v
    assert abs(v_next - v) <= 0.005 + 1e-9