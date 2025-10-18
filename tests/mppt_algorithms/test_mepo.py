"""Unit tests for MEPO (adaptive P&O) MPPT algorithm.

Run from repo root:
    python -m pytest -q

Relies on fixtures provided by tests/mppt_algorithms/conftest.py
"""

from __future__ import annotations

import math
import pytest

from core.mppt_algorithms.local.mepo import MEPO


def test_mepo_interface_and_cold_start(meas):
    """First step should seed a reference and return JSON‑safe debug."""
    algo = MEPO()  # defaults
    m = meas(t=0.0, v=20.0)

    a = algo.step(m)

    assert a.v_ref is not None, "cold start should output a reference"
    assert algo.vmin <= a.v_ref <= algo.vmax

    # Debug contract: JSON‑safe primitives present
    dbg = a.debug or {}
    assert dbg.get("algo") == "mepo"
    assert isinstance(dbg.get("p"), (int, float))
    assert isinstance(dbg.get("v_ref"), (int, float))
    assert isinstance(dbg.get("v_cmd"), (int, float))
    assert isinstance(dbg.get("cold_start"), bool)


def test_mepo_moves_toward_mpp_single_peak(single_peak, meas):
    """On a convex P–V curve, MEPO should improve power over time and reach a decent fraction of Pmax."""
    algo = MEPO(alpha=0.4, step_min=1e-3, step_max=0.05, vmin=0.0, vmax=42.0)

    v = 30.0
    p0 = v * single_peak(v)

    t = 0.0
    for _ in range(800):
        m = meas(t, v)
        a = algo.step(m)
        v = a.v_ref if a.v_ref is not None else v
        t += 1e-3

    p1 = v * single_peak(v)

    # Should not be worse than where we started
    assert p1 >= p0 * 0.95

    # Should reach a healthy fraction of the parabolic Pmax (≈ (ISC*VOC)/4 for the linear IV proxy)
    isc, voc = 8.0, 42.0
    pmax_proxy = (isc * voc) / 4.0
    assert p1 >= 0.6 * pmax_proxy


def test_mepo_respects_bounds_and_slew(meas):
    """Reference must remain within [vmin, vmax] and change no faster than slew limiter allows."""
    algo = MEPO(step_min=1e-3, step_max=0.05, vmin=0.0, vmax=42.0, slew=0.02)

    v = 41.5
    t = 0.0
    last_v = v

    for _ in range(100):
        a = algo.step(meas(t, v))
        v = a.v_ref if a.v_ref is not None else v
        # bounds
        assert 0.0 <= v <= 42.0
        # slew (allow tiny numerical slack)
        assert abs(v - last_v) <= algo._slew.max_step + 1e-9
        last_v = v
        t += 1e-3


def test_mepo_describe_and_config_roundtrip():
    """describe() provides UI metadata; update_params() mutates get_config()."""
    algo = MEPO(alpha=0.25, step_min=0.002, step_max=0.02, slew=0.05)

    spec = algo.describe()
    assert spec["key"] == "mepo"
    assert any(p["name"] == "alpha" for p in spec.get("params", []))

    cfg0 = algo.get_config()

    # Apply updates (including a bounds swap to test the guard)
    algo.update_params(alpha=0.5, step_min=0.06, step_max=0.01, slew=0.02)
    cfg1 = algo.get_config()

    assert math.isclose(cfg1["alpha"], 0.5, rel_tol=0, abs_tol=1e-12)
    # step_min/max should be swapped to maintain min <= max
    assert cfg1["step_min"] == pytest.approx(0.01)
    assert cfg1["step_max"] == pytest.approx(0.06)
    assert cfg1["slew"] == pytest.approx(0.02)


def test_mepo_live_slew_update(meas):
    """Changing slew via update_params should immediately clamp per‑step changes."""
    algo = MEPO(step_min=1e-3, step_max=0.05, vmin=0.0, vmax=42.0, slew=0.1)

    v = 10.0
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