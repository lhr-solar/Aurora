"""Unit tests for PANDO (classic P&O, fixed step) MPPT algorithm.

Run from repo root:
    python -m pytest -q tests/mppt_algorithms/test_pando.py

Relies on fixtures provided by tests/mppt_algorithms/conftest.py
"""

from __future__ import annotations

import math
import pytest

from core.mppt_algorithms.local.pando import PANDO


def test_pando_interface_and_cold_start(meas):
    """First step should seed a reference and include JSON‑safe debug fields."""
    algo = PANDO(step=0.01, vmin=0.0, vmax=42.0, slew=0.2)

    a = algo.step(meas(t=0.0, v=20.0))

    assert a.v_ref is not None, "cold start should output a reference"
    assert 0.0 <= a.v_ref <= 42.0

    dbg = a.debug or {}
    assert dbg.get("algo") == "pando"
    for key in ("p", "dP", "dir", "cold_start", "step", "v_ref", "v_cmd"):
        assert key in dbg, f"missing debug key: {key}"
    assert isinstance(dbg["cold_start"], bool)
    # numbers are JSON‑safe (plain ints/floats)
    for k in ("p", "dP", "dir", "step", "v_ref", "v_cmd"):
        assert isinstance(dbg[k], (int, float)), f"{k} not numeric"


def test_pando_flips_on_negative_dP(meas):
    """If last perturbation reduced power (dP < 0), direction should flip negative."""
    algo = PANDO(step=0.02, eps=0.0, vmin=0.0, vmax=42.0, slew=0.5)

    # 1) Cold start at a point on the right side of MPP
    v = 28.0
    t = 0.0
    a = algo.step(meas(t, v))

    # 2) Next measurement intentionally moves farther right (power drops → dP < 0)
    t += 1e-3
    v = 33.0
    a = algo.step(meas(t, v))

    # After a bad move, P&O should reverse direction (dir < 0)
    assert a.debug.get("dir", 0.0) < 0.0


def test_pando_respects_bounds_and_slew(meas):
    """Reference must remain within [vmin, vmax] and step changes obey the slew limiter."""
    algo = PANDO(step=0.03, vmin=0.0, vmax=42.0, slew=0.02)

    v = 41.5
    t = 0.0
    last_v = v
    for _ in range(120):
        a = algo.step(meas(t, v))
        v = a.v_ref if a.v_ref is not None else v
        assert 0.0 <= v <= 42.0
        assert abs(v - last_v) <= algo._slew.max_step + 1e-9
        last_v = v
        t += 1e-3


def test_pando_improves_power_on_single_peak(single_peak, meas):
    """On a convex single‑peak P–V, P&O should improve power over time."""
    algo = PANDO(step=0.04, vmin=0.0, vmax=42.0, slew=0.06)

    v = 8.0  # start far from MPP (~VOC/2)
    p0 = v * single_peak(v)

    t = 0.0
    for _ in range(1000):
        m = meas(t, v)
        a = algo.step(m)
        v = a.v_ref if a.v_ref is not None else v
        t += 1e-3

    p1 = v * single_peak(v)

    # Should not degrade and should reach a decent fraction of theoretical proxy Pmax
    assert p1 >= p0 * 1.10  # at least 10% improvement from a poor start
    isc, voc = 8.0, 42.0
    pmax_proxy = (isc * voc) / 4.0
    assert p1 >= 0.5 * pmax_proxy


def test_pando_describe_and_config_roundtrip():
    """describe() provides UI metadata; update_params() mutates get_config()."""
    algo = PANDO(step=0.01, eps=0.05, vmin=0.0, vmax=42.0, slew=0.05)

    spec = algo.describe()
    assert spec["key"] == "pando"
    names = {p.get("name") for p in spec.get("params", [])}
    assert {"step", "eps", "slew", "vmin", "vmax"}.issubset(names)

    cfg0 = algo.get_config()
    algo.update_params(step=0.02, eps=0.01, slew=0.02, vmin=0.0, vmax=40.0)
    cfg1 = algo.get_config()

    assert math.isclose(cfg1["step"], 0.02, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(cfg1["eps"], 0.01, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(cfg1["slew"], 0.02, rel_tol=0, abs_tol=1e-12)
    assert cfg1["vmax"] == pytest.approx(40.0)