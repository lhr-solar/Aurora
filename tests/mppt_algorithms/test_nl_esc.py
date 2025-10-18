"""Unit tests for NL-ESC (nonlinear extremum seeking control) hold algorithm.

Run from repo root:
    python -m pytest -q tests/mppt_algorithms/test_nl_esc.py

Relies on fixtures provided by tests/mppt_algorithms/conftest.py
"""

from __future__ import annotations

import math
import statistics as stats
import pytest

from core.mppt_algorithms.hold.nl_esc import NL_ESC


def test_nl_esc_interface_and_debug(meas):
    """First step should output a reference within bounds and include JSON-safe debug."""
    algo = NL_ESC(dither_amp=0.2, dither_hz=60.0, k=0.08, vmin=0.0, vmax=42.0, slew=0.5)

    a = algo.step(meas(t=0.0, v=20.0))

    assert a.v_ref is not None
    assert 0.0 <= a.v_ref <= 42.0

    dbg = a.debug or {}
    assert dbg.get("algo") == "nl_esc"
    # required keys present and JSON-friendly types
    for key in ("p", "grad", "v_base", "v_cmd", "theta", "dither_amp", "dither_hz"):
        assert key in dbg, f"missing debug key: {key}"
        assert isinstance(dbg[key], (int, float)), f"{key} not numeric"


def test_nl_esc_tracks_near_mpp(single_peak, meas, numeric_pmax):
    """On a single-peak curve, NL-ESC should settle near MPP with small ripple."""
    algo = NL_ESC(dither_amp=0.2, dither_hz=80.0, k=0.08, vmin=0.0, vmax=42.0, slew=0.3)

    v = 10.0
    t = 0.0
    samples = []  # (t, v, p)
    for _ in range(4000):  # 4 s at 1 kHz
        m = meas(t, v)
        a = algo.step(m)
        v = a.v_ref if a.v_ref is not None else v
        samples.append((t, v, m.v * m.i))
        t += 1e-3

    # Evaluate tail window (last 25%)
    tail = samples[int(0.75 * len(samples)) :]
    v_tail = [v for (_, v, _) in tail]
    p_tail = [p for (_, _, p) in tail]

    pmax = numeric_pmax(single_peak, vmin=0.0, vmax=42.0)
    p_mean = sum(p_tail) / max(1, len(p_tail))

    # Should achieve a healthy fraction of Pmax
    assert p_mean >= 0.80 * pmax

    # Ripple should be modest given 0.2 V dither
    v_std = stats.pstdev(v_tail) if len(v_tail) > 1 else 0.0
    assert v_std <= 1.2  # generous bound for this simple proxy

    v_mpp_est = 21.0
    v_tail_mean = sum(v_tail) / max(1, len(v_tail))
    assert abs(v_tail_mean - v_mpp_est) <= 2.5


def test_nl_esc_respects_bounds_and_slew(meas):
    """Reference must remain within [vmin, vmax] and step changes obey the slew limiter."""
    algo = NL_ESC(dither_amp=0.1, dither_hz=50.0, k=0.05, vmin=0.0, vmax=42.0, slew=0.02)

    v = 21.0
    t = 0.0
    last_v = v
    max_step = algo.get_config()["slew"]
    for _ in range(200):
        a = algo.step(meas(t, v))
        v = a.v_ref if a.v_ref is not None else v
        assert 0.0 <= v <= 42.0
        assert abs(v - last_v) <= max_step + 1e-9
        last_v = v
        t += 1e-3


essential_keys = {"dither_amp", "dither_hz", "k", "demod_alpha", "leak", "slew", "vmin", "vmax"}


def test_nl_esc_describe_and_config_roundtrip():
    """describe() provides UI metadata; update_params() mutates get_config()."""
    algo = NL_ESC(dither_amp=0.1, dither_hz=100.0, k=0.05, vmin=0.0, vmax=42.0, slew=0.1)

    spec = algo.describe()
    assert spec["key"] == "nl_esc"
    names = {p.get("name") for p in spec.get("params", [])}
    assert essential_keys.issubset(names)

    cfg0 = algo.get_config()
    # Apply updates
    algo.update_params(dither_amp=0.25, dither_hz=120.0, k=0.09, demod_alpha=0.2, leak=0.002, slew=0.03, vmin=0.0, vmax=40.0)
    cfg1 = algo.get_config()

    assert math.isclose(cfg1["dither_amp"], 0.25, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(cfg1["dither_hz"], 120.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(cfg1["k"], 0.09, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(cfg1["slew"], 0.03, rel_tol=0, abs_tol=1e-12)
    assert cfg1["vmax"] == pytest.approx(40.0)
