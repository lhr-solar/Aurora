

"""Unit tests for PSO (sequential global search) MPPT algorithm.

Run from repo root:
    python -m pytest -q tests/mppt_algorithms/test_pso.py

Relies on fixtures provided by tests/mppt_algorithms/conftest.py
"""

from __future__ import annotations

import math
import pytest

from core.mppt_algorithms.global_search.pso import PSO


def test_pso_interface_and_cold_start(meas):
    """First step should output a reference within bounds and include JSON-safe debug."""
    algo = PSO(window_pct=0.2, particles=6, iters=6, seed=7, vmin=0.0, vmax=42.0, slew=0.5)

    a = algo.step(meas(t=0.0, v=20.0))

    assert a.v_ref is not None
    assert 0.0 <= a.v_ref <= 42.0

    dbg = a.debug or {}
    assert dbg.get("algo") == "pso"
    # basic keys present and numeric
    for key in ("phase", "p", "center", "span"):
        assert key in dbg, f"missing debug key: {key}"
        assert isinstance(dbg[key], (int, float)), f"{key} not numeric"


def test_pso_explores_then_locks(single_peak, meas, numeric_pmax):
    """On a single-peak curve, PSO should explore and then settle with strong performance."""
    algo = PSO(window_pct=0.25, particles=8, iters=10, early_eps=5e-3, vmin=0.0, vmax=42.0, seed=11, slew=0.3)

    v = 10.0
    t = 0.0
    tail = []  # collect last 25% for mean power
    samples = []
    for k in range(1800):  # 1.8 s @ 1 kHz
        m = meas(t, v)
        a = algo.step(m)
        v = a.v_ref if a.v_ref is not None else v
        samples.append((t, v, m.v * m.i, a.debug.get("phase", -1)))
        t += 1e-3

    # Expect to have evaluated and improved best
    assert algo.gbest_v is not None
    assert algo.gbest_p > 0.0

    # Tail performance should be a healthy fraction of numeric Pmax
    tail = samples[int(0.75 * len(samples)) :]
    p_tail = [p for (_, _, p, _) in tail]
    p_mean = sum(p_tail) / max(1, len(p_tail))

    pmax = numeric_pmax(single_peak, vmin=0.0, vmax=42.0)
    assert p_mean >= 0.80 * pmax  # allow search overhead, should still be strong

    # Final phase is exploration (1) or lock (2); accept either but not 0
    last_phase = samples[-1][3]
    assert last_phase in (1, 2)


def test_pso_respects_bounds_and_slew(meas):
    """Reference remains within [vmin, vmax] and per-step delta obeys slew limiter."""
    algo = PSO(window_pct=0.2, particles=6, iters=6, vmin=0.0, vmax=42.0, slew=0.02)

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


def test_pso_window_debug_within_bounds(meas):
    """center/span reported in debug should define a window that stays within [vmin, vmax]."""
    algo = PSO(window_pct=0.35, particles=6, iters=6, vmin=0.0, vmax=42.0, seed=3)

    v = 22.0
    t = 0.0
    for _ in range(200):
        a = algo.step(meas(t, v))
        dbg = a.debug or {}
        c = dbg.get("center")
        s = dbg.get("span")
        if isinstance(c, (int, float)) and isinstance(s, (int, float)):
            lo, hi = c - s, c + s
            assert 0.0 <= lo <= 42.0
            assert 0.0 <= hi <= 42.0
        v = a.v_ref if a.v_ref is not None else v
        t += 1e-3


essential = {"window_pct", "particles", "iters", "early_eps", "slew", "vmin", "vmax", "seed"}


def test_pso_describe_and_config_roundtrip():
    """describe() provides UI metadata; update_params() mutates get_config()."""
    algo = PSO(window_pct=0.2, particles=7, iters=8, early_eps=1e-3, vmin=0.0, vmax=42.0, seed=5, slew=0.1)

    spec = algo.describe()
    assert spec["key"] == "pso"
    names = {p.get("name") for p in spec.get("params", [])}
    assert essential.issubset(names)

    cfg0 = algo.get_config()

    # Apply updates and verify round-trip
    algo.update_params(window_pct=0.3, particles=10, iters=5, early_eps=2e-3, vmin=0.0, vmax=40.0, seed=9, slew=0.2)
    cfg1 = algo.get_config()

    assert math.isclose(cfg1["window_pct"], 0.3, rel_tol=0, abs_tol=1e-12)
    assert cfg1["particles"] == 10
    assert cfg1["iters"] == 5
    assert math.isclose(cfg1["early_eps"], 2e-3, rel_tol=0, abs_tol=1e-12)
    assert cfg1["vmax"] == pytest.approx(40.0)
    assert cfg1.get("seed") == 9
    assert math.isclose(cfg1["slew"], 0.2, rel_tol=0, abs_tol=1e-12)