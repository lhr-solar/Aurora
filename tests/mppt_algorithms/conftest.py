"""Pytest fixtures for MPPT algorithm unit tests.

These provide simple IV models and helpers so tests can be concise and
deterministic without importing the full controller stack.

Run tests from the repo root:
    python -m pytest -q
"""

from __future__ import annotations

from typing import Callable
import pytest

from core.mppt_algorithms.types import Measurement

# Simple IV proxies used by unit tests
VOC: float = 42.0
ISC: float = 8.0


def iv_single_peak(v: float) -> float:
    """Linear IV -> parabolic P(V) with a single maximum.

    Clamps to [0, VOC] and returns current (A) for a given voltage (V).
    This is lightweight and deterministic, good enough for per‑algo tests.
    """
    v = max(0.0, min(VOC, v))
    return max(0.0, ISC * (1.0 - v / VOC))


def iv_multi_peak(v: float) -> float:
    """A crude two‑peak proxy to emulate partial shading on P(V).

    We superimpose two linear IV substrings with different effective VOC/ISC.
    Not physically exact, but it produces two distinct power maxima.
    """
    v = max(0.0, min(VOC, v))
    i1 = max(0.0, 0.6 * ISC * (1.0 - v / (0.6 * VOC)))  # shaded substring
    i2 = max(0.0, 0.8 * ISC * (1.0 - v / VOC))          # unshaded substring
    return i1 + i2

# Fixtures
@pytest.fixture(scope="session")
def single_peak() -> Callable[[float], float]:
    """Return a single‑peak IV function f(v) -> i."""
    return iv_single_peak


@pytest.fixture(scope="session")
def multi_peak() -> Callable[[float], float]:
    """Return a multi‑peak IV function f(v) -> i (partial shading proxy)."""
    return iv_multi_peak


@pytest.fixture
def meas(single_peak: Callable[[float], float]):
    """Factory that builds a Measurement using the *single‑peak* IV by default.

    Example:
        m = meas(t=0.0, v=20.0)  # i is computed from single_peak(20.0)
    """
    def _make(t: float, v: float, i: float | None = None, dt: float = 1e-3) -> Measurement:
        i_val = single_peak(v) if i is None else float(i)
        return Measurement(t=t, v=float(v), i=i_val, dt=float(dt))

    return _make


@pytest.fixture
def meas_with_iv():
    """Factory that builds a Measurement using an explicitly provided IV func.

    Example:
        m = meas_with_iv(iv_multi_peak, t=0.0, v=22.0)
    """
    def _make(iv_func: Callable[[float], float], t: float, v: float, dt: float = 1e-3) -> Measurement:
        return Measurement(t=float(t), v=float(v), i=float(iv_func(v)), dt=float(dt))

    return _make


@pytest.fixture(scope="session")
def numeric_pmax():
    """Coarse numeric Pmax estimator for a given IV function.

    Usage:
        pmax = numeric_pmax(iv_single_peak, vmin=0.0, vmax=VOC)
    """
    def _estimate(iv_func: Callable[[float], float], vmin: float = 0.0, vmax: float = VOC, n: int = 2000) -> float:
        best = 0.0
        step = (vmax - vmin) / float(n)
        for k in range(n + 1):
            v = vmin + k * step
            p = v * iv_func(v)
            if p > best:
                best = p
        return best

    return _estimate