"""
Common utilities for MPPT algorithms
- Small, dependency-free helpers used across multiple algorithms.

Includes:
- clamp(x, a, b): saturate a value into [a, b]
- within(x, a, b): check if x ∈ [a, b]
- sign(x): return -1.0, 0.0, or 1.0
- deadband(x, eps): zero-out small magnitudes
- safe_div(n, d, default): numerically safe division
- compute_power(v, i): convenience P = V·I
- EMA: exponential moving average (low-pass)
- MovingAverage: fixed-window mean
- SlewLimiter: per-step rate limiter for setpoints (e.g., v_ref)
"""
from __future__ import annotations

from typing import Optional, Deque
from collections import deque

# Scalar helpers
def clamp(x: float, a: float, b: float) -> float:
    """Clamp ``x`` into the closed interval [a, b].

    Assumes a <= b. If not, behavior still saturates using the provided bounds.
    """
    return a if x < a else b if x > b else x

def within(x: float, a: float, b: float) -> bool:
    """Return True iff ``x`` in [a, b]."""
    return a <= x <= b

def sign(x: float) -> float:
    """Return the sign of ``x`` as -1.0, 0.0, or 1.0."""
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0

def deadband(x: float, eps: float = 0.0) -> float:
    """Zero-out ``x`` when |x| <= eps; otherwise return x unchanged."""
    return 0.0 if abs(x) <= abs(eps) else x

def safe_div(n: float, d: float, default: float = 0.0) -> float:
    """Return n / d with a fallback when d is (near) zero."""
    return default if abs(d) < 1e-12 else n / d

def compute_power(v: float, i: float) -> float:
    """Convenience: compute electrical power P = V*I."""
    return v * i


# Filters
class EMA:
    """Exponential moving average (single-pole IIR low-pass).

    Parameters
    alpha : float
        Smoothing factor in [0, 1]. Higher = more weight on new samples.
    init : float
        Initial value (used for the first update).
    """

    def __init__(self, alpha: float, init: float = 0.0):
        self.a = clamp(alpha, 0.0, 1.0)
        self.y = init
        self.ready = False

    def reset(self, init: Optional[float] = None) -> None:
        """Reset the filter state; optionally set a new initial value."""
        if init is not None:
            self.y = init
        self.ready = False

    def update(self, x: float) -> float:
        """Feed a new sample and return the filtered value."""
        if not self.ready:
            self.y = x
            self.ready = True
        else:
            self.y = self.a * x + (1.0 - self.a) * self.y
        return self.y


class MovingAverage:
    """Fixed-size windowed average (simple FIR)."""

    def __init__(self, window: int, init: Optional[float] = None):
        if window <= 0:
            raise ValueError("window must be > 0")
        self.window = int(window)
        self.buf: Deque[float] = deque(maxlen=self.window)
        self.sum = 0.0
        if init is not None:
            for _ in range(self.window):
                self.buf.append(init)
                self.sum += init

    def reset(self, init: Optional[float] = None) -> None:
        self.buf.clear()
        self.sum = 0.0
        if init is not None:
            for _ in range(self.window):
                self.buf.append(init)
                self.sum += init

    def update(self, x: float) -> float:
        if len(self.buf) == self.window:
            self.sum -= self.buf[0]
        self.buf.append(x)
        self.sum += x
        return self.sum / len(self.buf)

    @property
    def value(self) -> float:
        return self.sum / len(self.buf) if self.buf else 0.0

# Rate limiting
class SlewLimiter:
    """Per-cycle rate limiter for a scalar setpoint (e.g., v_ref).

    Constrains the change between consecutive outputs to (+/-) max_step.
    Useful to protect the DC-DC stage and reduce EMI during global searches.
    """

    def __init__(self, max_step: float):
        self.max_step = abs(max_step)
        self._last: Optional[float] = None

    def reset(self, last: Optional[float] = None) -> None:
        """Reset the internal state; optionally seed with a last value."""
        self._last = last

    def step(self, target: float) -> float:
        if self._last is None:
            self._last = target
            return target
        dv = target - self._last
        if dv > self.max_step:
            self._last += self.max_step
        elif dv < -self.max_step:
            self._last -= self.max_step
        else:
            self._last = target
        return self._last

    @property
    def last(self) -> Optional[float]:
        return self._last