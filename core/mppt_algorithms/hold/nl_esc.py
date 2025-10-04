

"""
NL-ESC — Newton‑Like Extremum Seeking Control (S4 Lock/Hold)

Low‑ripple MPPT holder that injects a small sinusoidal dither around the
reference voltage, synchronously demodulates the resulting power ripple to
estimate the local gradient dP/dV, and integrates it (optionally with a
Newton‑like gain) to sit at the maximum power point.

This implementation is intentionally lightweight and dependency‑free. It is
meant to be used once you are already near the MPP (e.g., after S3 global
search), not as a global optimizer by itself.
"""
from __future__ import annotations

import math
from typing import Optional

from ..base import MPPTAlgorithm
from ..types import Measurement, Action
from ..common import clamp, EMA, SlewLimiter, compute_power


class NL_ESC(MPPTAlgorithm):
    """Small‑signal extremum seeking with synchronous demodulation.

    Parameters
    dither_amp : float
        Relative amplitude of the injected sinusoidal dither, in Volts (if
        you drive V directly). Keep small (e.g., 0.1–0.3% of Vref).
    dither_hz : float
        Dither frequency in Hz. Choose a value separated from PWM harmonics
        and within your sampling bandwidth.
    k : float
        Integration gain that maps demodulated gradient to a reference shift.
        This effectively includes normalization by dither amplitude.
    vmin, vmax : float
        Allowed command range for the voltage reference.
    slew : float
        Max change allowed per control step for the commanded reference (Volts).
    demod_alpha : float
        EMA smoothing factor [0,1] for the demodulated gradient estimate.
    leak : float
        Optional leak term applied to the internal base reference per step
        (use small values like 0.0–0.01 to bleed off drift if desired).
    """

    name = "nl_esc"
    uses_dither = True
    supports_global_search = False

    def __init__(
        self,
        dither_amp: float = 1e-3,
        dither_hz: float = 120.0,
        k: float = 0.4,
        vmin: float = 0.0,
        vmax: float = 100.0,
        slew: float = 0.05,
        demod_alpha: float = 0.1,
        leak: float = 0.0,
    ) -> None:
        self.A = abs(dither_amp)
        self.f = float(dither_hz)
        self.k = float(k)
        self.vmin, self.vmax = float(vmin), float(vmax)
        self.leak = float(leak)
        self._grad = EMA(alpha=demod_alpha)
        self._slew = SlewLimiter(max_step=slew)
        self.v_ref: Optional[float] = None  # base (undithered) reference
        self._theta: Optional[float] = None  # internal phase when m.t unavailable

    # Lifecycle
    def reset(self) -> None:
        self.v_ref = None
        self._grad.reset()
        self._slew = SlewLimiter(self._slew.max_step)
        self._theta = None

    # Core step
    def step(self, m: Measurement) -> Action:
        # Seed base reference on first call
        if self.v_ref is None:
            self.v_ref = clamp(m.v, self.vmin, self.vmax)

        # Compute dither phase (prefer absolute time; fall back to internal phase)
        omega = 2.0 * math.pi * self.f
        if m.t is not None:
            theta = omega * float(m.t)
        else:
            # Maintain our own phase using dt if available; else assume a small dt
            if self._theta is None:
                self._theta = 0.0
            dt = m.dt if m.dt is not None else 1e-3
            self._theta += omega * float(dt)
            theta = self._theta

        s = math.sin(theta)

        # Measure power and estimate gradient via synchronous demodulation
        p = compute_power(m.v, m.i)
        grad_est = self._grad.update(p * s)

        # Integrate gradient toward the extremum (Newton‑like behavior folded in k)
        base = (1.0 - self.leak) * self.v_ref + self.k * grad_est
        self.v_ref = clamp(base, self.vmin, self.vmax)

        # Command includes dither; then rate‑limit to protect converter
        v_target = self.v_ref + self.A * s
        v_cmd = self._slew.step(v_target)

        return Action(
            v_ref=v_cmd,
            debug={
                "p": p,
                "grad": grad_est,
                "v_base": float(self.v_ref),
                "v_cmd": float(v_cmd),
                "theta": theta,
            },
        )