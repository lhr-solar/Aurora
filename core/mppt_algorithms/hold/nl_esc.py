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
import math
from typing import Optional, Dict, Any

from ..base import MPPTAlgorithm
from ..types import Measurement, Action
from ..common import clamp, EMA, SlewLimiter, compute_power


class NL_ESC(MPPTAlgorithm):
    """Small‑signal extremum seeking with synchronous demodulation.

    Parameters
    dither_amp : float
        Absolute amplitude of the injected sinusoidal dither in Volts. Keep small (e.g., ≈0.1–0.3% of Vref as a rule of thumb).
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
                "algo": "nl_esc",
                "phase": "hold",
                "p": float(p),
                "grad": float(grad_est),
                "v_base": float(self.v_ref),
                "v_cmd": float(v_cmd),
                "theta": float(theta % (2.0 * math.pi)),
                "dither_amp": float(self.A),
                "dither_hz": float(self.f),
            },
        )

    # ---- Frontend helpers ----
    def describe(self) -> Dict[str, Any]:
        """Return UI metadata for tunable parameters."""
        return {
            "key": self.name,
            "label": "NL-ESC (hold)",
            "params": [
                {"name": "dither_amp",  "type": "number",  "min": 0.0,  "max": 2.0,   "step": 1e-3, "unit": "V",      "default": self.A,       "help": "Sinusoidal dither amplitude (Volts)"},
                {"name": "dither_hz",   "type": "number",  "min": 5.0,  "max": 2000.0, "step": 1.0,  "unit": "Hz",     "default": self.f,       "help": "Dither frequency"},
                {"name": "k",           "type": "number",  "min": 0.01, "max": 2.0,    "step": 0.01,              "default": self.k,       "help": "Integrator gain (includes normalization)"},
                {"name": "demod_alpha", "type": "number",  "min": 0.01, "max": 0.9,    "step": 0.01,              "default": self._grad.alpha, "help": "EMA smoothing for demodulated gradient"},
                {"name": "leak",        "type": "number",  "min": 0.0,  "max": 0.05,   "step": 1e-3,              "default": self.leak,    "help": "Optional leak on base reference per step"},
                {"name": "slew",        "type": "number",  "min": 0.001,"max": 2.0,    "step": 0.001, "unit": "V/step","default": self._slew.max_step, "help": "Max change allowed per control step"},
                {"name": "vmin",        "type": "number",                                      "default": self.vmin,    "unit": "V"},
                {"name": "vmax",        "type": "number",                                      "default": self.vmax,    "unit": "V"},
            ],
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            "dither_amp": self.A,
            "dither_hz": self.f,
            "k": self.k,
            "demod_alpha": self._grad.alpha,
            "leak": self.leak,
            "slew": self._slew.max_step,
            "vmin": self.vmin,
            "vmax": self.vmax,
        }

    def update_params(self, **kw: Any) -> None:
        if "dither_amp" in kw:
            self.A = abs(float(kw["dither_amp"]))
        if "dither_hz" in kw:
            self.f = max(0.1, float(kw["dither_hz"]))
        if "k" in kw:
            self.k = float(kw["k"])  # caller responsible for stability tuning
        if "demod_alpha" in kw:
            alpha = float(kw["demod_alpha"])
            alpha = max(1e-6, min(alpha, 0.999))
            # Re-create EMA with new alpha; state is reset to avoid bias
            self._grad = EMA(alpha=alpha)
        if "leak" in kw:
            self.leak = max(0.0, float(kw["leak"]))
        if "slew" in kw:
            self._slew.max_step = max(1e-6, float(kw["slew"]))
        if "vmin" in kw:
            self.vmin = float(kw["vmin"]) 
        if "vmax" in kw:
            self.vmax = float(kw["vmax"]) 