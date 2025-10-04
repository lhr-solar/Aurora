"""
MEPO — Modified Enhanced Perturb & Observe:

Adaptive-step hill-climb for S1 (everyday) MPPT. The voltage step scales with
recent power change magnitude (|change in P|), and the sign follows sign(change in P * change in V).
This yields faster convergence when far from the MPP and smaller ripple near it.
"""
from __future__ import annotations

from typing import Optional

from .base import MPPTAlgorithm
from .types import Measurement, Action
from .common import clamp, SlewLimiter, compute_power


class MEPO(MPPTAlgorithm):
    """Adaptive-step P&O with simple slew limiting.

    Parameters
    alpha : float
        Gain mapping |change in P| -> step size (scaled and clamped by step_[min,max]).
    step_min, step_max : float
        Bounds on the per-cycle voltage step (Volts).
    vmin, vmax : float
        Allowed voltage command range (Volts).
    slew : float
        Max change allowed per control step for the commanded reference (Volts).
    """

    name = "mepo"
    supports_global_search = False

    def __init__(
        self,
        alpha: float = 0.4,
        step_min: float = 1e-3,
        step_max: float = 2e-2,
        vmin: float = 0.0,
        vmax: float = 100.0,
        slew: float = 0.05,
    ) -> None:
        self.alpha = alpha
        self.step_min = step_min
        self.step_max = step_max
        self.vmin = vmin
        self.vmax = vmax
        self.prev_v: Optional[float] = None
        self.prev_p: Optional[float] = None
        self.v_ref: Optional[float] = None
        self._slew = SlewLimiter(max_step=slew)

    # Lifecycle
    def reset(self) -> None:
        self.prev_v = None
        self.prev_p = None
        self.v_ref = None
        # re-create to keep configured max_step but clear internal state
        self._slew = SlewLimiter(self._slew.max_step)
    
    # Core step
    def step(self, m: Measurement) -> Action:
        p = compute_power(m.v, m.i)

        # Cold start: seed to the present operating point; next call will step.
        if self.v_ref is None or self.prev_v is None or self.prev_p is None:
            self.v_ref = clamp(m.v, self.vmin, self.vmax)
        else:
            dV = m.v - self.prev_v
            dP = p - self.prev_p

            # Direction: follow sign(change in P * change in V)
            sgn = 1.0 if dP * dV > 0.0 else -1.0
            # Magnitude: proportional to |change in P|, then clamped
            raw = abs(dP) * self.alpha
            step = clamp(raw, self.step_min, self.step_max)

            self.v_ref = clamp(self.v_ref + sgn * step, self.vmin, self.vmax)

        # Update memory for next cycle
        self.prev_v, self.prev_p = m.v, p

        # Apply output slew limiting (protect converter, reduce EMI)
        v_cmd = self._slew.step(self.v_ref)

        return Action(
            v_ref=v_cmd,
            debug={
                "p": p,
                "v_ref": float(self.v_ref if self.v_ref is not None else m.v),
                "v_cmd": float(v_cmd),
            },
        )