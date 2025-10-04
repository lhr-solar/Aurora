"""
PANDO — Classic Perturb & Observe (P & O)

Fixed-step hill-climb MPPT for everyday, uniform-irradiance conditions.
This variant keeps a running perturbation direction and flips it whenever the
last perturbation reduced power. Simple, lightweight, and deterministic.

Notes
- Use small steps for low ripple; larger steps for faster convergence.
- Consider switching to MEPO/RUCA for adaptive steps, or to global search
  (PSO/Firefly/GA/ACO) when partial shading is detected.
"""
from __future__ import annotations

from typing import Optional

from ..base import MPPTAlgorithm
from ..types import Measurement, Action
from ..common import clamp, SlewLimiter, compute_power, deadband


class PANDO(MPPTAlgorithm):
    """Fixed‑step P&O with direction memory and slew limiting.

    Parameters
    step : float
        Fixed perturbation magnitude in Volts per control step.
    eps : float
        Power change deadband [W]; below this, treat as no change (noise guard).
    vmin, vmax : float
        Allowed voltage command range [V].
    slew : float
        Max allowed change per control step for the commanded reference [V].
    """

    name = "pando"
    supports_global_search = False

    def __init__(
        self,
        step: float = 8e-3,
        eps: float = 0.0,
        vmin: float = 0.0,
        vmax: float = 100.0,
        slew: float = 0.05,
    ) -> None:
        self.step = float(step)
        self.eps = float(eps)
        self.vmin, self.vmax = float(vmin), float(vmax)
        self._slew = SlewLimiter(max_step=float(slew))

        self.v_ref: Optional[float] = None
        self.prev_p: Optional[float] = None
        self.prev_v: Optional[float] = None
        self._dir: float = 1.0  # +1 increase V, -1 decrease V

    # Lifecycle
    def reset(self) -> None:
        self.v_ref = None
        self.prev_p = None
        self.prev_v = None
        self._dir = 1.0
        self._slew = SlewLimiter(self._slew.max_step)

    # Core step
    def step(self, m: Measurement) -> Action:
        p = compute_power(m.v, m.i)

        if self.v_ref is None or self.prev_p is None or self.prev_v is None:
            # cold start: seed reference to current operating point
            self.v_ref = clamp(m.v, self.vmin, self.vmax)
        else:
            dP = p - self.prev_p
            # Noise deadband: ignore tiny changes
            dP = deadband(dP, self.eps)

            if dP < 0.0:
                # Last perturbation hurt power -> flip direction
                self._dir = -self._dir
            # else keep the same direction

            self.v_ref = clamp(self.v_ref + self._dir * self.step, self.vmin, self.vmax)

        # memory for next cycle
        self.prev_p, self.prev_v = p, m.v

        # protect converter with slew limiting
        v_cmd = self._slew.step(self.v_ref)

        return Action(
            v_ref=v_cmd,
            debug={
                "p": p,
                "dP": p - (self.prev_p if self.prev_p is not None else p),
                "dir": self._dir,
                "v_ref": float(self.v_ref if self.v_ref is not None else m.v),
                "v_cmd": float(v_cmd),
            },
        )
