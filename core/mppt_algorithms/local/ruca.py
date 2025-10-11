"""
RUCA — Robust Unified Control Algorithm (S1 Everyday Tracker)

Sign-based MPPT that unifies voltage- and current-branch hill-climbing.
On each step it updates the reference using either:
  - Voltage branch:  sign(change in P * change in V)
  - Current branch:  sign(change P * change I)
You can alternate branches ("VI") or force a single branch ("V" or "I").

Why RUCA?
  - Fast convergence with very low compute (sign ops only)
  - Works well even at lower update rates
  - Generalizes P&O/IncCond behavior under one simple law

This implementation includes:
  - Optional light adaptivity (smaller step when |change P| is tiny)
  - Slew limiting to protect the converter (dv/dt)
  - Clean debug telemetry for logging/tuning
"""
from typing import Optional

from ..base import MPPTAlgorithm
from ..types import Measurement, Action
from ..common import clamp, SlewLimiter, compute_power


class RUCA(MPPTAlgorithm):
    """Unified sign-based tracker with optional branch alternation.

    Parameters
    step : float
        Base voltage step [V] added/subtracted each control cycle.
    alt_mode : str
        "V" → use only voltage branch; "I" -> only current branch; "VI" -> alternate.
    adapt_frac : float
        If |change P| < adapt_frac * |P|, step is halved (simple noise-aware trimming).
    vmin, vmax : float
        Clamp for commanded voltage reference [V].
    slew : float
        Max change allowed per control step for the commanded reference [V].
    """

    name = "ruca"
    supports_global_search = False

    def __init__(
        self,
        step: float = 8e-3,
        alt_mode: str = "VI",
        adapt_frac: float = 0.02,
        vmin: float = 0.0,
        vmax: float = 100.0,
        slew: float = 0.05,
    ) -> None:
        self.step0 = float(step)
        self.alt_mode = alt_mode.upper()  # "V", "I", or "VI"
        self.adapt_frac = float(adapt_frac)
        self.vmin, self.vmax = float(vmin), float(vmax)
        self._slew = SlewLimiter(max_step=float(slew))

        self.prev_v: Optional[float] = None
        self.prev_i: Optional[float] = None
        self.prev_p: Optional[float] = None
        self.v_ref: Optional[float] = None
        self._branch_toggle: int = 0

    # Lifecycle
    def reset(self) -> None:
        self.prev_v = None
        self.prev_i = None
        self.prev_p = None
        self.v_ref = None
        self._branch_toggle = 0
        self._slew = SlewLimiter(self._slew.max_step)

    # Core step
    def step(self, m: Measurement) -> Action:
        p = compute_power(m.v, m.i)

        if self.v_ref is None or self.prev_v is None or self.prev_i is None or self.prev_p is None:
            # Cold start: seed reference to the present operating point.
            self.v_ref = clamp(m.v, self.vmin, self.vmax)
        else:
            dV = m.v - self.prev_v
            dI = m.i - self.prev_i
            dP = p - self.prev_p

            # Decide which branch to use on this step
            use_v = (self.alt_mode == "V") or (self.alt_mode == "VI" and (self._branch_toggle % 2 == 0))

            # Direction is the sign of change P times change V (or change I)
            metric = dV if use_v else dI
            sgn = 1.0 if (dP * metric) > 0.0 else -1.0

            # Start from base step, trim if |change P| is tiny (noise/near-lock)
            step = self.step0
            if abs(dP) < self.adapt_frac * max(abs(p), 1.0):
                step *= 0.5

            self.v_ref = clamp(self.v_ref + sgn * step, self.vmin, self.vmax)
            self._branch_toggle += 1

        # Memory for next cycle
        self.prev_v, self.prev_i, self.prev_p = m.v, m.i, p

        # Rate-limit command to protect powerstage
        v_cmd = self._slew.step(self.v_ref)

        return Action(
            v_ref=v_cmd,
            debug={
                "p": float(p),
                "v_ref": float(self.v_ref if self.v_ref is not None else m.v),
                "v_cmd": float(v_cmd),
                "branch": ("V" if (self._branch_toggle % 2 == 1 or self.alt_mode == "V") else "I") if self.alt_mode != "I" else "I",
            },
        )