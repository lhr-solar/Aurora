"""Sweep — deterministic global search / GMPP reference

A simple deterministic voltage sweep intended for:
- Robust GMPP discovery under partial shading (multi-peak P–V curves)
- Ground-truth validation in simulation (compare against heuristic global search)

This is implemented in the same "one setpoint per control tick" style as PSO:
- On first call, build a coarse grid and command the first point.
- On each subsequent call, attribute the measured power to the previously
  commanded point, update the best-so-far, and command the next point.
- Optionally run a refinement sweep around the best coarse point.
- Finally lock to the best voltage found.

The algorithm assumes the plant/controller will settle sufficiently between
setpoint changes for the measured power to be meaningful. In simulation, this
is generally true.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any

from ..base import MPPTAlgorithm
from ..types import Measurement, Action
from ..common import clamp, SlewLimiter, compute_power


class Sweep(MPPTAlgorithm):
    """Deterministic voltage sweep for global MPP search.

    Parameters
    points : int
        Number of points in the coarse sweep.
    refine_points : int
        Number of points in the refinement sweep (0 disables refinement).
    refine_span_frac : float
        Half-span of the refinement window as a fraction of (vmax-vmin).
        Example: 0.05 => refine within ±5% of full range around coarse best.
    window_pct : float
        If > 0, sweep a window centered at the initial measured voltage:
        [Vseed - window_pct*Vseed, Vseed + window_pct*Vseed].
        A small absolute minimum span is enforced.
        If 0, sweep the full [vmin, vmax] range.
    vmin, vmax : float
        Allowed voltage command range.
    slew : float
        Max voltage change per control step (setpoint slew) [V/step].
    """

    name = "sweep"
    supports_global_search = True

    def __init__(
        self,
        points: int = 31,
        refine_points: int = 15,
        refine_span_frac: float = 0.05,
        window_pct: float = 0.0,
        vmin: float = 0.0,
        vmax: float = 100.0,
        slew: float = 0.25,
    ) -> None:
        self.points = max(3, int(points))
        self.refine_points = max(0, int(refine_points))
        self.refine_span_frac = float(refine_span_frac)
        self.window_pct = float(window_pct)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self._slew = SlewLimiter(max_step=float(slew))
        self.reset()

    # ---- lifecycle ----
    def reset(self) -> None:
        self.stage: str = "idle"  # idle|coarse|refine|lock
        self.grid: List[float] = []
        self.idx: int = 0
        self.last_cmd: Optional[float] = None
        self.v_ref: Optional[float] = None

        self.best_v: Optional[float] = None
        self.best_p: float = -1e18

        self._seed_v: Optional[float] = None
        self._coarse_done: bool = False

    def _apply_slew(self, prev: Optional[float], target: float) -> float:
        """Apply slew limiting using whichever API the limiter exposes."""
        s = self._slew
        fn = getattr(s, "limit", None)
        if callable(fn):
            return fn(prev, target)
        try:
            return s(prev, target)  # callable SlewLimiter
        except Exception:
            step = float(getattr(s, "max_step", 0.0) or 0.0)
            if prev is None:
                return float(target)
            dv = float(target) - float(prev)
            if dv > step:
                return float(prev) + step
            if dv < -step:
                return float(prev) - step
            return float(target)

    def _linspace(self, a: float, b: float, n: int) -> List[float]:
        if n <= 1:
            return [float(a)]
        a = float(a)
        b = float(b)
        step = (b - a) / float(n - 1)
        return [a + k * step for k in range(n)]

    def _build_coarse_grid(self, seed_v: float) -> None:
        self._seed_v = float(seed_v)

        if self.window_pct > 0.0:
            center = clamp(seed_v, self.vmin, self.vmax)
            abs_min = 0.05 * (self.vmax - self.vmin)
            span = max(self.window_pct * max(abs(center), 1e-6), abs_min)
            v_lo = clamp(center - span, self.vmin, self.vmax)
            v_hi = clamp(center + span, self.vmin, self.vmax)
        else:
            v_lo, v_hi = self.vmin, self.vmax

        # Avoid degenerate grids
        if v_hi - v_lo < 1e-6:
            v_hi = min(self.vmax, v_lo + 1e-3)

        self.grid = self._linspace(v_lo, v_hi, self.points)
        self.stage = "coarse"
        self.idx = 0
        self._coarse_done = False

    def _build_refine_grid(self, center_v: float) -> None:
        span = max(0.0, self.refine_span_frac) * (self.vmax - self.vmin)
        # Enforce a small minimum refinement span so we actually move
        span = max(span, 0.02 * (self.vmax - self.vmin))
        v_lo = clamp(center_v - span, self.vmin, self.vmax)
        v_hi = clamp(center_v + span, self.vmin, self.vmax)
        if v_hi - v_lo < 1e-6:
            v_hi = min(self.vmax, v_lo + 1e-3)

        self.grid = self._linspace(v_lo, v_hi, max(3, int(self.refine_points)))
        self.stage = "refine"
        self.idx = 0

    # ---- UI / config ----
    def describe(self) -> Dict[str, Any]:
        return {
            "key": self.name,
            "label": "Sweep (deterministic)",
            "params": [
                {"name": "points", "type": "integer", "min": 3, "max": 501, "step": 1,
                 "default": self.points, "help": "number of points in the coarse sweep"},
                {"name": "refine_points", "type": "integer", "min": 0, "max": 301, "step": 1,
                 "default": self.refine_points, "help": "points in the refinement sweep (0 disables)"},
                {"name": "refine_span_frac", "type": "number", "min": 0.0, "max": 0.5, "step": 0.005,
                 "default": self.refine_span_frac, "help": "refine half-span as fraction of full range"},
                {"name": "window_pct", "type": "number", "min": 0.0, "max": 0.9, "step": 0.01,
                 "default": self.window_pct, "help": "if >0, sweep ±window_pct*Vseed; else sweep full range"},
                {"name": "slew", "type": "number", "min": 0.01, "max": 5.0, "step": 0.01,
                 "unit": "V/step", "default": float(getattr(self._slew, "max_step", 0.0))},
                {"name": "vmin", "type": "number", "default": self.vmin, "unit": "V"},
                {"name": "vmax", "type": "number", "default": self.vmax, "unit": "V"},
            ],
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            "points": self.points,
            "refine_points": self.refine_points,
            "refine_span_frac": self.refine_span_frac,
            "window_pct": self.window_pct,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "slew": float(getattr(self._slew, "max_step", 0.0)),
        }

    def update_params(self, **kw: Any) -> None:
        if "points" in kw:
            self.points = max(3, int(kw["points"]))
        if "refine_points" in kw:
            self.refine_points = max(0, int(kw["refine_points"]))
        if "refine_span_frac" in kw:
            self.refine_span_frac = float(kw["refine_span_frac"])  # can be 0
        if "window_pct" in kw:
            self.window_pct = float(kw["window_pct"])  # can be 0
        if "vmin" in kw:
            self.vmin = float(kw["vmin"])
        if "vmax" in kw:
            self.vmax = float(kw["vmax"])
        if "slew" in kw:
            self._slew.max_step = float(kw["slew"])

    # ---- core ----
    def step(self, m: Measurement) -> Action:
        p = compute_power(m.v, m.i)

        # First call: build coarse grid and command the first point
        if self.stage == "idle":
            self._build_coarse_grid(m.v)
            target = self.grid[0]
            self.last_cmd = target
            raw = clamp(target, self.vmin, self.vmax)
            prev = self.v_ref if self.v_ref is not None else m.v
            v_out = self._apply_slew(prev, raw)
            self.v_ref = v_out
            # Initialize best to current operating point until we start assigning fitness
            self.best_v = float(m.v)
            self.best_p = float(p)
            return Action(
                v_ref=v_out,
                debug={
                    "algo": "sweep",
                    "phase": 1,
                    "done": False,
                    "stage": self.stage,
                    "idx": 0,
                    "n": len(self.grid),
                    "best_v": float(self.best_v),
                    "best_p": float(self.best_p),
                    "p": float(p),
                },
            )

        # Attribute measured power to the previously commanded point
        if self.last_cmd is not None:
            if p > self.best_p:
                self.best_p = float(p)
                self.best_v = float(self.last_cmd)

        # Advance index
        self.idx += 1

        # End-of-grid logic
        if self.idx >= len(self.grid):
            # Finished coarse; optionally refine once
            if (self.stage == "coarse") and (not self._coarse_done):
                self._coarse_done = True
                if self.refine_points > 0 and self.best_v is not None:
                    self._build_refine_grid(float(self.best_v))
                    target = self.grid[0]
                    self.last_cmd = target
                    raw = clamp(target, self.vmin, self.vmax)
                    prev = self.v_ref if self.v_ref is not None else m.v
                    v_out = self._apply_slew(prev, raw)
                    self.v_ref = v_out
                    return Action(
                        v_ref=v_out,
                        debug={
                            "algo": "sweep",
                            "phase": 1,
                            "done": False,
                            "stage": self.stage,
                            "idx": 0,
                            "n": len(self.grid),
                            "best_v": float(self.best_v or 0.0),
                            "best_p": float(self.best_p),
                            "p": float(p),
                        },
                    )

            # Otherwise lock to best
            self.stage = "lock"
            target = float(self.best_v if self.best_v is not None else m.v)
            self.last_cmd = target
            raw = clamp(target, self.vmin, self.vmax)
            prev = self.v_ref if self.v_ref is not None else m.v
            v_out = self._apply_slew(prev, raw)
            self.v_ref = v_out
            return Action(
                v_ref=v_out,
                debug={
                    "algo": "sweep",
                    "phase": 2,
                    "done": True,
                    "stage": self.stage,
                    "idx": len(self.grid),
                    "n": len(self.grid),
                    "best_v": float(self.best_v or target),
                    "best_p": float(self.best_p),
                    "p": float(p),
                },
            )

        # Normal sweep step
        target = self.grid[self.idx]
        self.last_cmd = target
        raw = clamp(target, self.vmin, self.vmax)
        prev = self.v_ref if self.v_ref is not None else m.v
        v_out = self._apply_slew(prev, raw)
        self.v_ref = v_out

        return Action(
            v_ref=v_out,
            debug={
                "algo": "sweep",
                "phase": 1 if self.stage != "lock" else 2,
                "done": (self.stage == "lock"),
                "stage": self.stage,
                "idx": int(self.idx),
                "n": len(self.grid),
                "best_v": float(self.best_v or 0.0),
                "best_p": float(self.best_p),
                "p": float(p),
            },
        )


__all__ = ["Sweep"]