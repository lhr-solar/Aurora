"""
HybridMPPT — S0..S5 state-machine orchestrator
Wires together S1 (everyday local tracker), S3 (global search), and S4 (lock/hold)
with a partial-shading detector (PSD) and basic safety checks.

This is a minimal, dependency-free controller intended for simulation/bench bring-up.
You can evolve it with richer PSD logic, limit handling, and telemetry.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from ..mppt_algorithms.types import Measurement, Action
from ..mppt_algorithms.registry import build
from .psd import PSDDetector
from .safety import SafetyLimits, check_limits


class State(str, Enum):
    INIT = "INIT"
    NORMAL = "NORMAL"
    GLOBAL_SEARCH = "GLOBAL_SEARCH"
    LOCK_HOLD = "LOCK_HOLD"
    SAFETY = "SAFETY"


@dataclass
class HybridConfig:
    """Configuration for the HybridMPPT controller.

    Selects algorithms for each state and their constructor kwargs, plus PSD
    and safety settings. All fields have sensible defaults.
    """

    # Algorithm selections and kwargs
    normal_name: str = "ruca"
    normal_kwargs: Dict[str, Any] = field(default_factory=lambda: {"step": 8e-3, "alt_mode": "VI"})

    global_name: str = "pso"
    global_kwargs: Dict[str, Any] = field(default_factory=lambda: {"particles": 6, "iters": 12, "window_pct": 0.2})

    hold_name: str = "nl_esc"
    hold_kwargs: Dict[str, Any] = field(default_factory=lambda: {"dither_amp": 1e-3, "dither_hz": 120.0, "k": 0.4})

    # PSD and Safety
    psd: Dict[str, Any] = field(default_factory=lambda: {"dp_frac": 0.04, "dv_frac": 0.01, "window": 5, "votes": 2})
    safety: SafetyLimits = field(default_factory=SafetyLimits)

    # Hysteresis / quiet cycles before S4→S1
    quiet_cycles: int = 3


class HybridMPPT:
    """Condition-aware hybrid MPPT controller with a simple state machine."""

    def __init__(self, cfg: HybridConfig = HybridConfig()):
        self.cfg = cfg
        # Build algorithms via the registry
        self.s1 = build(cfg.normal_name, **cfg.normal_kwargs)
        self.s3 = build(cfg.global_name, **cfg.global_kwargs)
        self.s4 = build(cfg.hold_name, **cfg.hold_kwargs)
        # Utilities
        self.psd = PSDDetector(**cfg.psd)
        # State
        self.state = State.INIT
        self._quiet = 0
        self._last_good_v: Optional[float] = None

    # Lifecycle
    def reset(self) -> None:
        """Reset controller state and underlying algorithms."""
        self.state = State.INIT
        self.s1.reset(); self.s3.reset(); self.s4.reset()
        self.psd.reset()
        self._quiet = 0
        self._last_good_v = None

    # Core step
    def step(self, m: Measurement) -> Action:
        """Advance the controller one cycle.

        Applies safety checks, runs the active state's algorithm, and performs
        state transitions based on PSD and lock/hold hysteresis.
        """
        # Safety first; if violated, enter SAFETY and hold a safe point
        status = check_limits(m, self.cfg.safety)
        if status != "OK":
            self.state = State.SAFETY
            safe_v = min(max(m.v, self.cfg.safety.vmin), self.cfg.safety.vmax)
            return Action(v_ref=safe_v, debug={"state": "SAFETY", "fault": status})

        if self.state == State.INIT:
            self.s1.reset(); self.s3.reset(); self.s4.reset(); self.psd.reset()
            self.state = State.NORMAL

        if self.state == State.NORMAL:
            a = self.s1.step(m)
            self._last_good_v = a.v_ref if a.v_ref is not None else m.v
            # Decide if we need a global search burst
            if self.psd.update_and_check(m):
                self.state = State.GLOBAL_SEARCH
                self.s3.reset()
            return self._with_state(a, State.NORMAL)

        if self.state == State.GLOBAL_SEARCH:
            a = self.s3.step(m)
            # Heuristic: PSO implementation sets debug["phase"] = 2 when locking to gbest
            phase = int(a.debug.get("phase", 1)) if a.debug else 1
            if phase == 2:
                self.state = State.LOCK_HOLD
                self.s4.reset()
            return self._with_state(a, State.GLOBAL_SEARCH)

        if self.state == State.LOCK_HOLD:
            a = self.s4.step(m)
            self._last_good_v = a.v_ref if a.v_ref is not None else m.v
            # Quiet-hold detection using ESC gradient magnitude
            grad = abs(float(a.debug.get("grad", 0.0))) if a.debug else 0.0
            self._quiet = self._quiet + 1 if grad < 1e-3 else 0
            if self._quiet >= self.cfg.quiet_cycles:
                self.state = State.NORMAL
                self.s1.reset(); self._quiet = 0
            # Re-check PSC in case conditions changed again
            if self.psd.update_and_check(m):
                self.state = State.GLOBAL_SEARCH
                self.s3.reset()
            return self._with_state(a, State.LOCK_HOLD)

        # Fallback (should not happen):
        return Action(v_ref=m.v, debug={"state": str(self.state)})

    # Helpers
    def _with_state(self, a: Action, state: State) -> Action:
        dbg = dict(a.debug) if a.debug else {}
        dbg["state"] = state.value
        a.debug = dbg
        return a
