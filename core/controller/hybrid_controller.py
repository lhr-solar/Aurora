"""
HybridMPPT — S0..S5 state-machine orchestrator
Wires together S1 (everyday local tracker), S3 (global search), and S4 (lock/hold)
with a partial-shading detector (PSD) and basic safety checks.

This is a minimal, dependency-free controller intended for simulation/bench bring-up.
You can evolve it with richer PSD logic, limit handling, and telemetry.
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Optional
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..mppt_algorithms.types import Measurement, Action
 
# Runtime import (Action is instantiated in this module)
from ..mppt_algorithms.types import Action, Measurement
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

    # LOCK_HOLD robustness (Option B): exit on drift + periodic revalidation
    hold_exit_dp_frac: float = 0.06   # exit HOLD if power drops >6% vs lock reference
    hold_exit_dv_frac: float = 0.03   # exit HOLD if voltage drifts >3% vs lock reference
    hold_revalidate_period_s: float = 0.10  # every 0.10s, run a few local steps to re-check slope
    hold_revalidate_cycles: int = 5         # number of NORMAL (S1) steps during revalidation


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

        # LOCK_HOLD reference + revalidation bookkeeping
        self._hold_p_ref: Optional[float] = None
        self._hold_v_ref: Optional[float] = None
        self._hold_elapsed_s: float = 0.0
        self._reval_remaining: int = 0

        # Event log for frontend (state changes, PSD triggers, safety trips)
        self.events: list[dict] = []

    # Lifecycle
    def reset(self) -> None:
        """Reset controller state and underlying algorithms."""
        self.state = State.INIT
        self.s1.reset(); self.s3.reset(); self.s4.reset()
        self.psd.reset()
        self._quiet = 0
        self._last_good_v = None

        self._hold_p_ref = None
        self._hold_v_ref = None
        self._hold_elapsed_s = 0.0
        self._reval_remaining = 0

    # Core step
    def step(self, m: "Measurement") -> "Action":
        """Advance the controller one cycle.

        Applies safety checks, runs the active state's algorithm, and performs
        state transitions based on PSD and lock/hold hysteresis.
        """
        # Safety first; if violated, enter SAFETY and hold a safe point
        status = check_limits(m, self.cfg.safety)
        if status != "OK":
            self.state = State.SAFETY
            safe_v = min(max(m.v, self.cfg.safety.vmin), self.cfg.safety.vmax)
            # Event: safety trip
            self.events.append({
                "t": m.t,
                "type": "safety_trip",
                "fault": status,
                "v": m.v,
                "i": m.i,
            })
            return Action(v_ref=safe_v, debug={"state": "SAFETY", "fault": status})

        if self.state == State.INIT:
            self.s1.reset(); self.s3.reset(); self.s4.reset(); self.psd.reset()
            self.state = State.NORMAL

        if self.state == State.NORMAL:
            a = self.s1.step(m)
            self._last_good_v = a.v_ref if a.v_ref is not None else m.v
            # Decide if we need a global search burst
            if self.psd.update_and_check(m):
                # Events: PSD trigger and state change NORMAL -> GLOBAL_SEARCH
                self.events.append({"t": m.t, "type": "psd_trigger"})
                self.events.append({"t": m.t, "type": "state_change", "from": State.NORMAL.value, "to": State.GLOBAL_SEARCH.value})
                self.state = State.GLOBAL_SEARCH
                self.s3.reset()
                # Mark the transition on the Action for UI/telemetry
                dbg = dict(a.debug) if a.debug else {}
                dbg.update({"state": State.NORMAL.value, "next_state": State.GLOBAL_SEARCH.value, "reason": "psd_trigger"})
                a.debug = dbg
                return a
            return self._with_state(a, State.NORMAL)

        if self.state == State.GLOBAL_SEARCH:
            a = self.s3.step(m)
            # Heuristic: PSO implementation sets debug["phase"] = 2 when locking to gbest
            phase = int(a.debug.get("phase", 1)) if a.debug else 1
            if phase == 2:
                self.events.append({"t": m.t, "type": "state_change", "from": State.GLOBAL_SEARCH.value, "to": State.LOCK_HOLD.value})
                self.state = State.LOCK_HOLD
                self.s4.reset()
                # Establish lock references for drift detection + periodic revalidation
                self._hold_p_ref = float(m.p)
                self._hold_v_ref = float(m.v)
                self._hold_elapsed_s = 0.0
                self._reval_remaining = 0
                # Mark the transition on the Action
                dbg = dict(a.debug) if a.debug else {}
                dbg.update({"state": State.GLOBAL_SEARCH.value, "next_state": State.LOCK_HOLD.value, "reason": "phase_lock"})
                a.debug = dbg
                return a
            return self._with_state(a, State.GLOBAL_SEARCH)

        if self.state == State.LOCK_HOLD:
            # Accumulate time spent in HOLD
            self._hold_elapsed_s += float(getattr(m, "dt", 0.0) or 0.0)

            # Exit HOLD if we drift materially away from the lock reference
            if self._hold_p_ref is not None and self._hold_p_ref > 1e-12:
                dp_frac = (float(self._hold_p_ref) - float(m.p)) / float(self._hold_p_ref)
                if dp_frac > float(self.cfg.hold_exit_dp_frac):
                    self.events.append({"t": m.t, "type": "hold_exit", "reason": "power_drop", "dp_frac": float(dp_frac)})
                    self.events.append({"t": m.t, "type": "state_change", "from": State.LOCK_HOLD.value, "to": State.GLOBAL_SEARCH.value})
                    self.state = State.GLOBAL_SEARCH
                    self.s3.reset()
                    return Action(v_ref=m.v, debug={"state": State.GLOBAL_SEARCH.value, "reason": "power_drop", "dp_frac": float(dp_frac)})

            if self._hold_v_ref is not None and abs(float(self._hold_v_ref)) > 1e-12:
                dv_frac = abs(float(m.v) - float(self._hold_v_ref)) / abs(float(self._hold_v_ref))
                if dv_frac > float(self.cfg.hold_exit_dv_frac):
                    self.events.append({"t": m.t, "type": "hold_exit", "reason": "voltage_drift", "dv_frac": float(dv_frac)})
                    self.events.append({"t": m.t, "type": "state_change", "from": State.LOCK_HOLD.value, "to": State.GLOBAL_SEARCH.value})
                    self.state = State.GLOBAL_SEARCH
                    self.s3.reset()
                    return Action(v_ref=m.v, debug={"state": State.GLOBAL_SEARCH.value, "reason": "voltage_drift", "dv_frac": float(dv_frac)})

            # Periodic revalidation: occasionally run a few local (S1) steps to re-check slope
            if self.cfg.hold_revalidate_period_s > 0 and self._hold_elapsed_s >= float(self.cfg.hold_revalidate_period_s) and self._reval_remaining == 0:
                self._reval_remaining = int(self.cfg.hold_revalidate_cycles)
                self._hold_elapsed_s = 0.0
                self.s1.reset()
                self.events.append({"t": m.t, "type": "hold_revalidate_start", "cycles": int(self.cfg.hold_revalidate_cycles)})

            if self._reval_remaining > 0:
                a = self.s1.step(m)
                self._reval_remaining -= 1
                # If PSD fires during revalidation, immediately jump to global search
                if self.psd.update_and_check(m):
                    self.events.append({"t": m.t, "type": "psd_trigger"})
                    self.events.append({"t": m.t, "type": "state_change", "from": State.LOCK_HOLD.value, "to": State.GLOBAL_SEARCH.value})
                    self.state = State.GLOBAL_SEARCH
                    self.s3.reset()
                    return self._with_state(a, State.GLOBAL_SEARCH)
                # When revalidation completes, refresh lock references
                if self._reval_remaining == 0:
                    self._hold_p_ref = float(m.p)
                    self._hold_v_ref = float(m.v)
                    self.events.append({"t": m.t, "type": "hold_revalidate_done"})
                return self._with_state(a, State.LOCK_HOLD)

            # Default HOLD behavior
            a = self.s4.step(m)
            self._last_good_v = a.v_ref if a.v_ref is not None else m.v

            # Quiet-hold detection using ESC gradient magnitude (allows S4→S1 in stable conditions)
            grad = abs(float(a.debug.get("grad", 0.0))) if a.debug else 0.0
            self._quiet = self._quiet + 1 if grad < 1e-3 else 0
            if self._quiet >= self.cfg.quiet_cycles:
                self.events.append({"t": m.t, "type": "state_change", "from": State.LOCK_HOLD.value, "to": State.NORMAL.value})
                self.state = State.NORMAL
                self.s1.reset(); self._quiet = 0
                # Also refresh lock refs so next HOLD entry uses current point
                self._hold_p_ref = None
                self._hold_v_ref = None
                self._hold_elapsed_s = 0.0
                self._reval_remaining = 0
                # Mark the transition on the Action
                dbg = dict(a.debug) if a.debug else {}
                dbg.update({"state": State.LOCK_HOLD.value, "next_state": State.NORMAL.value, "reason": "quiet_hold"})
                a.debug = dbg

            # Re-check PSC in case conditions changed again
            if self.psd.update_and_check(m):
                self.events.append({"t": m.t, "type": "psd_trigger"})
                self.events.append({"t": m.t, "type": "state_change", "from": State.LOCK_HOLD.value, "to": State.GLOBAL_SEARCH.value})
                self.state = State.GLOBAL_SEARCH
                self.s3.reset()
                # Mark the transition on the Action
                dbg = dict(a.debug) if a.debug else {}
                dbg.update({"state": State.LOCK_HOLD.value, "next_state": State.GLOBAL_SEARCH.value, "reason": "psd_trigger"})
                a.debug = dbg
                return a

            return self._with_state(a, State.LOCK_HOLD)

        # Fallback (should not happen):
        return Action(v_ref=m.v, debug={"state": str(self.state)})

    # Helpers
    def _with_state(self, a: "Action", state: State) -> "Action":
        dbg = dict(a.debug) if a.debug else {}
        dbg["state"] = state.value
        a.debug = dbg
        return a

    # Frontend helpers
    def get_state(self) -> str:
        """Return current state name for UI convenience."""
        return self.state.value

    def get_snapshot(self) -> Dict[str, Any]:
        """Minimal snapshot for the UI (state and last good reference)."""
        return {
            "state": self.state.value,
            "last_good_v": self._last_good_v,
        }

    def get_events(self, clear: bool = True) -> list[dict]:
        """Return accumulated events; clear by default for streaming semantics."""
        ev = list(self.events)
        if clear:
            self.events.clear()
        return ev

    def describe(self) -> Dict[str, Any]:
        """Compose describe() specs from S1/S3/S4 and include PSD/Safety schemas."""
        s1_desc = getattr(self.s1, "describe", lambda: {"key": "s1", "label": "S1", "params": []})()
        s3_desc = getattr(self.s3, "describe", lambda: {"key": "s3", "label": "S3", "params": []})()
        s4_desc = getattr(self.s4, "describe", lambda: {"key": "s4", "label": "S4", "params": []})()
        psd_schema = {
            "key": "psd",
            "label": "PSD Detector",
            "params": [
                {"name": "dp_frac", "type": "number", "min": 0.001, "max": 0.2, "step": 0.001, "default": float(self.cfg.psd.get("dp_frac", 0.04)), "help": "|dP|/P threshold"},
                {"name": "dv_frac", "type": "number", "min": 0.001, "max": 0.2, "step": 0.001, "default": float(self.cfg.psd.get("dv_frac", 0.01)), "help": "|dV|/V threshold"},
                {"name": "window",  "type": "integer","min": 1,     "max": 50,  "step": 1,     "default": int(self.cfg.psd.get("window", 5)),     "help": "samples per decision"},
                {"name": "votes",   "type": "integer","min": 1,     "max": 10,  "step": 1,     "default": int(self.cfg.psd.get("votes", 2)),      "help": "consecutive votes required"},
            ],
        }
        safety = self.cfg.safety
        safety_schema = {
            "key": "safety",
            "label": "Safety Limits",
            "params": [
                {"name": "vmin",    "type": "number", "unit": "V",     "default": float(safety.vmin)},
                {"name": "vmax",    "type": "number", "unit": "V",     "default": float(safety.vmax)},
                {"name": "imax",    "type": "number", "unit": "A",     "default": float(safety.imax)},
                {"name": "pmax",    "type": "number", "unit": "W",     "default": float(safety.pmax)},
                {"name": "tmod_max","type": "number", "unit": "°C",    "default": float(safety.tmod_max)},
                {"name": "quiet_cycles","type": "integer",              "default": int(self.cfg.quiet_cycles), "help": "S4→S1 hold hysteresis"},
            ],
        }
        return {"s1": s1_desc, "s3": s3_desc, "s4": s4_desc, "psd": psd_schema, "safety": safety_schema}

    def get_config(self) -> Dict[str, Any]:
        """Return current config snapshot suitable for JSON serialization."""
        return {
            "names": {"s1": self.cfg.normal_name, "s3": self.cfg.global_name, "s4": self.cfg.hold_name},
            "s1": getattr(self.s1, "get_config", lambda: {})(),
            "s3": getattr(self.s3, "get_config", lambda: {})(),
            "s4": getattr(self.s4, "get_config", lambda: {})(),
            "psd": dict(self.cfg.psd),
            "safety": asdict(self.cfg.safety),
            "quiet_cycles": self.cfg.quiet_cycles,
        }

    def update_params(self, section: str, **kw: Any) -> None:
        """Apply live parameter updates to a section ('s1'|'s3'|'s4'|'psd'|'safety'|'controller')."""
        sec = section.lower()
        if sec == "s1":
            up = getattr(self.s1, "update_params", None)
            if callable(up):
                up(**kw)
        elif sec == "s3":
            up = getattr(self.s3, "update_params", None)
            if callable(up):
                up(**kw)
        elif sec == "s4":
            up = getattr(self.s4, "update_params", None)
            if callable(up):
                up(**kw)
        elif sec == "psd":
            # Update active PSD detector and persist in cfg
            for k, v in kw.items():
                self.cfg.psd[k] = v
                if hasattr(self.psd, k):
                    setattr(self.psd, k, v)
        elif sec == "safety":
            for k, v in kw.items():
                if hasattr(self.cfg.safety, k):
                    setattr(self.cfg.safety, k, v)
        elif sec == "controller":
            if "quiet_cycles" in kw:
                self.cfg.quiet_cycles = int(kw["quiet_cycles"])
        # else: ignore unknown section silently (non-breaking)
