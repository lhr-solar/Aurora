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
            return self._with_state(a, State.NORMAL)

        if self.state == State.GLOBAL_SEARCH:
            a = self.s3.step(m)
            # Heuristic: PSO implementation sets debug["phase"] = 2 when locking to gbest
            phase = int(a.debug.get("phase", 1)) if a.debug else 1
            if phase == 2:
                self.events.append({"t": m.t, "type": "state_change", "from": State.GLOBAL_SEARCH.value, "to": State.LOCK_HOLD.value})
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
                self.events.append({"t": m.t, "type": "state_change", "from": State.LOCK_HOLD.value, "to": State.NORMAL.value})
                self.state = State.NORMAL
                self.s1.reset(); self._quiet = 0
            # Re-check PSC in case conditions changed again
            if self.psd.update_and_check(m):
                self.events.append({"t": m.t, "type": "psd_trigger"})
                self.events.append({"t": m.t, "type": "state_change", "from": State.LOCK_HOLD.value, "to": State.GLOBAL_SEARCH.value})
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

    # ---- Frontend helpers (optional) ----
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
