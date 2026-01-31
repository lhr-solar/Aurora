

"""single_controller.py

Run one MPPT algorithm for the entire simulation.

Aurora historically used a *hybrid* controller (a small state machine) that
invokes different algorithms depending on conditions.

For benchmarking and for clear algorithm attribution, we also support running
any single registry algorithm "as-is" for the whole run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from core.mppt_algorithms.registry import build
from core.mppt_algorithms.types import Action, Measurement


@dataclass(frozen=True)
class SingleConfig:
    """Configuration for :class:`SingleMPPT`."""

    algo_name: str
    algo_kwargs: Dict[str, Any] = field(default_factory=dict)


class SingleMPPT:
    """A thin wrapper that runs one registry algorithm for the entire run."""

    def __init__(self, cfg: SingleConfig):
        self.cfg = cfg
        # Registry returns an object that implements reset() and step(Measurement)->Action.
        self.algo = build(cfg.algo_name, **(cfg.algo_kwargs or {}))

    def reset(self) -> None:
        self.algo.reset()

    def step(self, m: Measurement) -> Action:
        action = self.algo.step(m)

        # Ensure a minimal debug payload so UIs can distinguish SINGLE runs
        # without depending on HybridMPPT state machinery.
        try:
            if hasattr(action, "debug") and isinstance(action.debug, dict):
                action.debug.setdefault("state", "SINGLE")
                action.debug.setdefault("algo", self.cfg.algo_name)
            else:
                # Action may be a simple dict-like or missing debug entirely
                dbg = {"state": "SINGLE", "algo": self.cfg.algo_name}
                try:
                    action.debug = dbg  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            # Debug is best-effort only; never interfere with control
            pass

        return action