"""
Base MPPT interface
- Defines the abstract contract that all MPPT algorithms must implement.

Design goals:
- Keep the API tiny and stable so algorithms remain swappable.
- Make algorithms pure w.r.t. hardware I/O: read Measurement → return Action.
- Allow the hybrid controller to introspect basic capabilities.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from .types import Measurement, Action


class MPPTAlgorithm(ABC):
    """Abstract base class for all MPPT algorithms.

    Implementations should:
    - Maintain any internal state needed for their logic.
    - Avoid direct hardware side-effects; only compute and return an Action.
    - Be robust to missing optional fields in :class:`Measurement`.

    Attributes
    name : str
        Short, stable identifier (e.g., "mepo", "ruca", "pso", "nl_esc").
    supports_global_search : bool
        True if the algorithm performs a multi-peak/global search (e.g., PSO).
    uses_dither : bool
        True if the algorithm injects a dither signal (e.g., NL-ESC).
    """

    name: str = "base"
    supports_global_search: bool = False
    uses_dither: bool = False

    # ---- Optional frontend/introspection helpers (safe defaults) ----
    def describe(self) -> Dict[str, Any]:
        """Return UI metadata for tunable parameters.

        Subclasses may override to enumerate tunables for auto‑rendered controls
        in the frontend. The default returns an empty spec so UIs can
        gracefully omit controls.
        """
        return {"key": self.name, "label": self.name.upper(), "params": []}

    def get_config(self) -> Dict[str, Any]:
        """Return the current tunable parameters (for snapshots/replay).

        Subclasses may override. Default is an empty mapping.
        """
        return {}

    def update_params(self, **kw: Any) -> None:
        """Apply parameter updates from the frontend.

        Subclasses may override. Default is a no‑op so callers can safely
        attempt updates even if an algorithm has nothing to tune.
        """
        return None

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state.

        Called when entering a new controller state (e.g., INIT → NORMAL) or
        when you want to re-initialize the algorithm between scenarios/tests.
        """
        ...

    @abstractmethod
    def step(self, m: Measurement) -> Action:
        """Compute the next reference from the current measurement.

        Parameters
        m : Measurement
            The latest filtered measurement (time `t`, voltage `v`, current `i`, etc.).

        Returns
        Action
            Desired reference(s) for the next control interval (typically `v_ref`).
        """
        ...


__all__ = ["MPPTAlgorithm"]