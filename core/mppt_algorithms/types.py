"""
Core types for MPPT algorithms

Minimal, dependency-free dataclasses shared by all algorithms and controllers.
Keep this file stable to avoid churn across the codebase.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Union, List

# JSON-friendly value type used by debug/meta payloads
JSONValue = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]


@dataclass
class Measurement:
    """Single control-cycle measurement from the PV/converter plant.

    Parameters
    t : float
        Absolute time in seconds (preferred). If unavailable, algorithms may
        fall back to `dt` to synthesize time.
    v : float
        Measured PV operating voltage [V].
    i : float
        Measured PV operating current [A].
    g : float, optional
        Plane-of-array irradiance [W/m^2] if available (used by detectors/models).
    t_mod : float, optional
        Module temperature [°C] if available.
    dt : float, optional
        Loop period in seconds for algorithms that need a timestep.
    meta : Dict[str, float]
        Free-form auxiliary signals (e.g., state_id, filtered signals, flags). JSON-friendly values only.
    """

    t: float
    v: float
    i: float
    g: Optional[float] = None
    t_mod: Optional[float] = None
    dt: Optional[float] = None
    meta: Dict[str, JSONValue] = field(default_factory=dict)

    @property
    def p(self) -> float:
        """Instantaneous electrical power [W] (convenience property)."""
        return self.v * self.i

    def to_dict(self) -> Dict[str, JSONValue]:
        """Shallow dict conversion suitable for JSON serialization."""
        return {
            "t": self.t,
            "v": self.v,
            "i": self.i,
            "g": self.g,
            "t_mod": self.t_mod,
            "dt": self.dt,
            "meta": self.meta,
        }


@dataclass
class Action:
    """Controller output for one control cycle.

    Prefer voltage-mode control (`v_ref`) when the converter supports it; use
    `duty_ref` for duty-based plants. Only set the field(s) you intend to drive.

    Parameters
    v_ref : float, optional
        Desired PV voltage reference [V].
    duty_ref : float, optional
        Desired duty cycle reference (0..1).
    debug : Dict[str, float]
        Arbitrary debug/telemetry entries to log with this action (JSON-friendly).
    """

    v_ref: Optional[float] = None
    duty_ref: Optional[float] = None
    debug: Dict[str, JSONValue] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, JSONValue]:
        """Shallow dict conversion suitable for JSON serialization."""
        return {
            "v_ref": self.v_ref,
            "duty_ref": self.duty_ref,
            "debug": self.debug,
        }


__all__ = ["Measurement", "Action", "JSONValue"]