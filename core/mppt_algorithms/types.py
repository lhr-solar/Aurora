"""
Core types for MPPT algorithms

Minimal, dependency-free dataclasses shared by all algorithms and controllers.
Keep this file stable to avoid churn across the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


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
        Free-form auxiliary signals (e.g., state_id, filtered signals, flags).
    """

    t: float
    v: float
    i: float
    g: Optional[float] = None
    t_mod: Optional[float] = None
    dt: Optional[float] = None
    meta: Dict[str, float] = field(default_factory=dict)

    @property
    def p(self) -> float:
        """Instantaneous electrical power [W] (convenience property)."""
        return self.v * self.i


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
        Arbitrary debug/telemetry entries to log with this action.
    """

    v_ref: Optional[float] = None
    duty_ref: Optional[float] = None
    debug: Dict[str, float] = field(default_factory=dict)


__all__ = ["Measurement", "Action"]