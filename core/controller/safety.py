"""
Safety checks for MPPT controller

Simple limit checking to guard the power stage. Extend with hardware-specific
faults and plausibility checks as needed.

This module is intentionally dependency‑free and uses only the shared
`Measurement` type from the algorithms package.
"""
from dataclasses import dataclass
from typing import Optional

from ..mppt_algorithms.types import Measurement


@dataclass
class SafetyLimits:
    """Static safety limits for the MPPT system.

    Parameters
    vmin, vmax : float
        Allowed PV voltage range [V].
    imax : float
        Maximum PV current [A].
    pmax : float
        Maximum PV power [W].
    tmod_max : float
        Maximum module temperature [deg C] (if provided by measurements).
    dvdt_max : float | None
        Optional maximum allowed slew of PV voltage [V/s].
    didt_max : float | None
        Optional maximum allowed slew of PV current [A/s].
    """

    vmin: float = 0.0
    vmax: float = 100.0
    imax: float = 100.0
    pmax: float = 5000.0
    tmod_max: float = 85.0
    dvdt_max: Optional[float] = None
    didt_max: Optional[float] = None


# Fault codes (short strings for telemetry/logging)
OK = "OK"
UVP = "UVP"    # Under‑voltage protection
OVP = "OVP"    # Over‑voltage protection
OCP = "OCP"    # Over‑current protection
OPP = "OPP"    # Over‑power protection
OTP = "OTP"    # Over‑temperature protection
DVDT = "DVDT"  # Excessive |dV/dt|
DIDT = "DIDT"  # Excessive |dI/dt|


def _abs(x: float) -> float:
    return -x if x < 0.0 else x


def check_limits(m: Measurement, lim: SafetyLimits, last: Optional[Measurement] = None) -> str:
    """Validate a measurement against the provided safety limits.

    Returns a short fault code (e.g., "OVP", "OCP") or "OK" if all checks pass.

    Notes
    - dV/dt and dI/dt checks require either `m.dt` or a previous measurement
      with a valid time `t`. If neither is available, those checks are skipped.
    - This function is side‑effect free; the controller decides how to react.
    """
    v, i = float(m.v), float(m.i)
    p = v * i

    # Static bounds
    if v < lim.vmin:
        return UVP
    if v > lim.vmax:
        return OVP
    if i > lim.imax:
        return OCP
    if p > lim.pmax:
        return OPP
    if m.t_mod is not None and m.t_mod > lim.tmod_max:
        return OTP

    # Rate limits (optional)
    if last is not None and (lim.dvdt_max is not None or lim.didt_max is not None):
        # Determine dt (prefer explicit dt; else use absolute time difference)
        dt = m.dt
        if (dt is None or dt <= 0.0) and (m.t is not None and last.t is not None):
            dt = float(m.t) - float(last.t)
        if dt is not None and dt > 0.0:
            if lim.dvdt_max is not None:
                dvdt = _abs(v - float(last.v)) / dt
                if dvdt > lim.dvdt_max:
                    return DVDT
            if lim.didt_max is not None:
                didt = _abs(i - float(last.i)) / dt
                if didt > lim.didt_max:
                    return DIDT

    return OK


def safe_voltage(m: Measurement, lim: SafetyLimits) -> float:
    """Clamp the present voltage into the safe range [vmin, vmax]."""
    v = float(m.v)
    if v < lim.vmin:
        return lim.vmin
    if v > lim.vmax:
        return lim.vmax
    return v


__all__ = [
    "SafetyLimits",
    "check_limits",
    "safe_voltage",
    "OK", "UVP", "OVP", "OCP", "OPP", "OTP", "DVDT", "DIDT",
]