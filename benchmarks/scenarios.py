

"""Benchmark scenario library for Aurora.

This module defines reusable environment profiles ("scenarios") for MPPT
benchmarking.

Design principles:
- Scenarios are pure data (no engine/controller imports)
- `env_profile` is passed directly into `SimulationConfig`
- Runner/UI adapt `Scenario` -> their own wrapper types

Environment profile schema:
Each scenario provides either:
- `env_profile = None`  → engine uses baseline irradiance/temperature
OR
- `env_profile = List[Dict]` with time-ordered events of the form:
    {
        "t": <float seconds>,
        "g": <float W/m^2>,            # optional
        "t_mod": <float °C>,           # optional
        "g_strings": <List[float]>,    # optional (partial shading)
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


# -----------------------------------------------------------------------------
# Event helpers
# -----------------------------------------------------------------------------


def event(
    t: float,
    *,
    g: Optional[float] = None,
    t_mod: Optional[float] = None,
    g_strings: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Create a single env_profile event."""
    e: Dict[str, Any] = {"t": float(t)}
    if g is not None:
        e["g"] = float(g)
    if t_mod is not None:
        e["t_mod"] = float(t_mod)
    if g_strings is not None:
        e["g_strings"] = [float(x) for x in g_strings]
    return e


def _sorted(profile: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(profile, key=lambda d: float(d.get("t", 0.0)))


# -----------------------------------------------------------------------------
# Scenario container
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    name: str
    env_profile: Any
    description: str = ""


# -----------------------------------------------------------------------------
# Scenario definitions
# -----------------------------------------------------------------------------


def steady() -> None:
    """Baseline: no environmental overrides."""
    return None


# --- Irradiance dynamics ------------------------------------------------------


def step_irradiance(
    *,
    t_step: float = 0.20,
    g0: float = 1000.0,
    g1: float = 450.0,
    t_total: float = 1.0,
) -> List[Dict[str, Any]]:
    return _sorted([
        event(0.0, g=g0),
        event(t_step, g=g1),
        event(t_total, g=g1),
    ])


def two_step_irradiance(
    *,
    t1: float = 0.20,
    t2: float = 0.60,
    g0: float = 1000.0,
    g1: float = 350.0,
    g2: float = 900.0,
    t_total: float = 1.0,
) -> List[Dict[str, Any]]:
    return _sorted([
        event(0.0, g=g0),
        event(t1, g=g1),
        event(t2, g=g2),
        event(t_total, g=g2),
    ])


def cloud_flicker(
    *,
    g_high: float = 1000.0,
    g_low: float = 250.0,
    period_s: float = 0.10,
    duty: float = 0.5,
    t_total: float = 1.0,
) -> List[Dict[str, Any]]:
    prof: List[Dict[str, Any]] = [event(0.0, g=g_high)]
    t = 0.0
    high = period_s * duty
    low = period_s - high
    toggle = True

    while t < t_total:
        t += high if toggle else low
        if t <= t_total:
            prof.append(event(t, g=(g_low if toggle else g_high)))
        toggle = not toggle

    prof.append(event(t_total, g=(g_high if not toggle else g_low)))
    return _sorted(prof)


# --- Temperature dynamics -----------------------------------------------------


def temp_step(
    *,
    t_step: float = 0.30,
    t0: float = 25.0,
    t1: float = 60.0,
    g: float = 1000.0,
    t_total: float = 1.0,
) -> List[Dict[str, Any]]:
    return _sorted([
        event(0.0, g=g, t_mod=t0),
        event(t_step, g=g, t_mod=t1),
        event(t_total, g=g, t_mod=t1),
    ])


# --- Startup / low-light ------------------------------------------------------


def low_light_startup(
    *,
    g_low: float = 80.0,
    g_high: float = 900.0,
    t_ramp_start: float = 0.10,
    t_ramp_end: float = 0.40,
    t_total: float = 1.0,
    steps: int = 10,
) -> List[Dict[str, Any]]:
    prof: List[Dict[str, Any]] = [event(0.0, g=g_low)]
    steps = max(2, int(steps))

    for k in range(steps + 1):
        a = k / float(steps)
        t = t_ramp_start + a * (t_ramp_end - t_ramp_start)
        g = g_low + a * (g_high - g_low)
        prof.append(event(t, g=g))

    prof.append(event(t_total, g=g_high))
    return _sorted(prof)


# --- Partial shading ----------------------------------------------------------


def partial_shading_step(
    *,
    t_step: float = 0.35,
    g_uniform: float = 1000.0,
    g_strings_shaded: Sequence[float] = (1000.0, 350.0, 350.0),
    t_total: float = 1.0,
) -> List[Dict[str, Any]]:
    return _sorted([
        event(0.0, g=g_uniform),
        event(t_step, g_strings=g_strings_shaded),
        event(t_total, g_strings=g_strings_shaded),
    ])


def partial_shading_toggle(
    *,
    t1: float = 0.25,
    t2: float = 0.55,
    g_strings_a: Sequence[float] = (1000.0, 500.0, 500.0),
    g_strings_b: Sequence[float] = (1000.0, 250.0, 250.0),
    t_total: float = 1.0,
) -> List[Dict[str, Any]]:
    return _sorted([
        event(0.0, g_strings=g_strings_a),
        event(t1, g_strings=g_strings_b),
        event(t2, g_strings=g_strings_a),
        event(t_total, g_strings=g_strings_a),
    ])


# -----------------------------------------------------------------------------
# Scenario catalog (public API)
# -----------------------------------------------------------------------------


def default_scenarios() -> List[Scenario]:
    """Canonical scenario set for MPPT benchmarking."""
    return [
        Scenario(
            name="steady",
            env_profile=steady(),
            description="Baseline steady-state conditions.",
        ),
        Scenario(
            name="step_g_down",
            env_profile=step_irradiance(),
            description="Single irradiance step down.",
        ),
        Scenario(
            name="two_step_g",
            env_profile=two_step_irradiance(),
            description="Irradiance down then up.",
        ),
        Scenario(
            name="cloud_flicker",
            env_profile=cloud_flicker(),
            description="Fast cloud-induced irradiance flicker.",
        ),
        Scenario(
            name="temp_step",
            env_profile=temp_step(),
            description="Module temperature step at fixed irradiance.",
        ),
        Scenario(
            name="low_light_startup",
            env_profile=low_light_startup(),
            description="Low-light startup then ramp to full sun.",
        ),
        Scenario(
            name="partial_shading_step",
            env_profile=partial_shading_step(),
            description="Uniform irradiance → partial shading.",
        ),
        Scenario(
            name="partial_shading_toggle",
            env_profile=partial_shading_toggle(),
            description="Toggle between partial shading patterns.",
        ),
    ]


def list_scenarios() -> List[str]:
    return [s.name for s in default_scenarios()]


def get_scenario(name: str) -> Scenario:
    for s in default_scenarios():
        if s.name == name:
            return s
    raise KeyError(f"Unknown scenario '{name}'. Available: {list_scenarios()}")