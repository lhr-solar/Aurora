"""
Simulation engine for Aurora.

This module glues together:
- the PV plant model (`core.src.*` -> Cell/Substrings/String/Array),
- the hybrid MPPT controller (`core.controller.hybrid_controller.HybridMPPT`),
- and a simple time-stepped loop that feeds measurements to the controller
  and applies its actions back to the plant.

The goal is to have one place that:
1. builds a default test array,
2. runs a control loop for N steps,
3. emits JSON-friendly dicts (so the desktop UI / frontend can plot them).

This is intentionally lightweight and dependency-free so it can run inside
bench scripts, Jupyter, or a desktop UI callback.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Iterable, Generator, Tuple, Callable

# PV plant imports
from core.src.cell import Cell
from core.src.substring import Substring
from core.src.string import PVString
from core.src.array import Array

# Controller imports
from core.controller.hybrid_controller import HybridMPPT, HybridConfig
from core.mppt_algorithms.types import Measurement, Action

# Plant construction helpers
def _make_default_cell() -> Cell:
    """
    Create a single PV cell with reasonable STC-like defaults.

    NOTE: tweak these to match LR Solar actual module characterization if available.
    """
    return Cell(
        isc_ref=5.84,
        voc_ref=0.621,
        diode_ideality=1.30,
        r_s=0.02,
        r_sh=200.0,
        voc_temp_coeff=-0.0023,
        isc_temp_coeff=0.00035,
        irradiance=1000.0,
        temperature_c=25.0,
        vmpp=0.5,
        impp=5.5,
        autofit=False,
    )

def build_default_array(
    n_strings: int = 2,
    substrings_per_string: int = 3,
    cells_per_substring: int = 18,
) -> Array:
    """
    Build a small but hierarchical PV array that exercises the whole stack.

    Layout:
        Array
          |- String 0
          │     |- Substring 0 (cells...)
          │     |- Substring 1
          │     |_ Substring 2
          |_ String 1
                |- ...
    """
    strings: List[PVString] = []
    for _ in range(n_strings):
        substrings: List[Substring] = []
        for _ in range(substrings_per_string):
            cells = [_make_default_cell() for __ in range(cells_per_substring)]
            substrings.append(Substring(cell_list=cells, bypass=None))
        strings.append(PVString(substrings=substrings))
    return Array(strings)

# Plant adapter
class PVPlant:
    """
    Simple adapter that exposes "apply controller action -> measure again" API.

    We assume the controller is voltage-mode: it gives us a v_ref, and we
    ask the array for current at that voltage.
    """

    def __init__(self, array: Array):
        self.array = array
        # Default environment
        self._irradiance = 1000.0  # W/m^2
        self._temperature = 25.0   # °C
        self.set_conditions(self._irradiance, self._temperature)

    def set_conditions(self, irradiance: float, temperature_c: float) -> None:
        self._irradiance = float(irradiance)
        self._temperature = float(temperature_c)
        # broadcast to array
        self.array.set_conditions(self._irradiance, self._temperature)

    def step(self, v: float) -> Tuple[float, float]:
        """
        Given a terminal voltage, return (v, i) as seen by the controller.
        """
        i = self.array.i_at_v(v)
        return float(v), float(i)

@dataclass
class EnvironmentState:
    """Shared environment knobs for irradiance/temperature/shading.

    This can be owned by the UI so that sliders / checkboxes update
    these fields, while the simulation engine's env_func reads them
    each step to mutate the Array.
    """
    global_g: float = 1000.0       # base irradiance (W/m^2)
    global_t: float = 25.0         # base module temperature (°C)
    partial_enabled: bool = False  # whether partial shading is on
    shade_factor: float = 0.4      # shaded region gets global_g * shade_factor

# Simulation config / result types
@dataclass
class SimulationConfig:
    total_time: float = 1.0        # seconds
    dt: float = 1e-3               # control / sample period
    start_v: float = 20.0          # initial array voltage guess
    irradiance: float = 1000.0     # W/m^2
    temperature_c: float = 25.0    # deg C
    array_kwargs: Dict[str, Any] = None
    controller_cfg: Optional[HybridConfig] = None

    # Optional: time-varying environment as list of (time_s, irradiance, temp_c)
    env_profile: Optional[List[Tuple[float, float, float]]] = None
    # Optional: per-sample callback for UIs/loggers
    on_sample: Optional[Callable[[Dict[str, Any]], None]] = None
    # Optional: fully flexible environment hook: env_func(array, t)
    env_func: Optional[Callable[[Array, float], None]] = None

    def build_array(self) -> Array:
        kwargs = self.array_kwargs or {}
        return build_default_array(**kwargs)

    def build_controller(self) -> HybridMPPT:
        if self.controller_cfg is None:
            return HybridMPPT()
        return HybridMPPT(self.controller_cfg)

    def env_at(self, t: float) -> Tuple[float, float]:
        """
        Return (irradiance, temperature_c) for time t.
        If no profile is provided, fall back to constant values.
        If profile is provided as a list of (t, G, T), we pick the last
        entry whose time <= t (stepwise hold).
        """
        if not self.env_profile:
            return self.irradiance, self.temperature_c
        latest_g = self.irradiance
        latest_t = self.temperature_c
        for tt, g, tc in self.env_profile:
            if tt <= t:
                latest_g = g
                latest_t = tc
            else:
                break
        return latest_g, latest_t

# Engine
class SimulationEngine:
    """
    Time-stepped MPPT simulation.

    Usage:
        cfg = SimulationConfig(total_time=0.5, dt=1e-3)
        eng = SimulationEngine(cfg)

        # Simple blocking loop (CLI / scripts):
        for sample in eng.run():
            print(sample)

        # Or step-wise (GUI-friendly):
        eng.start()
        while True:
            rec = eng.step()
            if rec is None:
                break
            print(rec)
    """

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.array = cfg.build_array()
        self.plant = PVPlant(self.array)
        self.ctrl = cfg.build_controller()
        self.on_sample = cfg.on_sample
        self._iter = None  # type: Optional[Generator[Dict[str, Any], None, None]]

        # environment
        # If a fully custom env_func or an env_profile is provided, we let
        # the run() loop drive conditions over time. Otherwise, broadcast
        # the simple constant (irradiance, temperature_c) once here.
        if self.cfg.env_func is None and not self.cfg.env_profile:
            self.plant.set_conditions(cfg.irradiance, cfg.temperature_c)

    def start(self) -> None:
        """Initialize internal state for step-wise simulation.

        This prepares an internal generator from :meth:`run` so that
        :meth:`step` can be called repeatedly from a GUI timer or other
        event loop without blocking.
        """
        self._iter = self.run()

    def step(self) -> Optional[Dict[str, Any]]:
        """Advance the simulation by one time step.

        Returns the record dict for this step, or ``None`` when the
        simulation has finished (i.e. the underlying generator is
        exhausted).
        """
        if self._iter is None:
            self.start()

        try:
            return next(self._iter)
        except StopIteration:
            return None

    def run(self) -> Generator[Dict[str, Any], None, None]:
        """
        Run the simulation and yield JSON-friendly dicts per timestep.
        """
        t = 0.0
        v = self.cfg.start_v

        # resolve environment at t=0
        if self.cfg.env_func is not None:
            # fully variable env: let the user mutate the array directly
            self.cfg.env_func(self.array, t)
            # For logging/telemetry, we still provide a scalar summary
            # using env_at (typically the baseline profile or constants).
            g0, t0 = self.cfg.env_at(t)
        else:
            g0, t0 = self.cfg.env_at(t)
            self.plant.set_conditions(g0, t0)

        # Prime the controller with an INIT measurement
        i = self.array.i_at_v(v)
        m = Measurement(t=t, v=v, i=i, g=g0, t_mod=t0, dt=self.cfg.dt)
        a = self.ctrl.step(m)
        rec = self._to_record(m, a)
        if self.on_sample is not None:
            self.on_sample(rec)
        yield rec

        steps = int(self.cfg.total_time / self.cfg.dt)
        for _ in range(steps):
            t += self.cfg.dt

            # apply controller action (voltage-mode)
            if a.v_ref is not None:
                v_cmd = float(a.v_ref)
            else:
                v_cmd = v

            # environment update
            if self.cfg.env_func is not None:
                # fully variable env: user mutates array based on time
                self.cfg.env_func(self.array, t)
                # Keep providing a scalar summary for logging/telemetry.
                g_now, t_now = self.cfg.env_at(t)
            else:
                g_now, t_now = self.cfg.env_at(t)
                self.plant.set_conditions(g_now, t_now)

            # measure plant at that voltage
            v, i = self.plant.step(v_cmd)

            m = Measurement(
                t=t,
                v=v,
                i=i,
                g=g_now,
                t_mod=t_now,
                dt=self.cfg.dt,
            )

            # controller next step
            a = self.ctrl.step(m)

            # emit a JSON-friendly record
            rec = self._to_record(m, a)
            if self.on_sample is not None:
                self.on_sample(rec)
            yield rec

    @staticmethod
    def _to_record(m: Measurement, a: Action) -> Dict[str, Any]:
        rec = {
            "t": m.t,
            "v": m.v,
            "i": m.i,
            "p": m.p,
            "g": m.g,
            "t_mod": m.t_mod,
            "dt": m.dt,
            "action": a.to_dict(),
        }
        return rec

#
# Example environment wiring for partial shading driven by EnvironmentState
def make_partial_shading_env(env_state: EnvironmentState):
    """Return an env_func(array, t) that uses a shared EnvironmentState.

    The returned callable matches the SimulationConfig.env_func signature
    and will be invoked by SimulationEngine.run() at each time step.

    Layout assumption (matching ``build_default_array``)::

        Array
          |- String 0
          │     |- Substring 0 (module 0)
          │     |- Substring 1 (module 1)
          │     |_ Substring 2 (module 2)
          |_ String 1
                |- Substring 0
                |- Substring 1
                |_ Substring 2
    """
    def env_func(array: Array, t: float) -> None:
        base_g = float(env_state.global_g)
        t_c = float(env_state.global_t)

        # If partial shading is enabled, shaded regions get reduced irradiance.
        shade_g = base_g * float(env_state.shade_factor)

        for s_idx, string in enumerate(array.strings):
            for sub_idx, substring in enumerate(string.substrings):
                if env_state.partial_enabled and s_idx == 0 and sub_idx == 0:
                    g_use = shade_g
                else:
                    g_use = base_g

                for cell in substring.cells:
                    # Prefer a Cell.set_conditions API if available
                    try:
                        if hasattr(cell, "set_conditions"):
                            cell.set_conditions(g_use, t_c)
                            continue
                    except TypeError:
                        # Fallback to direct attributes if signature mismatch
                        pass

                    if hasattr(cell, "irradiance"):
                        cell.irradiance = g_use
                    if hasattr(cell, "temperature_c"):
                        cell.temperature_c = t_c

    return env_func

# CLI demo
def _demo() -> None:
    env_state = EnvironmentState()
    cfg = SimulationConfig(
        total_time=0.2,
        dt=1e-3,
        start_v=18.0,
        array_kwargs={"n_strings": 2, "substrings_per_string": 3, "cells_per_substring": 18},
        env_func=make_partial_shading_env(env_state),
        on_sample=lambda rec: print(rec),
    )
    eng = SimulationEngine(cfg)
    for _ in eng.run():
        pass


if __name__ == "__main__":
    _demo()