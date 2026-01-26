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

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Iterable, Generator, Tuple, Callable, Union, Sequence

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

    def set_conditions(self, irradiance: Union[float, Sequence[float]], temperature_c: float) -> None:
        self._temperature = float(temperature_c)

        # Per-string irradiance support (partial shading)
        if isinstance(irradiance, (list, tuple)):
            g_list = [float(x) for x in irradiance]
            self._irradiance = float(sum(g_list) / max(len(g_list), 1))

            strings = getattr(self.array, "string_list", None)
            if strings is None:
                raise AttributeError("Array has no string_list; cannot apply per-string irradiance shading.")

            if len(g_list) != len(strings):
                raise ValueError(f"Per-string irradiance length {len(g_list)} != n_strings {len(strings)}")

            for s, g in zip(strings, g_list):
                s.set_conditions(float(g), self._temperature)

            # Clear IV cache if present (important)
            if hasattr(self.array, "cached_iv"):
                self.array.cached_iv = None
            return

        # Scalar irradiance (broadcast)
        self._irradiance = float(irradiance)
        self.array.set_conditions(self._irradiance, self._temperature)

    def step(self, v: float) -> Tuple[float, float]:
        """
        Given a terminal voltage, return (v, i) as seen by the controller.
        """
        # Array API compatibility: different versions may expose different method names
        i_fn = None
        for _name in ("i_at_v", "solve_i_at_v", "current_at_v", "i_of_v", "i_from_v"):
            _cand = getattr(self.array, _name, None)
            if callable(_cand):
                i_fn = _cand
                break
        if i_fn is None:
            raise AttributeError(
                "Array object has no callable method to compute current at a voltage. "
                "Tried: i_at_v, solve_i_at_v, current_at_v, i_of_v, i_from_v"
            )
        i = i_fn(v)
        return float(v), float(i)

@dataclass
class LiveOverrides:
    """Live environment overrides that can be mutated while a simulation runs.

    If a field is not None, it takes precedence over the configured constant
    values and any env_profile-derived values.
    """
    irradiance: Optional[Union[float, Sequence[float]]] = None
    temperature_c: Optional[float] = None

# Simulation config / result types
@dataclass
class SimulationConfig:
    total_time: float = 1.0        # seconds
    dt: float = 1e-3               # control / sample period
    start_v: float = 20.0          # initial array voltage guess
    irradiance: Union[float, Sequence[float]] = 1000.0     # W/m^2
    temperature_c: float = 25.0    # deg C
    array_kwargs: Dict[str, Any] = None
    controller_cfg: Optional[HybridConfig] = None
    # GMPP reference (ground truth) computation
    gmpp_ref: bool = False
    gmpp_ref_period_s: float = 0.05   # compute reference every 50ms
    gmpp_ref_points: int = 121        # sweep resolution for reference

    # Optional: time-varying environment as list of (time_s, irradiance, temp_c)
    env_profile: Optional[List[Tuple[float, Union[float, Sequence[float]], float]]] = None
    # Optional: per-sample callback for UIs/loggers
    on_sample: Optional[Callable[[Dict[str, Any]], None]] = None
    # Optional: live overrides (shared object mutated by UI)
    overrides: Optional[LiveOverrides] = None

    def build_array(self) -> Array:
        kwargs = self.array_kwargs or {}
        return build_default_array(**kwargs)

    def build_controller(self) -> HybridMPPT:
        if self.controller_cfg is None:
            return HybridMPPT()
        return HybridMPPT(self.controller_cfg)

    def env_at(self, t: float) -> Tuple[Union[float, Sequence[float]], float]:
        """Return (irradiance, temperature_c) for time t.

        Priority (highest first):
          1) live overrides (if provided and the specific field is not None)
          2) env_profile stepwise values
          3) constant irradiance/temperature_c from config
        """

        # Base values from config
        latest_g = self.irradiance
        latest_t = self.temperature_c

        # Stepwise-hold profile values (if provided)
        if self.env_profile:
            for tt, g, tc in self.env_profile:
                if tt <= t:
                    latest_g = g
                    latest_t = tc
                else:
                    break

        # Live overrides win
        if self.overrides is not None:
            if self.overrides.irradiance is not None:
                latest_g = self.overrides.irradiance  # keep scalar OR sequence
            if self.overrides.temperature_c is not None:
                latest_t = float(self.overrides.temperature_c)

        # Return irradiance as-is (float or sequence), temp as float
        return latest_g, float(latest_t)

# Engine
class SimulationEngine:
    """
    Time-stepped MPPT simulation.

    Usage:
        cfg = SimulationConfig(total_time=0.5, dt=1e-3)
        eng = SimulationEngine(cfg)
        for sample in eng.run():
            print(sample)
    """

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.array = cfg.build_array()
        self.plant = PVPlant(self.array)
        self.ctrl = cfg.build_controller()
        self.on_sample = cfg.on_sample

        # environment
        self.plant.set_conditions(cfg.irradiance, cfg.temperature_c)
    
    def _get_v_bounds(self) -> Tuple[float, float]:
        # Prefer controller cfg bounds if present
        vmin = 0.0
        vmax = 100.0
        cfg = getattr(self.ctrl, "cfg", None)
        if cfg is not None:
            if hasattr(cfg, "vmin"):
                vmin = float(cfg.vmin)
            if hasattr(cfg, "vmax"):
                vmax = float(cfg.vmax)

        # If array exposes voc(), prefer that as an upper bound
        voc = getattr(self.array, "voc", None)
        try:
            if callable(voc):
                vmax = max(vmax, float(voc()))
            elif voc is not None:
                vmax = max(vmax, float(voc))
        except Exception:
            pass

        return float(vmin), float(vmax)

    def _gmpp_reference(self, i_fn, points: int) -> Tuple[float, float]:
        """Return (v_gmp, p_gmp) via a deterministic sweep using array current-at-voltage."""
        vmin, vmax = self._get_v_bounds()
        n = max(11, int(points))
        if vmax - vmin < 1e-9:
            return float(vmin), 0.0

        best_v = vmin
        best_p = -1e18
        for k in range(n):
            v = vmin + (vmax - vmin) * (k / (n - 1))
            i = float(i_fn(v))
            p = float(v * i)
            if p > best_p:
                best_p = p
                best_v = v
        return float(best_v), float(best_p)

    def run(self) -> Generator[Dict[str, Any], None, None]:
        """
        Run the simulation and yield JSON-friendly dicts per timestep.
        """
        t = 0.0
        v = self.cfg.start_v
        p_best = -1e18
        v_best = v
        last_ref = None
        next_ref_t = 0.0

        # resolve environment at t=0
        g0, t0 = self.cfg.env_at(t)
        self.plant.set_conditions(g0, t0)
        g0_vec = list(g0) if isinstance(g0, (list, tuple)) else None

        # Prime the controller with an INIT measurement
        # Array API compatibility (same logic as PVPlant.step)
        i_fn = None
        for _name in ("i_at_v", "solve_i_at_v", "current_at_v", "i_of_v", "i_from_v"):
            _cand = getattr(self.array, _name, None)
            if callable(_cand):
                i_fn = _cand
                break
        if i_fn is None:
            raise AttributeError(
                "Array object has no callable method to compute current at a voltage. "
                "Tried: i_at_v, solve_i_at_v, current_at_v, i_of_v, i_from_v"
            )
        i = float(i_fn(v))
        g_meas = float(sum(g0) / len(g0)) if isinstance(g0, (list, tuple)) else float(g0)
        m = Measurement(t=t, v=v, i=i, g=g_meas, t_mod=t0, dt=self.cfg.dt)
        a = self.ctrl.step(m)
        if m.p > p_best:
            p_best = float(m.p)
            v_best = float(m.v)

        gmpp = None
        if self.cfg.gmpp_ref:
            v_gmp, p_gmp = self._gmpp_reference(i_fn, self.cfg.gmpp_ref_points)
            eff = float(p_best / max(p_gmp, 1e-12))
            gmpp = {
                "v_gmp_ref": float(v_gmp),
                "p_gmp_ref": float(p_gmp),
                "v_best": float(v_best),
                "p_best": float(p_best),
                "eff_best": float(eff),
            }
            last_ref = gmpp
            next_ref_t = t + float(self.cfg.gmpp_ref_period_s)
        extra0 = {"gmpp": gmpp} if gmpp else {}
        if g0_vec is not None:
            extra0["g_strings"] = g0_vec
        rec = self._to_record(m, a, extra=extra0 if extra0 else None)
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

            # measure plant at that voltage
            g_now, t_now = self.cfg.env_at(t)
            g_vec = list(g_now) if isinstance(g_now, (list, tuple)) else None
            self.plant.set_conditions(g_now, t_now)
            v, i = self.plant.step(v_cmd)

            g_meas = float(sum(g_now) / len(g_now)) if isinstance(g_now, (list, tuple)) else float(g_now)
            m = Measurement(
                t=t,
                v=v,
                i=i,
                g=g_meas,
                t_mod=t_now,
                dt=self.cfg.dt,
            )
            if m.p > p_best:
                p_best = float(m.p)
                v_best = float(m.v)
            gmpp = None
            if self.cfg.gmpp_ref and t >= next_ref_t:
                v_gmp, p_gmp = self._gmpp_reference(i_fn, self.cfg.gmpp_ref_points)
                eff = float(p_best / max(p_gmp, 1e-12))
                gmpp = {
                    "v_gmp_ref": float(v_gmp),
                    "p_gmp_ref": float(p_gmp),
                    "v_best": float(v_best),
                    "p_best": float(p_best),
                    "eff_best": float(eff),
                }
                last_ref = gmpp
                next_ref_t = t + float(self.cfg.gmpp_ref_period_s)
            elif last_ref is not None:
                # Reuse last reference volt/power, but keep best-so-far and efficiency up to date
                gmpp = dict(last_ref)
                p_gmp_ref = float(gmpp.get("p_gmp_ref", float("nan")))
                if p_gmp_ref == p_gmp_ref:  # not NaN
                    gmpp["v_best"] = float(v_best)
                    gmpp["p_best"] = float(p_best)
                    gmpp["eff_best"] = float(p_best / max(p_gmp_ref, 1e-12))
                last_ref = gmpp
            # controller next step
            a = self.ctrl.step(m)

            # emit a JSON-friendly record
            extra = {"gmpp": gmpp} if gmpp else {}
            if g_vec is not None:
                extra["g_strings"] = g_vec
            rec = self._to_record(m, a, extra=extra if extra else None)
            if self.on_sample is not None:
                self.on_sample(rec)
            yield rec

    @staticmethod
    def _to_record(m: Measurement, a: Action, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        if extra:
            rec.update(extra)
        return rec

# CLI demo
def _demo() -> None:
    # stepwise irradiance drop at 0.05s and 0.12s to emulate clouds
    profile = [
        (0.0, 1000.0, 25.0),
        (0.05, 650.0, 25.0),
        (0.12, 400.0, 27.0),
    ]
    cfg = SimulationConfig(
        total_time=0.2,
        dt=1e-3,
        start_v=18.0,
        array_kwargs={"n_strings": 2, "substrings_per_string": 3, "cells_per_substring": 18},
        env_profile=profile,
        on_sample=lambda rec: print(rec),
    )
    eng = SimulationEngine(cfg)
    for _ in eng.run():
        pass


if __name__ == "__main__":
    _demo()