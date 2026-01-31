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
import time
import random

# PV plant imports
from core.src.cell import Cell
from core.src.substring import Substring
from core.src.string import PVString
from core.src.array import Array

# Controller imports
from core.controller.hybrid_controller import HybridMPPT, HybridConfig
from core.controller.single_controller import SingleMPPT, SingleConfig
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
        # Resolve the array current-at-voltage function once (avoid per-step getattr scanning)
        self._i_fn = None
        for _name in ("i_at_v", "solve_i_at_v", "current_at_v", "i_of_v", "i_from_v"):
            _cand = getattr(self.array, _name, None)
            if callable(_cand):
                self._i_fn = _cand
                break
        if self._i_fn is None:
            raise AttributeError(
                "Array object has no callable method to compute current at a voltage. "
                "Tried: i_at_v, solve_i_at_v, current_at_v, i_of_v, i_from_v"
            )

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
        i = self._i_fn(v)
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
    # --- Internal: compiled env profile for fast env_at() ---
    _env_compiled: Optional[List[Tuple[float, Union[float, Sequence[float]], float]]] = None
    _env_idx: int = 0
    _env_last_t: float = -1e18
    def _compile_env_profile(self) -> None:
        """Compile env_profile into a sorted list of (t, g, t_mod) tuples for fast lookup."""
        prof = self.env_profile
        if not prof:
            self._env_compiled = None
            self._env_idx = 0
            self._env_last_t = -1e18
            return

        compiled: List[Tuple[float, Union[float, Sequence[float]], float]] = []
        for ev in prof:
            # Format A: legacy tuple/list (t, g, tc)
            if isinstance(ev, (tuple, list)):
                if len(ev) != 3:
                    continue
                tt, g, tc = ev
                try:
                    compiled.append((float(tt), g, float(tc)))
                except Exception:
                    continue

            # Format B: dict event
            elif isinstance(ev, dict):
                try:
                    tt = float(ev.get("t", 0.0))
                except Exception:
                    continue

                # per-string shading overrides scalar g if present
                g_val: Union[float, Sequence[float]]
                if "g_strings" in ev and ev.get("g_strings") is not None:
                    gs = ev.get("g_strings")
                    if isinstance(gs, (list, tuple)):
                        try:
                            g_val = [float(x) for x in gs]
                        except Exception:
                            continue
                    else:
                        continue
                elif "g" in ev and ev.get("g") is not None:
                    try:
                        g_val = float(ev.get("g"))
                    except Exception:
                        continue
                else:
                    # No irradiance info; skip
                    continue

                # temperature
                if "t_mod" in ev and ev.get("t_mod") is not None:
                    try:
                        t_val = float(ev.get("t_mod"))
                    except Exception:
                        continue
                else:
                    # If missing, keep the configured constant temperature
                    t_val = float(self.temperature_c)

                compiled.append((tt, g_val, t_val))

        compiled.sort(key=lambda x: x[0])
        self._env_compiled = compiled if compiled else None
        self._env_idx = 0
        self._env_last_t = -1e18
    total_time: float = 1.0        # seconds
    dt: float = 1e-3               # control / sample period
    start_v: float = 20.0          # initial array voltage guess
    irradiance: Union[float, Sequence[float]] = 1000.0     # W/m^2
    temperature_c: float = 25.0    # deg C
    array_kwargs: Dict[str, Any] = None
    controller_cfg: Optional[HybridConfig] = None

    # Controller selection
    #   - controller_mode = "hybrid" uses HybridMPPT (state machine)
    #   - controller_mode = "single" runs one registry algorithm for the full run
    controller_mode: str = "hybrid"
    algo_name: Optional[str] = None
    algo_kwargs: Dict[str, Any] = None

    # GMPP reference (ground truth) computation
    gmpp_ref: bool = False
    gmpp_ref_period_s: float = 0.05   # compute reference every 50ms
    gmpp_ref_points: int = 121        # sweep resolution for reference
    
    # --- Benchmarking / profiling ---
    perf_enabled: bool = True                    # include perf fields in output records
    perf_budget_us: Optional[float] = None       # if set, flag if ctrl step exceeds this

    # --- Measurement realism ---
    rng_seed: Optional[int] = 0                  # deterministic noise if desired

    noise_v_std: float = 0.0                     # volts (Gaussian)
    noise_i_std: float = 0.0                     # amps  (Gaussian)
    noise_g_std: float = 0.0                     # W/m^2 (Gaussian)

    adc_bits_v: Optional[int] = None             # e.g., 12 for 12-bit ADC
    adc_bits_i: Optional[int] = None
    adc_bits_g: Optional[int] = None

    # Full-scale ranges used for quantization (keep consistent with your sim bounds)
    v_full_scale: float = 100.0                  # volts
    i_full_scale: float = 20.0                   # amps
    g_full_scale: float = 1200.0                 # W/m^2

    # Optional: time-varying environment profile.
    # Supported formats:
    #   A) legacy tuples: List[(time_s, irradiance (float or per-string sequence), temp_c)]
    #   B) event dicts (benchmarks/scenarios.py): List[{"t":..., "g":..., "t_mod":..., "g_strings":...}]
    env_profile: Optional[List[Any]] = None
    # Optional: per-sample callback for UIs/loggers
    on_sample: Optional[Callable[[Dict[str, Any]], None]] = None
    # Optional: live overrides (shared object mutated by UI)
    overrides: Optional[LiveOverrides] = None

    def build_array(self) -> Array:
        kwargs = self.array_kwargs or {}
        return build_default_array(**kwargs)

    def build_controller(self):
        """Build the MPPT controller.

        Selection precedence:
          1) If controller_mode == "single" and algo_name is set, run a single registry
             algorithm for the entire run.
          2) Otherwise run the HybridMPPT controller. If controller_cfg is provided,
             it is used; else HybridMPPT defaults are used.

        This keeps legacy callers working (controller_cfg-only) while enabling the
        new controller selection API used by mppt_sim and the UI.
        """

        mode = (self.controller_mode or "hybrid").strip().lower()

        if mode == "single":
            if not self.algo_name:
                raise ValueError("SimulationConfig.controller_mode='single' requires algo_name")
            kwargs = self.algo_kwargs or {}
            return SingleMPPT(SingleConfig(algo_name=str(self.algo_name), algo_kwargs=kwargs))

        # Default: hybrid
        if self.controller_cfg is None:
            return HybridMPPT()
        return HybridMPPT(self.controller_cfg)

    def env_at(self, t: float) -> Tuple[Union[float, Sequence[float]], float]:
        """Return (irradiance, temperature_c) for time t.

        Priority (highest first):
          1) live overrides (if provided and the specific field is not None)
          2) env_profile stepwise-hold values
          3) constant irradiance/temperature_c from config

        Supports two env_profile formats:
          A) legacy tuples: (time_s, irradiance (float or per-string sequence), temp_c)
          B) event dicts: {"t":..., "g":..., "t_mod":..., "g_strings":...}
        """

        # Base values from config
        latest_g: Union[float, Sequence[float]] = self.irradiance
        latest_t: float = float(self.temperature_c)

        # Compile once lazily
        if self._env_compiled is None and self.env_profile:
            self._compile_env_profile()

        comp = self._env_compiled
        if comp:
            # If time goes backwards (new run / reset), restart cursor
            if t < self._env_last_t:
                self._env_idx = 0

            # Advance cursor while next event time is <= t
            idx = self._env_idx
            n = len(comp)
            while (idx + 1) < n and comp[idx + 1][0] <= t:
                idx += 1

            self._env_idx = idx
            self._env_last_t = float(t)

            # Apply the selected event if its time is <= t
            tt, g_ev, t_ev = comp[idx]
            if tt <= t:
                latest_g = g_ev
                latest_t = float(t_ev)

        # Live overrides win
        if self.overrides is not None:
            if self.overrides.irradiance is not None:
                latest_g = self.overrides.irradiance
            if self.overrides.temperature_c is not None:
                latest_t = float(self.overrides.temperature_c)

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
        # deterministic RNG for benchmark repeatability (noise/quantization)
        self._rng = random.Random(cfg.rng_seed) if cfg.rng_seed is not None else random.Random()

        # environment
        self.plant.set_conditions(cfg.irradiance, cfg.temperature_c)
        # Track last applied environment to avoid redundant set_conditions work
        self._last_env_key: Optional[Tuple[Any, float]] = None
    
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
    
    def _quantize_unsigned(self, x: float, bits: Optional[int], full_scale: float) -> float:
        """Uniform quantization for unsigned signals in [0, full_scale]."""
        if bits is None:
            return float(x)
        levels = (1 << int(bits)) - 1
        if levels <= 0 or full_scale <= 0:
            return float(x)

        x_clipped = max(0.0, min(float(x), float(full_scale)))
        step = float(full_scale) / float(levels)
        q = round(x_clipped / step) * step
        return float(q)

    def _quantize_signed(self, x: float, bits: Optional[int], full_scale: float) -> float:
        """Uniform quantization for signed signals in [-full_scale, +full_scale]."""
        if bits is None:
            return float(x)
        levels = (1 << int(bits)) - 1
        if levels <= 0 or full_scale <= 0:
            return float(x)

        fs = float(full_scale)
        x_clipped = max(-fs, min(float(x), fs))

        # Map [-fs, fs] -> [0, 2fs], quantize, then map back
        step = (2.0 * fs) / float(levels)
        q = round((x_clipped + fs) / step) * step - fs
        return float(q)

    def _apply_measurement_effects(self, v: float, i: float, g: float) -> Tuple[float, float, float]:
        """Apply optional measurement noise + ADC quantization."""
        cfg = self.cfg

        # Gaussian noise
        if cfg.noise_v_std > 0:
            v = float(v) + self._rng.gauss(0.0, float(cfg.noise_v_std))
        if cfg.noise_i_std > 0:
            i = float(i) + self._rng.gauss(0.0, float(cfg.noise_i_std))
        if cfg.noise_g_std > 0:
            g = float(g) + self._rng.gauss(0.0, float(cfg.noise_g_std))

        # Quantization
        v = self._quantize_unsigned(v, cfg.adc_bits_v, cfg.v_full_scale)  # PV voltage is non-negative
        i = self._quantize_signed(i, cfg.adc_bits_i, cfg.i_full_scale)    # CURRENT should be signed
        g = self._quantize_unsigned(g, cfg.adc_bits_g, cfg.g_full_scale)  # irradiance is non-negative

        return float(v), float(i), float(g)

    @staticmethod
    def _now_ns() -> int:
        return time.perf_counter_ns()
    
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
        k = 0  # step index (0 = initial sample)

        # Best-so-far should be tracked on TRUE plant power (pre noise/quantization)
        p_best_true = -1e18
        v_best_true = v
        
        p_best_meas = -1e18
        v_best_meas = v
        last_ref = None
        next_ref_t = 0.0
        last_ref_us = None

        # resolve environment at t=0
        g0, t0 = self.cfg.env_at(t)
        env_key0 = (tuple(g0) if isinstance(g0, (list, tuple)) else float(g0), float(t0))
        self._last_env_key = env_key0
        self.plant.set_conditions(g0, t0)
        g0_vec = list(g0) if isinstance(g0, (list, tuple)) else None

        # Prime the controller with an INIT measurement
        # Use PVPlant's resolved current-at-voltage function
        i_fn = self.plant._i_fn
        v_cmd = float(v)  # initial command equals start_v
        i = float(i_fn(v))
        g_meas = float(sum(g0) / len(g0)) if isinstance(g0, (list, tuple)) else float(g0)
        p_true = float(v * i)
        v_true = float(v)
        i_true = float(i)
        # p_true already computed above
        if p_true > p_best_true:
            p_best_true = p_true
            v_best_true = float(v)
        # apply optional sensor effects
        v_m, i_m, g_m = self._apply_measurement_effects(v, i, g_meas)
        m = Measurement(t=t, v=v_m, i=i_m, g=g_m, t_mod=t0, dt=self.cfg.dt)

        # time controller step
        ctrl_us = None
        over_budget = None
        if self.cfg.perf_enabled:
            t0_ns = self._now_ns()
            a = self.ctrl.step(m)
            t1_ns = self._now_ns()
            ctrl_us = (t1_ns - t0_ns) / 1000.0
            if self.cfg.perf_budget_us is not None:
                over_budget = bool(ctrl_us > float(self.cfg.perf_budget_us))
        else:
            a = self.ctrl.step(m)
        if m.p > p_best_meas:
            p_best_meas = float(m.p)
            v_best_meas = float(m.v)

        gmpp = None
        ref_us = None
        if self.cfg.gmpp_ref:
            if self.cfg.perf_enabled:
                r0 = self._now_ns()
                v_gmp, p_gmp = self._gmpp_reference(i_fn, self.cfg.gmpp_ref_points)
                r1 = self._now_ns()
                ref_us = (r1 - r0) / 1000.0
            else:
                v_gmp, p_gmp = self._gmpp_reference(i_fn, self.cfg.gmpp_ref_points)
            
            eff = float(p_best_true / max(p_gmp, 1e-12))
            gmpp = {
                "v_gmp_ref": float(v_gmp),
                "p_gmp_ref": float(p_gmp),
                "v_best": float(v_best_true),
                "p_best": float(p_best_true),
                "eff_best": float(eff),

                "v_best_meas": float(v_best_meas),
                "p_best_meas": float(p_best_meas),
                "eff_best_meas": float(p_best_meas / max(p_gmp, 1e-12)),
            }
            last_ref = gmpp
            last_ref_us = ref_us
            next_ref_t = t + float(self.cfg.gmpp_ref_period_s)
        extra0 = {"gmpp": gmpp} if gmpp else {}
        extra0["k"] = int(k)
        extra0["v_cmd"] = float(v_cmd)
        extra0["v_true"] = float(v_true)
        extra0["i_true"] = float(i_true)
        extra0["p_true"] = float(p_true)
        if self.cfg.perf_enabled:
            extra0["perf"] = {
                "ctrl_us": ctrl_us,
                "ref_us": ref_us,
                "budget_us": self.cfg.perf_budget_us,
                "over_budget": over_budget,
            }
        if g0_vec is not None:
            extra0["g_strings"] = g0_vec
        rec = self._to_record(m, a, extra=extra0 if extra0 else None)
        if self.on_sample is not None:
            self.on_sample(rec)
        yield rec

        steps = int(self.cfg.total_time / self.cfg.dt)
        for _ in range(steps):
            t += self.cfg.dt
            k += 1

            # apply controller action (voltage-mode)
            if a.v_ref is not None:
                v_cmd = float(a.v_ref)
            else:
                v_cmd = v

            # measure plant at that voltage
            g_now, t_now = self.cfg.env_at(t)
            g_vec = list(g_now) if isinstance(g_now, (list, tuple)) else None

            env_key = (tuple(g_now) if isinstance(g_now, (list, tuple)) else float(g_now), float(t_now))
            if env_key != self._last_env_key:
                self.plant.set_conditions(g_now, t_now)
                self._last_env_key = env_key
            v, i = self.plant.step(v_cmd)
            p_true = float(v * i)
            v_true = float(v)
            i_true = float(i)
            # p_true already computed above
            if p_true > p_best_true:
                p_best_true = p_true
                v_best_true = float(v)

            g_meas = float(sum(g_now) / len(g_now)) if isinstance(g_now, (list, tuple)) else float(g_now)

            # apply optional sensor effects
            v_m, i_m, g_m = self._apply_measurement_effects(v, i, g_meas)
            m = Measurement(
                t=t,
                v=v_m,
                i=i_m,
                g=g_m,
                t_mod=t_now,
                dt=self.cfg.dt,
            )
            if m.p > p_best_meas:
                p_best_meas = float(m.p)
                v_best_meas = float(m.v)
            gmpp = None
            ref_us = None
            if self.cfg.gmpp_ref and t >= next_ref_t:
                if self.cfg.perf_enabled:
                    r0 = self._now_ns()
                    v_gmp, p_gmp = self._gmpp_reference(i_fn, self.cfg.gmpp_ref_points)
                    r1 = self._now_ns()
                    ref_us = (r1 - r0) / 1000.0
                else:
                    v_gmp, p_gmp = self._gmpp_reference(i_fn, self.cfg.gmpp_ref_points)
                eff = float(p_best_true / max(p_gmp, 1e-12))
                gmpp = {
                    "v_gmp_ref": float(v_gmp),
                    "p_gmp_ref": float(p_gmp),
                    "v_best": float(v_best_true),
                    "p_best": float(p_best_true),
                    "eff_best": float(eff),
                    
                    "v_best_meas": float(v_best_meas),
                    "p_best_meas": float(p_best_meas),
                    "eff_best_meas": float(p_best_meas / max(p_gmp, 1e-12)),
                }
                last_ref = gmpp
                last_ref_us = ref_us
                next_ref_t = t + float(self.cfg.gmpp_ref_period_s)
            elif last_ref is not None:
                # Reuse last reference volt/power, but keep best-so-far and efficiency up to date
                gmpp = dict(last_ref)
                p_gmp_ref = float(gmpp.get("p_gmp_ref", float("nan")))
                if p_gmp_ref == p_gmp_ref:  # not NaN
                    gmpp["v_best"] = float(v_best_true)
                    gmpp["p_best"] = float(p_best_true)
                    gmpp["eff_best"] = float(p_best_true / max(p_gmp_ref, 1e-12))
                    gmpp["v_best_meas"] = float(v_best_meas)
                    gmpp["p_best_meas"] = float(p_best_meas)
                    gmpp["eff_best_meas"] = float(p_best_meas / max(p_gmp_ref, 1e-12))
                last_ref = gmpp
                ref_us = last_ref_us
            # controller next step
            ctrl_us = None
            over_budget = None
            if self.cfg.perf_enabled:
                c0 = self._now_ns()
                a = self.ctrl.step(m)
                c1 = self._now_ns()
                ctrl_us = (c1 - c0) / 1000.0
                if self.cfg.perf_budget_us is not None:
                    over_budget = bool(ctrl_us > float(self.cfg.perf_budget_us))
            else:
                a = self.ctrl.step(m)

            # emit a JSON-friendly record
            extra = {"gmpp": gmpp} if gmpp else {}
            extra["k"] = int(k)
            extra["v_cmd"] = float(v_cmd)
            extra["v_true"] = float(v_true)
            extra["i_true"] = float(i_true)
            extra["p_true"] = float(p_true)
            if self.cfg.perf_enabled:
                extra["perf"] = {
                    "ctrl_us": ctrl_us,
                    "ref_us": ref_us,
                    "budget_us": self.cfg.perf_budget_us,
                    "over_budget": over_budget,
                }
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
            for k, v in extra.items():
                if k in rec:
                    rec.setdefault("extra", {})[k] = v
                else:
                    rec[k] = v
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