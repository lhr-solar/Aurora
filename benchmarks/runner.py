"""Benchmark runner for Aurora MPPT algorithms.

This module is intentionally *thin* on assumptions about your project structure.
It provides:
  - A small benchmark harness (Algorithm × Scenario × Budget)
  - Collection of per-tick records from `SimulationEngine.run()`
  - Basic metrics extraction from the emitted `gmpp` and `perf` fields
  - JSONL output suitable for later aggregation/plotting

Expected engine output fields (added in your engine benchmarking upgrade):
  rec["gmpp"]["eff_step"]          # true per-step efficiency (p_true / p_gmp_ref)
  rec["gmpp"]["eff_step_meas"]     # measured per-step efficiency (p / p_gmp_ref)
  # legacy keys eff_best/eff_best_meas may also exist
  rec["perf"]["ctrl_us"]           # controller step time (microseconds)
  rec["perf"]["ref_us"]            # GMPP reference sweep time (microseconds)
  rec["perf"]["over_budget"]        # bool

If your engine uses different names, adjust `extract_*` helpers below.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Callable

# Session helpers
from benchmarks.session import ensure_session_dir, make_suite_payload, compute_suite_signature

from core.controller.hybrid_controller import HybridConfig
from core.mppt_algorithms import registry as mppt_registry


# --- Imports from Aurora ------------------------------------------------------

# NOTE: keep imports local-friendly. If your project uses different package paths,
# change these two imports (everything else can remain intact).
try:
    from simulators.engine import SimulationEngine, SimulationConfig  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Could not import SimulationEngine/SimulationConfig. "
        "Update imports in benchmarks/runner.py to match your repo structure."
    ) from e


# --- Spec types ---------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioSpec:
    """A named environment profile to run."""

    name: str
    env_profile: Any  # passthrough to SimulationConfig.env_profile
    description: str = ""


@dataclass(frozen=True)
class BudgetSpec:
    """Resource + sensing constraints."""

    name: str
    dt: float

    # Compute budget (microseconds): flags overruns in engine perf output
    perf_budget_us: Optional[float] = None

    # Measurement realism (optional)
    rng_seed: Optional[int] = 0
    noise_v_std: float = 0.0
    noise_i_std: float = 0.0
    noise_g_std: float = 0.0
    adc_bits_v: Optional[int] = None
    adc_bits_i: Optional[int] = None
    adc_bits_g: Optional[int] = None


@dataclass(frozen=True)
class AlgorithmSpec:
    """A named MPPT controller configuration.

    Aurora supports two controller modes:
      - hybrid: HybridMPPT state machine (uses HybridConfig)
      - single: run exactly one registry algorithm for the full run

    For backward compatibility, `controller_cfg` can still be provided; if
    controller_mode is not specified we treat it as a hybrid config.
    """

    name: str

    # New controller selection API (preferred)
    controller_mode: str = "hybrid"          # "hybrid" or "single"
    algo_name: Optional[str] = None           # required when controller_mode == "single"
    algo_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Legacy hybrid config (still supported)
    controller_cfg: Any = None

    description: str = ""


@dataclass
class RunSummary:
    """Scalar results from one Algorithm × Scenario × Budget run."""

    algo: str
    scenario: str
    budget: str

    # MPPT quality
    eff_true_final: Optional[float] = None
    eff_true_median: Optional[float] = None
    eff_meas_final: Optional[float] = None
    eff_meas_median: Optional[float] = None

    # Perf
    ctrl_us_p50: Optional[float] = None
    ctrl_us_p95: Optional[float] = None
    ctrl_us_p99: Optional[float] = None
    ref_us_p50: Optional[float] = None
    ref_us_p95: Optional[float] = None
    budget_violations: int = 0

    # Run info
    n_ticks: int = 0
    wall_s: float = 0.0

    # Optional arbitrary extras
    extra: Dict[str, Any] = field(default_factory=dict)


# --- Helpers ------------------------------------------------------------------


def percentile(xs: Sequence[float], q: float) -> Optional[float]:
    """Simple percentile (q in [0,1]) with linear interpolation."""
    if not xs:
        return None
    if q <= 0:
        return float(min(xs))
    if q >= 1:
        return float(max(xs))
    s = sorted(float(x) for x in xs)
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def extract_gmpp_eff(rec: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Return (eff_true, eff_meas) from a record.

    Supports both legacy keys (eff_best/eff_best_meas) and the newer per-step keys
    (eff_step/eff_step_meas).
    """
    gmpp = rec.get("gmpp") or {}

    # Prefer legacy names if present, otherwise fall back to per-step names
    eff_true = gmpp.get("eff_best")
    if eff_true is None:
        eff_true = gmpp.get("eff_step")

    eff_meas = gmpp.get("eff_best_meas")
    if eff_meas is None:
        eff_meas = gmpp.get("eff_step_meas")

    try:
        eff_true = None if eff_true is None else float(eff_true)
    except Exception:
        eff_true = None
    try:
        eff_meas = None if eff_meas is None else float(eff_meas)
    except Exception:
        eff_meas = None

    return eff_true, eff_meas


def extract_perf(rec: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], bool]:
    """Return (ctrl_us, ref_us, over_budget) from a record."""
    perf = rec.get("perf") or {}
    ctrl = perf.get("ctrl_us")
    ref = perf.get("ref_us")
    over = bool(perf.get("over_budget", False))
    try:
        ctrl = None if ctrl is None else float(ctrl)
    except Exception:
        ctrl = None
    try:
        ref = None if ref is None else float(ref)
    except Exception:
        ref = None
    return ctrl, ref, over


# --- Per-string irradiance normalization -------------------------------------

def _fit_g_strings(g: Sequence[float], n_strings: int) -> List[float]:
    """Fit a per-string irradiance sequence to exactly n_strings.

    Rules:
      - if len == n_strings: unchanged
      - if len == 1: repeat the single value
      - if len > n_strings: trim
      - if len < n_strings: pad with last value
    """
    g_list = [float(x) for x in g]
    if n_strings <= 0:
        return []
    if len(g_list) == n_strings:
        return g_list
    if len(g_list) == 1:
        return [g_list[0]] * n_strings
    if len(g_list) > n_strings:
        return g_list[:n_strings]
    # len(g_list) < n_strings: pad with last value
    return g_list + [g_list[-1]] * (n_strings - len(g_list))


def normalize_env_profile_for_strings(env_profile: Any, n_strings: int) -> Any:
    """Normalize any per-string irradiance lists in env_profile to match n_strings.

    Supports:
      - event dict list: [{"t":..., "g":..., "t_mod":..., "g_strings":[...]}]
      - legacy tuple list: [(t, irradiance (float or seq), temp)]
    """
    if env_profile is None:
        return None

    # Format B: event dicts
    if isinstance(env_profile, list) and env_profile and isinstance(env_profile[0], dict):
        out = []
        for ev in env_profile:
            ev2 = dict(ev)
            gs = ev2.get("g_strings", None)
            if gs is not None:
                ev2["g_strings"] = _fit_g_strings(gs, n_strings)
            out.append(ev2)
        return out

    # Format A: legacy tuples List[(t, irradiance (float or seq), temp)]
    if isinstance(env_profile, list) and env_profile and isinstance(env_profile[0], (list, tuple)):
        out = []
        for item in env_profile:
            t, irr, temp = item
            if isinstance(irr, (list, tuple)):
                irr = _fit_g_strings(irr, n_strings)
            out.append((t, irr, temp))
        return out

    return env_profile


# --- Formatting and streaming helpers -----------------------------------------

def _fmt(x: Any, nd: int = 4) -> str:
    try:
        if x is None:
            return "None"
        xf = float(x)
        if xf != xf:  # NaN
            return "nan"
        return f"{xf:.{nd}f}"
    except Exception:
        return str(x)


def format_bench_line(rec: Dict[str, Any], *, tick: int) -> str:
    gmpp = rec.get("gmpp") or {}
    perf = rec.get("perf") or {}
    action = rec.get("action") or {}
    dbg = {}
    if isinstance(action, dict):
        dbg = action.get("debug") or {}
    state = None
    reason = None
    if isinstance(dbg, dict):
        state = dbg.get("state")
        # Some controllers don't emit an explicit "reason"; fall back to other useful debug fields.
        reason = dbg.get("reason")
        if reason is None:
            if bool(dbg.get("cold_start", False)):
                reason = "cold_start"
            else:
                branch = dbg.get("branch")
                if branch not in (None, "", "NA"):
                    reason = branch
                else:
                    algo = dbg.get("algo")
                    if algo not in (None, ""):
                        reason = f"algo={algo}"

    v_cmd = rec.get("v_cmd")
    if v_cmd is None:
        v_cmd = (action.get("v_ref") if isinstance(action, dict) else None)

    msg = (
        "[bench] "
        f"tick={tick} "
        f"t={_fmt(rec.get('t'), 4)} dt={_fmt(rec.get('dt'), 6)} | "
        f"g={_fmt(rec.get('g'), 1)} t_mod={_fmt(rec.get('t_mod'), 2)} | "
        f"v_cmd={_fmt(v_cmd, 4)} | "
        f"meas(v,i,p)=({_fmt(rec.get('v'), 4)},{_fmt(rec.get('i'), 4)},{_fmt(rec.get('p'), 4)}) | "
        f"eff={_fmt(gmpp.get('eff_best', gmpp.get('eff_step')), 4)} | "
        f"ctrl_us={_fmt(perf.get('ctrl_us'), 1)} over={bool(perf.get('over_budget', False))}"
    )
    if state is not None or reason is not None:
        msg += f" | state={state} reason={reason}"
    return msg


def summarize_records(
    algo: AlgorithmSpec,
    scenario: ScenarioSpec,
    budget: BudgetSpec,
    records: Sequence[Dict[str, Any]],
    wall_s: float,
) -> RunSummary:
    eff_true_series: List[float] = []
    eff_meas_series: List[float] = []
    ctrl_us: List[float] = []
    ref_us: List[float] = []
    budget_viol = 0

    for rec in records:
        et, em = extract_gmpp_eff(rec)
        if et is not None:
            eff_true_series.append(et)
        if em is not None:
            eff_meas_series.append(em)
        c, r, over = extract_perf(rec)
        if c is not None:
            ctrl_us.append(c)
        if r is not None:
            ref_us.append(r)
        if over:
            budget_viol += 1

    summary = RunSummary(
        algo=algo.name,
        scenario=scenario.name,
        budget=budget.name,
        n_ticks=len(records),
        wall_s=float(wall_s),
        budget_violations=int(budget_viol),
    )

    # Final-eff: last available value (not necessarily last tick if gmpp is absent early)
    if eff_true_series:
        summary.eff_true_final = float(eff_true_series[-1])
        summary.eff_true_median = float(median(eff_true_series))
    if eff_meas_series:
        summary.eff_meas_final = float(eff_meas_series[-1])
        summary.eff_meas_median = float(median(eff_meas_series))

    if ctrl_us:
        summary.ctrl_us_p50 = float(percentile(ctrl_us, 0.50) or 0.0)
        summary.ctrl_us_p95 = float(percentile(ctrl_us, 0.95) or 0.0)
        summary.ctrl_us_p99 = float(percentile(ctrl_us, 0.99) or 0.0)
    if ref_us:
        summary.ref_us_p50 = float(percentile(ref_us, 0.50) or 0.0)
        summary.ref_us_p95 = float(percentile(ref_us, 0.95) or 0.0)

    # Store budget/scenario details for downstream reporting
    summary.extra["scenario_description"] = scenario.description
    summary.extra["budget"] = asdict(budget)
    return summary


def run_once(
    algo: AlgorithmSpec,
    scenario: ScenarioSpec,
    budget: BudgetSpec,
    *,
    total_time: float,
    gmpp_ref: bool,
    gmpp_ref_period_s: float,
    gmpp_ref_points: int,
    perf_enabled: bool = True,
    log: Optional[Callable[[str], None]] = None,
    log_every_s: float = 0.25,
    log_first_n: int = 5,
    log_last_n: int = 5,
    keep_records: bool = True,
    records_path: Optional[Path] = None,
    cancel: Optional[Callable[[], bool]] = None,
) -> Tuple[List[Dict[str, Any]], RunSummary]:
    """Run one simulation and return (records, summary)."""

    cfg = SimulationConfig(
        total_time=float(total_time),
        dt=float(budget.dt),
        # Controller selection
        controller_mode=str(getattr(algo, "controller_mode", "hybrid")),
        algo_name=getattr(algo, "algo_name", None),
        algo_kwargs=getattr(algo, "algo_kwargs", {}) or {},
        controller_cfg=(algo.controller_cfg if str(getattr(algo, "controller_mode", "hybrid")).strip().lower() != "single" else None),
        env_profile=scenario.env_profile,
        gmpp_ref=bool(gmpp_ref),
        gmpp_ref_period_s=float(gmpp_ref_period_s),
        gmpp_ref_points=int(gmpp_ref_points),
        # Perf + realism knobs
        perf_enabled=bool(perf_enabled),
        perf_budget_us=budget.perf_budget_us,
        rng_seed=budget.rng_seed,
        noise_v_std=float(budget.noise_v_std),
        noise_i_std=float(budget.noise_i_std),
        noise_g_std=float(budget.noise_g_std),
        adc_bits_v=budget.adc_bits_v,
        adc_bits_i=budget.adc_bits_i,
        adc_bits_g=budget.adc_bits_g,
    )

    # Robust fix: adapt any per-string irradiance lists to the configured array's n_strings
    try:
        n_strings = None

        # Preferred: SimulationConfig can build an array without running the simulation
        build_array = getattr(cfg, "build_array", None)
        if callable(build_array):
            arr = build_array()
            n_strings = getattr(arr, "n_strings", None)
            if n_strings is None:
                n_strings = len(getattr(arr, "string_list", None) or [])

        # Fallback: some configs expose n_strings directly
        if n_strings is None:
            n_strings = getattr(cfg, "n_strings", None)

        if n_strings is not None and cfg.env_profile is not None:
            cfg.env_profile = normalize_env_profile_for_strings(cfg.env_profile, int(n_strings))
            # If the config caches a compiled env profile, force a rebuild
            if hasattr(cfg, "_env_compiled"):
                setattr(cfg, "_env_compiled", None)
    except Exception:
        # Never fail benchmarks due to normalization
        pass

    t0 = time.perf_counter()
    eng = SimulationEngine(cfg)

    records: List[Dict[str, Any]] = []
    eff_true_series: List[float] = []
    eff_meas_series: List[float] = []
    ctrl_us: List[float] = []
    ref_us: List[float] = []
    budget_viol = 0

    tick = 0
    last_log_wall = time.monotonic()

    f = None
    if records_path is not None:
        records_path.parent.mkdir(parents=True, exist_ok=True)
        f = records_path.open("w", encoding="utf-8")

    try:
        for rec in eng.run():
            if cancel is not None and cancel():
                raise RuntimeError("Cancelled")
            # Optionally retain full per-tick records
            if keep_records:
                records.append(rec)

            # Optionally stream-save records to JSONL
            if f is not None:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Online accumulation for summary (so keep_records can be False)
            et, em = extract_gmpp_eff(rec)
            if et is not None:
                eff_true_series.append(et)
            if em is not None:
                eff_meas_series.append(em)
            c, r, over = extract_perf(rec)
            if c is not None:
                ctrl_us.append(c)
            if r is not None:
                ref_us.append(r)
            if over:
                budget_viol += 1

            tick += 1

            # Streaming terminal logs (throttled)
            if log is not None:
                now = time.monotonic()
                if tick <= log_first_n:
                    log(format_bench_line(rec, tick=tick))
                    last_log_wall = now
                elif log_every_s > 0 and (now - last_log_wall) >= log_every_s:
                    log(format_bench_line(rec, tick=tick))
                    last_log_wall = now
    finally:
        if f is not None:
            f.close()

    wall_s = time.perf_counter() - t0

    # Build summary from online series (works whether or not we kept records)
    summary = RunSummary(
        algo=algo.name,
        scenario=scenario.name,
        budget=budget.name,
        n_ticks=int(tick),
        wall_s=float(wall_s),
        budget_violations=int(budget_viol),
    )

    if eff_true_series:
        summary.eff_true_final = float(eff_true_series[-1])
        summary.eff_true_median = float(median(eff_true_series))
    if eff_meas_series:
        summary.eff_meas_final = float(eff_meas_series[-1])
        summary.eff_meas_median = float(median(eff_meas_series))

    if ctrl_us:
        summary.ctrl_us_p50 = float(percentile(ctrl_us, 0.50) or 0.0)
        summary.ctrl_us_p95 = float(percentile(ctrl_us, 0.95) or 0.0)
        summary.ctrl_us_p99 = float(percentile(ctrl_us, 0.99) or 0.0)
    if ref_us:
        summary.ref_us_p50 = float(percentile(ref_us, 0.50) or 0.0)
        summary.ref_us_p95 = float(percentile(ref_us, 0.95) or 0.0)

    summary.extra["scenario_description"] = scenario.description
    summary.extra["budget"] = asdict(budget)

    # If requested, log the last N ticks (using retained records if available,
    # otherwise reload from JSONL if we saved)
    if log is not None and log_last_n > 0:
        tail: List[Dict[str, Any]] = []
        if keep_records and records:
            tail = records[-min(log_last_n, len(records)) :]
        elif records_path is not None and records_path.exists():
            try:
                # Load only the last N lines
                with records_path.open("r", encoding="utf-8") as rf:
                    lines = rf.readlines()
                for line in lines[-min(log_last_n, len(lines)) :]:
                    tail.append(json.loads(line))
            except Exception:
                tail = []
        if tail:
            log(f"[bench] ... last {len(tail)} ticks:")
            start_tick = int(tick) - int(len(tail)) + 1
            for j, rec in enumerate(tail):
                true_tick = start_tick + j
                log(format_bench_line(rec, tick=true_tick))

    # Attach richer metrics (energy ratio, settle time, ripple) if available.
    # Requires full records; if keep_records=False but records were saved, reload them.
    try:
        from benchmarks.metrics import compute_metrics  # type: ignore

        recs_for_metrics: Optional[List[Dict[str, Any]]] = None
        if keep_records:
            recs_for_metrics = records
        elif records_path is not None and records_path.exists():
            recs_for_metrics = []
            with records_path.open("r", encoding="utf-8") as rf:
                for line in rf:
                    recs_for_metrics.append(json.loads(line))

        if recs_for_metrics is None:
            raise RuntimeError("metrics require records; set keep_records=True or save_records=True")

        m = compute_metrics(recs_for_metrics)
        m_dict = asdict(m)
        summary.extra["metrics"] = m_dict

        for k in (
            "energy_true_ratio",
            "energy_meas_ratio",
            "settle_time_s_true",
            "settle_time_s_meas",
            "t_disturb",
            "settle_time_s_true_post",
            "settle_time_s_meas_post",
            "recovery_settle_s_true",
            "recovery_settle_s_meas",
            "ripple_rms_true",
            "ripple_rms_meas",
            "tracking_error_area_true",
            "tracking_error_area_meas",
            "score",
        ):
            if k in m_dict:
                summary.extra[k] = m_dict[k]
    except Exception as e:
        summary.extra["metrics_error"] = str(e)

        # Fallback metrics directly from records (robust to naming differences)
        try:
            # Acquire records for metrics (same logic as above)
            recs_for_metrics: Optional[List[Dict[str, Any]]] = None
            if keep_records:
                recs_for_metrics = records
            elif records_path is not None and records_path.exists():
                recs_for_metrics = []
                with records_path.open("r", encoding="utf-8") as rf:
                    for line in rf:
                        recs_for_metrics.append(json.loads(line))

            if not recs_for_metrics:
                raise RuntimeError("no records available for fallback metrics")

            # Energy ratios vs GMPP reference power
            num_true = 0.0
            num_meas = 0.0
            den_ref = 0.0

            # Identify a change point for 'settle' measurement (first time g_strings changes)
            t_start = float(recs_for_metrics[0].get("t", 0.0) or 0.0)
            first_gs = recs_for_metrics[0].get("g_strings")
            if isinstance(first_gs, list):
                for r in recs_for_metrics[1:]:
                    gs = r.get("g_strings")
                    if gs is not None and gs != first_gs:
                        t_start = float(r.get("t", t_start) or t_start)
                        break

            # Helper to pull eff values with legacy/new naming
            def _eff_pair(r: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
                g = r.get("gmpp") or {}
                et = g.get("eff_best")
                if et is None:
                    et = g.get("eff_step")
                em = g.get("eff_best_meas")
                if em is None:
                    em = g.get("eff_step_meas")
                try:
                    et = None if et is None else float(et)
                except Exception:
                    et = None
                try:
                    em = None if em is None else float(em)
                except Exception:
                    em = None
                return et, em

            eff_thresh = 0.98
            settle_true = None
            settle_meas = None

            # Build power series for ripple calc
            p_true_series: List[Tuple[float, float]] = []
            p_meas_series: List[Tuple[float, float]] = []

            for r in recs_for_metrics:
                dt = float(r.get("dt", 0.0) or 0.0)
                gmpp = r.get("gmpp") or {}
                p_ref = gmpp.get("p_gmp_ref")
                try:
                    p_ref_f = 0.0 if p_ref is None else float(p_ref)
                except Exception:
                    p_ref_f = 0.0

                try:
                    p_true = float(r.get("p_true"))
                except Exception:
                    p_true = None
                try:
                    p_meas = float(r.get("p"))
                except Exception:
                    p_meas = None

                if dt > 0 and p_ref_f > 0:
                    den_ref += p_ref_f * dt
                    if p_true is not None:
                        num_true += p_true * dt
                    if p_meas is not None:
                        num_meas += p_meas * dt

                t = float(r.get("t", 0.0) or 0.0)
                if p_true is not None:
                    p_true_series.append((t, p_true))
                if p_meas is not None:
                    p_meas_series.append((t, p_meas))

                # Settle times (first time after t_start crossing threshold)
                et, em = _eff_pair(r)
                if t >= t_start:
                    if settle_true is None and et is not None and et >= eff_thresh:
                        settle_true = t - t_start
                    if settle_meas is None and em is not None and em >= eff_thresh:
                        settle_meas = t - t_start

            if den_ref > 0:
                summary.extra["energy_true_ratio"] = float(num_true / den_ref)
                summary.extra["energy_meas_ratio"] = float(num_meas / den_ref)
            else:
                summary.extra["energy_true_ratio"] = 0.0
                summary.extra["energy_meas_ratio"] = 0.0

            summary.extra["settle_time_s_true"] = settle_true
            summary.extra["settle_time_s_meas"] = settle_meas

            # Ripple RMS: compute RMS deviation of power over last window
            # Window = last min(0.2s, 20% of run time)
            t_end = float(recs_for_metrics[-1].get("t", 0.0) or 0.0)
            window_s = min(0.2, 0.2 * t_end) if t_end > 0 else 0.0
            t0_win = t_end - window_s

            def _ripple_rms(series: List[Tuple[float, float]]) -> Optional[float]:
                if not series:
                    return None
                win = [p for (tt, p) in series if tt >= t0_win]
                if len(win) < 2:
                    return None
                mean_p = sum(win) / float(len(win))
                var = sum((p - mean_p) ** 2 for p in win) / float(len(win))
                return float(var ** 0.5)

            summary.extra["ripple_rms_true"] = _ripple_rms(p_true_series)
            summary.extra["ripple_rms_meas"] = _ripple_rms(p_meas_series)

        except Exception as e2:
            summary.extra["metrics_fallback_error"] = str(e2)

    return records, summary


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write JSONL, truncating any existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Append rows to a JSONL file (create if missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --- Defaults (safe placeholders) --------------------------------------------


def default_budgets() -> List[BudgetSpec]:
    return [
        BudgetSpec(name="ideal_1kHz", dt=1e-3, perf_budget_us=None),
        BudgetSpec(name="noisy_12bit_200Hz", dt=5e-3, adc_bits_v=12, adc_bits_i=12, adc_bits_g=12, noise_v_std=0.05, noise_i_std=0.02),
        BudgetSpec(name="tight_compute_200Hz", dt=5e-3, perf_budget_us=50.0),
    ]


def default_scenarios() -> List[ScenarioSpec]:
    """Load default benchmark scenarios from `benchmarks.scenarios`.

    This keeps scenario definitions centralized and reusable by both CLI and UI.
    If the import fails due to package path differences, fall back to a single
    steady-state scenario.
    """

    try:
        from benchmarks.scenarios import default_scenarios as _default_scenarios  # type: ignore
    except Exception:
        return [
            ScenarioSpec(
                name="steady",
                env_profile=None,
                description="No env_profile; uses cfg baseline G/T.",
            ),
        ]

    out: List[ScenarioSpec] = []
    for s in _default_scenarios():
        out.append(
            ScenarioSpec(
                name=str(getattr(s, "name", "")),
                env_profile=getattr(s, "env_profile", None),
                description=str(getattr(s, "description", "")),
            )
        )

    # Safety: never return empty
    if not out:
        out = [
            ScenarioSpec(
                name="steady",
                env_profile=None,
                description="No env_profile; uses cfg baseline G/T.",
            )
        ]

    return out


def default_algorithms() -> List[AlgorithmSpec]:
    """Default algorithm set.

    Includes:
      - "hybrid": HybridMPPT controller with default HybridConfig
      - a few single-algorithm baselines (true single-controller runs)

    This is used when you don't enable --use-registry.
    """
    return [
        AlgorithmSpec(name="hybrid", controller_mode="hybrid", controller_cfg=HybridConfig()),
        AlgorithmSpec(name="ruca", controller_mode="single", algo_name="ruca"),
        AlgorithmSpec(name="mepo", controller_mode="single", algo_name="mepo"),
        AlgorithmSpec(name="pando", controller_mode="single", algo_name="pando"),
    ]


def _load_mppt_registry_map() -> Dict[str, str]:
    """Load the MPPT algorithm registry as name -> "module:Class"."""
    return {str(k): str(v) for k, v in (mppt_registry.available() or {}).items()}


def list_registry_algorithms() -> List[str]:
    """Return sorted MPPT algorithm keys from the registry plus the special 'hybrid' choice."""
    keys = sorted(_load_mppt_registry_map().keys())
    # 'hybrid' is a controller, not a registry algorithm, but we expose it as a top-level benchmark choice.
    return ["hybrid"] + keys


def _parse_algo_spec(spec: str) -> HybridConfig:
    """Parse a CLI algo spec into a HybridConfig.

    Supported forms:
      - "ruca"                 -> sets normal_name
      - "normal=ruca"          -> sets normal_name
      - "normal=ruca,global=pso,hold=nl_esc" -> sets names per state

    Any unspecified fields use HybridConfig defaults.
    """
    spec = spec.strip()
    if not spec:
        return HybridConfig()

    cfg = HybridConfig()

    if "=" not in spec:
        cfg.normal_name = spec
        return cfg

    parts = [p.strip() for p in spec.split(",") if p.strip()]
    kv = {}
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Invalid algo spec segment '{p}'. Expected key=value.")
        k, v = p.split("=", 1)
        kv[k.strip().lower()] = v.strip()

    if "normal" in kv:
        cfg.normal_name = kv["normal"]
    if "global" in kv:
        cfg.global_name = kv["global"]
    if "hold" in kv:
        cfg.hold_name = kv["hold"]

    return cfg


def resolve_algorithms_from_registry(names: Sequence[str]) -> List[AlgorithmSpec]:
    """Resolve user-provided algorithm selections into AlgorithmSpec.

    Rules:
      - "hybrid" (or hybrid aliases) => HybridMPPT with default HybridConfig
      - "normal=ruca,global=pso,..." => HybridMPPT with an explicit HybridConfig
      - "ruca" / "p&o" / "gmpp" etc => SINGLE controller running that registry algo

    This matches the UI semantics: selecting an algo runs it for the entire run.
    """

    reg_keys = _load_mppt_registry_map()
    reg_lc = {k.lower(): k for k in reg_keys.keys()}

    resolved: List[AlgorithmSpec] = []
    missing: List[str] = []

    for raw in names:
        raw = raw.strip()
        if not raw:
            continue

        raw_l = raw.lower().strip()

        # Special controller choice
        if raw_l in ("hybrid", "hybrid_mppt", "hybridmppt"):
            resolved.append(AlgorithmSpec(name="hybrid", controller_mode="hybrid", controller_cfg=HybridConfig()))
            continue

        # Hybrid spec (explicit per-state algorithms)
        if "=" in raw:
            try:
                hc = _parse_algo_spec(raw)
            except Exception as e:
                raise RuntimeError(f"Invalid algorithm spec '{raw}': {e}") from e

            # Validate each chosen algorithm name exists in MPPT registry (case-insensitive)
            for field_name, val in (
                ("normal", hc.normal_name),
                ("global", hc.global_name),
                ("hold", hc.hold_name),
            ):
                key = val
                if key not in reg_keys:
                    hit = reg_lc.get(str(key).lower())
                    if hit is None:
                        missing.append(f"{field_name}={val}")
                    else:
                        if field_name == "normal":
                            hc.normal_name = hit
                        elif field_name == "global":
                            hc.global_name = hit
                        else:
                            hc.hold_name = hit

            display_name = raw.replace(" ", "")
            resolved.append(AlgorithmSpec(name=display_name, controller_mode="hybrid", controller_cfg=hc))
            continue

        # Plain name => single algo (alias-aware)
        if not mppt_registry.is_valid(raw):
            missing.append(raw)
            continue

        canonical = mppt_registry.resolve_key(raw)
        resolved.append(AlgorithmSpec(name=canonical, controller_mode="single", algo_name=canonical))

    if missing:
        raise RuntimeError(
            f"Unknown MPPT algorithm(s): {missing}. Available: {sorted(reg_keys.keys())}"
        )

    return resolved





# --- Suite runner -------------------------------------------------------------


def run_suite(
    algorithms: Sequence[AlgorithmSpec],
    scenarios: Sequence[ScenarioSpec],
    budgets: Sequence[BudgetSpec],
    *,
    total_time: float,
    gmpp_ref: bool,
    gmpp_ref_period_s: float,
    gmpp_ref_points: int,
    out_dir: Path,
    save_records: bool = False,
    log: Optional[Callable[[str], None]] = None,
    log_every_s: float = 0.25,
    keep_records: bool = True,
    cancel: Optional[Callable[[], bool]] = None,
    use_session: bool = True,
) -> List[RunSummary]:
    """Run Algorithm × Scenario × Budget and write JSONL output."""

    ts = time.strftime("%Y%m%d_%H%M%S")

    # Build a canonical payload for scenario/budget comparability.
    scenarios_payload: List[Dict[str, Any]] = [
        {
            "name": s.name,
            "description": s.description,
            "env_profile": s.env_profile,
        }
        for s in scenarios
    ]
    budgets_payload: List[Dict[str, Any]] = [asdict(b) for b in budgets]

    payload = make_suite_payload(
        scenarios=scenarios_payload,
        budgets=budgets_payload,
        total_time=float(total_time),
        gmpp_ref=bool(gmpp_ref),
        gmpp_ref_period_s=float(gmpp_ref_period_s),
        gmpp_ref_points=int(gmpp_ref_points),
    )
    signature = compute_suite_signature(payload)

    # Session folder is reused while the suite signature remains unchanged.
    if use_session:
        sd = ensure_session_dir(out_dir=out_dir, signature=signature, create_if_missing=True)
        if sd is None:
            raise RuntimeError("Failed to create or resolve benchmark session directory")
        session_dir = sd
    else:
        # Legacy behavior: one folder per invocation
        sig_hash = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:8]
        session_dir = out_dir / f"bench_{ts}__{sig_hash}"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "session_meta.json").write_text(
            json.dumps({"created_at": ts, "signature": signature}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Unique id for THIS invocation (so records don't overwrite)
    invocation_id = f"run_{ts}"

    summaries: List[RunSummary] = []
    summary_rows: List[Dict[str, Any]] = []

    for algo in algorithms:
        for sc in scenarios:
            for bd in budgets:
                if cancel is not None and cancel():
                    raise RuntimeError("Cancelled")
                if log is not None:
                    log(
                        f"[bench] START algo={algo.name} scenario={sc.name} budget={bd.name} dt={bd.dt} total_time={total_time}"
                    )
                rec_path = None
                if save_records:
                    rec_path = session_dir / "records" / f"{algo.name}__{sc.name}__{bd.name}__{invocation_id}.jsonl"
                records, summary = run_once(
                    algo,
                    sc,
                    bd,
                    total_time=total_time,
                    gmpp_ref=gmpp_ref,
                    gmpp_ref_period_s=gmpp_ref_period_s,
                    gmpp_ref_points=gmpp_ref_points,
                    perf_enabled=True,
                    log=log,
                    log_every_s=log_every_s,
                    keep_records=keep_records,
                    records_path=rec_path if save_records else None,
                    cancel=cancel,
                )

                summaries.append(summary)
                summary.extra["invocation_id"] = invocation_id
                summary_rows.append(asdict(summary))

                if log is not None:
                    log(
                        "[bench] DONE "
                        f"algo={summary.algo} scenario={summary.scenario} budget={summary.budget} "
                        f"ticks={summary.n_ticks} wall_s={summary.wall_s:.3f} "
                        f"eff_true_final={summary.eff_true_final} eff_true_med={summary.eff_true_median} "
                        f"ctrl_p50={summary.ctrl_us_p50} ctrl_p95={summary.ctrl_us_p95} "
                        f"viol={summary.budget_violations}"
                    )

    out_path = session_dir / "summaries.jsonl"
    append_jsonl(out_path, summary_rows)

    return summaries


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aurora MPPT benchmark runner")

    p.add_argument("--out", type=str, default="data/benchmarks", help="Output directory")
    p.add_argument("--total-time", type=float, default=1.0, help="Simulation duration (s)")
    p.add_argument("--gmpp-ref", action="store_true", help="Compute GMPP reference sweeps")
    p.add_argument("--gmpp-period", type=float, default=0.25, help="GMPP reference period (s)")
    p.add_argument("--gmpp-points", type=int, default=300, help="Points per reference sweep")
    p.add_argument("--save-records", action="store_true", help="Also save per-tick records")
    p.add_argument("--log-every", type=float, default=0.25, help="Terminal/log update period (s)")
    p.add_argument("--no-keep-records", action="store_true", help="Do not keep full records in memory (requires --save-records for metrics)")

    # Optional: hook into registry
    p.add_argument("--use-registry", action="store_true", help="Resolve algorithms by name from a registry")
    p.add_argument("--algs", type=str, default="", help="Comma-separated algorithm names (registry mode)")
    p.add_argument("--list-algs", action="store_true", help="List available MPPT algorithm keys and exit")

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    out_dir = Path(args.out)

    # Scenarios + budgets: can be replaced by your own module later
    scenarios = default_scenarios()
    budgets = default_budgets()

    if args.use_registry:
        if getattr(args, "list_algs", False):
            for k in list_registry_algorithms():
                print(k)
            return 0

        names = [s.strip() for s in args.algs.split(",") if s.strip()]
        if not names:
            raise SystemExit("--use-registry requires --algs name1,name2,...")
        algorithms = resolve_algorithms_from_registry(names)
    else:
        algorithms = default_algorithms()

    run_suite(
        algorithms=algorithms,
        scenarios=scenarios,
        budgets=budgets,
        total_time=float(args.total_time),
        gmpp_ref=bool(args.gmpp_ref),
        gmpp_ref_period_s=float(args.gmpp_period),
        gmpp_ref_points=int(args.gmpp_points),
        out_dir=out_dir,
        save_records=bool(args.save_records),
        log=None,
        log_every_s=float(args.log_every),
        keep_records=not bool(args.no_keep_records),
        use_session=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())