"""Benchmark runner for Aurora MPPT algorithms.

This module is intentionally *thin* on assumptions about your project structure.
It provides:
  - A small benchmark harness (Algorithm × Scenario × Budget)
  - Collection of per-tick records from `SimulationEngine.run()`
  - Basic metrics extraction from the emitted `gmpp` and `perf` fields
  - JSONL output suitable for later aggregation/plotting

Expected engine output fields (added in your engine benchmarking upgrade):
  rec["gmpp"]["eff_best"]          # true best-so-far / true GMPP reference
  rec["gmpp"]["eff_best_meas"]     # measured best-so-far / true GMPP reference
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.controller.hybrid_controller import HybridConfig


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

    `controller_cfg` is passed directly to SimulationConfig. In your repo this is
    typically a dataclass/config object consumed by the hybrid controller.

    If you want to support string names -> configs, add a resolver.
    """

    name: str
    controller_cfg: Any
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
    """Return (eff_true, eff_meas) from a record."""
    gmpp = rec.get("gmpp") or {}
    eff_true = gmpp.get("eff_best")
    eff_meas = gmpp.get("eff_best_meas")
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
) -> Tuple[List[Dict[str, Any]], RunSummary]:
    """Run one simulation and return (records, summary)."""

    cfg = SimulationConfig(
        total_time=float(total_time),
        dt=float(budget.dt),
        controller_cfg=algo.controller_cfg,
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

    t0 = time.perf_counter()
    eng = SimulationEngine(cfg)
    records = list(eng.run())
    wall_s = time.perf_counter() - t0

    summary = summarize_records(algo, scenario, budget, records, wall_s)
    return records, summary


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
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
    """Return a minimal set of scenarios.

    NOTE: `env_profile` is passed through to your engine; use the representation
    your engine expects (e.g., list of events, callable, etc.).

    If you already have `benchmarks/scenarios.py`, import from there and delete
    these placeholders.
    """

    return [
        ScenarioSpec(name="steady", env_profile=None, description="No env_profile; uses cfg baseline G/T."),
    ]


def default_algorithms() -> List[AlgorithmSpec]:
    """Fallback algorithms if you don't use --use-registry.

    These use HybridConfig defaults but vary the NORMAL tracker.
    """
    return [
        AlgorithmSpec(name="ruca", controller_cfg=HybridConfig(normal_name="ruca")),
        AlgorithmSpec(name="mepo", controller_cfg=HybridConfig(normal_name="mepo")),
        AlgorithmSpec(name="pando", controller_cfg=HybridConfig(normal_name="pando")),
    ]


def _load_mppt_registry_map() -> Dict[str, str]:
    """Load the MPPT algorithm registry as name -> "module:Class".

    Uses Aurora's lazy MPPT registry at `core.mppt_algorithms.registry`.
    Returns a dict like {"ruca": "core....:RUCA", ...}.
    """
    mod = importlib.import_module("core.mppt_algorithms.registry")
    available_fn = getattr(mod, "available", None)
    if not callable(available_fn):
        raise RuntimeError(
            "core.mppt_algorithms.registry.available() not found/callable. "
            "Ensure you are using Aurora's MPPT registry module."
        )
    reg_map = available_fn()
    if not isinstance(reg_map, dict):
        raise RuntimeError(
            "core.mppt_algorithms.registry.available() did not return a dict. "
            f"Got: {type(reg_map)}"
        )
    return {str(k): str(v) for k, v in reg_map.items()}


def list_registry_algorithms() -> List[str]:
    """Return sorted MPPT algorithm keys from the registry."""
    return sorted(_load_mppt_registry_map().keys())


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
    """Resolve algorithm names (or specs) into HybridConfig controller_cfg objects.

    In Aurora, `SimulationConfig.controller_cfg` expects a `HybridConfig`.
    We treat each provided `name` as an MPPT registry key for the NORMAL tracker
    unless the user provides a key=value spec.
    """

    reg_keys = _load_mppt_registry_map()
    reg_lc = {k.lower(): k for k in reg_keys.keys()}

    resolved: List[AlgorithmSpec] = []
    missing: List[str] = []

    for raw in names:
        raw = raw.strip()
        if not raw:
            continue

        # Build HybridConfig from a spec, then validate selected algos exist
        try:
            hc = _parse_algo_spec(raw)
        except Exception as e:
            raise RuntimeError(f"Invalid algorithm spec '{raw}': {e}") from e

        # Validate each chosen algorithm name exists in MPPT registry
        for field_name, val in (
            ("normal", hc.normal_name),
            ("global", hc.global_name),
            ("hold", hc.hold_name),
        ):
            key = val
            if key not in reg_keys:
                # case-insensitive match
                hit = reg_lc.get(key.lower())
                if hit is None:
                    missing.append(f"{field_name}={val}")
                else:
                    # normalize casing
                    if field_name == "normal":
                        hc.normal_name = hit
                    elif field_name == "global":
                        hc.global_name = hit
                    else:
                        hc.hold_name = hit

        # Name used for output filenames / tables
        display_name = raw.replace(" ", "")
        resolved.append(AlgorithmSpec(name=display_name, controller_cfg=hc))

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
) -> List[RunSummary]:
    """Run Algorithm × Scenario × Budget and write JSONL output."""

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"bench_{ts}"

    summaries: List[RunSummary] = []
    summary_rows: List[Dict[str, Any]] = []

    for algo in algorithms:
        for sc in scenarios:
            for bd in budgets:
                records, summary = run_once(
                    algo,
                    sc,
                    bd,
                    total_time=total_time,
                    gmpp_ref=gmpp_ref,
                    gmpp_ref_period_s=gmpp_ref_period_s,
                    gmpp_ref_points=gmpp_ref_points,
                    perf_enabled=True,
                )

                summaries.append(summary)
                summary_rows.append(asdict(summary))

                if save_records:
                    rec_path = out_dir / run_id / "records" / f"{algo.name}__{sc.name}__{bd.name}.jsonl"
                    write_jsonl(rec_path, records)

    out_path = out_dir / run_id / "summaries.jsonl"
    write_jsonl(out_path, summary_rows)

    return summaries


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aurora MPPT benchmark runner")

    p.add_argument("--out", type=str, default="data/benchmarks", help="Output directory")
    p.add_argument("--total-time", type=float, default=1.0, help="Simulation duration (s)")
    p.add_argument("--gmpp-ref", action="store_true", help="Compute GMPP reference sweeps")
    p.add_argument("--gmpp-period", type=float, default=0.25, help="GMPP reference period (s)")
    p.add_argument("--gmpp-points", type=int, default=300, help="Points per reference sweep")
    p.add_argument("--save-records", action="store_true", help="Also save per-tick records")

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
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())