"""
source_sim.py

Scenario / source-driven simulations for Aurora.

This runner focuses on CHANGING THE SOURCE CONDITIONS over time:
- irradiance ramps
- temperature ramps
- step changes to emulate partial shading
- custom CSV-like profiles (time, G, T)

It is meant to be the place where we test "what happens to the array + controller
when the sun / panel temperature changes like THIS?"

Under the hood it uses the same SimulationEngine you already have.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Callable

from simulators.engine import SimulationConfig, SimulationEngine
from core.controller.hybrid_controller import HybridConfig
from core.mppt_algorithms import registry as mppt_registry

# Built-in source profiles
def profile_stc() -> List[Tuple[float, float, float]]:
    """Constant STC-like conditions."""
    return [(0.0, 1000.0, 25.0)]


def profile_cloudy() -> List[Tuple[float, float, float]]:
    """Simple multi-step cloud / PSC pattern."""
    return [
        (0.0, 1000.0, 25.0),
        (0.05, 650.0, 25.0),
        (0.12, 450.0, 26.0),
        (0.18, 800.0, 27.0),
        (0.23, 1000.0, 27.0),
    ]


def profile_ramp(start_g: float = 200.0, end_g: float = 1000.0, duration: float = 0.25, temp_c: float = 25.0) -> List[Tuple[float, float, float]]:
    """
    Build a linear irradiance ramp from start_g → end_g over `duration` seconds.
    We discretize in 5 segments (good enough for controller testing).
    """
    steps = 5
    out: List[Tuple[float, float, float]] = []
    for k in range(steps + 1):
        t = duration * k / steps
        g = start_g + (end_g - start_g) * (k / steps)
        out.append((t, g, temp_c))
    return out


def profile_hot_panel() -> List[Tuple[float, float, float]]:
    """Keep irradiance high but increase temperature; useful for Voc / Pmax shifts."""
    return [
        (0.0, 1000.0, 25.0),
        (0.05, 1000.0, 35.0),
        (0.10, 1000.0, 45.0),
        (0.15, 1000.0, 55.0),
    ]


def load_profile_from_csv(path: str) -> List[Tuple[float, float, float]]:
    """
    Load a simple CSV with 3 columns: time_s, irradiance, temp_c
    No header is required. If present, it's ignored.
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[source_sim] CSV profile file not found: {path}")

    rows: List[Tuple[float, float, float]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("time"):
                # header
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            t = float(parts[0])
            g = float(parts[1])
            tc = float(parts[2])
            rows.append((t, g, tc))
    if not rows:
        raise SystemExit(f"[source_sim] CSV profile is empty or invalid: {path}")
    # ensure sorted
    rows.sort(key=lambda x: x[0])
    return rows


def get_profile(name: str, csv_path: Optional[str] = None) -> List[Tuple[float, float, float]]:
    name = (name or "stc").lower()
    if name in ("stc", "const", "fixed"):
        return profile_stc()
    if name in ("cloud", "cloudy", "psc", "shade"):
        return profile_cloudy()
    if name in ("ramp", "sunrise"):
        return profile_ramp()
    if name in ("hot", "temperature", "hot-panel"):
        return profile_hot_panel()
    if name == "csv":
        if not csv_path:
            raise SystemExit("[source_sim] CSV profile selected but no --csv-path provided.")
        return load_profile_from_csv(csv_path)
    # fallback
    return profile_stc()

# Controller builder
def build_hybrid_with(algo_name: Optional[str]) -> HybridConfig:
    """
    Optionally force the hybrid controller to use a particular local MPPT
    from the registry (e.g. local.pando, local.ruca, global_search.pso).
    If algo_name is None, we just return a plain HybridConfig().
    """
    if not algo_name:
        return HybridConfig()
    catalog = mppt_registry.catalog()
    if algo_name not in catalog:
        raise SystemExit(
            f"[source_sim] Unknown algorithm '{algo_name}'. "
            f"Available: {', '.join(sorted(catalog.keys()))}"
        )
    return HybridConfig(local_algo=algo_name)

# Runner
def run_source_sim(
    profile_name: str = "stc",
    algo: Optional[str] = None,
    csv_path: Optional[str] = None,
    total_time: float = 0.3,
    dt: float = 1e-3,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run a simulation where the SOURCE is the main changing variable.
    Returns the list of per-sample records.
    """
    env_profile = get_profile(profile_name, csv_path)

    # we still use the hybrid controller; can be extended to accept MPPT choice
    hcfg = build_hybrid_with(algo)

    records: List[Dict[str, Any]] = []

    def _collect(rec: Dict[str, Any]) -> None:
        records.append(rec)
        if verbose:
            print(rec)

    cfg = SimulationConfig(
        total_time=total_time,
        dt=dt,
        start_v=18.0,
        array_kwargs={"n_strings": 2, "substrings_per_string": 3, "cells_per_substring": 18},
        env_profile=env_profile,
        controller_cfg=hcfg,
        on_sample=_collect,
    )

    eng = SimulationEngine(cfg)
    for _ in eng.run():
        pass

    return records

# CLI
def main() -> None:
    parser = argparse.ArgumentParser(description="Run a source-driven (irradiance/temperature) simulation.")
    parser.add_argument("--algo", type=str, default=None,
                        help="MPPT algorithm key from core.mppt_algorithms.registry (e.g. local.pando)")
    parser.add_argument("--profile", type=str, default="stc",
                        help="Built-in profile: stc, cloud, ramp, hot, csv")
    parser.add_argument("--csv-path", type=str, default=None,
                        help="If --profile=csv, path to CSV with time_s,irradiance,temp_c")
    parser.add_argument("--dt", type=float, default=1e-3,
                        help="Simulation step (s)")
    parser.add_argument("--time", type=float, default=0.3,
                        help="Total simulation time (s)")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print each sample")

    args = parser.parse_args()

    run_source_sim(
        profile_name=args.profile,
        algo=args.algo,
        csv_path=args.csv_path,
        total_time=args.time,
        dt=args.dt,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()