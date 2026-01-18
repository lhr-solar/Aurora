"""
mppt_sim.py

Convenience runner for MPPT experiments.

This script is a *thinner* wrapper than `engine.py`. It focuses on:
- choosing / building a specific MPPT algorithm (via the registry),
- running it on a standard array for a fixed time,
- printing / returning JSON-friendly samples.

Use this when you want to compare algorithms or run batches from the CLI.

Examples
--------
# run with default hybrid controller
python -m simulators.mppt_sim

# run with a specific algorithm from the registry
python -m simulators.mppt_sim --algo local.pando

# run with a step-down irradiance profile
python -m simulators.mppt_sim --profile cloud
"""

import argparse
from typing import Any, Dict, List, Tuple, Optional, Callable
from pathlib import Path

from simulators.engine import (
    SimulationConfig,
    SimulationEngine,
)

from core.controller.hybrid_controller import HybridConfig
from core.mppt_algorithms import registry as mppt_registry
import csv

# Profiles
def get_profile(name: str) -> List[Tuple[float, float, float]]:
    """
    Return a simple environment (t, G, T) profile by name.
    """
    name = name.lower()
    if name in ("none", "const", "stc"):
        return [(0.0, 1000.0, 25.0)]
    if name in ("cloud", "psc", "shade"):
        # emulate a cloud / partial shading
        return [
            (0.0, 1000.0, 25.0),
            (0.05, 650.0, 25.0),
            (0.12, 450.0, 27.0),
            (0.16, 800.0, 26.0),
        ]
    # default
    return [(0.0, 1000.0, 25.0)]

# Controller builders
def build_hybrid_with(algo_name: Optional[str]) -> HybridConfig:
    """
    Build a HybridConfig that forces the local tracker to the chosen algo.
    If algo_name is None, we just use whatever defaults the HybridMPPT has.
    """
    if not algo_name:
        return HybridConfig()
    # We don't want to break if the user typoed the name; so we validate
    cat = mppt_registry.catalog()
    if algo_name not in cat:
        raise SystemExit(
            f"[mppt_sim] Unknown algorithm '{algo_name}'. "
            f"Available: {', '.join(sorted(cat.keys()))}"
        )
    # HybridConfig takes a mapping of phase -> algo name; we at least set the local one
    return HybridConfig(normal_name=algo_name)

# Runner
def run_mppt_sim(
    algo: Optional[str] = None,
    profile_name: str = "stc",
    total_time: float = 0.25,
    dt: float = 1e-3,
    verbose: bool = True,
    csv_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run a single MPPT scenario and return the list of records.
    """
    profile = get_profile(profile_name)
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
        env_profile=profile,
        controller_cfg=hcfg,
        on_sample=_collect,
    )
    eng = SimulationEngine(cfg)
    for _ in eng.run():
        pass

    if csv_path and records:
        # Ensure output directory exists: Aurora/data/runs
        out_dir = Path("data") / "runs"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / csv_path

        # Flatten records (including nested action dict if present)
        rows = []
        for rec in records:
            row = dict(rec)
            action = row.pop("action", None)
            if isinstance(action, dict):
                for k, v in action.items():
                    row[f"action_{k}"] = v
            rows.append(row)

        fieldnames = sorted({k for r in rows for k in r.keys()})
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        if verbose:
            print(f"[mppt_sim] CSV written to {out_path}")

    return records

# CLI
def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single MPPT simulation.")
    parser.add_argument("--algo", type=str, default=None, help="Algorithm key from core.mppt_algorithms.registry (e.g. local.pando, local.ruca, global_search.pso)")
    parser.add_argument("--profile", type=str, default="stc", help="Environment profile name: stc, cloud, shade")
    parser.add_argument("--dt", type=float, default=1e-3, help="Simulation step (s)")
    parser.add_argument("--time", type=float, default=0.25, help="Total simulation time (s)")
    parser.add_argument("--quiet", action="store_true", help="Do not print each sample")
    parser.add_argument("--csv", type=str, default=None, help="Path to write CSV results")
    args = parser.parse_args()

    run_mppt_sim(
        algo=args.algo,
        profile_name=args.profile,
        total_time=args.time,
        dt=args.dt,
        verbose=not args.quiet,
        csv_path=args.csv,
    )


if __name__ == "__main__":
    main()
