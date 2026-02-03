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
import time
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


# Controller selection

def parse_controller_selection(selection: Optional[str]) -> Tuple[str, Optional[str], HybridConfig]:
    """Return (controller_mode, algo_name, hybrid_cfg).

    Contract:
      - None / 'hybrid' -> run HybridMPPT controller
      - anything else  -> run that *single* algorithm for the whole run

    Notes:
      - We always return a HybridConfig for backward compatibility with older
        `SimulationConfig` signatures; newer engines can ignore it for single-mode.
    """
    if selection is None:
        return "hybrid", None, HybridConfig()

    sel = str(selection).strip()
    sel_l = sel.lower()

    if sel_l in ("hybrid", "hybrid_mppt", "hybridmppt"):
        return "hybrid", None, HybridConfig()

    if not mppt_registry.is_valid(sel):
        # Show canonical keys (not aliases) for clarity
        keys = mppt_registry.ALGORITHMS
        raise SystemExit(
            f"[mppt_sim] Unknown algorithm '{sel}'. "
            f"Available: {', '.join(keys)}"
        )

    # Normalize to canonical key so downstream (engine/controller) uses consistent naming
    sel = mppt_registry.resolve_key(sel)

    # For backward compatibility: older engines only support HybridConfig, so we
    # set normal_name=sel as a best-effort fallback.
    return "single", sel, HybridConfig(normal_name=sel)

# Runner
def run_mppt_sim(
    algo: Optional[str] = None,
    profile_name: str = "stc",
    total_time: float = 0.25,
    dt: float = 1e-3,
    verbose: bool = True,
    csv_path: Optional[str] = None,
    realtime: bool = False,
    tick_ms: int = 33,
) -> List[Dict[str, Any]]:
    """
    Run a single MPPT scenario and return the list of records.
    """
    profile = get_profile(profile_name)
    controller_mode, algo_name, hcfg = parse_controller_selection(algo)

    records: List[Dict[str, Any]] = []

    def _collect(rec: Dict[str, Any]) -> None:
        records.append(rec)
        if verbose:
            print(rec)

    # Prefer the newer controller selection API if available; fall back to the
    # legacy HybridConfig-only API.
    try:
        cfg = SimulationConfig(
            total_time=total_time,
            dt=dt,
            start_v=18.0,
            array_kwargs={"n_strings": 2, "substrings_per_string": 3, "cells_per_substring": 18},
            env_profile=profile,
            controller_mode=controller_mode,
            algo_name=algo_name,
            controller_cfg=(hcfg if controller_mode == "hybrid" else None),
            on_sample=_collect,
        )
    except TypeError:
        # Legacy engines: still run, but single-mode will behave like
        # "hybrid with normal_name=<algo>".
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

    if realtime:
        # Wall-clock paced stepping (useful for demos / interactive observation)
        tick_s = max(0.0, float(tick_ms) / 1000.0)
        eng.reset()  # emits initial record via on_sample
        while True:
            rec = eng.step_once()
            if rec is None:
                break
            if tick_s > 0:
                time.sleep(tick_s)
    else:
        # Batch mode (as fast as possible)
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

            # Flatten nested dicts for nicer CSVs
            action = row.pop("action", None)
            if isinstance(action, dict):
                for k, v in action.items():
                    row[f"action_{k}"] = v

            gmpp = row.pop("gmpp", None)
            if isinstance(gmpp, dict):
                for k, v in gmpp.items():
                    row[f"gmpp_{k}"] = v

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
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        help=(
            "Controller selection. Use 'hybrid' for the HybridMPPT controller, "
            "or pass an algorithm key from core.mppt_algorithms.registry "
            "(e.g. local.pando, local.ruca, global_search.pso) to run that single "
            "algorithm for the entire run."
        ),
    )
    parser.add_argument("--profile", type=str, default="stc", help="Environment profile name: stc, cloud, shade")
    parser.add_argument("--dt", type=float, default=1e-3, help="Simulation step (s)")
    parser.add_argument("--time", type=float, default=0.25, help="Total simulation time (s)")
    parser.add_argument("--quiet", action="store_true", help="Do not print each sample")
    parser.add_argument("--csv", type=str, default=None, help="Path to write CSV results")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Pace the simulation in wall-clock time (one step per tick).",
    )
    parser.add_argument(
        "--tick-ms",
        type=int,
        default=33,
        help="Tick interval in milliseconds for --realtime (default: 33ms ~ 30Hz).",
    )
    args = parser.parse_args()

    run_mppt_sim(
        algo=args.algo,
        profile_name=args.profile,
        total_time=args.time,
        dt=args.dt,
        verbose=not args.quiet,
        csv_path=args.csv,
        realtime=args.realtime,
        tick_ms=args.tick_ms,
    )


if __name__ == "__main__":
    main()
