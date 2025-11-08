"""Aurora Config Creator CLI

Standalone command-line tool for building Aurora array/layout config JSON files.

Run this file directly:

    python ConfigCreator.py

It will prompt you for general plotting/simulation settings, array shape,
default cell parameters, and then save the generated JSON config file.

The generated JSON matches the structure expected by
ArrayPlotterWindow.load_and_apply_config, i.e.::

    {
      "title": "...",
      "points": 400,
      "live": true,
      "live_interval_ms": 500,
      "layout": {
        "strings": [
          { "substrings": [ { "cells": [ { ...cell params... }, ... ] } ] },
          ...
        ]
      }
    }
"""

import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass
class CellDefaults:
    """Default parameters for a single cell.

    These roughly mirror what Aurora's core Cell model expects. You can
    derive these values from a module datasheet or measurements.
    """

    isc_ref: float = 5.84
    voc_ref: float = 0.621
    diode_ideality: float = 1.3
    r_s: float = 0.02
    r_sh: float = 800.0
    vmpp: float = 0.521
    impp: float = 5.4
    irradiance: float = 1000.0
    temperature_c: float = 25.0
    autofit: bool = False


def prompt_float(prompt: str, default: float, min_val: float = None, max_val: float = None) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            val = float(raw)
            if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
                print(f"Value must be between {min_val} and {max_val}. Please try again.")
                continue
            return val
        except ValueError:
            print("Invalid number. Please try again.")


def prompt_int(prompt: str, default: int, min_val: int = None, max_val: int = None) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            val = int(raw)
            if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
                print(f"Value must be between {min_val} and {max_val}. Please try again.")
                continue
            return val
        except ValueError:
            print("Invalid integer. Please try again.")


def prompt_bool(prompt: str, default: bool) -> bool:
    yes = {"yes", "y"}
    no = {"no", "n"}
    default_str = "y" if default else "n"
    while True:
        raw = input(f"{prompt} (y/n) [{default_str}]: ").strip().lower()
        if not raw:
            return default
        if raw in yes:
            return True
        if raw in no:
            return False
        print("Please enter 'y' or 'n'.")


def main() -> int:
    print("Aurora Config Creator CLI")
    print("=========================\n")

    title = input("Title [My Array Config]: ").strip()
    if not title:
        title = "My Array Config"

    points = prompt_int("Points", 400, 10, 5000)
    live = prompt_bool("Start in Live mode", True)
    live_interval_ms = prompt_int("Live interval (ms)", 500, 50, 10000)

    n_strings = prompt_int("# Strings", 1, 1, 100)
    n_substrings = prompt_int("# Substrings per string", 1, 1, 100)
    n_cells = prompt_int("# Cells per substring", 2, 1, 200)

    print("\nEnter cell default parameters (per cell):")
    cell_defaults = CellDefaults()
    cell_defaults.isc_ref = prompt_float("Isc_ref (A)", cell_defaults.isc_ref, 0.0, 50.0)
    cell_defaults.voc_ref = prompt_float("Voc_ref (V)", cell_defaults.voc_ref, 0.0, 5.0)
    cell_defaults.impp = prompt_float("Impp (A)", cell_defaults.impp, 0.0, 50.0)
    cell_defaults.vmpp = prompt_float("Vmpp (V)", cell_defaults.vmpp, 0.0, 5.0)
    cell_defaults.r_s = prompt_float("R_s (Ω)", cell_defaults.r_s, 0.0, 5.0)
    cell_defaults.r_sh = prompt_float("R_sh (Ω)", cell_defaults.r_sh, 1.0, 10000.0)
    cell_defaults.diode_ideality = prompt_float("Diode ideality n", cell_defaults.diode_ideality, 0.5, 3.0)
    cell_defaults.irradiance = prompt_float("Irradiance (W/m²)", cell_defaults.irradiance, 0.0, 1500.0)
    cell_defaults.temperature_c = prompt_float("Temperature (°C)", cell_defaults.temperature_c, -40.0, 100.0)
    cell_defaults.autofit = prompt_bool("Autofit IV params", cell_defaults.autofit)

    cfg: Dict[str, Any] = {
        "title": title,
        "points": points,
        "live": live,
        "live_interval_ms": live_interval_ms,
    }

    # Build layout: uniform cell defaults everywhere for now
    layout_strings: List[Dict[str, Any]] = []
    for _ in range(n_strings):
        substrings: List[Dict[str, Any]] = []
        for _ in range(n_substrings):
            cells = [asdict(cell_defaults) for _ in range(n_cells)]
            substrings.append({"cells": cells})
        layout_strings.append({"substrings": substrings})

    cfg["layout"] = {"strings": layout_strings}

    print("\nGenerated config JSON:")
    print(json.dumps(cfg, indent=2))

    while True:
        save_path = input("\nEnter filename to save config (e.g. config.json): ").strip()
        if not save_path:
            print("Filename cannot be empty.")
            continue
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            print(f"Config saved to {save_path}")
            break
        except Exception as e:
            print(f"Error saving file: {e}")
            retry = prompt_bool("Try again?", True)
            if not retry:
                print("Exiting without saving.")
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())