

# Aurora Usage Guide

This document explains **how to run, configure, and use Aurora correctly**.
It is written for users who want to run simulations, debug MPPT behavior, or generate reproducible benchmarks.

If you are new, read this once end-to-end before running experiments.

---

## Prerequisites

### System Requirements
- Python **3.10+** (3.11 works)
- macOS or Linux recommended  
  (Windows can work, but Qt and OpenGL backends are more fragile)

### Python Environment
Aurora is designed to run inside a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Ways to Run Aurora

Aurora can be run in three primary modes:
1. **Desktop UI** (interactive exploration)
2. **CLI simulation** (algorithm debugging)
3. **Benchmarking** (controlled comparison)

All three modes share the same engine, physics model, and controllers.

---

## 1. Desktop UI (Recommended Starting Point)

### Launching the UI

From the repo root:

```bash
python -m ui.desktop.main_window
```

This launches the main window with three tabs:
- **Live Bench**
- **Benchmarks**
- **Glossary / Docs**

---

### Live Bench Tab

The Live Bench is designed for **interactive debugging and intuition building**.

#### Controls
- **Irradiance slider** (W/m²)
- **Temperature slider** (°C)
- **MPPT algorithm selector**
- **Time step and simulation controls**
- **CSV profile toggle (optional)**

#### Behavior
- Slider changes immediately affect the environment
- IV and PV curves update in real time
- MPPT operating point and power output are visualized continuously

This mode is *not deterministic* and should not be used for formal comparison.

---

### CSV Profiles (Environment Playback)

CSV profiles define **time-indexed environment inputs**.

When a CSV profile is active:
- Sliders are disabled
- Environment values are driven exclusively by the profile
- Simulation becomes deterministic and replayable

Typical CSV columns:
```text
time, irradiance, temperature
```

Profiles live in:
```text
profiles/
```

CSV profiles are required for benchmarking and reproducible experiments.

---

### Glossary / Docs Tab

The Docs tab renders all files in `docs/` inside the application.

Start here if:
- You are new to Aurora
- You want to understand architecture or terminology
- You are adding new algorithms or metrics

---

## 2. Command-Line Simulation (CLI)

CLI simulation is useful for **fast iteration and debugging** without UI overhead.

### Basic MPPT Simulation

```bash
python -m simulators.mppt_sim --algo pando
```

If an invalid algorithm is passed, the simulator prints all available options.

### Common Use Cases
- Verifying algorithm behavior
- Debugging convergence issues
- Testing under fixed environment conditions
- Running batch experiments via scripts

CLI simulation uses the same engine and controller code as the UI.

---

### Source Simulation Utility

```bash
python -m simulators.source_sim
```

This utility isolates **environment and source behavior** from MPPT logic.
It is helpful when debugging profiles or irradiance/temperature handling.

---

## 3. Benchmarking

Benchmarking is used to **compare MPPT algorithms under identical conditions**.

### Benchmark Tab (UI)

The Benchmarks tab:
- Executes predefined scenarios
- Runs multiple algorithms under identical inputs
- Computes standardized metrics
- Displays comparative plots and tables

This is the preferred method for performance comparison.

---

### Benchmark Components

Benchmarks are defined by three elements:

1. **Scenario**
   - Time-indexed environment definition
   - Implemented in `benchmarks/scenarios.py`

2. **Metric**
   - Quantitative performance measure
   - Implemented in `benchmarks/metrics.py`

3. **Runner**
   - Executes simulations and aggregates results
   - Implemented in `benchmarks/runner.py`

All benchmarks enforce:
- identical initialization
- identical environment inputs
- identical simulation parameters

---

## Configuration Files

Aurora may save configuration state (when enabled) to:

```text
configs/
```

These files capture:
- communication settings
- capture parameters
- UI state needed for reproducibility

---

## Output Data

### Logged Runs

Simulation outputs are written to:

```text
data/runs/
```

Each run contains:
- timestamped measurements
- environment values
- controller state outputs

### Benchmark Outputs

Benchmark results are written to:

```text
data/benchmarks/
```

These outputs are:
- machine-readable
- deterministic (given identical inputs)
- suitable for offline analysis

---

## Recommended Workflow

1. **Explore**
   - Use Live Bench with sliders
   - Build intuition for algorithm behavior

2. **Stabilize**
   - Capture conditions with CSV profiles

3. **Compare**
   - Run benchmarks across algorithms

4. **Analyze**
   - Inspect logs and metrics
   - Iterate on controller design

---

## Common Mistakes

- Using Live Bench results for formal comparison
- Comparing algorithms under different profiles
- Modifying UI code to change simulation behavior
- Forgetting to register new algorithms

---

## Troubleshooting

### UI does not launch
- Confirm virtual environment is activated
- Verify PyQt6 and pyqtgraph are installed
- Run from repo root

### Algorithm not listed
- Ensure it is registered in the algorithm registry
- Check naming consistency

### Inconsistent benchmark results
- Verify CSV profiles and scenarios are identical
- Confirm timestep and initialization settings

---

## Where to Go Next

- `docs/glossary.md` — concepts and terminology
- `docs/architecture.md` — execution model and boundaries
- `docs/api.md` — controller interfaces and contracts

This guide should evolve as new features are added.