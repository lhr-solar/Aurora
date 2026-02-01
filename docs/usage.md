

# Aurora Usage Guide

This document explains **how to run, configure, and use Aurora correctly**.
It is written for users who want to explore MPPT behavior, debug algorithms, or generate reproducible benchmark results.

If you are new, read this once end-to-end before running experiments.

---

## Prerequisites

### System Requirements
- Python **3.10+** (3.11 supported)
- macOS or Linux recommended  
  (Windows can work, but Qt/OpenGL backends are more fragile)

### Python Environment
Aurora is designed to run inside a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Ways to Run Aurora

Aurora supports three execution modes:

1. **Desktop UI** — interactive exploration and debugging  
2. **CLI simulation** — fast, scriptable algorithm testing  
3. **Benchmarking** — controlled, reproducible comparison  

All three modes share the same engine, physics model, and controller code.

---

## 1. Desktop UI (Recommended Starting Point)

### Launching the UI

From the repository root:

```bash
python -m ui.desktop.main_window
```

This opens the main window with three tabs:
- **Live Bench**
- **Benchmarks**
- **Glossary / Docs**

---

### Live Bench Tab

The Live Bench is intended for **intuition building and debugging**, not formal evaluation.

#### Controls
- **Irradiance slider** (W/m²)
- **Temperature slider** (°C)
- **MPPT algorithm selector**
- **Simulation timestep and run controls**
- **CSV profile toggle (optional)**

#### Behavior
- Slider changes immediately affect the environment
- IV and PV curves update continuously
- MPPT operating point and power output are visualized in real time

⚠️ **Important:**  
Live Bench runs are *not deterministic* due to UI event timing and manual interaction.  
Do not use Live Bench results for benchmarking or algorithm comparison.

---

### CSV Profiles (Deterministic Environment Playback)

CSV profiles define **time-indexed environment inputs** and enable reproducible simulations.

When a CSV profile is active:
- Sliders are disabled
- Environment values come exclusively from the profile
- Simulation becomes deterministic (given fixed `dt`)

#### CSV Format

Required columns (header names must match):

```text
time, irradiance, temperature
```

Conventions:
- `time` is in seconds and must be monotonically increasing
- `irradiance` is in W/m²
- `temperature` is in °C

Profiles are stored in:
```text
profiles/
```

CSV profiles are **required** for benchmarking and fair algorithm comparison.

---

### Glossary / Docs Tab

The Docs tab renders all Markdown files in `docs/` directly inside the application.

Use this tab to:
- understand architecture and terminology
- review controller contracts
- follow extension guides

---

## 2. Command-Line Simulation (CLI)

CLI simulation is useful for **fast iteration and debugging** without UI overhead.

### Basic MPPT Simulation

```bash
python -m simulators.mppt_sim --algo pando
```

If an invalid algorithm name is provided, the simulator prints the list of available algorithms.

### Typical CLI Use Cases
- Verifying algorithm logic
- Debugging convergence or oscillation
- Testing under fixed or profile-driven environments
- Running batch experiments from scripts

CLI simulation uses the same engine and controller code as the UI.

---

### Source / Environment Simulator

```bash
python -m simulators.source_sim
```

This utility isolates **environment playback** from MPPT logic.
It is useful when debugging CSV profiles or irradiance/temperature handling.

---

## 3. Benchmarking

Benchmarking is used to **compare MPPT algorithms under identical conditions**.

### Benchmark Tab (UI)

The Benchmarks tab:
- Executes predefined scenarios
- Runs multiple algorithms with identical initialization
- Computes standardized metrics
- Displays comparative plots and tables

This is the preferred method for performance comparison.

---

### Benchmark Components

Benchmarks are defined by three elements:

1. **Scenario**
   - Deterministic, time-indexed environment definition
   - Implemented in `benchmarks/scenarios.py`

2. **Metric**
   - Quantitative performance measure
   - Implemented in `benchmarks/metrics.py`

3. **Runner**
   - Executes simulations and aggregates results
   - Implemented in `benchmarks/runner.py`

All benchmarks enforce:
- identical environment inputs
- identical simulation parameters
- isolated controller instances

---

## Configuration Files

Aurora may persist configuration state (when enabled) to:

```text
configs/
```

These files capture UI and simulation configuration needed for reproducibility.

---

## Output Data

### Logged Runs

Simulation outputs are written to:

```text
data/runs/
```

Each run typically contains:
- time series of `(t, G, T, V, I, P)`
- controller outputs or exposed internal signals
- metadata (algorithm name, timestep, profile/scenario)

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
   - Capture conditions using CSV profiles

3. **Compare**
   - Run benchmarks across algorithms

4. **Analyze**
   - Inspect logs and metrics
   - Iterate on controller design

---

## Common Mistakes

- Using Live Bench results for formal comparison
- Comparing algorithms under different profiles or scenarios
- Modifying UI code to affect simulation behavior
- Forgetting to register a new algorithm

---

## Troubleshooting

### UI does not launch
- Confirm the virtual environment is activated
- Verify PyQt6 and pyqtgraph are installed
- Ensure you are running from the repo root

### Algorithm not listed
- Ensure it is registered in the algorithm registry
- Check naming consistency between file and registration

### Inconsistent benchmark results
- Verify scenarios and profiles are deterministic
- Confirm timestep and initialization settings are identical
- Ensure controllers do not retain state across runs

---

## Where to Go Next

- `docs/glossary.md` — concepts and terminology
- `docs/architecture.md` — execution model and invariants
- `docs/api.md` — controller interfaces and contracts

This guide should evolve alongside the codebase.