# Aurora

Aurora is a **modular solar array simulation and analysis platform** developed for Longhorn Racing Solar.  
It models photovoltaic (PV) systems from the cell level up through full arrays, simulates MPPT algorithms under dynamic conditions, and provides a desktop UI for interactive experimentation, benchmarking, and debugging.

Aurora is designed to be:
- **Educational** – easy to explore and reason about
- **Extensible** – straightforward to add new models, algorithms, and UI panels
- **Research-grade** – suitable for controlled experiments and benchmarking

---

## What Aurora Does

At a high level, Aurora simulates how a solar array behaves over time while an MPPT controller attempts to extract maximum power under changing conditions such as irradiance, temperature, and partial shading.

Aurora answers questions like:
- How does an MPPT algorithm behave under non-uniform irradiance?
- How quickly does it converge to the global MPP?
- How do algorithm choices affect efficiency, stability, and tracking error?
- How does array topology influence power curves?

---

## Core Features

### Physics-Based Modeling
- Single-diode PV cell model
- Explicit temperature and irradiance dependence
- Hierarchical structure:
  ```
  Cell → Substring → String → Array
  ```

### MPPT Algorithms
- Perturb & Observe (P&O)
- Incremental Conductance
- Metaheuristic and hybrid controllers (configurable and extensible)

### Simulation Engine
- Fixed-timestep discrete simulation loop
- Real-time and accelerated execution modes
- Deterministic scenario playback
- Thread-safe communication between engine and UI
- Automatic run logging (`data/runs/`)

### Desktop UI (PyQt6)
- Live IV and PV curves
- Irradiance and temperature profiles (slider or CSV-driven)
- MPPT state and power tracking visualization
- Benchmarking and comparison tools
- Integrated **Documentation / Glossary tab**

---

## Repository Structure

```
Aurora/
├── engine/           # Core simulation loop and array evaluation
├── array/            # PV cell, substring, string, and array models
├── controllers/      # MPPT algorithm implementations
├── simulators/       # CLI-based simulation entry points
├── benchmarks/       # Scenarios, metrics, and benchmarking runner
├── ui/
│   └── desktop/      # PyQt dashboards (Live Bench, Benchmarks, Docs)
├── docs/             # Markdown documentation rendered in the UI
├── data/
│   └── runs/         # Saved simulation and benchmark outputs
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/Aurora.git
cd Aurora
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Aurora

### Desktop UI (Recommended)
Launch the interactive desktop application:

```bash
python -m ui.desktop.main
```

This provides:
- Live simulation controls
- Interactive plots
- Benchmarking dashboards
- Built-in documentation and glossary

### Command-Line Simulation
Run a simulation directly from the terminal:

```bash
python -m simulators.mppt_sim --algo pando
```

Available algorithms can be listed with invalid input or via the docs tab.

---

## Documentation

Aurora includes **in-app documentation** accessible via the **“Glossary / Docs”** tab in the desktop UI.

This documentation explains:
- How to run and configure simulations
- How the architecture fits together
- How to add or modify MPPT algorithms
- How benchmarking works
- Definitions of core concepts and terminology

The same content lives in Markdown files under:
```
docs/
```

Key files:
- `docs/usage.md`
- `docs/architecture.md`
- `docs/api.md`
- `docs/glossary.md`

> Documentation is intentionally part of the application so it stays in sync with the codebase.

---

## Extending Aurora

Common extension points:
- **New MPPT algorithm** → `controllers/`
- **New benchmark scenario** → `benchmarks/scenarios.py`
- **New UI dashboard** → `ui/desktop/`
- **New metrics** → `benchmarks/metrics.py`

Each extension path is documented in the Glossary / Docs tab.

---

## Data & Outputs

- Simulation and benchmark runs are stored in:
  ```
  data/runs/
  ```
- Outputs are designed to be reproducible and machine-readable
- This directory is ignored by Git by default

---

## Contributing

Aurora is structured to support iterative experimentation.  
If you are modifying or extending the system, start by reading:
1. `docs/architecture.md`
2. `docs/glossary.md`

Then follow the existing patterns in the codebase.

---

## Acknowledgements

Developed for **Longhorn Racing Solar**  
University of Texas at Austin
# Aurora

Aurora is a **time-stepped photovoltaic (PV) simulation and MPPT experimentation platform** developed for Longhorn Racing Solar.

It combines:
- a physics-based PV model (cell → substring → string → array)
- a discrete-time simulation engine
- pluggable MPPT algorithms
- a PyQt6 + **pyqtgraph** desktop UI for interactive debugging and benchmarking

If you’re new, the shortest path is:
1. Run the desktop app
2. Open the **Glossary / Docs** tab
3. Use Live Bench first, then Benchmarks once you’re comfortable

---

## What Aurora Simulates

Aurora simulates PV behavior under **time-varying environment conditions** and evaluates how an MPPT controller tracks the maximum power point.

In realistic conditions (especially **partial shading**), the PV power curve can develop **multiple local maxima**, which is exactly where MPPT algorithms can fail in interesting ways (oscillation, slow convergence, local trapping, etc.). Aurora is built to make those failure modes observable and measurable.

---

## Core Concepts (quick glossary)

- **IV curve**: current vs voltage at a fixed instant.
- **PV curve**: power vs voltage at a fixed instant; the peak is the MPP.
- **MPP vs GMPP**: local peak vs global peak (important under shading).
- **Environment**: irradiance + temperature, supplied by sliders, CSV profiles, or benchmark scenarios.

For the detailed version, see `docs/glossary.md` (also rendered inside the UI).

---

## Key Features

### PV Physics Model
- Single-diode cell model with explicit irradiance and temperature dependence
- Hierarchical electrical aggregation:
  ```
  Cell → Substring → String → Array
  ```
- Deterministic evaluation (same inputs → same curves)

### MPPT Algorithms
Aurora includes multiple MPPT implementations (and is designed to make adding more straightforward). The CLI and UI expose registered algorithms such as:
- `pando` (Perturb & Observe)
- `pso`, `mepo`, `nl_esc`, `ruca`, `sweep` (see `core/mppt_algorithms/`)

### Simulation Engine
- Discrete time-step simulation loop (engine orchestrates: environment → array evaluation → controller → logging)
- Run logging for post-analysis (see `data/runs/`)

### Desktop UI (PyQt6 + pyqtgraph)
- Live IV/PV plotting
- Irradiance/temperature input via **sliders** or **CSV profiles**
- MPPT state visualization
- Benchmark dashboard for comparative evaluation
- Embedded documentation tab (**Glossary / Docs**) rendered from `docs/*.md`

---

## Repository Structure (actual)

```text
Aurora/
├── core/
│   ├── src/                 # PV physics + array hierarchy (cell/substring/string/array)
│   ├── controller/          # Controller interface / glue layer
│   └── mppt_algorithms/     # MPPT implementations (local/, global_search/, hold/, ...)
├── simulators/              # CLI simulators (engine.py, mppt_sim.py, source_sim.py)
├── benchmarks/              # Scenarios + metrics + benchmark runner
├── ui/
│   └── desktop/             # PyQt dashboards (Live Bench, Benchmarks, Glossary)
├── profiles/                # CSV profiles (environment timelines)
├── configs/                 # Saved UI/sim configs (if enabled)
├── docs/                    # Markdown docs rendered in the UI
├── data/
│   ├── runs/                # Logged simulation runs
│   └── benchmarks/          # Benchmark outputs
└── requirements.txt
```

---

## Installation

### Requirements
- Python **3.10+** recommended
- macOS / Linux works best (Windows can work, but Qt tooling is pickier)

### Setup

```bash
git clone https://github.com/<your-org-or-username>/Aurora.git
cd Aurora

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Running Aurora

### Desktop UI (recommended)

```bash
python -m ui.desktop.main_window
```

This launches the main window with tabs:
- **Live Bench**: interactive environment control + plots
- **Benchmarks**: scenario-driven comparison + metrics
- **Glossary / Docs**: in-app documentation rendered from `docs/`

### Command-line MPPT simulation

```bash
python -m simulators.mppt_sim --algo pando
```

If you pass an invalid algorithm name, the simulator prints the valid options.

### Engine/source simulator (utility)

```bash
python -m simulators.source_sim
```

(Useful when you’re isolating environment/profile behavior from MPPT behavior.)

---

## Documentation

Aurora keeps documentation **in the repo** and **inside the app**.

- In-app: open **Glossary / Docs** tab
- In-repo: `docs/`

Key docs:
- `docs/usage.md` — how to run / common workflows
- `docs/architecture.md` — data flow + module boundaries
- `docs/api.md` — controller interfaces and expectations
- `docs/glossary.md` — vocabulary + “how to extend” playbook

---

## Extending Aurora (common paths)

### Add a new MPPT algorithm
- Implement in `core/mppt_algorithms/`
- Register it with the algorithm registry used by the simulator/UI (see existing algorithms for the exact pattern)
- Test via:
  - Live Bench (UI)
  - `python -m simulators.mppt_sim --algo <name>`
  - Benchmarks

### Add a benchmark scenario or metric
- Scenarios: `benchmarks/scenarios.py`
- Metrics: `benchmarks/metrics.py`
- Execution: `benchmarks/runner.py`

### Add a new dashboard tab
- Implement a `QWidget` in `ui/desktop/`
- Wire it into `ui/desktop/main_window.py`

---

## Data & Outputs

- Logged runs: `data/runs/`
- Benchmark outputs: `data/benchmarks/`

These outputs are intended to be:
- deterministic (given identical inputs)
- machine-readable
- comparable across algorithms and runs

---

## Contributing / Onboarding

If you’re modifying the system, read these in order:
1. `docs/glossary.md`
2. `docs/architecture.md`

Aurora stays maintainable when these boundaries remain clean:
- **Engine orchestrates**
- **PV model computes**
- **Controller decides**
- **UI displays**
- **Benchmarks compare**

---

## Acknowledgements

Developed for **Longhorn Racing Solar** (The University of Texas at Austin)