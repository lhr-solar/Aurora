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