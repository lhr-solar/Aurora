

# Aurora

Aurora is a **time-stepped photovoltaic (PV) simulation and MPPT experimentation platform** developed for Longhorn Racing Solar.

It combines:
- a physics-based PV model (cell → substring → string → array)
- a discrete-time simulation engine
- pluggable MPPT algorithms
- a PyQt6 + **pyqtgraph** desktop UI for interactive debugging and benchmarking

Aurora is designed to be:
- **Educational** – easy to explore and reason about
- **Extensible** – straightforward to add new algorithms, scenarios, and UI panels
- **Research-grade** – suitable for controlled experiments and fair benchmarking

If you’re new, the shortest path is:
1. Run the desktop app
2. Open the **Glossary / Docs** tab
3. Use Live Bench for intuition, then Benchmarks for comparison

---

## What Aurora Simulates

Aurora simulates PV behavior under **time-varying environment conditions** and evaluates how an MPPT controller tracks the maximum power point.

In realistic conditions—especially **partial shading**—the PV power–voltage curve can develop **multiple local maxima**. These non-convexities are where MPPT algorithms can fail in interesting ways (oscillation, slow convergence, local trapping). Aurora is built to make those behaviors observable and measurable.

---

## Core Concepts (Quick Glossary)

- **IV curve**: current vs voltage at a fixed instant
- **PV curve**: power vs voltage at a fixed instant; the peak is the MPP
- **MPP vs GMPP**: local maximum vs global maximum (critical under shading)
- **Environment**: irradiance + temperature, supplied by sliders, CSV profiles, or benchmark scenarios

For precise definitions and workflows, see `docs/glossary.md` (also rendered inside the UI).

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
Aurora includes multiple MPPT implementations and is designed to make adding new ones straightforward. Algorithms exposed through the CLI and UI include:
- `pando` (Perturb & Observe)
- `pso`, `mepo`, `nl_esc`, `ruca`, `sweep`

(See `core/mppt_algorithms/` for implementations.)

### Simulation Engine
- Fixed time-step discrete simulation loop
- Engine orchestrates: environment → array evaluation → controller → logging
- Deterministic playback for profiles and benchmarks
- Run logging for post-analysis (`data/runs/`)

### Desktop UI (PyQt6 + pyqtgraph)
- Live IV/PV plotting
- Irradiance/temperature input via **sliders** or **CSV profiles**
- MPPT operating point and power visualization
- Benchmark dashboard for comparative evaluation
- Embedded documentation tab (**Glossary / Docs**)

---

## Repository Structure (Canonical)

```text
Aurora/
├── Aurora.py              # Primary application entry point
├── core/
│   ├── src/               # PV physics + array hierarchy
│   ├── controller/        # Controller interfaces / glue
│   └── mppt_algorithms/   # MPPT implementations
├── simulators/            # CLI simulators (engine, mppt_sim, source_sim)
├── benchmarks/            # Scenarios, metrics, benchmark runner
├── ui/
│   └── desktop/           # PyQt dashboards (Live Bench, Benchmarks, Docs)
├── profiles/              # CSV environment profiles
├── configs/               # Saved UI/simulation configs (if enabled)
├── docs/                  # Markdown documentation rendered in the UI
├── data/
│   ├── runs/              # Logged simulation runs
│   └── benchmarks/        # Benchmark outputs
└── requirements.txt
```

---

## Installation

### Requirements
- Python **3.10+** recommended
- macOS or Linux preferred (Windows can work, but Qt tooling is more fragile)

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

### Desktop UI (Recommended)

```bash
python3 Aurora.py
```

This launches the main window with tabs:
- **Live Bench** — interactive environment control and plots
- **Benchmarks** — scenario-driven comparison and metrics
- **Glossary / Docs** — in-app documentation from `docs/`

### Command-Line MPPT Simulation

```bash
python -m simulators.mppt_sim --algo pando
```

Passing an invalid algorithm name prints the list of available options.

### Source / Environment Simulator

```bash
python -m simulators.source_sim
```

Useful for isolating and debugging environment or profile behavior.

---

## Documentation

Aurora keeps documentation **in the repository and inside the application**.

- In-app: open the **Glossary / Docs** tab
- In-repo: `docs/`

Key documents:
- `docs/usage.md` — how to run and reproduce experiments
- `docs/architecture.md` — execution model and boundaries
- `docs/api.md` — controller contracts and interfaces
- `docs/glossary.md` — concepts, terminology, and extension guides

---

## Extending Aurora

Common extension paths:
- **New MPPT algorithm** → `core/mppt_algorithms/`
- **New benchmark scenario** → `benchmarks/scenarios.py`
- **New metric** → `benchmarks/metrics.py`
- **New UI dashboard** → `ui/desktop/`

Each path is documented in the Glossary / Docs tab.

---

## Data & Outputs

- Logged simulation runs: `data/runs/`
- Benchmark outputs: `data/benchmarks/`

Outputs are designed to be:
- deterministic (given identical inputs)
- machine-readable
- comparable across algorithms and runs

---

## Contributing / Onboarding

If you are modifying the system, read these first:
1. `docs/glossary.md`
2. `docs/architecture.md`

Aurora stays maintainable when these boundaries remain clean:
- **Engine orchestrates**
- **Physics computes**
- **Controller decides**
- **UI displays**
- **Benchmarks compare**

---

## Acknowledgements

Developed for **Longhorn Racing Solar**  
The University of Texas at Austin