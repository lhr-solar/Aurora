# Aurora Glossary & Developer Guide

This document is the **authoritative reference** for understanding, using, and extending Aurora.
It is intentionally technical, but written to be readable by someone new to the codebase.

This file is rendered directly inside the Aurora UI under the **Glossary / Docs** tab.

If you are onboarding:
1. Read **Conceptual Overview**
2. Study **Simulation Lifecycle**
3. Skim **Codebase Mapping**
4. Use **Extension Guides** when modifying the system

---

## Conceptual Overview

### What is Aurora?

Aurora is a **discrete‑time photovoltaic (PV) system simulator** with modular MPPT controllers and an interactive desktop UI.

It is designed to support:
- algorithm development and comparison
- controlled experiments under partial shading
- debugging MPPT behavior at runtime
- reproducible benchmarking

Aurora is not just a plotting tool — it is a **deterministic experimentation framework**.

---

### What Aurora Is Not

Aurora is intentionally **not**:
- a real-time embedded controller
- a hardware-in-the-loop simulator
- a grid-level PV system model
- a UI-driven control system

Aurora is a **discrete-time, software-only experimentation framework**.
Its purpose is to study controller behavior under controlled conditions, not to emulate deployment environments.

---

### Core Problem Being Studied

Real PV arrays exhibit:
- spatially non‑uniform irradiance
- temperature‑dependent electrical behavior
- bypass‑diode‑induced non‑convex PV curves
- multiple local maxima in the power–voltage relationship

Under these conditions, MPPT algorithms may:
- converge slowly
- oscillate near optima
- become trapped in local MPPs
- trade efficiency for stability

Aurora exists to make these behaviors **explicit, observable, and measurable**.

---

## Fundamental Concepts

### PV Cell
The smallest modeled electrical unit.
Aurora uses a **single‑diode equivalent circuit model** with explicit irradiance and temperature dependence.

Outputs:
- IV curve
- temperature‑adjusted electrical parameters

---

### Substring
A series group of PV cells protected by a bypass diode.
Substrings are the primary source of:
- curve discontinuities
- multiple MPPs under partial shading

---

### String
A series connection of substrings.
- Voltage aggregates across substrings
- Current is constrained by the weakest element

---

### Array
A collection of strings.
This is the abstraction exposed to the MPPT controller.

---

### IV Curve
Instantaneous current–voltage relationship of the array at a fixed environment state.

---

### PV Curve
Instantaneous power–voltage relationship.
The **maximum power point (MPP)** corresponds to a local or global maximum of this curve at a fixed environment state.

---

### MPP vs GMPP
- **MPP**: any local maximum of the PV curve
- **GMPP**: the global maximum across the entire voltage domain

Under partial shading, distinguishing between the two is non‑trivial.

---

### MPPT (Maximum Power Point Tracking)
A control algorithm that adjusts the array operating point (typically voltage) over time to maximize extracted power.

In Aurora, MPPT algorithms are implemented as **stateful controllers**, not black boxes.

---

## Simulation Lifecycle (How Aurora Actually Runs)

Aurora advances in **fixed discrete time steps**.

At each simulation step:

1. **Environment Sampling**
   - Irradiance and temperature are sampled
   - Source is one of:
     - UI sliders (interactive)
     - CSV profile (deterministic)
     - Benchmark scenario (scripted)

2. **Array Evaluation**
   - The PV array model computes IV and PV curves
   - Evaluation is deterministic and stateless

3. **Controller Update**
   - The MPPT controller receives measurements
   - Internal controller state is updated
   - A new operating point is proposed

4. **State Application**
   - Voltage/current are applied
   - Instantaneous power is computed

5. **Logging and Visualization**
   - Measurements are logged
   - UI plots and indicators update

This loop is orchestrated by the **simulation engine** and shared across UI, CLI, and benchmark modes.

No lifecycle step mutates state owned by another layer.

---

## Codebase Mapping (Current Structure)

### `core/src/`
Physics and electrical modeling layer:
- PV cells
- substrings
- strings
- arrays

Changes here affect **all simulation modes** and should be treated carefully.

---

### `core/mppt_algorithms/`
MPPT algorithm implementations.

Characteristics:
- stateful
- deterministic given identical inputs
- independent of UI and benchmarking context

Algorithms are registered and exposed to both the CLI and UI.

---

### `core/controller/`
Defines controller interfaces and glue logic between the engine and MPPT implementations.

If an algorithm can “plug in,” it passes through here.

---

### `simulators/`
Command‑line entry points.

Used for:
- fast algorithm iteration
- debugging without UI overhead
- batch experiments

Examples:
- `mppt_sim.py`
- `source_sim.py`

---

### `benchmarks/`
Comparative evaluation framework.

Components:
- `scenarios.py` — environment timelines
- `metrics.py` — performance metrics
- `runner.py` — execution and aggregation

Benchmarks enforce:
- identical initial conditions
- identical environment inputs
- identical simulation parameters

This guarantees fair comparison.

---

### `ui/desktop/`
PyQt6 dashboards.

Dashboards are **pure views**:
- they do not own physics
- they do not implement control logic
- they consume engine outputs

Tabs include:
- Live Bench
- Benchmarks
- Glossary / Docs

---

## Practical Usage Notes

### Sliders vs CSV Profiles
- Sliders are interactive and exploratory
- CSV profiles are deterministic and reproducible

When a CSV profile is active:
- sliders are disabled
- the environment is fully defined by the profile

This distinction is critical for benchmarking.

---

### Typical Workflow
1. Explore behavior using sliders
2. Capture scenarios using CSV profiles
3. Benchmark algorithms under identical conditions
4. Analyze logs in `data/runs/` or `data/benchmarks/`

---

## Extension Guides

### Adding a New MPPT Algorithm
1. Implement the algorithm in `core/mppt_algorithms/`
2. Follow the existing controller interface
3. Register the algorithm
4. Validate via:
   - CLI simulation
   - Live Bench
   - Benchmarks

---

### Adding Metrics
1. Define the metric in `benchmarks/metrics.py`
2. Ensure it consumes logged data only
3. Integrate into the benchmark runner and UI

---

### Adding a UI Dashboard
1. Create a new `QWidget` in `ui/desktop/`
2. Keep it presentation‑only
3. Register it in `main_window.py`

---

### Modifying Physics
Edits in `core/src/` affect **all algorithms and benchmarks**.

Only make these changes if you intend to alter the physical model itself.

---

## Common Pitfalls

- Treating MPP and GMPP as interchangeable
- Letting UI state leak into simulation logic
- Comparing algorithms across mismatched scenarios
- Introducing implicit global state

If a change feels convenient but violates separation of concerns, it is probably incorrect.

---

## Design Philosophy (In One Page)

Aurora prioritizes:
- determinism
- explicit state
- clean boundaries
- extensibility over clever abstractions

A useful mental model:
- **Engine orchestrates**
- **Physics computes**
- **Controller decides**
- **UI displays**
- **Benchmarks compare**

---

## Where to Go Next

- `docs/architecture.md` — detailed system design
- `docs/api.md` — controller contracts
- `docs/usage.md` — execution details

This document should evolve alongside the codebase.