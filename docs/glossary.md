

# Aurora Glossary & Developer Guide

This document is the **single source of truth** for understanding, using, and extending Aurora.

If you are new:
1. Start at **Conceptual Overview**
2. Read **How Aurora Runs**
3. Then jump to **How to Modify or Extend Aurora**

This page is rendered directly inside the Aurora UI under the **Glossary / Docs** tab.

---

## Conceptual Overview

### What is Aurora?
Aurora is a **time-stepped photovoltaic (PV) system simulator** with pluggable MPPT controllers and a real-time desktop UI. It allows users to study how solar arrays behave under changing conditions and how control algorithms respond.

Aurora is not just a visualizer — it is a **controlled experimentation platform**.

---

### Core Problem Aurora Solves
In real solar arrays:
- Irradiance is non-uniform
- Temperature varies spatially and temporally
- IV curves can have **multiple local maxima**
- MPPT algorithms can get trapped or oscillate

Aurora lets you **observe, measure, and compare** MPPT behavior under these conditions.

---

## Key Concepts (Plain English)

### PV Cell
The smallest modeled unit. Uses a **single-diode model** to convert irradiance and temperature into current–voltage behavior.

### Substring
A group of PV cells protected by a bypass diode. Substrings are the source of **partial shading effects** and multi-peak PV curves.

### String
A series connection of substrings. Voltage adds, current is shared.

### Array
A collection of strings. This is what the MPPT controller interacts with.

---

### IV Curve
A curve showing current vs voltage for the array at a fixed moment in time.

### PV Curve
A curve showing power vs voltage.  
The **maximum power point (MPP)** lies at the peak of this curve.

---

### MPP vs GMPP
- **MPP**: Any local maximum in the PV curve
- **GMPP**: The *global* maximum power point

Under partial shading, multiple MPPs can exist.

---

### MPPT (Maximum Power Point Tracking)
An algorithm that adjusts operating voltage (or duty cycle) to track the MPP over time.

Aurora treats MPPT algorithms as **controllers**, not black boxes.

---

## How Aurora Runs (Simulation Lifecycle)

Aurora advances in **discrete time steps**.

At each time step:

1. **Environment Update**
   - Irradiance and temperature are read
   - Values come from:
     - Manual sliders
     - CSV profiles
     - Benchmark scenarios

2. **Array Evaluation**
   - The array model computes IV and PV curves
   - Electrical state is fully deterministic

3. **Controller Step**
   - The MPPT algorithm receives measurements
   - It proposes a new operating point

4. **State Update**
   - Voltage/current are applied
   - Power is computed

5. **Logging & UI Update**
   - Values are logged
   - UI plots are refreshed

This loop lives in the **engine**.

---

## Important Files & Directories

### `engine/`
Contains the simulation loop and array evaluation logic.

- Responsible for:
  - Time stepping
  - Calling controllers
  - Producing measurements

If something affects *everything*, it probably lives here.

---

### `array/`
Physical PV modeling:
- Cells
- Substrings
- Strings
- Arrays

Changes here affect **electrical realism**.

---

### `controllers/`
MPPT algorithms live here.

Each controller:
- Inherits from a common base
- Implements a step/update method
- Maintains internal state

If you want to add an algorithm, start here.

---

### `simulators/`
Command-line entry points.

Useful for:
- Batch runs
- Debugging algorithms without the UI
- Automated testing

---

### `benchmarks/`
Everything related to **comparative evaluation**.

- `scenarios.py` → defines test environments
- `metrics.py` → defines performance metrics
- `runner.py` → executes and aggregates results

---

### `ui/desktop/`
All PyQt dashboards:
- Live Bench
- Benchmarks
- Glossary / Docs

UI dashboards are **views**, not logic owners.

---

## How to Use Aurora (Practically)

### Sliders vs CSV Profiles
- **Sliders**: manual, interactive control
- **CSV profiles**: deterministic, reproducible environments

When a CSV profile is active:
- Sliders are disabled
- Environment comes entirely from the profile

---

### Typical Workflow
1. Start with sliders to understand behavior
2. Switch to CSV for repeatability
3. Benchmark algorithms
4. Analyze logs

---

## How to Add a New MPPT Algorithm

1. Create a new file in `controllers/`
2. Inherit from the base controller
3. Implement the `step()` (or equivalent) method
4. Register the algorithm
5. Test via:
   - CLI (`simulators/`)
   - Live Bench
   - Benchmarks tab

Follow existing controllers as templates.

---

## How to Modify or Extend Aurora

### Add a New UI Panel
1. Create a new dashboard in `ui/desktop/`
2. Inherit from `QWidget`
3. Wire it into `main_window.py`

---

### Add New Metrics
1. Edit `benchmarks/metrics.py`
2. Define the metric computation
3. Ensure it is logged and displayed

---

### Change Physical Modeling
Modify files in `array/`.

⚠️ Changes here affect *all* simulations.

---

## Common Pitfalls

- Confusing **MPP** with **GMPP**
- Forgetting to register a new controller
- Modifying UI code to change simulation behavior
- Comparing algorithms without identical scenarios

---

## Design Philosophy

Aurora follows these principles:
- **Separation of concerns**
- **Deterministic simulation**
- **Explicit state**
- **Extensibility over cleverness**

If you feel lost, re-read this document and `architecture.md`.

---

## Where to Go Next

- `docs/usage.md` → Running Aurora
- `docs/architecture.md` → Deep system design
- `docs/api.md` → Controller interfaces

This glossary is meant to evolve with the codebase.
# Aurora Glossary & Developer Guide

This document is the **authoritative reference** for understanding, using, and extending Aurora.
It is intentionally technical, but written to be readable by someone new to the codebase.

If you are onboarding:
1. Read **Conceptual Overview**
2. Study **Simulation Lifecycle**
3. Skim **Codebase Mapping**
4. Use **Extension Guides** when modifying the system

This file is rendered directly inside the Aurora UI under the **Glossary / Docs** tab.

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
The **maximum power point (MPP)** corresponds to the global or local maximum of this curve.

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