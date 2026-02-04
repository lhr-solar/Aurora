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

## Live Bench UI Control Reference

This section defines every user-facing control in the Live Bench panel.
Each control maps directly to simulation state, engine behavior, or logging configuration.

---

### Source Controls

#### Irradiance (W/m²)
Controls the incident solar irradiance applied to the PV array.

- Represents plane-of-array irradiance
- Directly affects:
  - Photocurrent generation
  - Available electrical power
- Higher values increase current and power
- Lower values reduce current and can expose local maxima under partial shading

When a CSV profile is active, this value is ignored and slider input is disabled.

---

#### Temperature (°C)
Controls the PV cell temperature used in the electrical model.

- Affects semiconductor behavior
- Higher temperatures:
  - Reduce open-circuit voltage
  - Shift the MPP to lower voltages
- Lower temperatures increase voltage and efficiency

Applied uniformly across the array unless overridden by a CSV profile.

---

### MPPT Controls

#### Algorithm Selector
Dropdown used to select the active MPPT controller.

- Determines the control law used to adjust operating voltage
- Switching algorithms:
  - Resets internal controller state
  - Restarts tracking from initial conditions

Only registered algorithms appear in this list.

---

#### Continuous Mode
Enables continuous controller updates during simulation runtime.

- When enabled:
  - The controller updates every simulation timestep
- When disabled:
  - The controller may operate in stepped or episodic modes
  - Used primarily for debugging or algorithm development

Most algorithms are designed to run in continuous mode.

---

### CSV Profile Controls

#### Use CSV Profile
Toggles deterministic environment playback using a CSV file.

- When enabled:
  - Irradiance and temperature sliders are disabled
  - Environment inputs are sourced exclusively from the CSV
  - Simulation becomes reproducible
- When disabled:
  - Environment inputs are controlled by UI sliders

This option is required for benchmarking and fair comparison.

---

#### CSV Path
File path pointing to the active CSV profile.

- CSV must define time-indexed irradiance and temperature values
- Path is relative to the project root by default
- Invalid or malformed CSVs will prevent simulation start

---

#### Browse…
Opens a file picker to select a CSV profile from disk.

---

#### Profile Editor…
Launches the CSV Profile Editor.

- Used to:
  - Create new environment profiles
  - Edit existing profiles
  - Visualize irradiance and temperature timelines

Changes made here do not affect the simulation until the profile is saved and selected.

---

#### Save as Profile…
Saves the current environment configuration as a new CSV profile.

- Useful for capturing exploratory scenarios
- Generated profiles can be reused for benchmarking

---

### Simulation Timing Controls

#### Time (s)
Total simulated time horizon.

- Defines how long the simulation runs in simulated seconds
- Does not represent wall-clock time
- Used for:
  - Benchmark normalization
  - Transient response analysis

---

#### Timestep (`dt`, seconds)
Simulation resolution.

- Smaller `dt`:
  - Higher numerical fidelity
  - Increased computational cost
- Larger `dt`:
  - Faster execution
  - Risk of instability or missed dynamics

`dt` directly influences MPPT behavior and must be identical across benchmark runs.

---

### Output and Logging Controls

#### CSV Output Filename
Specifies the filename used to save simulation results.

- Output is written to the data directory
- Contains:
  - Time
  - Voltage
  - Current
  - Power
  - Environment variables
  - Optional reference data

Used for post-processing and analysis.

---

#### Compute GMPP Reference
Enables computation of the Global Maximum Power Point reference.

- Performs a full PV curve evaluation at each timestep
- Used to:
  - Quantify MPPT tracking error
  - Measure convergence quality
- Increases computational cost

Required for benchmarking accuracy metrics.

---

#### Terminal: Stream Samples
Streams simulation samples to the terminal during runtime.

- Intended for debugging
- Can be disabled to reduce console noise

---

#### Terminal Period (s)
Controls how frequently samples are printed to the terminal.

- Smaller values:
  - Higher verbosity
  - Greater overhead
- Larger values:
  - Reduced output
  - Cleaner logs

---

#### Tick (ms)
UI refresh interval.

- Controls how often plots and indicators update
- Does not affect simulation physics or controller logic
- Larger values improve performance on slower machines

Purely a visualization parameter.

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