# Aurora Architecture

This document provides a **technical, end‑to‑end description of Aurora’s internal architecture** and execution model.
It is intended for contributors who want to modify system behavior, add algorithms, or reason precisely about simulation correctness and reproducibility.

If you only read one technical document before touching code, read this one.

---

## Architectural Intent

Aurora is designed as a **deterministic, discrete‑time simulation framework** for studying PV systems and MPPT control algorithms.
The architecture prioritizes correctness, reproducibility, and extensibility over convenience.

Four guiding objectives shape all design decisions:

1. **Deterministic execution**  
   Given identical environment inputs, configuration, timestep, and algorithm, Aurora produces identical numerical outputs.

2. **Strict separation of concerns**  
   Physics, control, orchestration, UI, and benchmarking are isolated by design.

3. **Explicit state ownership**  
   Every piece of mutable state has a clear owner; state transitions occur in known locations.

4. **Extensibility under constraint**  
   New algorithms, scenarios, metrics, and dashboards should be easy to add *without weakening invariants*.

---

## High‑Level System Decomposition

Aurora can be understood as five interacting layers:

```
Environment Sources (sliders / CSV profiles / benchmark scenarios)
                ↓
Simulation Engine (time orchestration)
                ↓
PV Physics Model (array evaluation)
                ↓
MPPT Controller (decision logic)
                ↓
Logging, Metrics, and UI
```

Each layer has a single responsibility and communicates through explicit interfaces.
Violating these boundaries is the most common source of subtle bugs.

---

## Time‑Stepped Execution Model

Aurora runs as a **fixed‑timestep discrete‑time simulation** with timestep `Δt`.

At simulation step *k → k+1*, the engine executes the following invariant sequence:

1. **Environment sampling**  
   Sample irradiance `G_k` and temperature `T_k`.

2. **Array electrical evaluation**  
   Evaluate IV and PV characteristics at the current operating point.

3. **Controller update**  
   Pass measurements to the MPPT controller and update its internal state.

4. **Operating‑point application**  
   Apply the controller’s proposed control action (typically a target array voltage).

5. **Logging and visualization**  
   Persist measurements and emit state to UI consumers.

This ordering must not be changed.
Controller behavior, benchmark metrics, and convergence properties all assume this sequence.
Each step only mutates state it owns; no step mutates state owned by another layer.

---

## Simulation Engine (`simulators/engine.py`)

The simulation engine is the **central orchestrator**.

### Responsibilities
- Maintain simulation time and timestep
- Select and query the active environment source
- Invoke PV array evaluation
- Call the MPPT controller interface
- Emit structured measurements to loggers and observers

### Non‑Responsibilities
- Physics calculations
- Control logic
- UI rendering
- Benchmark metric computation

The engine is intentionally *thin*: it defines **when** things happen, not **how** they work.

---

## Environment Sources

Environment inputs consist of:
- irradiance `G` (W/m²)
- temperature `T` (°C)

Exactly **one environment source** is active at runtime:

- **UI sliders**  
  Interactive, exploratory, non‑deterministic (event‑loop driven)

- **CSV profiles**  
  Deterministic, replayable, time‑indexed inputs

- **Benchmark scenarios**  
  Scripted, deterministic environment functions

Once selected, an environment source behaves as a **pure function of simulation time**.
The engine treats all sources identically.

This abstraction enables:
- reproducible experiments
- fair algorithm comparison
- strict separation between UI and simulation logic

---

## PV Physics Layer (`core/src/`)

This layer implements the **electrical model of the PV system**.

### Structural Hierarchy
```
Cell → Substring → String → Array
```

### Properties
- Stateless with respect to simulation time
- Deterministic for a given `(G, T)` pair
- Safe to evaluate multiple times per timestep

### Responsibilities
- Convert `(G, T)` into IV characteristics
- Aggregate electrical behavior hierarchically
- Expose array‑level voltage/current/power behavior

Any modification here affects *all* controllers, benchmarks, and UI visualizations and should be treated as a physics‑level change.

---

## MPPT Controllers (`core/mppt_algorithms/`)

MPPT algorithms are implemented as **stateful controllers**.

### Responsibilities
- Consume electrical measurements `(V, I, P, …)`
- Maintain algorithm‑specific internal state
- Propose the next operating point (typically voltage)

### Constraints
- Controllers do **not** evaluate physics
- Controllers do **not** read or mutate UI state
- Controllers are agnostic to execution context (UI, CLI, benchmarks)

This isolation ensures controllers are:
- directly comparable
- testable in isolation
- reusable across all execution contexts

---

## Controller Integration Layer (`core/controller/`)

This layer defines the **formal contract** between the engine and MPPT algorithms.

It is responsible for:
- enforcing a consistent controller interface
- adapting controller outputs into engine‑safe control signals
- mediating measurement bundles passed to algorithms

If an algorithm can be selected from the CLI or UI, it passes through this layer.

---

## Execution Contexts

Aurora supports three execution contexts that share the same engine, physics, and controllers:

### Live UI
- Real‑time stepping
- Interactive environment manipulation
- Visualization‑first feedback

### CLI Simulation
- Scripted execution
- Fast iteration and debugging
- No UI overhead

### Benchmarking
- Controlled, deterministic environment scenarios
- Identical initialization across algorithms
- Metric‑driven comparison

Differences between contexts exist only in **environment sources and observers**, not in simulation logic.

---

## Benchmarking Architecture (`benchmarks/`)

Benchmarking is treated as a **first‑class system**, not a bolt‑on.

### Components
- `scenarios.py` — deterministic environment definitions over time
- `metrics.py` — functions over logged simulation outputs
- `runner.py` — orchestration and aggregation

### Guarantees
- identical environment inputs
- identical simulation parameters
- isolated controller instances per run

These guarantees are required for meaningful and fair comparison.

---

## UI Architecture (`ui/desktop/`)

The desktop UI is a **passive consumer of simulation state**.

### Rules
- UI code never modifies physics
- UI code never implements control logic
- UI reflects state; it does not own it

### Dashboards
- **Live Bench** — real‑time inspection and debugging
- **Benchmarks** — comparative metrics and plots
- **Glossary / Docs** — embedded documentation

Keeping the UI passive prevents hidden state coupling and nondeterministic behavior.

---

## Logging and Data Outputs

Simulation outputs are:
- timestamped
- structured
- written to `data/runs/` and `data/benchmarks/`

Design goals:
- machine‑readable
- reproducible
- suitable for offline analysis and regression testing

Logs are the canonical source of truth for benchmarking and metrics.

---

## Determinism Caveats

Aurora guarantees determinism **given identical inputs**, but contributors should be aware of:
- floating‑point sensitivity across platforms
- UI event timing (Live UI only)
- controller algorithms that introduce randomness (must be seeded or avoided in benchmarks)

Benchmarks must use deterministic environment sources and controllers.
Contributors should assume that any nondeterminism invalidates benchmark results unless explicitly justified.

---

## Common Architectural Failure Modes

Avoid:
- embedding logic in UI files
- letting controllers depend on UI state
- mixing benchmarking logic into the engine
- sharing controller instances across runs
- altering timestep ordering

If a change feels convenient but blurs ownership boundaries, it is likely incorrect.

---

## Mental Model (Compressed)

If you remember nothing else:

- **Engine orchestrates time**
- **Physics computes electrical behavior**
- **Controller decides control actions**
- **UI visualizes state**
- **Benchmarks compare outcomes**

Maintaining these roles is what keeps Aurora correct and extensible.

---

## Pointers

- `docs/glossary.md` — terminology and workflows
- `docs/api.md` — controller contracts
- `docs/usage.md` — execution and configuration

This document should evolve alongside the system.