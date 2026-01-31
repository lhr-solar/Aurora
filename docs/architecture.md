# Aurora Architecture

This document provides a **technical, end‑to‑end description of Aurora’s internal architecture** and execution model.
It is intended for contributors who want to modify system behavior, add algorithms, or reason precisely about simulation correctness.

If you only read one technical document before touching code, read this one.

---

## Design Objectives

Aurora’s architecture is guided by four explicit goals:

1. **Deterministic execution**  
   Given identical inputs (environment, configuration, algorithm), the simulation produces identical outputs.

2. **Strict separation of concerns**  
   Physics, control logic, orchestration, UI, and benchmarking are isolated by design.

3. **Explicit state and data flow**  
   All state transitions occur in known locations; no implicit globals or hidden coupling.

4. **Extensibility under constraint**  
   New algorithms, scenarios, metrics, and dashboards should be easy to add *without* weakening correctness.

---

## High‑Level System Decomposition

Aurora can be understood as five interacting layers:

```
Environment Sources (profiles / scenarios / sliders)
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
Violating these boundaries is the most common source of bugs.

---

## Time‑Stepped Execution Model

Aurora runs as a **discrete‑time simulation** with a fixed timestep `Δt`.

At simulation time step *k → k+1*, the following sequence occurs:

1. **Environment sampling**
2. **Array electrical evaluation**
3. **Controller state update**
4. **Operating point application**
5. **Measurement logging and visualization**

This ordering is **intentional and invariant**.
Reordering steps breaks controller assumptions and invalidates benchmarks.

---

## Simulation Engine (`simulators/engine.py`)

The engine is the **central orchestrator** of the simulation.

### Responsibilities
- Maintain simulation time
- Invoke environment sources
- Trigger PV array evaluation
- Call the MPPT controller
- Emit measurements to loggers and UI consumers

### Non‑Responsibilities
- No physics calculations
- No control logic
- No UI rendering
- No benchmarking logic

The engine is deliberately “thin”: it coordinates *when* things happen, not *how* they work.

---

## Environment Sources

Environment inputs consist of:
- irradiance
- temperature

These values originate from exactly **one active source** at runtime:

- UI sliders (interactive, non‑deterministic)
- CSV profiles (deterministic, replayable)
- Benchmark scenarios (scripted, controlled)

Once selected, an environment source behaves as a **pure function of time**.
The engine does not care where the data came from.

This abstraction enables:
- reproducible experiments
- fair benchmarking
- clean separation between UI and simulation logic

---

## PV Physics Layer (`core/src/`)

This layer implements the **electrical model of the PV system**.

### Hierarchy
```
Cell → Substring → String → Array
```

### Properties
- Stateless with respect to simulation time
- Deterministic for a given environment state
- Safe to evaluate multiple times per timestep

### Responsibilities
- Compute IV curves from irradiance and temperature
- Aggregate electrical behavior hierarchically
- Expose array‑level electrical characteristics

Any modification here affects *every* controller, benchmark, and UI visualization.

---

## MPPT Controllers (`core/mppt_algorithms/`)

Controllers implement **decision‑making logic**.

### Responsibilities
- Consume electrical measurements (V, I, P, history)
- Maintain internal algorithm state
- Propose the next operating voltage (or equivalent control variable)

### Constraints
- Controllers do **not** evaluate physics
- Controllers do **not** interact with UI state
- Controllers are agnostic to execution context (UI vs CLI vs benchmarks)

This isolation makes controllers:
- directly comparable
- testable in isolation
- reusable across all simulation modes

---

## Controller Integration (`core/controller/`)

This layer defines the **contract** between the engine and MPPT algorithms.

It is responsible for:
- enforcing a common controller interface
- adapting controller outputs into engine‑usable control signals
- mediating data passed to algorithms

If a controller can be “plugged in,” it passes through this layer.

---

## Execution Contexts

Aurora supports three execution contexts that all share the same core logic:

### Live UI
- Real‑time stepping
- Interactive environment changes
- Visualization‑first feedback

### CLI Simulation
- Scripted execution
- Fast iteration for debugging
- No UI overhead

### Benchmarking
- Controlled environment scenarios
- Identical initial conditions across algorithms
- Metric‑driven comparison

The engine, physics model, and controllers are identical in all three cases.

---

## Benchmarking Architecture (`benchmarks/`)

Benchmarking is treated as a **first‑class system**, not a bolt‑on.

### Components
- `scenarios.py` — defines time‑indexed environment inputs
- `metrics.py` — computes performance metrics from logged data
- `runner.py` — executes simulations and aggregates results

### Guarantees
- identical environment inputs
- identical simulation parameters
- identical initialization

These guarantees are required for meaningful algorithm comparison.

---

## UI Architecture (`ui/desktop/`)

The desktop UI is a **pure consumer of simulation outputs**.

### Rules
- UI code never modifies physics
- UI code never implements control logic
- UI reflects state, it does not own it

### Dashboards
- **Live Bench** — real‑time simulation inspection
- **Benchmarks** — comparative results and metrics
- **Glossary / Docs** — embedded documentation

Keeping the UI passive prevents subtle correctness bugs.

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

Logs are the canonical source of truth for benchmarking.

---

## Common Architectural Failure Modes

Avoid:
- embedding logic in UI files
- letting controllers depend on UI state
- mixing benchmarking logic into the engine
- introducing implicit or global state
- altering timestep ordering

If a change feels convenient but blurs boundaries, it is likely incorrect.

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

This document should evolve as the system evolves.