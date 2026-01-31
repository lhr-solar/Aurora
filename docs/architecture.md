

# Aurora Architecture

This document explains **how Aurora is structured internally** and **how data flows through the system** during a simulation.

If you want to modify behavior, add algorithms, or debug unexpected results, this is the document to read.

---

## Architectural Goals

Aurora is designed around the following principles:

- **Deterministic simulation**  
  Same inputs produce the same outputs.

- **Separation of concerns**  
  Physics, control, UI, and benchmarking are isolated.

- **Explicit state**  
  No hidden globals, no magic coupling.

- **Extensibility first**  
  New algorithms, scenarios, and dashboards should be easy to add.

---

## High-Level System View

At a high level, Aurora is composed of five layers:

```
Environment (Profiles / Scenarios)
        ↓
Simulation Engine
        ↓
PV Array Model
        ↓
MPPT Controller
        ↓
Logging + UI
```

Each layer has a single responsibility and communicates through well-defined interfaces.

---

## Core Data Flow (Single Time Step)

Aurora runs in **discrete time steps**.  
Each step follows the same sequence:

1. Environment values are updated
2. The array electrical state is evaluated
3. The controller proposes a new operating point
4. Power and state are updated
5. Results are logged and visualized

This order is **intentional** and should not be rearranged casually.

---

## Engine (`engine/`)

The engine owns the **simulation loop**.

Responsibilities:
- Advance simulation time
- Pull environment inputs (irradiance, temperature)
- Evaluate the PV array
- Invoke the MPPT controller
- Emit measurements to UI and loggers

The engine does **not**:
- Know about UI widgets
- Care where environment data originates
- Implement control logic itself

Think of the engine as the **orchestrator**.

---

## Environment Sources

Environment values (irradiance, temperature) can come from:

- Manual UI sliders
- CSV profiles
- Benchmark scenarios

These are mutually exclusive at runtime.

Once an environment source is selected, the engine treats it as a **pure data provider**.

This ensures:
- Repeatability
- Clean benchmarking
- No UI-side state leakage

---

## PV Array Model (`array/`)

This layer implements the **physics**.

Hierarchy:
```
Cell → Substring → String → Array
```

Responsibilities:
- Convert irradiance + temperature into IV curves
- Aggregate electrical behavior hierarchically
- Produce deterministic electrical outputs

Key properties:
- Stateless with respect to time
- Fully deterministic
- Safe to evaluate repeatedly

Any change here affects *every* simulation mode.

---

## MPPT Controllers (`controllers/`)

Controllers implement **decision-making logic**.

Responsibilities:
- Consume measurements (V, I, P, history)
- Maintain internal algorithm state
- Propose the next operating point

Controllers:
- Do **not** evaluate physics
- Do **not** touch UI code
- Are agnostic to execution context (UI vs CLI vs benchmark)

This makes controllers:
- Testable
- Comparable
- Swappable

---

## Simulation Modes

Aurora supports three execution contexts:

### Live UI
- Real-time stepping
- Interactive environment changes
- Visualization-first

### CLI Simulation
- Scriptable execution
- Fast iteration for debugging
- No UI overhead

### Benchmarking
- Controlled scenarios
- Identical conditions across algorithms
- Metrics-driven comparison

All three share the **same engine and controller code**.

---

## Benchmarking Architecture (`benchmarks/`)

Benchmarking is a **first-class system**, not an afterthought.

Components:
- `scenarios.py` → defines environment timelines
- `metrics.py` → computes performance metrics
- `runner.py` → executes simulations and aggregates results

Benchmarks enforce:
- Identical initial conditions
- Identical environment inputs
- Identical simulation parameters

This guarantees fair comparison.

---

## UI Architecture (`ui/desktop/`)

UI code is strictly a **consumer** of simulation outputs.

Rules:
- UI never modifies physics directly
- UI never implements control logic
- UI reflects state, it does not own it

Dashboards:
- Live Bench → real-time simulation view
- Benchmarks → comparative analysis
- Glossary / Docs → embedded documentation

This separation keeps the system debuggable.

---

## Logging & Outputs

Simulation outputs are:
- Timestamped
- Structured
- Saved to `data/runs/`

Properties:
- Machine-readable
- Reproducible
- Comparable across runs

Logs are designed to support:
- Offline analysis
- Visualization
- Regression testing

---

## Common Architecture Mistakes

Avoid:
- Putting logic in UI files
- Letting controllers read UI state
- Mixing benchmarking logic into the engine
- Introducing implicit global state

If something feels “convenient” but breaks separation, it is probably wrong.

---

## Mental Model Summary

If you remember nothing else:

- **Engine orchestrates**
- **Array computes physics**
- **Controller decides**
- **UI displays**
- **Benchmarks compare**

Keep these roles clean, and Aurora stays easy to evolve.

---

## Where to Go Next

- `docs/glossary.md` → terminology & workflows
- `docs/api.md` → controller interfaces
- `benchmarks/` → example comparative experiments