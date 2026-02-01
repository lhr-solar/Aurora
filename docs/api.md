# Aurora API & Contracts

This document defines the **formal interfaces and invariants** that make Aurora composable, testable, and benchmarkable.

It is the authoritative reference for contributors writing or modifying:
- MPPT algorithms (`core/mppt_algorithms/`)
- controller glue (`core/controller/`)
- simulators (`simulators/`)
- benchmark tooling (`benchmarks/`)

If an algorithm “sort of works” in the UI but behaves inconsistently in CLI or benchmarks, a contract in this document is almost certainly being violated.

---

## Global Conventions

### Units (Mandatory)

Aurora assumes the following units at all public interfaces:

- Time: **seconds (s)**
- Irradiance: **W/m²**
- Temperature: **°C**  
  (may be converted internally to Kelvin where required by the physics model)
- Voltage: **V**
- Current: **A**
- Power: **W**

All conversions must occur **at module boundaries**. No implicit unit changes are allowed across interfaces.

---

### Determinism

- The **PV physics layer** (`core/src/`) must be deterministic for a given `(G, T)` pair.
- MPPT controllers must be deterministic given an identical measurement stream.
- Live UI execution may appear nondeterministic due to event-loop timing and user interaction.
- **Benchmarks and CLI simulations must be deterministic**.

If a controller uses randomness, it must:
- expose an explicit seed, and
- be seeded deterministically for benchmarks.

---

## Controller Model (Conceptual)

Aurora models MPPT algorithms as **stateful discrete-time controllers**.

At each timestep:

1. The engine evaluates the PV array at the current operating point
2. A measurement bundle is produced
3. The controller consumes measurements and updates internal state
4. The controller proposes the next operating point

Controllers do **not**:
- evaluate physics
- access UI state
- depend on execution context

---

## Controller Interface (Required Contract)

Each MPPT algorithm must implement a controller class that conforms to the following **logical interface**.

> Exact base-class names may evolve, but the semantics below are invariant.

### Required Methods

#### Initialization
```python
controller = Controller(config: dict)
```

- Must not perform expensive computation
- Must not depend on environment values
- Must not allocate global state

---

#### Reset / Initialize
```python
controller.reset(initial_voltage: float)
```

- Clears all internal state
- Sets the controller to a known starting condition
- Called at the beginning of **every run** (UI, CLI, benchmarks)

---

#### Step
```python
V_next = controller.step(measurement: dict)
```

Consumes measurements from the engine and returns the next control action.

**Input (`measurement`)** — minimum required fields:
```text
t   : float   # simulation time (s)
dt  : float   # timestep (s)
G   : float   # irradiance (W/m²)
T   : float   # temperature (°C)
V   : float   # current operating voltage (V)
I   : float   # current operating current (A)
P   : float   # current operating power (W)
```

Optional fields (may or may not be present):
- rolling history buffers
- estimated gradients (e.g., dP/dV)
- IV or PV curve samples

Controllers **must not assume** optional fields are present.

**Output (`V_next`)**
- Must be a finite scalar voltage (float)
- Represents the controller’s proposed next operating point

---

#### Finalize (Optional)
```python
summary = controller.finalize()
```

- Called at end of run (if implemented)
- May return diagnostic or summary information
- Must not mutate controller state after finalization

---

## Control Signal Semantics

Aurora operates primarily in **voltage-control mode**.

### Bounds and Clipping
- The engine/controller glue enforces voltage bounds:
  ```
  0 ≤ V ≤ V_oc_estimate
  ```
- Controllers must tolerate clipping gracefully:
  - no integrator windup
  - no assumption that `V_next` was applied exactly

Returning NaN, inf, or nonsensical voltages is a contract violation.

---

## Minimal Reference Controller (Canonical Example)

The example below illustrates the **minimum correct implementation** of an MPPT controller in Aurora.
It is intentionally simple and should be used as a structural reference, not a performant algorithm.

```python
class ExampleController:
    """
    Minimal voltage-based MPPT controller example.
    Demonstrates the required lifecycle and contracts.
    """

    def __init__(self, config: dict):
        # Read configuration only (no environment access here)
        self.step_size = float(config.get("step_size", 0.5))
        self.direction = +1
        self.prev_power = None

    def reset(self, initial_voltage: float):
        # Clear all internal state between runs
        self.V = float(initial_voltage)
        self.prev_power = None
        self.direction = +1

    def step(self, measurement: dict) -> float:
        # Required fields (guaranteed by contract)
        V = measurement["V"]
        P = measurement["P"]

        # First step: no comparison possible yet
        if self.prev_power is None:
            self.prev_power = P
            return V + self.direction * self.step_size

        # Simple perturb-and-observe logic
        if P < self.prev_power:
            self.direction *= -1

        self.prev_power = P
        return V + self.direction * self.step_size

    def finalize(self):
        # Optional: return diagnostics
        return {
            "final_voltage": self.V,
            "final_power": self.prev_power,
        }
```

### Why This Example Matters

This example demonstrates several **non-negotiable contracts**:
- All controller state is owned by the controller instance
- `reset()` is the only place state is initialized per run
- `step()` is pure with respect to external state
- Returned voltages may be clipped by the engine and must tolerate it

If your controller follows this structure, it will work correctly in:
- the desktop UI
- CLI simulations
- benchmark runs

---

## Algorithm Registration

Algorithms must be registered to be discoverable by both the CLI and the UI.

### Naming Rules
- Lowercase
- CLI-safe (no spaces)
- Stable over time

Examples:
- `pando`
- `ruca`
- `nl_esc`
- `sweep`

### Registry Requirements
To be discoverable:
- The module must be importable without side effects
- The controller class must be registered in the algorithm registry
- Registration must occur at import time

If an algorithm does not appear in the UI or CLI, it is almost always unregistered.

---

## Simulator Interfaces

### MPPT Simulator (`simulators/mppt_sim.py`)

The MPPT simulator is the canonical non-UI execution path.

Responsibilities:
- Parse CLI arguments (algorithm, duration, timestep, environment source)
- Construct engine and controller instances
- Execute the discrete-time loop
- Persist outputs to `data/runs/`

**Invariant:**  
Given deterministic inputs, CLI runs must be reproducible bit-for-bit (within floating-point tolerance).

---

### Source Simulator (`simulators/source_sim.py`)

Used to debug and validate environment sources independently of MPPT logic.

Responsibilities:
- Load and validate CSV profiles or scripted sources
- Step through time deterministically
- Emit `(t, G, T)` streams and diagnostics

---

## Benchmarking Contracts (`benchmarks/`)

Benchmarking relies on **strict isolation and repeatability**.

### Scenarios (`benchmarks/scenarios.py`)
A scenario defines environment as a deterministic function of time.

Requirements:
- Deterministic mapping `t → (G, T)`
- Fixed duration and recommended timestep
- No dependence on controller behavior

Scenarios must never:
- inspect controller internals
- adapt to controller outputs

---

### Metrics (`benchmarks/metrics.py`)
Metrics are pure functions over logged outputs.

A metric must:
- consume only run logs or measurement streams
- return a scalar or structured summary
- be comparable across algorithms

Examples:
- convergence time to within ε of GMPP
- harvested energy relative to oracle
- steady-state oscillation magnitude
- average tracking efficiency

---

### Runner (`benchmarks/runner.py`)
The runner orchestrates benchmarking.

Responsibilities:
- Instantiate fresh controllers per run
- Execute scenarios across algorithms
- Aggregate metrics without state leakage

Sharing controller instances across runs is a correctness violation.

---

## Data Model: Runs and Outputs

### Run Logs (`data/runs/`)

Run logs are the **canonical record** of a simulation.

A run typically includes:
- time series of `(t, G, T, V, I, P)`
- controller outputs and exposed diagnostics
- metadata (algorithm name, timestep, profile/scenario, timestamp)

---

### Benchmark Outputs (`data/benchmarks/`)

Benchmark outputs include:
- per-algorithm per-scenario metric tables
- aggregate summaries
- plots (if enabled)

---

## Safety and Correctness Requirements

Controllers must never:
- crash on NaNs (they must detect and recover)
- emit voltages outside physical bounds
- assume monotonic PV curves

Physics code must never:
- mutate global configuration
- depend on wall-clock time

Benchmark code must never:
- reuse controller objects across runs
- mix environment sources within a comparison

---

## Extension Checklists

### Adding a New MPPT Algorithm
1. Implement controller in `core/mppt_algorithms/`
2. Conform strictly to the controller interface
3. Register the algorithm
4. Validate in order:
   - CLI with constant environment
   - CLI with deterministic CSV profile
   - Live Bench (sanity only)
   - Benchmarks (comparison)

---

### Adding a New Scenario
1. Define deterministic `(G(t), T(t))` in `benchmarks/scenarios.py`
2. Specify duration and timestep
3. Register with the benchmark runner/UI

---

### Adding a New Metric
1. Implement metric in `benchmarks/metrics.py`
2. Consume only logs (no live state)
3. Integrate into runner aggregation and UI display

---

## Practical Debugging Guidance

- UI vs CLI mismatch → check environment source and timestep
- Algorithm not listed → check registration
- Noisy benchmarks → verify determinism and state isolation

---

## Pointers

- `docs/glossary.md` — terminology and workflows
- `docs/architecture.md` — execution model and invariants
- `docs/usage.md` — running and reproducing experiments

This document should evolve alongside the codebase.