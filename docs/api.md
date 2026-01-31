

# Aurora API & Contracts

This document specifies the **interfaces (contracts) that make Aurora composable**.

It is aimed at contributors writing or modifying:
- MPPT algorithms (`core/mppt_algorithms/`)
- controller glue (`core/controller/`)
- simulators (`simulators/`)
- benchmark tooling (`benchmarks/`)

If you implement a new algorithm and it “kind of works” but behaves strangely in the UI or benchmarks, it’s usually because one of these contracts is being violated.

---

## Conventions

### Units
Aurora assumes the following default units throughout the simulation stack:

- Time: **seconds (s)**
- Irradiance: **W/m²**
- Temperature: **°C** (internally may be converted to K where needed)
- Voltage: **V**
- Current: **A**
- Power: **W**

If a module uses different units internally, it must convert at the boundary.

### Determinism
- The **PV physics layer** (`core/src/`) must be deterministic for a given `(G, T)`.
- MPPT algorithms should be deterministic given the same measurement stream.
- Live UI runs can appear nondeterministic due to event-loop timing; benchmarking must be deterministic.

---

## Controller Contract (What MPPT Algorithms Must Implement)

Aurora treats each MPPT algorithm as a **stateful controller** that receives measurements and proposes a new operating point.

Conceptually:

1. The engine evaluates the array at the current operating point
2. The controller receives measurements (V, I, P, plus metadata)
3. The controller returns the next target operating point

### Required Properties
A controller must:
- Maintain its own internal state (no global state)
- Be able to reset cleanly between runs
- Never depend on UI widgets or Qt timing
- Return outputs that are safe (finite, within allowable bounds)

### Expected Lifecycle
A controller is expected to support this lifecycle:

1. **Construction**: instantiate with config
2. **Reset/Initialize**: clear state, set initial operating point
3. **Step**: consume measurements, return next control action
4. **Finalize** (optional): produce summary

---

## Control Signal: Operating Point

Aurora primarily operates controllers in a **voltage-control** style.

### Control Output
A controller returns one of:
- a **target voltage** `V_target`
- (in some controllers) a structured object describing a step/update

**Invariant:** controller outputs must be finite and physically meaningful.

### Bounds
The engine/controller glue enforces voltage bounds based on array configuration (e.g., `0 ≤ V ≤ V_oc_estimate`).

Controllers should still behave well when clipped:
- avoid integrator windup
- avoid assuming the returned voltage was accepted exactly

---

## Measurement Contract

At each time step, the engine provides the controller with a measurement bundle.

### Minimum Measurements
Controllers can assume these exist:
- `t`: simulation time (s)
- `dt`: timestep (s)
- `G`: irradiance (W/m²)
- `T`: temperature (°C)
- `V`: operating voltage (V)
- `I`: operating current (A)
- `P`: operating power (W)

### Optional / Advanced Measurements
Depending on configuration and UI mode, the controller may also receive:
- recent history buffers (rolling windows)
- estimated gradients (e.g., dP/dV)
- IV/PV curve samples (for algorithms that do global search)

Controllers must gracefully handle missing optional fields.

---

## Algorithm Registration

Algorithms are exposed to both the CLI and the UI through a registration mechanism.

### Naming Rules
- Algorithm names must be lowercase and CLI-safe (no spaces)
- Name should match the folder/file name where possible

Examples:
- `pando`
- `ruca`
- `nl_esc`

### Registry Contract
To be discoverable:
- The algorithm must be importable without side effects
- The algorithm must be registered in the algorithm registry used by `simulators/mppt_sim.py` and the UI selector

If an algorithm is not listed, 99% of the time it was never registered.

---

## Simulator APIs

### `python -m simulators.mppt_sim`
The MPPT simulator is the canonical CLI entry point.

Responsibilities:
- parse arguments (algorithm name, run length, dt, environment source)
- construct the engine + controller
- execute the time-step loop
- write results to `data/runs/`

**Contract:** CLI runs should be reproducible when given deterministic environment inputs.

### `python -m simulators.source_sim`
The source simulator is used to isolate/debug environment playback.

Responsibilities:
- load and validate environment sources (CSV)
- step through time
- output the environment stream (and optionally diagnostics)

---

## Benchmark Contracts (`benchmarks/`)

Benchmarking relies on strict contracts to produce fair comparisons.

### Scenarios (`benchmarks/scenarios.py`)
A scenario defines environment as a function of time:

- A scenario must be deterministic
- It must return `(G(t), T(t))` for each `t`
- It should define its duration and recommended `dt`

Scenarios should not:
- inspect controller internals
- adapt to algorithm outputs

### Metrics (`benchmarks/metrics.py`)
Metrics are functions over logged outputs.

A metric must:
- consume only run logs / measurement streams
- return a scalar or structured summary
- be comparable across algorithms

Examples of useful metric families:
- convergence time to within ε of GMPP
- energy harvested relative to oracle
- steady-state oscillation amplitude
- tracking efficiency over the run

### Runner (`benchmarks/runner.py`)
The runner orchestrates:
- selecting scenarios
- selecting algorithms
- executing runs
- aggregating metrics

**Invariant:** the runner must not leak state across runs.

---

## Data Model: Runs and Outputs

### Run Logs (`data/runs/`)
Run logs are the canonical record of a simulation.

A run typically contains:
- time series of `(t, G, T, V, I, P)`
- controller outputs and any exposed internal signals
- metadata (algorithm name, scenario/profile, dt, timestamp)

### Benchmark Outputs (`data/benchmarks/`)
Benchmark outputs contain:
- per-algorithm per-scenario metric tables
- summaries and rankings
- plots (if enabled)

---

## Safety and Correctness Requirements

Controllers must never:
- crash on NaNs (they should detect and recover)
- emit voltages outside allowable bounds
- assume monotonic PV curves

Physics code must never:
- mutate global configuration in-place
- depend on wall-clock time

Benchmark code must never:
- reuse controller objects across different algorithms
- mix different environment streams in a comparison

---

## Extension Checklists

### Adding a New MPPT Algorithm
1. Implement in `core/mppt_algorithms/`
2. Confirm it adheres to the measurement + output contract
3. Register it in the algorithm registry
4. Validate in this order:
   - CLI with a constant environment
   - CLI with a deterministic profile
   - Live Bench (sanity)
   - Benchmarks (comparison)

### Adding a New Scenario
1. Define it in `benchmarks/scenarios.py`
2. Ensure determinism and fixed duration
3. Add it to any scenario registry used by the runner/UI

### Adding a New Metric
1. Implement in `benchmarks/metrics.py`
2. Ensure it uses only logs
3. Add to runner aggregation and UI display

---

## Practical Debugging Tips

- If UI and CLI results differ: check environment source (sliders vs profile) and timestep.
- If an algorithm "doesn't show up": it’s almost always missing registration.
- If benchmarking looks noisy: verify all scenarios are deterministic and you are not sharing state between runs.

---

## Pointers

- `docs/glossary.md` — terminology + workflows
- `docs/architecture.md` — execution model and boundaries
- `docs/usage.md` — how to run and reproduce experiments

This API document should evolve as contracts evolve.