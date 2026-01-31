

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