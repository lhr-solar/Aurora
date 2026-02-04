

# Output Interpretation Guide

This document explains how to correctly interpret Aurora’s simulation outputs.  
It is intended to prevent common misreadings of plots, metrics, and controller behavior—especially when comparing MPPT algorithms.

If results appear unintuitive, unrealistic, or “too perfect,” this document should be consulted before debugging code.

---

## 1. Core Output Signals

Aurora produces three primary physical outputs at every timestep:

- **Voltage (`V_pv`)** – actual PV terminal voltage
- **Current (`I_pv`)** – PV output current
- **Power (`P`)** – electrical power delivered by the array

Power is **not independently controlled**. It is derived as:

\[
Power = Voltage * Current
\]

The MPPT controller directly influences **voltage**, while current is largely dictated by environmental conditions.

---

## 2. How to Read Voltage, Current, and Power Together

Correct interpretation requires viewing these signals jointly.

### Key relationships
- Voltage is the **control variable**
- Current is primarily **environment-driven**
- Power is a **dependent outcome**

### Interpretation rules
- Stable voltage + fluctuating power  
  → Environment (irradiance / temperature) is changing  
- Oscillating voltage + flat power  
  → Controller probing near MPP (often expected)  
- Voltage converges but power remains low  
  → Likely trapped in a local MPP  

---

## 3. Convergence vs Oscillation

Not all oscillation indicates failure.

### Convergence
- Voltage approaches a stable operating region
- Tracking error decreases over time

### Limit-cycle oscillation (expected)
- Small, bounded voltage oscillations near MPP
- Indicates active exploration and robustness

### Instability
- Increasing oscillation amplitude
- Runaway voltage behavior
- Power degradation over time

**Rule of thumb:**  
Small oscillations near peak power are acceptable; growing oscillations are not.

---

## 4. Local MPP vs Global MPP Behavior

Under partial shading or complex profiles, multiple maxima may exist.

### Signs of local MPP trapping
- Voltage plateaus early
- Power flattens below known GMPP reference
- Minimal voltage exploration after convergence

### Signs of GMPP escape
- Sudden voltage relocation
- Followed by a sustained power increase
- Often accompanied by brief instability

This behavior is **desirable** in robust MPPT designs.

---

## 5. Transient vs Steady-State Behavior

Startup behavior visually dominates plots but should not dominate evaluation.

### Transient phase
- Initial convergence
- High exploration
- Non-representative oscillations

### Steady state
- Stable voltage region
- Bounded oscillation
- Meaningful performance comparison

Benchmarks should either:
- Exclude the transient window, or
- Report metrics separately for transient and steady-state phases

---

## 6. Tracking Metrics and Performance Evaluation

Single-point metrics can be misleading.

### Common metrics
- Mean power
- Peak power
- Mean tracking error
- Time-to-convergence
- Total energy harvested (integral of power)

### Critical insight
> Two algorithms can reach the same peak power but harvest very different total energy.

Energy-based metrics are often the most representative of real performance.

---

## 7. Environment-Driven vs Controller-Driven Changes

Understanding causality is critical.

### Environment-driven changes
- Irradiance step → current changes first
- Temperature drift → slow voltage shift
- Power responds indirectly

### Controller-driven changes
- Voltage moves first
- Power follows
- Current adjusts as a consequence

If voltage changes without environmental input, the controller is acting.

---

## 8. CSV Profile Output Interpretation

When CSV profiles are enabled:

- Inputs are deterministic
- Noise is minimized
- Repeatability is expected

### Implications
- Smooth output does **not** imply superior control
- Identical traces across runs are correct behavior
- Differences between algorithms reflect control logic, not randomness

---

## 9. Diagnostic Overlays (Advanced)

Some outputs may include optional overlays:

- Reference voltage (`V_ref`)
- Oracle MPP voltage (`V_mpp`)
- Error bands or convergence markers

These overlays are **diagnostic tools**, not performance signals, and should not be confused with physical outputs.

---

## 10. When Results Are Suspicious

Certain outcomes indicate configuration or interpretation errors.

### Red flags
- Voltage exactly equals `V_mpp` at all timesteps
- Zero oscillation with high update frequency
- Identical performance across all algorithms under shading
- Power exceeding the PV model’s theoretical maximum

### Likely causes
- Oracle voltage accidentally applied
- UI / debug override still enabled
- Reference voltage plotted instead of applied voltage
- Controller state resetting each timestep

---

## 11. Quick Interpretation Cheat Sheet

- Power drops but voltage is steady → environment change
- Voltage jumps suddenly → controller action
- Smooth traces under CSV → expected
- “Perfect” tracking → verify no oracle leakage
- Large oscillations → tuning or stability issue

---

## Final Guidance

Aurora’s outputs are physically grounded.  
Correct interpretation depends on understanding **what is controlled**, **what is constrained**, and **what is derived**.

When in doubt, verify:
1. Which voltage is being observed
2. Whether overrides are enabled
3. Whether behavior is transient or steady-state