

# Aurora Benchmarking

Aurora’s benchmarking system evaluates MPPT algorithms under controlled scenarios and hardware-style constraints. It produces per-tick logs, derives standardized metrics, and ranks runs in the Benchmarks tab using a composite score.

This doc covers how to **run**, **update**, and **interpret** results.

---

## Files to know

**Core benchmarking**
- `benchmarks/runner.py` — runs suites, writes outputs
- `benchmarks/metrics.py` — converts per-tick records → scalar metrics + score
- `benchmarks/session.py` — session folder + signature utilities

**UI**
- `ui/desktop/benchmarks_dashboard.py` — leaderboard UI (loads session summaries and ranks by score)

---

## What gets written to disk

Benchmark output lives under a benchmarks output directory (default is `data/benchmarks/`).

### Session folders

Aurora groups runs into a persistent **session folder** that is reused while the app is running **as long as the suite signature does not change**.

A new session folder is created when:
- the program restarts, or
- scenarios/budgets (or key suite knobs) change.

Typical layout:

```text
data/benchmarks/
  current_session.json
  latest_session_path.txt
  session_YYYYMMDD_HHMMSS__<hash>/
    session_meta.json
    summaries.jsonl
    records/
      <algo>__<scenario>__<budget>__run_YYYYMMDD_HHMMSS.jsonl
```

What each file is:
- `current_session.json` — active session metadata + signature
- `latest_session_path.txt` — convenience pointer to the active session folder
- `session_meta.json` — metadata for the session folder itself
- `summaries.jsonl` — append-only summary rows (one per algo×scenario×budget run)
- `records/*.jsonl` — per-tick records (optional depending on settings)

---

## Running benchmarks

### From the UI

1. Open the **Benchmarks** tab.
2. Set your output directory (the **Out:** field). Default is `data/benchmarks/`.
3. Select algorithms, scenarios, and budgets.
4. Click **Run Benchmarks**.

After completion:
- the dashboard resolves the active session folder (via `latest_session_path.txt`)
- loads `<session>/summaries.jsonl`
- displays a leaderboard sorted by **Score**

### Loading an older session

In the Benchmarks tab:
- Use the **Session:** path field.
- Click **Browse session…** to pick a session folder.
- It will auto-load that folder’s `summaries.jsonl` and rank by score.

---

## CLI usage (optional)

If you run benchmarks from the command line (if enabled in your repo):

```bash
python -m benchmarks.runner --help
```

A typical run looks like:

```bash
python -m benchmarks.runner \
  --algos hybrid \
  --scenarios partial_shading_step \
  --budgets ideal_1kHz \
  --total-time 1.0
```

The output directory behavior is the same (session folder reuse).

---

## Per-tick record schema (what metrics consume)

Each tick emits a JSON dict. The important keys are:

- `t` — time in seconds
- `v`, `i`, `p` — measured values (post-noise/quant if enabled)
- `p_true` — optional true power (ideal reference)
- `g_strings` — optional list[float] per-string irradiance
- `gmpp` — optional dict with GMPP reference info:
  - `p_gmp_ref` — true reference power (GMPP reference)
  - `eff_step` — true per-step efficiency (`p_true / p_gmp_ref`)
  - `eff_step_meas` — measured per-step efficiency (`p / p_gmp_ref`)
  - legacy: `eff_best`, `eff_best_meas` may also exist
- `perf` — optional compute timing dict:
  - `ctrl_us` — controller compute time (µs)
  - `ref_us` — reference update compute time (µs)
  - `over_budget` — whether the tick exceeded a compute budget

---

## Metrics (what the leaderboard shows)

Metrics are produced by `benchmarks/metrics.py::compute_metrics(records)`.

### Energy ratios

These estimate captured energy relative to the GMPP reference:

- `energy_true_ratio` — integrates true power (uses `p_true` if present)
- `energy_meas_ratio` — integrates measured power (`p`)

Interpretation:
- **1.0** = perfect tracking
- **0.9** = missed ~10% of available energy

### Disturbance detection

- `t_disturb` is detected as the first time:
  1) `g_strings` changes, else
  2) `p_gmp_ref` changes

This is used to define “recovery after a step / disturbance”.

### Settle time

Settle uses efficiency staying above a threshold for a hold window:
- threshold: **0.98**
- hold time: **0.05 s**

Outputs:
- `settle_time_s_true`, `settle_time_s_meas` — global settle time from run start
- `settle_time_s_true_post`, `settle_time_s_meas_post` — absolute settle time computed on the post-disturb slice
- `recovery_settle_s_true`, `recovery_settle_s_meas` — settle time relative to first post-disturb tick

Why recovery may be blank:
- If the scenario keeps changing (e.g., flicker), efficiency may never stay ≥0.98 for a full 0.05 s window.

### Ripple RMS (stability)

- `ripple_rms_true`, `ripple_rms_meas`

Computed as RMS ripple of **power** over the last **0.2 s** window:
- build the power series (true or measured)
- subtract mean
- compute RMS

Interpretation:
- lower is better (less steady-state oscillation)

### Tracking error area (dynamic quality)

- `tracking_error_area_true`, `tracking_error_area_meas`

Defined as efficiency deficit area:

\[
\int \max(0, 1 - \mathrm{eff}(t))\, dt
\]

By default this integrates starting at `t_disturb` (if detected). Lower is better.

### Compute/performance

- `ctrl_us_p50`, `ctrl_us_p95`, `ctrl_us_p99`
- `ref_us_p50`, `ref_us_p95`
- `budget_violations`

Interpretation:
- large `ctrl_us_p95` means the algorithm is heavier/less embedded-feasible
- `budget_violations > 0` means it exceeded the budget on some ticks

---

## Composite score (leaderboard ranking)

Score is computed in `benchmarks/metrics.py::composite_score()`.

Current formula:

\[
\text{score} = 100\cdot E 
- 2\cdot T_{rec}
- 5\cdot R
- 10\cdot A
- 0.001\cdot C
\]

Where:
- \(E\) = `energy_meas_ratio` (default)
- \(T_{rec}\) = `recovery_settle_s_meas`
- \(R\) = `ripple_rms_meas`
- \(A\) = `tracking_error_area_meas`
- \(C\) = `ctrl_us_p95`

### Why score can be negative

Only energy is positive; everything else is a penalty. A score goes negative when penalties outweigh `100 * energy`.

Negative does **not** mean the run failed — it just means it ranks poorly under the current weights.

### Tuning score weights

Edit `benchmarks/metrics.py` in `composite_score()`:
- increase `w_energy` if you want energy capture to dominate
- increase `w_settle` if you care more about recovery speed
- increase `w_ripple` if you care more about stability
- increase `w_area` if you want stronger penalties for spending time off-GMPP
- increase `w_ctrl` if you care more about compute cost

---

## Interpreting results (common patterns)

### High energy, low ripple, low tracking area
Best-in-class tracking and stability.

### High energy, high ripple
Algorithm reaches GMPP but oscillates (often aggressive perturbing).

### Lower energy, low ripple
Stable but may be stuck near local MPP or converges too slowly.

### High tracking error area
Spends significant time away from GMPP (slow recovery, repeated overshoots, or never fully converges).

### Recovery blank (empty)
Usually means:
- disturbance detected, but the settle condition never held long enough (flicker scenarios), or
- threshold/hold are too strict for that scenario.

In those scenarios, prefer interpreting:
- `tracking_error_area_meas`
- `energy_meas_ratio`

---

## Updating scenarios and budgets

### Scenarios
Scenarios define environmental evolution (irradiance/temp, shading, steps/flicker).

Update where scenarios are defined (commonly in benchmarking config or runner scenario registry). Changing scenarios affects the suite signature, so a **new session** is created.

### Budgets
Budgets define sampling frequency, noise/quantization settings, and compute budgets.

Update where budgets are defined (commonly in runner budget registry). Changing budgets also creates a **new session**.

---

## Extending metrics

To add a new metric:

1. Add fields to `MetricSummary` in `benchmarks/metrics.py`.
2. Compute it inside `compute_metrics()`.
3. Ensure `benchmarks/runner.py` copies the new key into `summary.extra` (optional but recommended so UI can show it directly).
4. Add a column to the Benchmarks dashboard table if you want it visible.

Tip: Keep `compute_metrics()` robust (missing fields should degrade gracefully rather than crash).

---

## Debugging

### “reason” shows `None` in terminal
Some controllers don’t emit `debug["reason"]`. The runner’s bench-line formatter falls back to other debug fields like `cold_start`, `branch`, or `algo`.

### Benchmark errors about per-string irradiance length
If you see something like:

```text
ValueError: Per-string irradiance length X != n_strings Y
```

That means scenario shading data doesn’t match the modeled array string count. Fix by ensuring `g_strings` length matches `n_strings` for the scenario, or normalize shading inputs.

---

---

## Score Tuning Cookbook

The composite score is a weighted tradeoff between energy capture, recovery speed, stability, tracking quality, and compute cost.

Current structure:

\[
\text{score} = w_E E - w_T T_{rec} - w_R R - w_A A - w_C C
\]

Where:
- \(E\) = energy ratio
- \(T_{rec}\) = recovery settle time
- \(R\) = ripple RMS
- \(A\) = tracking error area
- \(C\) = ctrl p95 compute time

Below are recommended weight profiles depending on your objective.

### 1️⃣ Energy-first (field performance priority)

Use when maximizing energy harvest is the only objective.

```python
w_energy = 150
w_settle = 1
w_ripple = 2
w_area   = 5
w_ctrl   = 0.0005
```

Effect:
- Energy dominates ranking
- Slow recovery is mildly penalized
- Compute cost barely matters

---

### 2️⃣ Stability-first (hardware-friendly / low oscillation)

Use when ripple and smoothness matter (converter stress, EMI, longevity).

```python
w_energy = 100
w_settle = 2
w_ripple = 15
w_area   = 8
w_ctrl   = 0.001
```

Effect:
- High ripple is strongly punished
- Oscillatory controllers rank lower

---

### 3️⃣ Fast-recovery (dynamic shading environments)

Use for partial shading step scenarios.

```python
w_energy = 100
w_settle = 8
w_ripple = 5
w_area   = 15
w_ctrl   = 0.001
```

Effect:
- Slow convergence heavily penalized
- Algorithms that quickly find GMPP rank higher

---

### 4️⃣ Embedded-constrained (real firmware target)

Use when controller must meet tight compute budgets.

```python
w_energy = 100
w_settle = 2
w_ripple = 5
w_area   = 10
w_ctrl   = 0.01
```

Effect:
- Heavy algorithms are strongly penalized
- Encourages lightweight implementations

---

### Normalizing Score (optional)

If negative scores are confusing, you may normalize scores per session:

\[
score_{norm} = 100 \cdot \frac{score - min}{max - min}
\]

This preserves ranking but forces a 0–100 scale.

---

## Troubleshooting

### Recovery column is blank

Likely causes:
- Disturbance detected, but efficiency never stayed ≥0.98 for 0.05 s.
- Scenario keeps changing (e.g., flicker).

What to do:
- Lower threshold or hold time in `metrics.py`.
- Rely more on `tracking_error_area_meas`.

---

### Score is negative

Not an error.

Means penalties outweigh `100 * energy`.

Fixes (if desired):
- Increase `w_energy`.
- Reduce ripple/area weights.
- Normalize scores per session.

---

### All scores look very similar

Possible reasons:
- Algorithms behave similarly under that scenario.
- Weights are too small relative to energy.
- Scenario is too easy (static irradiance).

Try:
- More aggressive shading steps.
- Increase dynamic penalties (`w_settle`, `w_area`).

---

### Disturbance not detected

If `t_disturb` is None:
- Scenario may be static.
- `g_strings` may not change.
- `p_gmp_ref` may not be computed.

Recovery metrics will not populate in this case.

---

### Benchmark crashes with irradiance length error

Example:

```text
ValueError: Per-string irradiance length X != n_strings Y
```

Cause:
- Scenario shading array does not match modeled module string count.

Fix:
- Ensure `g_strings` length matches `n_strings`.
- Normalize scenario inputs before simulation.

---

### Leaderboard not updating

Check:
- `latest_session_path.txt` exists.
- `summaries.jsonl` exists in the selected session folder.
- The Session path in the UI points to a valid session directory.

---

## Quick checklist

- Want to compare algorithms fairly?
  - Keep them in the **same session** (same scenarios/budgets/knobs)
- Want to see old results?
  - Use the **Session** path box and load the folder
- Want meaningful rankings?
  - Tune score weights in `benchmarks/metrics.py`
- Flicker scenarios?
  - Recovery may be blank; rely on energy + tracking error area