

"""Metric extraction for Aurora MPPT benchmarking.

This module converts per-tick simulation records (emitted by `SimulationEngine.run()`)
into scalar metrics suitable for ranking MPPT algorithms.

It is deliberately standalone:
- no UI imports
- no runner imports
- no controller imports

Expected record schema (from your engine benchmarking upgrade):
  rec["t"]: float seconds
  rec["v"], rec["i"], rec["p"]: measured values (post-noise/quant if enabled)
  rec["gmpp"]: optional dict
      - "p_gmp_ref": float (true reference power)
      - "eff_best": float (true best-so-far / ref)
      - "eff_best_meas": float (measured best-so-far / ref)
  rec["perf"]: optional dict
      - "ctrl_us": float
      - "ref_us": float
      - "over_budget": bool

If some fields are missing (e.g. gmpp_ref disabled), metrics degrade gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------


Record = Dict[str, Any]


@dataclass
class MetricSummary:
    """Scalar metrics for one (algo, scenario, budget) run."""

    # Energy / efficiency
    energy_true_ratio: Optional[float] = None
    energy_meas_ratio: Optional[float] = None

    # Convergence
    settle_time_s_true: Optional[float] = None
    settle_time_s_meas: Optional[float] = None

    # Stability (lower is better)
    ripple_rms_true: Optional[float] = None
    ripple_rms_meas: Optional[float] = None

    # Performance / compute
    ctrl_us_p50: Optional[float] = None
    ctrl_us_p95: Optional[float] = None
    ctrl_us_p99: Optional[float] = None
    ref_us_p50: Optional[float] = None
    ref_us_p95: Optional[float] = None
    budget_violations: int = 0

    # Diagnostics
    n_ticks: int = 0
    t_start: Optional[float] = None
    t_end: Optional[float] = None

    extra: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def percentile(xs: Sequence[float], q: float) -> Optional[float]:
    if not xs:
        return None
    if q <= 0:
        return float(min(xs))
    if q >= 1:
        return float(max(xs))
    s = sorted(float(x) for x in xs)
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def _rms(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    m2 = sum(float(x) * float(x) for x in xs) / float(len(xs))
    return float(m2 ** 0.5)


# -----------------------------------------------------------------------------
# Record extraction
# -----------------------------------------------------------------------------


def extract_time(rec: Record) -> Optional[float]:
    return _safe_float(rec.get("t"))


def extract_measured_power(rec: Record) -> Optional[float]:
    # engine record contains rec["p"] as measured (post-noise/quant)
    return _safe_float(rec.get("p"))


def extract_gmpp_ref_power(rec: Record) -> Optional[float]:
    gmpp = rec.get("gmpp") or {}
    return _safe_float(gmpp.get("p_gmp_ref"))


def extract_eff(rec: Record) -> Tuple[Optional[float], Optional[float]]:
    gmpp = rec.get("gmpp") or {}
    return _safe_float(gmpp.get("eff_best")), _safe_float(gmpp.get("eff_best_meas"))


def extract_perf(rec: Record) -> Tuple[Optional[float], Optional[float], bool]:
    perf = rec.get("perf") or {}
    ctrl = _safe_float(perf.get("ctrl_us"))
    ref = _safe_float(perf.get("ref_us"))
    over = bool(perf.get("over_budget", False))
    return ctrl, ref, over


# -----------------------------------------------------------------------------
# Core metrics
# -----------------------------------------------------------------------------


def energy_ratio(
    records: Sequence[Record],
    *,
    use_true: bool,
    dt_fallback: Optional[float] = None,
) -> Optional[float]:
    """Estimate captured energy / reference energy.

    - Captured energy is integrated from a power series.
      - For `use_true=True`, this uses the reference power times `eff_best`.
        (Because true power is not directly emitted in rec["p"] when measurement
         effects are enabled.)
      - For `use_true=False`, this uses rec["p"] (measured) directly.

    - Reference energy integrates `p_gmp_ref`.

    Requires gmpp_ref enabled to be meaningful.
    """

    if not records:
        return None

    # Build time grid
    ts: List[float] = []
    p_ref: List[float] = []

    # For numerator
    p_num: List[float] = []

    for rec in records:
        t = extract_time(rec)
        if t is None:
            continue
        pref = extract_gmpp_ref_power(rec)
        if pref is None:
            continue

        ts.append(t)
        p_ref.append(pref)

        if use_true:
            eff_true, _ = extract_eff(rec)
            if eff_true is None:
                # if eff absent, fall back to 0 contribution
                p_num.append(0.0)
            else:
                p_num.append(float(pref) * float(eff_true))
        else:
            pm = extract_measured_power(rec)
            p_num.append(float(pm) if pm is not None else 0.0)

    if len(ts) < 2:
        return None

    # integrate via trapezoid
    e_ref = 0.0
    e_num = 0.0
    for k in range(1, len(ts)):
        dt = ts[k] - ts[k - 1]
        if dt <= 0:
            continue
        e_ref += 0.5 * (p_ref[k] + p_ref[k - 1]) * dt
        e_num += 0.5 * (p_num[k] + p_num[k - 1]) * dt

    if e_ref <= 1e-12:
        return None
    return float(e_num / e_ref)


def settle_time(
    records: Sequence[Record],
    *,
    use_true: bool,
    threshold: float = 0.98,
    hold_time_s: float = 0.05,
) -> Optional[float]:
    """Return first time the efficiency stays >= threshold for hold_time_s.

    This is a robust convergence metric:
    - uses eff_best (true) or eff_best_meas (measured)
    - requires gmpp_ref enabled
    """

    if not records:
        return None

    # Collect (t, eff)
    pts: List[Tuple[float, float]] = []
    for rec in records:
        t = extract_time(rec)
        if t is None:
            continue
        et, em = extract_eff(rec)
        eff = et if use_true else em
        if eff is None:
            continue
        pts.append((float(t), float(eff)))

    if len(pts) < 2:
        return None

    # Walk and find earliest window that stays above threshold
    start_idx = 0
    while start_idx < len(pts):
        t0, e0 = pts[start_idx]
        if e0 < threshold:
            start_idx += 1
            continue

        # Extend until time window is satisfied or a drop occurs
        t_end = t0
        ok = True
        j = start_idx
        while j < len(pts) and (t_end - t0) < hold_time_s:
            t_end, ej = pts[j]
            if ej < threshold:
                ok = False
                break
            j += 1

        if ok and (t_end - t0) >= hold_time_s:
            return float(t0)

        start_idx += 1

    return None


def ripple_rms(
    records: Sequence[Record],
    *,
    use_true: bool,
    window_s: float = 0.20,
) -> Optional[float]:
    """Compute RMS ripple of efficiency over the last `window_s` seconds."""

    if not records:
        return None

    # Determine end time
    t_end = extract_time(records[-1])
    if t_end is None:
        return None
    t0 = float(t_end) - float(window_s)

    xs: List[float] = []
    for rec in records:
        t = extract_time(rec)
        if t is None or t < t0:
            continue
        et, em = extract_eff(rec)
        eff = et if use_true else em
        if eff is None:
            continue
        xs.append(float(eff))

    if len(xs) < 3:
        return None

    # Detrend by mean and compute RMS
    mu = mean(xs)
    return _rms([x - mu for x in xs])


def perf_summary(records: Sequence[Record]) -> Dict[str, Any]:
    ctrl: List[float] = []
    ref: List[float] = []
    viol = 0

    for rec in records:
        c, r, over = extract_perf(rec)
        if c is not None:
            ctrl.append(c)
        if r is not None:
            ref.append(r)
        if over:
            viol += 1

    return {
        "ctrl_us_p50": percentile(ctrl, 0.50),
        "ctrl_us_p95": percentile(ctrl, 0.95),
        "ctrl_us_p99": percentile(ctrl, 0.99),
        "ref_us_p50": percentile(ref, 0.50),
        "ref_us_p95": percentile(ref, 0.95),
        "budget_violations": int(viol),
    }


# -----------------------------------------------------------------------------
# Top-level API
# -----------------------------------------------------------------------------


def compute_metrics(
    records: Sequence[Record],
    *,
    settle_threshold: float = 0.98,
    settle_hold_s: float = 0.05,
    ripple_window_s: float = 0.20,
) -> MetricSummary:
    """Compute a standard metric bundle from per-tick records."""

    out = MetricSummary(n_ticks=len(records))

    if records:
        out.t_start = extract_time(records[0])
        out.t_end = extract_time(records[-1])

    # Energy ratios (require gmpp_ref)
    out.energy_true_ratio = energy_ratio(records, use_true=True)
    out.energy_meas_ratio = energy_ratio(records, use_true=False)

    # Convergence
    out.settle_time_s_true = settle_time(
        records,
        use_true=True,
        threshold=settle_threshold,
        hold_time_s=settle_hold_s,
    )
    out.settle_time_s_meas = settle_time(
        records,
        use_true=False,
        threshold=settle_threshold,
        hold_time_s=settle_hold_s,
    )

    # Stability
    out.ripple_rms_true = ripple_rms(records, use_true=True, window_s=ripple_window_s)
    out.ripple_rms_meas = ripple_rms(records, use_true=False, window_s=ripple_window_s)

    # Perf
    ps = perf_summary(records)
    out.ctrl_us_p50 = ps["ctrl_us_p50"]
    out.ctrl_us_p95 = ps["ctrl_us_p95"]
    out.ctrl_us_p99 = ps["ctrl_us_p99"]
    out.ref_us_p50 = ps["ref_us_p50"]
    out.ref_us_p95 = ps["ref_us_p95"]
    out.budget_violations = ps["budget_violations"]

    return out