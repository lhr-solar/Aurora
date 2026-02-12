"""Metric extraction for Aurora MPPT benchmarking.

This module converts per-tick simulation records (emitted by `SimulationEngine.run()`)
into scalar metrics suitable for ranking MPPT algorithms.

It is deliberately standalone:
- no UI imports
- no runner imports
- no controller imports

Expected record schema:
  rec["t"]: float seconds
  rec["v"], rec["i"], rec["p"]: measured values (post-noise/quant if enabled)
  rec["p_true"]: optional float (true power)
  rec["g_strings"]: optional list[float] (per-string irradiance)
  rec["gmpp"]: optional dict
      - "p_gmp_ref": float (true reference power)
      - "eff_step": float (true per-step efficiency, p_true / p_gmp_ref)
      - "eff_step_meas": float (measured per-step efficiency, p / p_gmp_ref)
      - legacy "eff_best" and "eff_best_meas" may also exist
  rec["perf"]: optional dict
      - "ctrl_us": float
      - "ref_us": float
      - "over_budget": bool

If some fields are missing (e.g. gmpp_ref disabled), metrics degrade gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple


Record = Dict[str, Any]


@dataclass
class MetricSummary:
    """Scalar metrics for one (algo, scenario, budget) run."""

    # Energy / efficiency
    energy_true_ratio: Optional[float] = None
    energy_meas_ratio: Optional[float] = None

    # Convergence (absolute time in seconds)
    settle_time_s_true: Optional[float] = None
    settle_time_s_meas: Optional[float] = None

    # Disturbance detection + disturbance-relative convergence
    t_disturb: Optional[float] = None
    settle_time_s_true_post: Optional[float] = None
    settle_time_s_meas_post: Optional[float] = None
    recovery_settle_s_true: Optional[float] = None
    recovery_settle_s_meas: Optional[float] = None

    # Stability (lower is better)
    ripple_rms_true: Optional[float] = None
    ripple_rms_meas: Optional[float] = None

    # Tracking quality (lower is better)
    tracking_error_area_true: Optional[float] = None
    tracking_error_area_meas: Optional[float] = None

    # Composite score (higher is better)
    score: Optional[float] = None

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


def extract_time(rec: Record) -> Optional[float]:
    return _safe_float(rec.get("t"))


def extract_measured_power(rec: Record) -> Optional[float]:
    return _safe_float(rec.get("p"))


def extract_true_power(rec: Record) -> Optional[float]:
    return _safe_float(rec.get("p_true"))


def extract_g_strings(rec: Record) -> Optional[List[float]]:
    gs = rec.get("g_strings")
    if isinstance(gs, list):
        try:
            return [float(x) for x in gs]
        except Exception:
            return None
    return None


def extract_gmpp_ref_power(rec: Record) -> Optional[float]:
    gmpp = rec.get("gmpp") or {}
    return _safe_float(gmpp.get("p_gmp_ref"))


def extract_eff(rec: Record) -> Tuple[Optional[float], Optional[float]]:
    gmpp = rec.get("gmpp") or {}

    # Prefer legacy keys if present; fall back to per-step keys
    eff_true = _safe_float(gmpp.get("eff_best"))
    if eff_true is None:
        eff_true = _safe_float(gmpp.get("eff_step"))

    eff_meas = _safe_float(gmpp.get("eff_best_meas"))
    if eff_meas is None:
        eff_meas = _safe_float(gmpp.get("eff_step_meas"))

    return eff_true, eff_meas


def extract_perf(rec: Record) -> Tuple[Optional[float], Optional[float], bool]:
    perf = rec.get("perf") or {}
    ctrl = _safe_float(perf.get("ctrl_us"))
    ref = _safe_float(perf.get("ref_us"))
    over = bool(perf.get("over_budget", False))
    return ctrl, ref, over


def detect_disturb_time(records: Sequence[Record]) -> Optional[float]:
    """Detect the first 'disturbance' time.

    Primary signal: first change in g_strings.
    Fallback: first change in p_gmp_ref.
    """
    if not records:
        return None

    t0 = extract_time(records[0])
    if t0 is None:
        t0 = 0.0

    gs0 = extract_g_strings(records[0])
    if gs0 is not None:
        for rec in records[1:]:
            gs = extract_g_strings(rec)
            if gs is not None and gs != gs0:
                tt = extract_time(rec)
                return float(tt) if tt is not None else float(t0)

    pref0 = extract_gmpp_ref_power(records[0])
    if pref0 is not None:
        for rec in records[1:]:
            pref = extract_gmpp_ref_power(rec)
            if pref is None:
                continue
            if abs(float(pref) - float(pref0)) > 1e-9:
                tt = extract_time(rec)
                return float(tt) if tt is not None else float(t0)

    return None


def tracking_error_area(
    records: Sequence[Record],
    *,
    use_true: bool,
    t_start: Optional[float] = None,
    clamp_below_zero: bool = True,
) -> Optional[float]:
    """Integrate efficiency deficit area: ∫ (1 - eff(t)) dt.

    - If clamp_below_zero, integrate max(0, 1-eff).
    - If t_start is provided, integrate only for t >= t_start.

    Lower is better.
    """
    if not records:
        return None

    pts: List[Tuple[float, float]] = []
    for rec in records:
        t = extract_time(rec)
        if t is None:
            continue
        if t_start is not None and float(t) < float(t_start):
            continue
        et, em = extract_eff(rec)
        eff = et if use_true else em
        if eff is None:
            continue
        pts.append((float(t), float(eff)))

    if len(pts) < 2:
        return None

    area = 0.0
    for k in range(1, len(pts)):
        t1, e1 = pts[k - 1]
        t2, e2 = pts[k]
        dt = t2 - t1
        if dt <= 0:
            continue
        d1 = 1.0 - e1
        d2 = 1.0 - e2
        if clamp_below_zero:
            d1 = max(0.0, d1)
            d2 = max(0.0, d2)
        area += 0.5 * (d1 + d2) * dt

    return float(area)


def composite_score(m: MetricSummary, *, prefer_measured: bool = True) -> Tuple[Optional[float], Dict[str, float]]:
    """Compute a composite score (higher is better)."""
    energy = m.energy_meas_ratio if prefer_measured else m.energy_true_ratio
    rec_settle = m.recovery_settle_s_meas if prefer_measured else m.recovery_settle_s_true
    ripple = m.ripple_rms_meas if prefer_measured else m.ripple_rms_true
    area = m.tracking_error_area_meas if prefer_measured else m.tracking_error_area_true

    energy = float(energy) if energy is not None else 0.0
    rec_settle = float(rec_settle) if rec_settle is not None else 10.0
    ripple = float(ripple) if ripple is not None else 0.0
    area = float(area) if area is not None else 0.0
    ctrl_p95 = float(m.ctrl_us_p95) if m.ctrl_us_p95 is not None else 0.0

    # Weights (tweak as needed)
    w_energy = 100.0
    w_settle = 2.0
    w_ripple = 5.0
    w_area = 10.0
    w_ctrl = 0.001

    score = (
        w_energy * energy
        - w_settle * rec_settle
        - w_ripple * ripple
        - w_area * area
        - w_ctrl * ctrl_p95
    )

    comps = {
        "energy": w_energy * energy,
        "recovery_settle": -w_settle * rec_settle,
        "ripple": -w_ripple * ripple,
        "tracking_area": -w_area * area,
        "ctrl_p95": -w_ctrl * ctrl_p95,
    }

    return float(score), comps


def energy_ratio(records: Sequence[Record], *, use_true: bool) -> Optional[float]:
    """Estimate captured energy / reference energy (trapezoid integration)."""
    if not records:
        return None

    ts: List[float] = []
    p_ref: List[float] = []
    p_num: List[float] = []

    for rec in records:
        t = extract_time(rec)
        if t is None:
            continue
        pref = extract_gmpp_ref_power(rec)
        if pref is None:
            continue

        ts.append(float(t))
        p_ref.append(float(pref))

        if use_true:
            ptrue = extract_true_power(rec)
            if ptrue is not None:
                p_num.append(float(ptrue))
            else:
                eff_true, _ = extract_eff(rec)
                if eff_true is None:
                    p_num.append(0.0)
                else:
                    p_num.append(float(pref) * float(eff_true))
        else:
            pm = extract_measured_power(rec)
            p_num.append(float(pm) if pm is not None else 0.0)

    if len(ts) < 2:
        return None

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
    """Return first time the efficiency stays >= threshold for hold_time_s."""
    if not records:
        return None

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

    start_idx = 0
    while start_idx < len(pts):
        t0, e0 = pts[start_idx]
        if e0 < threshold:
            start_idx += 1
            continue

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


def ripple_rms(records: Sequence[Record], *, use_true: bool, window_s: float = 0.20) -> Optional[float]:
    """Compute RMS ripple of power over the last window."""
    if not records:
        return None

    t_end = extract_time(records[-1])
    if t_end is None:
        return None
    t0 = float(t_end) - float(window_s)

    xs: List[float] = []
    for rec in records:
        t = extract_time(rec)
        if t is None or float(t) < t0:
            continue

        if use_true:
            ptrue = extract_true_power(rec)
            if ptrue is not None:
                xs.append(float(ptrue))
                continue
            pref = extract_gmpp_ref_power(rec)
            et, _ = extract_eff(rec)
            if pref is None or et is None:
                continue
            xs.append(float(pref) * float(et))
        else:
            pm = extract_measured_power(rec)
            if pm is None:
                continue
            xs.append(float(pm))

    if len(xs) < 3:
        return None

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
        out.t_disturb = detect_disturb_time(records)

    # Energy
    out.energy_true_ratio = energy_ratio(records, use_true=True)
    out.energy_meas_ratio = energy_ratio(records, use_true=False)

    # Global settle time
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

    # Post-disturb settle + recovery settle (relative to first post-disturb timestamp)
    if out.t_disturb is not None:
        t_dist = float(out.t_disturb)
        records_post: List[Record] = []
        for rec in records:
            t = extract_time(rec)
            if t is None:
                continue
            if float(t) >= t_dist:
                records_post.append(rec)

        t0_post = extract_time(records_post[0]) if records_post else None
        t_settle_true_post = settle_time(
            records_post,
            use_true=True,
            threshold=settle_threshold,
            hold_time_s=settle_hold_s,
        )
        t_settle_meas_post = settle_time(
            records_post,
            use_true=False,
            threshold=settle_threshold,
            hold_time_s=settle_hold_s,
        )

        if t0_post is not None and t_settle_true_post is not None:
            out.settle_time_s_true_post = float(t_settle_true_post)
            out.recovery_settle_s_true = max(0.0, float(t_settle_true_post) - float(t0_post))
        if t0_post is not None and t_settle_meas_post is not None:
            out.settle_time_s_meas_post = float(t_settle_meas_post)
            out.recovery_settle_s_meas = max(0.0, float(t_settle_meas_post) - float(t0_post))

    # Ripple
    out.ripple_rms_true = ripple_rms(records, use_true=True, window_s=ripple_window_s)
    out.ripple_rms_meas = ripple_rms(records, use_true=False, window_s=ripple_window_s)

    # Tracking error area (post-disturb preferred)
    t_area_start = out.t_disturb
    out.tracking_error_area_true = tracking_error_area(records, use_true=True, t_start=t_area_start)
    out.tracking_error_area_meas = tracking_error_area(records, use_true=False, t_start=t_area_start)

    # Perf
    ps = perf_summary(records)
    out.ctrl_us_p50 = ps["ctrl_us_p50"]
    out.ctrl_us_p95 = ps["ctrl_us_p95"]
    out.ctrl_us_p99 = ps["ctrl_us_p99"]
    out.ref_us_p50 = ps["ref_us_p50"]
    out.ref_us_p95 = ps["ref_us_p95"]
    out.budget_violations = ps["budget_violations"]

    # Composite score
    score, comps = composite_score(out, prefer_measured=True)
    out.score = score
    out.extra["score_components"] = comps

    return out