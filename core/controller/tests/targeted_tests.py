"""
Targeted behavioral tests for MPPT algorithms using the Hybrid controller.

Run from the repo root:
    python -m pytest -q
"""
# Ensure project root is on sys.path (since this test lives under core/controller/tests)
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
ROOT = None
p = _THIS
for _ in range(8):
    if (p / "core").is_dir():
        ROOT = p
        break
    p = p.parent
if ROOT is None:
    ROOT = _THIS.parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataclasses import dataclass
from typing import Callable, Iterable

from core.controller import HybridMPPT, HybridConfig, SafetyLimits
from core.mppt_algorithms import Measurement

# Toy plants & scenarios

def power_single_peak(v: float, voc: float = 42.0, isc: float = 8.0) -> float:
    """Convex-ish P(V) with a single maximum (good for local trackers)."""
    v = max(0.0, min(v, voc))
    i = isc * max(0.0, 1.0 - (v / voc)) ** 1.2
    return v * i


def power_multi_peak(v: float) -> float:
    """Two-peak toy curve; global optimum near ~32 V for defaults below."""
    def hump(vv: float, v0: float, w: float, a: float) -> float:
        x = (vv - v0) / w
        return a * max(0.0, 1.0 - x * x)
    return max(hump(v, 18.0, 10.0, 115.0), hump(v, 32.0, 7.0, 140.0))


@dataclass
class FirstOrderPlant:
    """Discrete plant that follows v_ref with a first‑order lag.

    Plant uses the previous command to evolve this step; controller then
    computes a new command for the next step.
    """
    v: float = 20.0
    alpha: float = 0.25  # larger = faster response

    def step(self, v_ref: float, power_of_v: Callable[[float], float]) -> tuple[float, float, float]:
        self.v += self.alpha * (v_ref - self.v)
        v = max(self.v, 1e-6)
        p = power_of_v(v)
        i = p / v
        return v, i, p


# Helpers

def numeric_pmax(power_of_v: Callable[[float], float], v_lo: float = 0.0, v_hi: float = 42.0, n: int = 2000) -> float:
    """Coarse numeric max P over [v_lo, v_hi] for acceptance thresholds."""
    step = (v_hi - v_lo) / max(1, n)
    v = v_lo
    pmax = 0.0
    for _ in range(n + 1):
        p = power_of_v(v)
        if p > pmax:
            pmax = p
        v += step
    return pmax


def run_hybrid(cfg: HybridConfig, power_of_v: Callable[[float], float], t_final: float, dt: float = 1e-3, v0: float = 20.0):
    """Run Hybrid controller closed loop; return trace and summary.

    Trace rows: (t, state, v, i, p, v_ref)
    """
    ctrl = HybridMPPT(cfg)
    plant = FirstOrderPlant(v=v0)
    t = 0.0
    hist: list[tuple[float, str, float, float, float, float]] = []

    seed = getattr(ctrl, "_last_good_v", None)
    last_v_ref = seed if seed is not None else v0

    while t < t_final:
        v, i, p = plant.step(last_v_ref, power_of_v)
        m = Measurement(t=t, v=v, i=i, dt=dt)
        a = ctrl.step(m)
        v_ref = a.v_ref if a.v_ref is not None else v
        state = (a.debug or {}).get("state", "NA")
        hist.append((t, state, v, i, p, v_ref))
        last_v_ref = v_ref
        t += dt

    return hist


def summarize(trace: Iterable[tuple[float, str, float, float, float, float]], tail: float = 0.2) -> dict:
    import statistics as stats
    tr = list(trace)
    n = len(tr)
    tail_n = max(10, int(n * tail))
    tail = tr[-tail_n:]
    v_vals = [x[2] for x in tail]
    p_vals = [x[4] for x in tail]
    return {
        "p_mean": sum(p_vals) / len(p_vals),
        "p_std": stats.pstdev(p_vals),
        "v_mean": sum(v_vals) / len(v_vals),
        "v_std": stats.pstdev(v_vals),
    }


def tail_after_state(trace, state_name: str, min_len: int = 200) -> list:
    """Return the portion of the trace at/after first occurrence of state_name.
    If state not found or tail too short, return the last `min_len` samples.
    """
    for idx, (_, s, *_rest) in enumerate(trace):
        if s == state_name:
            tail = trace[idx:]
            if len(tail) >= min_len:
                return tail
            break
    # fallback: last min_len samples
    return trace[-min_len:]


# Tests

def test_hybrid_with_s1_variants_on_single_peak():
    """With RUCA/MEPO/PANDO as S1, Hybrid should stay NORMAL and reach high power."""
    pmax = numeric_pmax(power_single_peak)
    for name in ("ruca", "mepo", "pando"):
        cfg = HybridConfig()
        cfg.normal_name = name
        # Avoid passing algo-specific kwargs (e.g., 'step') that some S1 variants don't accept
        cfg.normal_kwargs = {}
        trace = run_hybrid(cfg, power_single_peak, t_final=1.2)
        states = {s for _, s, *_ in trace}
        assert "GLOBAL_SEARCH" not in states, f"Unexpected search for S1={name} on single-peak"
        tail = summarize(trace)
        assert tail["p_mean"] >= 0.80 * pmax, f"{name}: mean power too low"
        assert tail["p_std"] < 4.0, f"{name}: ripple too high"


def test_hybrid_psc_search_and_hold_quality():
    """Under PSC, Hybrid should enter GLOBAL_SEARCH then LOCK_HOLD and perform well in hold."""
    # Sensitive PSD and safety clamp below Voc to keep away from open circuit
    cfg = HybridConfig()
    cfg.psd.update({"dp_frac": 0.02, "dv_frac": 0.02, "window": 3, "votes": 1})
    cfg.safety = SafetyLimits(vmin=0.0, vmax=35.9, imax=100.0, pmax=1e6, tmod_max=85.0)

    # Scenario switch at t=0.4s
    def scenario_power(t: float) -> Callable[[float], float]:
        return (lambda v: power_single_peak(v)) if t < 0.4 else (lambda v: power_multi_peak(v))

    # Run with time-varying scenario
    t_final, dt = 2.5, 1e-3
    t = 0.0
    plant = FirstOrderPlant(v=22.0)
    ctrl = HybridMPPT(cfg)
    trace: list[tuple[float, str, float, float, float, float]] = []
    seed = getattr(ctrl, "_last_good_v", None)
    last_v_ref = seed if seed is not None else plant.v
    while t < t_final:
        v, i, p = plant.step(last_v_ref, scenario_power(t))
        a = ctrl.step(Measurement(t=t, v=v, i=i, dt=dt))
        v_ref = a.v_ref if a.v_ref is not None else v
        state = (a.debug or {}).get("state", "NA")
        trace.append((t, state, v, i, p, v_ref))
        last_v_ref = v_ref
        t += dt

    states = {s for _, s, *_ in trace}
    assert "GLOBAL_SEARCH" in states, "Never entered GLOBAL_SEARCH"
    assert "LOCK_HOLD" in states, "Never entered LOCK_HOLD"

    # Evaluate only the portion after first LOCK_HOLD
    hold_tail = tail_after_state(trace, "LOCK_HOLD", min_len=300)
    pmax_multi = numeric_pmax(power_multi_peak)
    metrics = summarize(hold_tail)
    assert metrics["p_mean"] >= 0.65 * pmax_multi, "LOCK_HOLD mean power too low"
    assert metrics["v_std"] < 2.0, "LOCK_HOLD voltage ripple too high"


def test_hybrid_safety_clamps_ovp():
    """If a measurement violates limits, Hybrid must clamp and stay in SAFETY."""
    cfg = HybridConfig()
    cfg.safety = SafetyLimits(vmin=0.0, vmax=20.0, imax=100.0, pmax=5000.0, tmod_max=85.0)
    ctrl = HybridMPPT(cfg)

    m = Measurement(t=0.0, v=30.0, i=1.0, dt=1e-3)
    a = ctrl.step(m)
    assert (a.debug or {}).get("state") == "SAFETY", "Expected SAFETY on OVP"
    assert a.v_ref == 20.0, f"Expected v_ref clamp to 20.0, got {a.v_ref}"