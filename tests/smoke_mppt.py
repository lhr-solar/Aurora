# 
from dataclasses import dataclass
import math

from core.mppt_algorithms import Measurement
from core.mppt_algorithms import build  # uses your registry
from core.controller import HybridMPPT, HybridConfig

# ---------- Simple plant models (lightweight, good for A/B tests) ----------

def power_single_peak(v: float, voc=42.0, isc=8.0) -> float:
    """Convex-ish P(V): I ~ Isc*(1 - (V/Voc))^1.2   (clamped at 0)"""
    v = max(0.0, min(v, voc))
    i = isc * max(0.0, 1.0 - (v / voc)) ** 1.2
    return v * i

def power_multi_peak(v: float) -> float:
    """
    Toy multi-peak: max of two smooth 'humps', roughly mimicking PSC.
    Tune heights/centers to change which peak is global.
    """
    def hump(v, v0, w, a):
        x = (v - v0) / w
        return a * max(0.0, 1.0 - x * x)  # concave parabola clipped at 0
    return max(hump(v, 18.0, 10.0, 115.0), hump(v, 32.0, 7.0, 140.0))  # global near 32V

@dataclass
class FirstOrderPlant:
    """Follows v_ref with first-order lag; i is backsolved from P(V)/V."""
    v: float = 20.0
    alpha: float = 0.25  # larger = faster plant response

    def step(self, v_ref: float, power_func) -> tuple[float, float, float]:
        self.v += self.alpha * (v_ref - self.v)
        v = max(self.v, 1e-6)
        p = power_func(v)
        i = p / v
        return v, i, p

# ---------- Generic runner + metrics ----------

def run_controller(ctrl, power_of_v, t_final=1.5, dt=1e-3, v0=20.0):
    plant = FirstOrderPlant(v=v0)
    t = 0.0
    energy = 0.0
    history = []  # (t, state, v, i, p, v_ref)
    last_state = None

    while t < t_final:
        # In case you want a mid-run scenario change, swap power_of_v() here.
        v, i, p = plant.v, plant.step(getattr(ctrl, "_last_good_v", v0), power_of_v)[1], power_of_v(plant.v)

        m = Measurement(t=t, v=v, i=i, dt=dt)
        a = ctrl.step(m)
        v_ref = a.v_ref if a.v_ref is not None else v

        # Advance plant with the controller's new command
        v, i, p = plant.step(v_ref, power_of_v)

        energy += p * dt
        state = (a.debug or {}).get("state", "NA")
        history.append((t, state, v, i, p, v_ref))
        last_state = state
        t += dt

    return {
        "energy": energy,
        "trace": history,
        "final_state": last_state,
    }

def summarize(history, tail=0.2):
    """Basic scorecard on the last fraction of the run."""
    import statistics as stats
    n = len(history)
    tail_n = max(10, int(n * tail))
    tail_hist = history[-tail_n:]
    v_vals = [x[2] for x in tail_hist]
    p_vals = [x[4] for x in tail_hist]
    return {
        "v_mean": sum(v_vals) / len(v_vals),
        "v_std": stats.pstdev(v_vals),
        "p_mean": sum(p_vals) / len(p_vals),
        "p_std": stats.pstdev(p_vals),
    }

# ---------- 1A. Single-peak: local trackers (RUCA/MEPO/P&O) should settle fast ----------

def test_local_trackers_single_peak():
    for name in ["ruca", "mepo", "pando"]:
        algo = build(name)  # use your defaults
        # Tiny controller wrapper to run an algo alone:
        class Wrapper:
            def __init__(self, s1): self.s1 = s1; self._last_good_v = 20.0
            def step(self, m: Measurement):
                a = self.s1.step(m); self._last_good_v = a.v_ref or m.v; return a
        ctrl = Wrapper(algo)
        out = run_controller(ctrl, power_single_peak, t_final=1.0)
        tail = summarize(out["trace"])
        assert tail["p_std"] < 3.0, f"{name}: ripple too high"
        print(f"[single] {name}: P_mean={tail['p_mean']:.2f}W, V_mean={tail['v_mean']:.2f}V, ripple(P)={tail['p_std']:.2f}")

# ---------- 1B. Partial shading: Hybrid should jump to GLOBAL_SEARCH then LOCK_HOLD ----------

def test_hybrid_under_psc():
    cfg = HybridConfig()
    ctrl = HybridMPPT(cfg)

    # Scenario: start single-peak, then switch to multi-peak at 0.4s
    def scenario_power(v, t):
        return power_single_peak(v) if t < 0.4 else power_multi_peak(v)

    # Run with time-varying power function
    t_final, dt = 1.5, 1e-3
    plant = FirstOrderPlant(v=22.0)
    t = 0.0
    energy = 0.0
    states = []
    while t < t_final:
        v_curr = plant.v
        p_curr = scenario_power(v_curr, t)
        i_curr = p_curr / max(v_curr, 1e-6)

        m = Measurement(t=t, v=v_curr, i=i_curr, dt=dt)
        a = ctrl.step(m)
        v_ref = a.v_ref or v_curr

        v_next, i_next, p_next = plant.step(v_ref, lambda vv: scenario_power(vv, t))
        energy += p_next * dt
        states.append((t, (a.debug or {}).get("state", "NA")))
        t += dt

    # Basic assertions: we should have visited GLOBAL_SEARCH and then LOCK_HOLD
    visited = {s for _, s in states}
    assert "GLOBAL_SEARCH" in visited, "Hybrid never entered GLOBAL_SEARCH under PSC"
    assert "LOCK_HOLD" in visited, "Hybrid never entered LOCK_HOLD after PSO convergence"
    print("Hybrid visited states:", sorted(visited))

if __name__ == "__main__":  # manual run
    test_local_trackers_single_peak()
    test_hybrid_under_psc()
    print("Smoke tests completed.")