"""
Aurora MPPT Algorithms — single-file bootstrap

This module provides a minimal, self-contained scaffolding so you can:
  • Swap algorithms behind a unified API (MPPTAlgorithm)
  • Run a hybrid state-machine controller by picking algos via a registry
  • Start with working skeletons for MEPO (S1), RUCA (S1), PSO (S3), NL_ESC (S4)

You can later split this file into `types.py`, `base.py`, `common.py`, `mepo.py`,
`ruca.py`, `pso.py`, `nl_esc.py`, and `registry.py` without changing imports
if you re-export the same names here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from abc import ABC, abstractmethod
import math
import random

# ================================
# Types
# ================================

@dataclass
class Measurement:
    """Single control-cycle measurement from the PV/converter.

    t: seconds, v: volts, i: amps. Optional: irradiance (g), module temp (t_mod),
    dt: loop period. `meta` is for any aux signals (e.g., state id).
    """
    t: float
    v: float
    i: float
    g: Optional[float] = None
    t_mod: Optional[float] = None
    dt: Optional[float] = None
    meta: Dict[str, float] = field(default_factory=dict)

@dataclass
class Action:
    """Controller output for this cycle. Prefer voltage mode (v_ref) if possible.
    `debug` carries telemetry for logs.
    """
    v_ref: Optional[float] = None
    duty_ref: Optional[float] = None
    debug: Dict[str, float] = field(default_factory=dict)

# ================================
# Common helpers
# ================================

def clamp(x: float, a: float, b: float) -> float:
    return a if x < a else b if x > b else x

class EMA:
    """Exponential moving average (for quick low-pass filtering)."""
    def __init__(self, alpha: float, init: float = 0.0):
        self.a = clamp(alpha, 0.0, 1.0)
        self.y = init
        self.ready = False
    def update(self, x: float) -> float:
        if not self.ready:
            self.y = x
            self.ready = True
        else:
            self.y = self.a * x + (1.0 - self.a) * self.y
        return self.y

class SlewLimiter:
    """Limits per-cycle change (rate limiting for v_ref/duty_ref)."""
    def __init__(self, max_step: float):
        self.max_step = abs(max_step)
        self._last: Optional[float] = None
    def step(self, target: float) -> float:
        if self._last is None:
            self._last = target
            return target
        dv = target - self._last
        if dv > self.max_step:
            self._last += self.max_step
        elif dv < -self.max_step:
            self._last -= self.max_step
        else:
            self._last = target
        return self._last

# ================================
# Base class API
# ================================

class MPPTAlgorithm(ABC):
    name: str = "base"
    supports_global_search: bool = False
    uses_dither: bool = False

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def step(self, m: Measurement) -> Action:
        """Compute next setpoint from current measurement.
        Must be PURE WRT hardware (no direct I/O side effects).
        """
        ...

# ================================
# S1 — MEPO (Modified Enhanced P&O)
# ================================

class MEPO(MPPTAlgorithm):
    """Adaptive-step P&O: step ∝ |ΔP| with sign(ΔP * ΔV).
    Faster when far, gentler near MPP.
    """
    name = "mepo"
    supports_global_search = False

    def __init__(self, alpha: float = 0.4, step_min: float = 1e-3, step_max: float = 2e-2,
                 vmin: float = 0.0, vmax: float = 100.0, slew: float = 0.05):
        self.alpha, self.step_min, self.step_max = alpha, step_min, step_max
        self.vmin, self.vmax = vmin, vmax
        self.prev_v: Optional[float] = None
        self.prev_p: Optional[float] = None
        self.v_ref: Optional[float] = None
        self.slew = SlewLimiter(max_step=slew)

    def reset(self) -> None:
        self.prev_v = None
        self.prev_p = None
        self.v_ref = None
        self.slew = SlewLimiter(self.slew.max_step)

    def step(self, m: Measurement) -> Action:
        p = m.v * m.i
        if self.v_ref is None or self.prev_v is None or self.prev_p is None:
            # cold start: hold current V; next cycle will compute a real step
            self.v_ref = clamp(m.v, self.vmin, self.vmax)
        else:
            dV = m.v - self.prev_v
            dP = p - self.prev_p
            sgn = 1.0 if dV * dP > 0 else -1.0
            step = clamp(abs(dP) * self.alpha, self.step_min, self.step_max)
            self.v_ref = clamp(self.v_ref + sgn * step, self.vmin, self.vmax)
        self.prev_v, self.prev_p = m.v, p
        v_cmd = self.slew.step(self.v_ref)
        return Action(v_ref=v_cmd, debug={"p": p, "v_ref": self.v_ref, "v_cmd": v_cmd})

# ================================
# S1 — RUCA (Robust Unified Control Algorithm)
# ================================

class RUCA(MPPTAlgorithm):
    """Unified sign-based controller; alternates voltage and current branches.
    Voltage branch approximates MEPO; current branch approximates Modified IncCond.
    """
    name = "ruca"
    supports_global_search = False

    def __init__(self, step: float = 8e-3, alt_mode: str = "VI", vmin: float = 0.0, vmax: float = 100.0,
                 slew: float = 0.05):
        self.step0 = step
        self.alt_mode = alt_mode.upper()  # "V", "I", or "VI" (alternate)
        self.vmin, self.vmax = vmin, vmax
        self.prev_v: Optional[float] = None
        self.prev_i: Optional[float] = None
        self.prev_p: Optional[float] = None
        self.v_ref: Optional[float] = None
        self._branch_toggle = 0
        self.slew = SlewLimiter(max_step=slew)

    def reset(self) -> None:
        self.prev_v = self.prev_i = self.prev_p = None
        self.v_ref = None
        self._branch_toggle = 0
        self.slew = SlewLimiter(self.slew.max_step)

    def step(self, m: Measurement) -> Action:
        p = m.v * m.i
        if self.v_ref is None or self.prev_v is None or self.prev_i is None or self.prev_p is None:
            self.v_ref = clamp(m.v, self.vmin, self.vmax)
        else:
            dV = m.v - self.prev_v
            dI = m.i - self.prev_i
            dP = p - self.prev_p

            use_v_branch = (self.alt_mode == "V") or (self.alt_mode == "VI" and (self._branch_toggle % 2 == 0))

            sgn = 1.0
            if use_v_branch:
                # Voltage branch: sign(ΔP * ΔV)
                sgn = 1.0 if dP * dV > 0 else -1.0
            else:
                # Current branch: sign(ΔP * ΔI)
                sgn = 1.0 if dP * dI > 0 else -1.0

            step = self.step0
            # Light adaptive trimming: smaller step if |dP| small
            if abs(dP) < 0.02 * max(abs(p), 1e-6):
                step *= 0.5

            self.v_ref = clamp(self.v_ref + sgn * step, self.vmin, self.vmax)
            self._branch_toggle += 1

        self.prev_v, self.prev_i, self.prev_p = m.v, m.i, p
        v_cmd = self.slew.step(self.v_ref)
        return Action(v_ref=v_cmd, debug={"p": p, "v_ref": self.v_ref, "branch": self._branch_toggle % 2})

# ================================
# S3 — PSO (Particle Swarm Optimization)
# ================================

class PSO(MPPTAlgorithm):
    """Windowed, sequential PSO suitable for real-time control.

    We evaluate one particle per control step: the setpoint commanded in the
    *previous* call is assigned the fitness from the *current* measurement.
    """
    name = "pso"
    supports_global_search = True

    def __init__(self, window_pct: float = 0.2, particles: int = 6, iters: int = 12,
                 early_eps: float = 5e-3, vmin: float = 0.0, vmax: float = 100.0, seed: int = 7,
                 slew: float = 0.1):
        self.window_pct = window_pct
        self.np = max(3, particles)
        self.iters = max(1, iters)
        self.early_eps = early_eps
        self.vmin, self.vmax = vmin, vmax
        self.rng = random.Random(seed)
        self.slew = SlewLimiter(max_step=slew)
        self.reset()

    def reset(self) -> None:
        self.center = None   # window center (seeded by first measurement's V)
        self.span = None     # absolute window span
        self.pos: List[float] = []
        self.vel: List[float] = []
        self.pbest_v: List[float] = []
        self.pbest_p: List[float] = []
        self.gbest_v: Optional[float] = None
        self.gbest_p: float = -1e9
        self.k_eval = 0      # how many particle evaluations done
        self.last_cmd: Optional[float] = None
        self._iter = 0
        self._streak_small = 0

    def _init_swarm(self, seed_v: float) -> None:
        self.center = clamp(seed_v, self.vmin, self.vmax)
        span = max(self.window_pct * max(self.center, 1e-3), 0.05 * (self.vmax - self.vmin))
        v_lo = clamp(self.center - span, self.vmin, self.vmax)
        v_hi = clamp(self.center + span, self.vmin, self.vmax)
        self.span = (v_hi - v_lo)
        self.pos = [self.rng.uniform(v_lo, v_hi) for _ in range(self.np)]
        self.vel = [0.0 for _ in range(self.np)]
        self.pbest_v = self.pos.copy()
        self.pbest_p = [-1e9 for _ in range(self.np)]
        self.gbest_v, self.gbest_p = None, -1e9
        self.k_eval = 0
        self._iter = 0
        self._streak_small = 0

    def step(self, m: Measurement) -> Action:
        p = m.v * m.i
        # Seed swarm on first call
        if self.center is None:
            self._init_swarm(m.v)
            # Command first particle right away
            self.last_cmd = self.pos[0]
            v_cmd = self.slew.step(self.last_cmd)
            return Action(v_ref=v_cmd, debug={"phase": 0, "p": p})

        # Assign fitness to the particle we commanded in the previous step
        idx = self.k_eval % self.np
        self.pbest_p[idx] = max(self.pbest_p[idx], p)
        if p >= self.pbest_p[idx]:
            self.pbest_v[idx] = self.last_cmd if self.last_cmd is not None else self.pos[idx]
        if p > self.gbest_p:
            self.gbest_p, self.gbest_v = p, (self.last_cmd if self.last_cmd is not None else m.v)

        self.k_eval += 1
        # Advance particle positions once per particle (velocity update)
        if self.k_eval % self.np == 0:
            self._iter += 1
            w, c1, c2 = 0.5, 1.0, 1.0
            for i in range(self.np):
                r1, r2 = self.rng.random(), self.rng.random()
                cognitive = c1 * r1 * (self.pbest_v[i] - self.pos[i])
                social = c2 * r2 * ((self.gbest_v if self.gbest_v is not None else self.pos[i]) - self.pos[i])
                self.vel[i] = 0.95 * clamp(w * self.vel[i] + cognitive + social, -self.span, self.span)
                self.pos[i] = clamp(self.pos[i] + self.vel[i], self.vmin, self.vmax)
            # Early stop check
            if self.gbest_p > -1e9 and abs(self.vel[0]) + abs(self.vel[-1]) < self.early_eps * max(abs(self.gbest_p), 1.0):
                self._streak_small += 1
            else:
                self._streak_small = 0

        # Choose next command: cycle through particles unless early-stop reached
        if self._iter >= self.iters or self._streak_small >= 3:
            # Lock to best found
            target = self.gbest_v if self.gbest_v is not None else self.pos[idx]
        else:
            target = self.pos[self.k_eval % self.np]
        self.last_cmd = target
        v_cmd = self.slew.step(target)
        return Action(v_ref=v_cmd, debug={"gbest_v": float(self.gbest_v or 0.0), "gbest_p": self.gbest_p, "iter": self._iter})

# ================================
# S4 — NL_ESC (Newton-Like Extremum Seeking Control)
# ================================

class NL_ESC(MPPTAlgorithm):
    """Small sinusoidal dither + synchronous demodulation to estimate gradient.
    Integrates gradient (with optional Newton-like gain) to sit at the maximum
    with very low steady-state ripple.
    """
    name = "nl_esc"
    uses_dither = True

    def __init__(self, dither_amp: float = 1e-3, dither_hz: float = 120.0, k: float = 0.4,
                 vmin: float = 0.0, vmax: float = 100.0, slew: float = 0.05):
        self.A = abs(dither_amp)
        self.f = dither_hz
        self.k = k
        self.vmin, self.vmax = vmin, vmax
        self.v_ref: Optional[float] = None
        self.demod = EMA(alpha=0.1)
        self.slew = SlewLimiter(max_step=slew)

    def reset(self) -> None:
        self.v_ref = None
        self.demod = EMA(alpha=0.1)
        self.slew = SlewLimiter(self.slew.max_step)

    def step(self, m: Measurement) -> Action:
        if self.v_ref is None:
            self.v_ref = clamp(m.v, self.vmin, self.vmax)
        # Sinusoidal dither and synchronous demodulation
        omega = 2.0 * math.pi * self.f
        s = math.sin(omega * m.t)
        p = m.v * m.i
        grad_est = self.demod.update(p * s)
        self.v_ref = clamp(self.v_ref + self.k * grad_est, self.vmin, self.vmax)
        v_cmd = self.slew.step(self.v_ref + self.A * s)
        return Action(v_ref=v_cmd, debug={"p": p, "grad": grad_est, "v_base": self.v_ref, "v_cmd": v_cmd})

# ================================
# Registry
# ================================

_REGISTRY = {
    MEPO.name: MEPO,
    RUCA.name: RUCA,
    PSO.name: PSO,
    NL_ESC.name: NL_ESC,
}

def build(name: str, **kwargs) -> MPPTAlgorithm:
    """Factory: build an algorithm by registry name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown MPPT algorithm '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)

def available() -> Dict[str, type]:
    return dict(_REGISTRY)
