"""
PSO — Particle Swarm Optimization

Windowed, sequential PSO tailored for real‑time MPPT. Instead of evaluating the
whole swarm in one control tick, this implementation evaluates **one particle
per control step** (the setpoint you commanded last step), which keeps compute
light and integrates naturally with live hardware.

Workflow
1) On first call, seed the swarm around the current voltage and command particle 0.
2) Each subsequent call assigns the measured power to the particle commanded in
   the *previous* call, updates personal/global bests, advances the swarm once
   per completed sweep, and commands the next particle (or the gbest when done).

Notes
- Use a **window** around a seed Vmp guess ((+/-)15–25%) to reduce time and enforce
  safety limits.  
- Add dv/dt and di/dt **rate limits** at the setpoint layer (we provide
  SlewLimiter here for v_ref).  
- The early‑stop criterion is intentionally simple; tune `early_eps` to your
  converter/plant.
"""
from typing import Optional, List
import random

from ..base import MPPTAlgorithm
from ..types import Measurement, Action
from ..common import clamp, SlewLimiter, compute_power


class PSO(MPPTAlgorithm):
    """Sequential PSO suitable for S3 GLOBAL_SEARCH.

    Parameters
    window_pct : float
        Half‑span of the initial search window as a fraction of the seed voltage
        (e.g., 0.2 → ±20%). A small absolute minimum span is enforced.
    particles : int
        Number of particles in the swarm (>=3 recommended).
    iters : int
        Max number of position updates (sweeps) to run before locking to gbest.
    early_eps : float
        Early‑stop threshold; when gbest power improves by < early_eps×|gbest|
        for several consecutive iterations, we stop early.
    vmin, vmax : float
        Allowed command range for the voltage reference [V].
    seed : int
        RNG seed for reproducible behavior in tests.
    slew : float
        Max change allowed per control step for the commanded reference [V].
    """

    name = "pso"
    supports_global_search = True

    def __init__(
        self,
        window_pct: float = 0.2,
        particles: int = 6,
        iters: int = 12,
        early_eps: float = 5e-3,
        vmin: float = 0.0,
        vmax: float = 100.0,
        seed: int = 7,
        slew: float = 0.1,
    ) -> None:
        self.window_pct = float(window_pct)
        self.np = max(3, int(particles))
        self.iters = max(1, int(iters))
        self.early_eps = float(early_eps)
        self.vmin, self.vmax = float(vmin), float(vmax)
        self.rng = random.Random(int(seed))
        self._slew = SlewLimiter(max_step=float(slew))
        self.reset()

    # Lifecycle
    def reset(self) -> None:
        self.center: Optional[float] = None
        self.span: Optional[float] = None
        self.pos: List[float] = []
        self.vel: List[float] = []
        self.pbest_v: List[float] = []
        self.pbest_p: List[float] = []
        self.gbest_v: Optional[float] = None
        self.gbest_p: float = -1e18
        self.prev_gbest_p: float = -1e18
        self.k_eval: int = 0
        self._iter: int = 0
        self._streak_small: int = 0
        self.last_cmd: Optional[float] = None

    # Internals
    def _init_swarm(self, seed_v: float) -> None:
        self.center = clamp(seed_v, self.vmin, self.vmax)
        # Enforce a small absolute span to avoid a degenerate window
        abs_min = 0.05 * (self.vmax - self.vmin)
        span = max(self.window_pct * max(abs(self.center), 1e-6), abs_min)
        v_lo = clamp(self.center - span, self.vmin, self.vmax)
        v_hi = clamp(self.center + span, self.vmin, self.vmax)
        self.span = max(v_hi - v_lo, 1e-6)

        self.pos = [self.rng.uniform(v_lo, v_hi) for _ in range(self.np)]
        self.vel = [0.0 for _ in range(self.np)]
        self.pbest_v = self.pos.copy()
        self.pbest_p = [-1e18 for _ in range(self.np)]
        self.gbest_v, self.gbest_p = None, -1e18
        self.prev_gbest_p = -1e18
        self.k_eval = 0
        self._iter = 0
        self._streak_small = 0

    # Core step
    def step(self, m: Measurement) -> Action:
        p = compute_power(m.v, m.i)

        # First call: seed and command particle 0
        if self.center is None:
            self._init_swarm(m.v)
            self.last_cmd = self.pos[0]
            v_cmd = self._slew.step(self.last_cmd)
            return Action(v_ref=v_cmd, debug={"phase": 0, "p": p})

        # Assign fitness to the previously commanded particle
        idx = self.k_eval % self.np
        # Update personal best
        if p >= self.pbest_p[idx]:
            self.pbest_p[idx] = p
            self.pbest_v[idx] = self.last_cmd if self.last_cmd is not None else self.pos[idx]
        # Update global best
        if p > self.gbest_p:
            self.prev_gbest_p = self.gbest_p
            self.gbest_p = p
            self.gbest_v = self.last_cmd if self.last_cmd is not None else m.v

        self.k_eval += 1

        # After a full sweep, advance the swarm once (velocity + position update)
        if self.k_eval % self.np == 0:
            self._iter += 1
            w, c1, c2 = 0.5, 1.0, 1.0
            for i in range(self.np):
                r1, r2 = self.rng.random(), self.rng.random()
                cognitive = c1 * r1 * (self.pbest_v[i] - self.pos[i])
                social = c2 * r2 * ((self.gbest_v if self.gbest_v is not None else self.pos[i]) - self.pos[i])
                # Soft clamp the velocity step to the window span for stability
                self.vel[i] = 0.95 * clamp(w * self.vel[i] + cognitive + social, -self.span, self.span)
                self.pos[i] = clamp(self.pos[i] + self.vel[i], self.vmin, self.vmax)

            # Simple early‑stop: small improvement in gbest power over consecutive iters
            if self.prev_gbest_p > -1e17:  # ensure we had at least one update
                gain = abs(self.gbest_p - self.prev_gbest_p)
                threshold = self.early_eps * max(abs(self.gbest_p), 1.0)
                self._streak_small = self._streak_small + 1 if gain < threshold else 0
            else:
                self._streak_small = 0

        # Decide next command
        done = (self._iter >= self.iters) or (self._streak_small >= 3)
        if done and self.gbest_v is not None:
            target = self.gbest_v
            phase = 2  # lock phase
        else:
            target = self.pos[self.k_eval % self.np]
            phase = 1  # exploring

        self.last_cmd = target
        v_cmd = self._slew.step(target)

        return Action(
            v_ref=v_cmd,
            debug={
                "phase": phase,
                "iter": self._iter,
                "idx": self.k_eval % self.np,
                "gbest_v": float(self.gbest_v or 0.0),
                "gbest_p": float(self.gbest_p),
                "p": float(p),
            },
        )