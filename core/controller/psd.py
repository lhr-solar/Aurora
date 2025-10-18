"""
PSD — Partial Shading Detector (lightweight heuristics)

Detects when the PV curve is likely multi‑peaked, signalling the controller to
run a global search (S3). Heuristics are intentionally simple and cheap.

Heuristics (votes):
  H1. Large |change P| while |change V| is tiny (suggests a distant peak at similar voltage)
  H2. Conflicting local slopes dP/dV within a short window (sign flips)

If the number of positive heuristics >= `votes`, we deem the condition PSC=true.
"""

from dataclasses import dataclass
from typing import Deque, Dict, Any
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..mppt_algorithms.types import Measurement  # type: ignore
from collections import deque


@dataclass
class PSDConfig:
    dp_frac: float = 0.04   # large power change while voltage barely changed (fraction of |P|)
    dv_frac: float = 0.01   # "barely changed" voltage threshold as fraction of |V|
    window: int = 5         # number of recent samples to consider
    votes: int = 2          # how many heuristic votes required to trigger


class PSDDetector:
    """Simple, low‑cost partial shading detector.

    Parameters
    dp_frac : float
        Minimum fraction of |P| that |change P| must exceed (with tiny |change V|) to vote H1.
    dv_frac : float
        Fraction of |V| used to define "tiny" |change V| for H1.
    window : int
        Number of samples to keep for H2 (slope‑sign consistency check).
    votes : int
        Number of positive heuristic votes needed to flag PSC.
    """

    def __init__(self, dp_frac: float = 0.04, dv_frac: float = 0.01, window: int = 5, votes: int = 2):
        if window < 3:
            raise ValueError("window must be ≥ 3")
        self.cfg = PSDConfig(dp_frac=dp_frac, dv_frac=dv_frac, window=window, votes=votes)
        self.buf: Deque["Measurement"] = deque(maxlen=self.cfg.window)

    # Tunable properties (proxy to cfg)
    @property
    def dp_frac(self) -> float:
        return self.cfg.dp_frac

    @dp_frac.setter
    def dp_frac(self, v: float) -> None:
        self.cfg.dp_frac = float(v)

    @property
    def dv_frac(self) -> float:
        return self.cfg.dv_frac

    @dv_frac.setter
    def dv_frac(self, v: float) -> None:
        self.cfg.dv_frac = float(v)

    @property
    def window(self) -> int:
        return self.cfg.window

    @window.setter
    def window(self, w: int) -> None:
        w = max(3, int(w))
        if w != self.cfg.window:
            self.cfg.window = w
            # Rebuild deque with new maxlen, preserving recent samples
            self.buf = deque(self.buf, maxlen=w)

    @property
    def votes(self) -> int:
        return self.cfg.votes

    @votes.setter
    def votes(self, n: int) -> None:
        self.cfg.votes = max(1, int(n))

    # Lifecycle
    def reset(self) -> None:
        self.buf.clear()

    # Streaming API
    def update(self, m: "Measurement") -> None:
        """Append a new sample to the window (no decision)."""
        self.buf.append(m)

    def update_and_check(self, m: "Measurement") -> bool:
        """Append and return PSC decision in one call."""
        self.update(m)
        return self.is_psc()

    # Logic
    # Frontend helpers
    def describe(self) -> Dict[str, Any]:
        return {
            "key": "psd",
            "label": "PSD Detector",
            "params": [
                {"name": "dp_frac", "type": "number",  "min": 0.001, "max": 0.2,  "step": 0.001, "default": float(self.cfg.dp_frac), "help": "|dP|/P threshold for H1"},
                {"name": "dv_frac", "type": "number",  "min": 0.001, "max": 0.2,  "step": 0.001, "default": float(self.cfg.dv_frac), "help": "|dV|/V small-change threshold"},
                {"name": "window",  "type": "integer", "min": 3,     "max": 100, "step": 1,     "default": int(self.cfg.window),  "help": "Samples kept for H2 slope consistency"},
                {"name": "votes",   "type": "integer", "min": 1,     "max": 5,   "step": 1,     "default": int(self.cfg.votes),   "help": "Votes required (H1/H2) to flag PSC"},
            ],
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            "dp_frac": float(self.cfg.dp_frac),
            "dv_frac": float(self.cfg.dv_frac),
            "window": int(self.cfg.window),
            "votes": int(self.cfg.votes),
        }

    def update_params(self, **kw: Any) -> None:
        if "dp_frac" in kw:
            self.dp_frac = float(kw["dp_frac"])  # via property
        if "dv_frac" in kw:
            self.dv_frac = float(kw["dv_frac"])  # via property
        if "window" in kw:
            self.window = int(kw["window"])      # resizes buffer
        if "votes" in kw:
            self.votes = int(kw["votes"])        # via property

    def is_psc(self) -> bool:
        """Return True if current window indicates partial shading.

        Requires at least 3 samples. Uses H1 (change P big, change V small) and H2 (slope
        sign disagreement) to cast votes and compares against `cfg.votes`.
        """
        if len(self.buf) < 3:
            return False

        votes = 0

        # H1: big |change P| while |change V| small over the last step
        a, b = self.buf[-2], self.buf[-1]
        p_a, p_b = a.v * a.i, b.v * b.i
        dP = abs(p_b - p_a)
        v_scale = max(abs(b.v), 1.0)
        dV_small = abs(b.v - a.v) <= self.cfg.dv_frac * v_scale
        if dV_small and dP >= self.cfg.dp_frac * max(abs(p_b), 1.0):
            votes += 1

        # H2: conflicting local dP/dV signs within window
        pos, neg = 0, 0
        for i in range(1, len(self.buf)):
            m0, m1 = self.buf[i - 1], self.buf[i]
            dV = m1.v - m0.v
            if abs(dV) < 1e-9:
                continue  # ignore near-vertical pairs
            dP = (m1.v * m1.i) - (m0.v * m0.i)
            slope_sign = 1 if dP / dV > 0 else -1
            if slope_sign > 0:
                pos += 1
            else:
                neg += 1
        if pos > 0 and neg > 0:
            votes += 1

        return votes >= self.cfg.votes