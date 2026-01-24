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
    consecutive: int = 2    # require PSC true this many checks in a row to trigger
    latch: int = 5          # once triggered, hold PSC true for this many samples
    cooldown: int = 10      # after trigger, ignore new triggers for this many samples


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
        self._streak = 0        # consecutive raw-PSC positives
        self._latch_left = 0    # how many samples PSC stays latched true
        self._cooldown_left = 0 # how many samples until we allow a new trigger
        self._last_debug: Dict[str, Any] = {}

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
        self._streak = 0
        self._latch_left = 0
        self._cooldown_left = 0
        self._last_debug = {}

    # Streaming API
    def update(self, m: "Measurement") -> None:
        """Append a new sample to the window (no decision)."""
        self.buf.append(m)

    def update_and_check(self, m: "Measurement") -> bool:
        """Append and return PSC decision in one call."""
        self.update(m)
        return self.is_psc()
    
    def last_debug(self) -> Dict[str, Any]:
        return dict(self._last_debug)

    # Logic
    # Frontend helpers
    def describe(self) -> Dict[str, Any]:
        return {
            "key": "psd",
            "label": "PSD Detector",
            "params": [
                {"name": "dp_frac", "type": "number",  "min": 0.001, "max": 0.2,  "step": 0.001, 
                    "default": float(self.cfg.dp_frac), "help": "|dP|/P threshold for H1"},
                {"name": "dv_frac", "type": "number",  "min": 0.001, "max": 0.2,  "step": 0.001, 
                    "default": float(self.cfg.dv_frac), "help": "|dV|/V small-change threshold"},
                {"name": "window",  "type": "integer", "min": 3,     "max": 100, "step": 1,     
                    "default": int(self.cfg.window),  "help": "Samples kept for H2 slope consistency"},
                {"name": "votes",   "type": "integer", "min": 1,     "max": 5,   "step": 1,     
                    "default": int(self.cfg.votes),   "help": "Votes required (H1/H2) to flag PSC"},
                {"name": "consecutive", "type": "integer", "min": 1, "max": 20, "step": 1,
                    "default": int(self.cfg.consecutive), "help": "Debounce: require PSC positive this many consecutive checks"},
                {"name": "latch", "type": "integer", "min": 0, "max": 200, "step": 1,
                    "default": int(self.cfg.latch), "help": "Latch: after trigger, keep PSC true for N samples"},
                {"name": "cooldown", "type": "integer", "min": 0, "max": 500, "step": 1,
                    "default": int(self.cfg.cooldown), "help": "Cooldown: after trigger, block new triggers for N samples"},
            ],
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            "dp_frac": float(self.cfg.dp_frac),
            "dv_frac": float(self.cfg.dv_frac),
            "window": int(self.cfg.window),
            "votes": int(self.cfg.votes),
            "consecutive": int(self.cfg.consecutive),
            "latch": int(self.cfg.latch),
            "cooldown": int(self.cfg.cooldown),
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
        if "consecutive" in kw: self.cfg.consecutive = max(1, int(kw["consecutive"]))
        if "latch" in kw: self.cfg.latch = max(0, int(kw["latch"]))
        if "cooldown" in kw: self.cfg.cooldown = max(0, int(kw["cooldown"]))

    def is_psc(self) -> bool:
        """Return True if current window indicates partial shading.

        Requires at least 3 samples. Uses H1 (change P big, change V small) and H2 (slope
        sign disagreement) to cast votes and compares against `cfg.votes`.
        """
        n = len(self.buf)
        if n < 3:
            self._last_debug = {"psc": False, "mode": "insufficient_samples", "n": n, "votes": 0}
            return False

        votes = 0
        h1 = False
        h2 = False
        h1_max_dp = 0.0
        h1_pair = None

        # -------------------------
        # H1: scan window
        # big |dP| while |dV| is tiny for ANY adjacent pair in the window
        for i in range(1, n):
            a, b = self.buf[i - 1], self.buf[i]
            p_a, p_b = a.v * a.i, b.v * b.i
            dP = abs(p_b - p_a)

            v_scale = max(abs(b.v), 1.0)
            if abs(b.v - a.v) <= self.cfg.dv_frac * v_scale:
                if dP >= self.cfg.dp_frac * max(abs(p_b), 1.0):
                    h1 = True
                    if dP > h1_max_dp:
                        h1_max_dp = dP
                        h1_pair = (i - 1, i)

        if h1:
            votes += 1

        # -------------------------
        # H2: conflicting local slope signs within window
        pos, neg = 0, 0
        for i in range(1, n):
            m0, m1 = self.buf[i - 1], self.buf[i]
            dV = m1.v - m0.v
            if abs(dV) < 1e-9:
                continue
            dP = (m1.v * m1.i) - (m0.v * m0.i)
            # sign of slope dP/dV
            if (dP / dV) > 0:
                pos += 1
            else:
                neg += 1

        if pos > 0 and neg > 0:
            h2 = True
            votes += 1

        raw_psc = (votes >= self.cfg.votes)

        # -------------------------
        # Latch / cooldown / debounce
        # 1) If latched: stay true
        if self._latch_left > 0:
            self._latch_left -= 1
            self._last_debug = {
                "psc": True, "mode": "latched",
                "votes": votes, "required_votes": self.cfg.votes,
                "h1": h1, "h2": h2, "h1_pair": h1_pair, "h1_max_dp": float(h1_max_dp),
                "pos": pos, "neg": neg,
            }
            return True

        # 2) If in cooldown: stay false and reset streak
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            self._streak = 0
            self._last_debug = {
                "psc": False, "mode": "cooldown",
                "votes": votes, "required_votes": self.cfg.votes,
                "h1": h1, "h2": h2, "h1_pair": h1_pair, "h1_max_dp": float(h1_max_dp),
                "pos": pos, "neg": neg,
            }
            return False

        # 3) Debounce: require consecutive raw positives
        if raw_psc:
            self._streak += 1
        else:
            self._streak = 0

        if self._streak >= int(self.cfg.consecutive):
            # Trigger PSC, then latch + cooldown
            self._latch_left = int(self.cfg.latch)
            self._cooldown_left = int(self.cfg.cooldown)
            self._streak = 0
            self._last_debug = {
                "psc": True, "mode": "triggered",
                "votes": votes, "required_votes": self.cfg.votes,
                "h1": h1, "h2": h2, "h1_pair": h1_pair, "h1_max_dp": float(h1_max_dp),
                "pos": pos, "neg": neg,
            }
            return True

        self._last_debug = {
            "psc": False, "mode": "idle",
            "votes": votes, "required_votes": self.cfg.votes,
            "h1": h1, "h2": h2, "h1_pair": h1_pair, "h1_max_dp": float(h1_max_dp),
            "pos": pos, "neg": neg, "streak": int(self._streak),
        }
        return False