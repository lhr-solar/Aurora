"""
Aurora MPPT Algorithms package

Thin re-export layer for the public API. Keep this file minimal; algorithms
live in their own modules. Import from here for stability.
"""
from __future__ import annotations

from .types import Measurement, Action
from .base import MPPTAlgorithm

# Algorithms
from .mepo import MEPO
from .ruca import RUCA
from .pso import PSO
from .nl_esc import NL_ESC
from .pando import PANDO  # classic P&O

# Registry helpers
from .registry import build, available, register, get_class

__all__ = [
    "Measurement", "Action",
    "MPPTAlgorithm",
    # algorithms
    "MEPO", "RUCA", "PSO", "NL_ESC", "PANDO",
    # registry
    "build", "available", "register", "get_class",
]
