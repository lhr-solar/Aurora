"""
Aurora MPPT Algorithms package

Thin re-export layer for the public API. Keep this file minimal; algorithms
live in their own modules. Import from here for stability.
"""
from __future__ import annotations

from .types import Measurement, Action
from .base import MPPTAlgorithm

# Algorithms
from .local.mepo import MEPO
from .local.ruca import RUCA
from .global_search.pso import PSO
from .hold.nl_esc import NL_ESC
from .local.pando import PANDO  # classic P&O

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
