"""
Algorithm registry (lazy import)

Central place to look up and construct MPPT algorithms by a short name.

Why lazy? Importing heavy modules (future ANN, etc.) only when needed keeps
startup light and avoids circular imports.
"""
from importlib import import_module
from typing import Dict, Tuple, Type

from .base import MPPTAlgorithm

# name -> (module_path, class_name)
_REGISTRY = {
    "mepo":   ("core.mppt_algorithms.local.mepo", "MEPO"),
    "ruca":   ("core.mppt_algorithms.local.ruca", "RUCA"),
    "pando":  ("core.mppt_algorithms.local.pando", "PANDO"),
    "pso":    ("core.mppt_algorithms.global_search.pso", "PSO"),
    "nl_esc": ("core.mppt_algorithms.hold.nl_esc", "NL_ESC"),
}

def register(name: str, module_path: str, class_name: str) -> None:
    """Add or override a registry entry.

    Example
    >>> register("my_algo", "my_pkg.my_mod", "MyAlgo")
    """
    if not name:
        raise ValueError("name must be a non-empty string")
    _REGISTRY[name] = (module_path, class_name)


def get_class(name: str) -> Type[MPPTAlgorithm]:
    """Resolve a registry name to a class object (imports lazily)."""
    try:
        mod_path, cls_name = _REGISTRY[name]
    except KeyError as e:
        raise KeyError(
            f"Unknown MPPT algorithm '{name}'. Available: {list(_REGISTRY)}"
        ) from e

    cls = getattr(import_module(mod_path), cls_name)
    if not issubclass(cls, MPPTAlgorithm):
        raise TypeError(
            f"Resolved class {cls} is not a subclass of MPPTAlgorithm"
        )
    return cls



# --- Runtime guard to fix accidental method shadowing on algorithm instances ---

def _fix_shadowed_methods(obj: MPPTAlgorithm) -> MPPTAlgorithm:
    """If an instance defines a non‑callable attribute with the same name as a
    method (e.g., `step`), move that attribute aside so the method is callable.

    This makes the system resilient to algorithms that used `self.step` for a
    step size, which would otherwise shadow the `step()` method.
    """
    for name in ("step", "reset"):
        cls_attr = getattr(type(obj), name, None)
        inst_attr = getattr(obj, name, None)
        if callable(cls_attr) and not callable(inst_attr):
            try:
                # Preserve the value under a safer name for potential later use
                if name == "step" and isinstance(inst_attr, (int, float)):
                    setattr(obj, "step_size", float(inst_attr))
                else:
                    setattr(obj, f"_{name}_attr", inst_attr)
                delattr(obj, name)
            except Exception:
                # Best-effort; if we cannot move/delete, leave as-is
                pass
    return obj

def build(name: str, **kwargs) -> MPPTAlgorithm:
    """Instantiate an algorithm by name.

    Parameters
    name : str
        Registry key (e.g., "mepo", "ruca", "pso", "nl_esc").
    **kwargs
        Constructor parameters forwarded to the algorithm class.
    """
    cls = get_class(name)
    obj = cls(**kwargs)
    # Repair any accidental method shadowing (e.g., self.step = 0.01)
    obj = _fix_shadowed_methods(obj)
    return obj


def available() -> Dict[str, str]:
    """Return a mapping of registry names -> "module:Class" strings."""
    return {k: f"{mod}:{cls}" for k, (mod, cls) in _REGISTRY.items()}


__all__ = ["build", "available", "register", "get_class"]