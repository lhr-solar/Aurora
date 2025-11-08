"""Aurora entrypoint

This file is the *single* starting point for the Aurora application.

Running:

    python Aurora.py

will launch the Qt-based frontend defined in ``ui/desktop/main_window.py``.
From there you can:

- Choose a configuration/layout file.
- Open the IV/PV plotter.
- Start and stop a live MPPT simulation using the SimulationEngine.

All of the heavy lifting (core models, simulators, and detailed UI logic)
lives in the package modules; this file simply wires them together as an
explicit entrypoint.
"""

import sys
from pathlib import Path

# Ensure the repository root is on sys.path so that ``ui.*`` and
# ``simulators.*`` imports work when running this file directly.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    """Launch the Aurora main window and start the Qt event loop.

    This delegates to ``ui.desktop.main_window.main`` so that all
    application setup (high-DPI handling, window construction, etc.)
    remains in one place.
    """
    try:
        from ui.desktop.main_window import main as main_window_main
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Failed to import ui.desktop.main_window. Make sure you are "
            "running Aurora from the repository root, or that the package "
            "is installed in your environment."
        ) from exc

    return main_window_main()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())