"""ui.desktop.main_window

Main application window for Aurora.

- Central area: tabs for 'Live Bench' (LabDashboard) and 'Benchmarks' (BenchmarksDashboard)
- Bottom dock: shared TerminalPanel (run log)
- Shared LiveOverrides instance for Option A live control

Run:
    python -m ui.desktop.main_window
"""

import sys
import importlib
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QDockWidget,
)

# Reusable terminal panel
from ui.desktop.terminal_panel import TerminalPanel
from simulators.engine import LiveOverrides


def _placeholder(title: str, subtitle: str = "Not implemented yet") -> QWidget:
    """Return a simple placeholder widget for an empty dashboard."""
    w = QWidget()
    layout = QVBoxLayout(w)

    header = QLabel(title)
    header.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
    header.setStyleSheet("font-size: 18px; font-weight: 600;")

    sub = QLabel(subtitle)
    sub.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
    sub.setWordWrap(True)
    sub.setStyleSheet("color: #666;")

    layout.addStretch(1)
    layout.addWidget(header)
    layout.addWidget(sub)
    layout.addStretch(2)
    return w


class MainWindow(QMainWindow):
    """Aurora desktop application shell."""

    def __init__(self, *, title: str = "Aurora") -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1200, 800)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self.setCentralWidget(self._tabs)

        # Global terminal/log dock (shared across dashboards)
        self.terminal = TerminalPanel(title="Terminal")
        dock = QDockWidget("Terminal", self)
        dock.setObjectName("aurora_terminal_dock")
        dock.setWidget(self.terminal)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
            | Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

        # Shared live overrides (mutated by UI controls; read by in-process simulations)
        self.overrides = LiveOverrides()

        self._lab_tab = self._load_dashboard(
            module_path="ui.desktop.lab_dashboard",
            class_name="LabDashboard",
            fallback_title="Live Bench",
        )
        self._tabs.addTab(self._lab_tab, "Live Bench")

        self._bench_tab = self._load_dashboard(
            module_path="ui.desktop.benchmarks_dashboard",
            class_name="BenchmarksDashboard",
            fallback_title="Benchmarks",
        )
        self._tabs.addTab(self._bench_tab, "Benchmarks")

        # Status bar message to confirm the shell is alive
        self.statusBar().showMessage("Ready")

        self.terminal.append_line("[ui] Aurora UI started")
        self.terminal.append_line("[ui] LiveOverrides ready")
        self.terminal.append_line("[ui] Benchmarks tab ready")

    def _load_dashboard(
        self,
        *,
        module_path: str,
        class_name: str,
        fallback_title: str,
    ) -> QWidget:
        """Attempt to import and instantiate a dashboard widget.

        Dashboards are expected to be QWidget subclasses.
        If the module/class doesn't exist yet (or is empty), we fall back to a
        placeholder so the app shell still runs.
        """
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            # Prefer dependency injection: if the dashboard accepts a `terminal=` kwarg,
            # pass the shared terminal panel. Otherwise, fall back to a no-arg constructor.
            try:
                widget = cls(
                    terminal=getattr(self, "terminal", None),
                    overrides=getattr(self, "overrides", None),
                )  # type: ignore[call-arg]
            except TypeError:
                # Dashboard may not accept dependency injection yet
                widget = cls()  # type: ignore[call-arg]
            if not isinstance(widget, QWidget):
                raise TypeError(f"{module_path}.{class_name} is not a QWidget")
            return widget
        except Exception as e:
            # Keep the shell running; show details in a tooltip.
            if hasattr(self, "terminal") and self.terminal is not None:
                try:
                    self.terminal.append_line(
                        f"[ui] Failed to load {module_path}.{class_name}: {type(e).__name__}: {e}"
                    )
                except Exception:
                    pass
            w = _placeholder(
                fallback_title,
                subtitle=(
                    "This dashboard isn't wired yet.\n\n"
                    f"Expected: {module_path}.{class_name}\n"
                    f"Error: {type(e).__name__}: {e}"
                ),
            )
            w.setToolTip(f"{module_path}.{class_name} import failed: {e}")
            return w


def run(argv: Optional[list[str]] = None) -> int:
    """Entry point for launching the desktop app."""
    app = QApplication(argv or sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())