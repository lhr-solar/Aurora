import json
import sys
from pathlib import Path
from typing import Optional

try:
    from PyQt6 import QtWidgets
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
        QFileDialog,
        QGroupBox,
    )
    from PyQt6.QtCore import Qt
except Exception as e:
    raise RuntimeError(
        "PyQt6 must be installed to use the Aurora desktop front end. "
        "Install it with `pip install PyQt6`."
    ) from e

# Relative import within the Aurora.ui.desktop package
try:
    from .array_plotter import ArrayPlotterWindow
except ImportError:
    # Fallback for running directly from the repo root without packages
    # (e.g. `python ui/desktop/main_window.py`)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ui.desktop.array_plotter import ArrayPlotterWindow  # type: ignore


class AuroraMainWindow(QMainWindow):
    """
    Main desktop front-end for Aurora.

    Responsibilities:
    - Let the user choose a JSON/YAML layout/config file.
    - Set basic plotting options (points, live toggle, live interval).
    - Launch and control the ArrayPlotterWindow.
    - Provide a placeholder panel for future MPPT/source simulators.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Aurora Frontend – PV / MPPT Simulation")

        self._plotter: Optional[ArrayPlotterWindow] = None
        self._config_path: Optional[Path] = None

        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        # --- Config file section -------------------------------------------------
        cfg_group = QGroupBox("Array / Simulation Configuration", self)
        cfg_layout = QVBoxLayout(cfg_group)

        path_row = QHBoxLayout()
        self.config_edit = QLineEdit()
        self.config_edit.setPlaceholderText("No config selected (JSON/YAML layout)...")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse_config)

        path_row.addWidget(QLabel("Config file:", self))
        path_row.addWidget(self.config_edit, 1)
        path_row.addWidget(browse_btn)

        cfg_layout.addLayout(path_row)

        # --- Plotting options ----------------------------------------------------
        opts_row = QHBoxLayout()

        # Points
        opts_row.addWidget(QLabel("IV/PV Points:", self))
        self.points_spin = QSpinBox(self)
        self.points_spin.setRange(10, 5000)
        self.points_spin.setValue(400)
        opts_row.addWidget(self.points_spin)

        # Live toggle
        self.live_check = QCheckBox("Live mode", self)
        self.live_check.setChecked(False)
        opts_row.addWidget(self.live_check)

        # Live interval
        opts_row.addWidget(QLabel("Interval (ms):", self))
        self.live_interval_spin = QSpinBox(self)
        self.live_interval_spin.setRange(50, 10_000)
        self.live_interval_spin.setSingleStep(50)
        self.live_interval_spin.setValue(500)
        opts_row.addWidget(self.live_interval_spin)

        opts_row.addStretch(1)
        cfg_layout.addLayout(opts_row)

        root_layout.addWidget(cfg_group)

        # --- Actions / launchers -------------------------------------------------
        actions_row = QHBoxLayout()

        self.launch_plotter_btn = QPushButton("Open IV / PV Plotter", self)
        self.launch_plotter_btn.clicked.connect(self._on_open_plotter)
        actions_row.addWidget(self.launch_plotter_btn)

        self.reload_cfg_btn = QPushButton("Reload config in plotter", self)
        self.reload_cfg_btn.clicked.connect(self._on_reload_into_plotter)
        self.reload_cfg_btn.setEnabled(False)
        actions_row.addWidget(self.reload_cfg_btn)

        actions_row.addStretch(1)

        quit_btn = QPushButton("Quit", self)
        quit_btn.clicked.connect(self.close)
        actions_row.addWidget(quit_btn)

        root_layout.addLayout(actions_row)

        # --- Placeholder: simulators panel --------------------------------------
        sim_group = QGroupBox("MPPT / Source Simulators (coming soon)", self)
        sim_layout = QVBoxLayout(sim_group)
        sim_label = QLabel(
            "This panel is reserved for tying in `simulators/mppt_sim.py`, "
            "`engine.py`, and `source_sim.py`.\n\n"
            "- You can expose controls here (irradiance, temperature, profiles, etc.).\n"
            "- The simulators should update the underlying Array instance used by "
            "ArrayPlotterWindow.\n"
            "- With Live mode enabled, the plotter will continuously re-fetch "
            "IV curves and show the evolving state.",
            self,
        )
        sim_label.setWordWrap(True)
        sim_layout.addWidget(sim_label)
        root_layout.addWidget(sim_group)

        # Status bar
        self.statusBar().showMessage("Ready")

    # -------------------------------------------------------------------------
    # UI Handlers
    # -------------------------------------------------------------------------

    def _on_browse_config(self) -> None:
        """Open a file dialog to pick a JSON/YAML configuration."""
        dlg = QFileDialog(self, "Select layout/config file")
        dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
        dlg.setNameFilters([
            "Config files (*.json *.cfg *.yaml *.yml)",
            "JSON files (*.json)",
            "All files (*)",
        ])

        if dlg.exec():
            files = dlg.selectedFiles()
            if not files:
                return
            path = Path(files[0])
            self._config_path = path
            self.config_edit.setText(str(path))
            self.statusBar().showMessage(f"Selected config: {path}")
            self.reload_cfg_btn.setEnabled(self._plotter is not None)

    def _on_open_plotter(self) -> None:
        """
        Create (or re-show) the ArrayPlotterWindow and apply current UI settings.

        If a config path is set and exists, we load it into the plotter using
        ArrayPlotterWindow.load_and_apply_config().
        """
        if self._plotter is None:
            try:
                self._plotter = ArrayPlotterWindow(self)
            except RuntimeError as e:
                QMessageBox.critical(self, "Plotter error", str(e))
                return

        # Apply title & basic settings
        self._apply_settings_to_plotter()

        # If config file is set, load it
        if self._config_path is not None and self._config_path.exists():
            try:
                self._plotter.load_and_apply_config(str(self._config_path))
                self.statusBar().showMessage(
                    f"Loaded config into plotter: {self._config_path}"
                )
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Config load failed",
                    f"Failed to load config into plotter:\n{e}",
                )
        else:
            # No config: rely on ArrayPlotterWindow's own fallback demo Array
            self.statusBar().showMessage("Opened plotter (using its internal demo array)")

        self.reload_cfg_btn.setEnabled(True)
        self._plotter.show()
        self._plotter.raise_()
        self._plotter.activateWindow()

    def _on_reload_into_plotter(self) -> None:
        """
        Reload the currently selected config file into an existing plotter window.
        """
        if self._plotter is None:
            QMessageBox.information(
                self,
                "No plotter",
                "The IV/PV plotter is not open yet. Click 'Open IV / PV Plotter' first.",
            )
            return

        if self._config_path is None or not self._config_path.exists():
            QMessageBox.warning(
                self,
                "No valid config",
                "Please select a valid JSON/YAML config file first.",
            )
            return

        try:
            # Update points/live settings first
            self._apply_settings_to_plotter()
            # Then load the config
            self._plotter.load_and_apply_config(str(self._config_path))
            self.statusBar().showMessage(
                f"Reloaded config into plotter: {self._config_path}"
            )
        except Exception as e:
            QMessageBox.warning(
                self,
                "Config load failed",
                f"Failed to reload config into plotter:\n{e}",
            )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _apply_settings_to_plotter(self) -> None:
        """
        Push the current frontend settings (points, live, interval) into
        ArrayPlotterWindow, without needing to go through a config file.
        """
        if self._plotter is None:
            return

        # Points
        try:
            self._plotter.points_spin.setValue(self.points_spin.value())
        except Exception:
            pass

        # Live interval
        try:
            interval_ms = int(self.live_interval_spin.value())
            self._plotter._live_interval_ms = interval_ms  # type: ignore[attr-defined]
        except Exception:
            pass

        # Live toggle: call the same handler ArrayPlotterWindow uses
        try:
            desired_live = self.live_check.isChecked()
            is_active = self._plotter._timer.isActive()  # type: ignore[attr-defined]
            if desired_live and not is_active:
                self._plotter.live_btn.setChecked(True)
                self._plotter._on_toggle_live(True)
            elif not desired_live and is_active:
                self._plotter.live_btn.setChecked(False)
                self._plotter._on_toggle_live(False)
        except Exception:
            pass


def main() -> int:
    """
    Entry point for running the Aurora frontend directly:

        python -m ui.desktop.main_window

    or (depending on your repo layout):

        python ui/desktop/main_window.py
    """
    app = QApplication(sys.argv)

    # Best-effort high-DPI handling
    try:
        from PyQt6.QtGui import QGuiApplication
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            QGuiApplication.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass

    win = AuroraMainWindow()
    win.resize(900, 400)
    win.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())