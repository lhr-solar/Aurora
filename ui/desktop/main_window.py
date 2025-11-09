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
    from PyQt6.QtCore import Qt, QTimer
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

# Simulation engine (core + simulators)
try:
    from simulators.engine import (
        SimulationConfig,
        SimulationEngine,
        EnvironmentState,
        make_partial_shading_env,
    )
except Exception as e:
    raise RuntimeError(
        "Failed to import simulators.engine. Make sure you are running from the "
        "Aurora repository root or that the package is installed."
    ) from e


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
        self._sim_engine: Optional[SimulationEngine] = None
        self._sim_timer: QTimer = QTimer(self)
        self._sim_timer.timeout.connect(self._on_sim_tick)
        # Shared environment state (used by the simulation engine and, later, the plotter)
        self._env_state: EnvironmentState = EnvironmentState()

        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        # Config file section
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

        # Plotting options
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

        # Actions / launchers
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

        # Simulators panel
        sim_group = QGroupBox("MPPT / Source Simulators", self)
        sim_layout = QVBoxLayout(sim_group)
        sim_label = QLabel(
            "Controls for tying in `simulators/engine.py`.\n\n"
            "- Use 'Run Simulation' to start a time-stepped MPPT simulation.\n"
            "- The simulation updates the underlying Array used by the plotter.\n"
            "- With Live mode enabled in the plotter, IV/PV curves will refresh as "
            "conditions change.\n"
            "- You can stop the simulation from here or by pressing Esc/Space.",
            self,
        )
        sim_label.setWordWrap(True)
        sim_layout.addWidget(sim_label)

        sim_btn_row = QHBoxLayout()
        self.run_sim_btn = QPushButton("Run Simulation", self)
        self.run_sim_btn.clicked.connect(self._on_run_simulation)
        sim_btn_row.addWidget(self.run_sim_btn)

        self.stop_sim_btn = QPushButton("Stop Simulation", self)
        self.stop_sim_btn.clicked.connect(self._on_stop_simulation)
        self.stop_sim_btn.setEnabled(False)
        sim_btn_row.addWidget(self.stop_sim_btn)

        sim_btn_row.addStretch(1)
        sim_layout.addLayout(sim_btn_row)

        root_layout.addWidget(sim_group)

        # Status bar
        self.statusBar().showMessage("Ready")

    # UI Handlers
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
                # Build an env_func bound to the shared EnvironmentState so that
                # the plotter's sliders/checkboxes and the simulation engine
                # both drive the same environment model.
                env_func = make_partial_shading_env(self._env_state)
                self._plotter = ArrayPlotterWindow(
                    self,
                    env_state=self._env_state,
                    env_func=env_func,
                )
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

    # Helpers
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
        
    # Simulation control
    def _on_run_simulation(self) -> None:
        """Start a time-stepped simulation and drive the front-end from a QTimer.

        This wires the simulation engine to use the same layout as the
        currently-selected config file (when available) and enables a
        fully variable environment via EnvironmentState + make_partial_shading_env.
        """
        # Ensure the plotter exists and has an Array to work with
        if self._plotter is None:
            self._on_open_plotter()
            if self._plotter is None:
                return

        # Try to derive array_kwargs from the currently-selected config
        array_kwargs = None
        if self._config_path is not None and self._config_path.exists():
            try:
                with self._config_path.open("r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                layout = cfg_data.get("layout")
                if layout is not None:
                    array_kwargs = {"layout": layout}
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Config parse warning",
                    f"Failed to parse config for simulation:\n{e}\n"
                    "Falling back to engine's default demo array.",
                )

        # (Re)create a fresh simulation engine
        try:
            env_func = make_partial_shading_env(self._env_state)
            cfg = SimulationConfig(
                total_time=1.0,
                dt=1e-3,
                array_kwargs=array_kwargs,
                # Use a fully variable environment driven by EnvironmentState
                env_func=env_func,
            )
            self._sim_engine = SimulationEngine(cfg)
            self._sim_engine.start()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Simulation error",
                f"Failed to start simulation:\n{e}",
            )
            self._sim_engine = None
            return

        # Start GUI timer to step the simulation
        if not self._sim_timer.isActive():
            # 10 ms UI tick; actual simulation dt is controlled by SimulationConfig
            self._sim_timer.start(10)

        self.run_sim_btn.setEnabled(False)
        self.stop_sim_btn.setEnabled(True)
        self.statusBar().showMessage(
            "Simulation running… (press Stop or Esc/Space to halt)."
        )

    def _on_stop_simulation(self) -> None:
        """Stop the simulation and reset controls."""
        if self._sim_timer.isActive():
            self._sim_timer.stop()

        self._sim_engine = None
        self.run_sim_btn.setEnabled(True)
        self.stop_sim_btn.setEnabled(False)
        self.statusBar().showMessage("Simulation stopped.")

    def _on_sim_tick(self) -> None:
        """Advance the simulation by one step and update the front-end."""
        if self._sim_engine is None:
            if self._sim_timer.isActive():
                self._sim_timer.stop()
            self.run_sim_btn.setEnabled(True)
            self.stop_sim_btn.setEnabled(False)
            return

        try:
            rec = self._sim_engine.step()
        except Exception as e:
            # Any unexpected error stops the simulation
            if self._sim_timer.isActive():
                self._sim_timer.stop()
            self._sim_engine = None
            self.run_sim_btn.setEnabled(True)
            self.stop_sim_btn.setEnabled(False)
            QMessageBox.critical(
                self,
                "Simulation error",
                f"Simulation step failed:\n{e}",
            )
            return

        # None indicates the generator is exhausted (simulation finished)
        if rec is None:
            if self._sim_timer.isActive():
                self._sim_timer.stop()
            self._sim_engine = None
            self.run_sim_btn.setEnabled(True)
            self.stop_sim_btn.setEnabled(False)
            self.statusBar().showMessage("Simulation finished.")
            return

        # Update the IV/PV plot and side panels with the current operating point
        try:
            if self._plotter is not None:
                v = rec.get("v")
                i = rec.get("i")
                if v is not None and i is not None:
                    self._plotter.set_operating_point(v, i)

                # Update telemetry (V, I, P, G, T) if supported
                try:
                    self._plotter.update_telemetry(rec)  # type: ignore[attr-defined]
                except Exception:
                    pass

                # Append MPPT status log line, if data is present and supported
                try:
                    self._plotter.append_mppt_log(rec)  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            # Ignore UI update errors so they don't break the simulation loop
            pass

    # -------------------------------------------------------------------------
    # Key handling: allow Esc/Space to stop the simulation
    # -------------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        try:
            key = event.key()
        except Exception:
            return super().keyPressEvent(event)

        if key in (Qt.Key.Key_Escape, Qt.Key.Key_Space):
            self._on_stop_simulation()
        else:
            super().keyPressEvent(event)


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