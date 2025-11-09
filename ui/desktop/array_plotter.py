from typing import Any, Tuple
import numpy as np

try:
    # Prefer PyQt6
    from PyQt6 import QtWidgets
    from PyQt6.QtWidgets import (
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QSpinBox,
        QLabel,
        QFileDialog,
        QGroupBox,
        QSlider,
        QTextEdit,
        QCheckBox,
    )
    from PyQt6.QtCore import Qt, QTimer
    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False

try:
    import pyqtgraph as pg
    from pyqtgraph import PlotWidget, GraphicsLayoutWidget, InfiniteLine
    PG_AVAILABLE = True
except Exception:
    PG_AVAILABLE = False


def _normalize_iv_arrays(a1: np.ndarray, a2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Try to determine which array is V and which is I and return (V, I).

    Many implementations return (V, I) but some (older Array.iv_curve) return (I, V).
    Heuristics: voltage often starts higher and decreases as index increases, while
    current usually starts near zero and increases. We'll check monotonicity and
    value ranges to pick the right order.
    """
    # Coerce to numpy arrays
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)

    # If a1 is strictly decreasing and a2 is non-decreasing, a1 is V
    if a1.size >= 2 and a2.size >= 2:
        if a1[0] > a1[-1] and a2[0] <= a2[-1]:
            return a1, a2
        if a2[0] > a2[-1] and a1[0] <= a1[-1]:
            return a2, a1

    # Fallback: choose the array with larger max as voltage (voltages usually larger)
    if np.nanmax(a1) >= np.nanmax(a2):
        return a1, a2
    return a2, a1





class ArrayPlotterWindow(QMainWindow):
    """A small PyQt window that plots IV and PV curves for an Array-like object.

    Usage:
      win = ArrayPlotterWindow()
      win.set_array(my_array)
      win.show()
    """

    def __init__(self, parent=None, env_state=None, env_func=None):
        """Create the window. This plotter uses pyqtgraph for rendering.

        ``env_state`` is an optional shared environment object (e.g.
        simulators.engine.EnvironmentState) that the UI can mutate via
        sliders / checkboxes, while ``env_func`` is an optional callable
        with signature ``env_func(array, t)`` that applies that state to
        the attached Array. Both are optional for backward compatibility.
        """
        if not PYQT_AVAILABLE or not PG_AVAILABLE:
            raise RuntimeError("PyQt6 and pyqtgraph must be installed to use the GUI")
        super().__init__(parent)
        self.setWindowTitle("Array IV / PV Plotter")
        self.array = None
        # Shared environment hooks (may be None)
        self._env_state = env_state
        self._env_func = env_func

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Top row: left telemetry, center plot, right MPPT log
        top_row = QHBoxLayout()
        root.addLayout(top_row, 1)

        # Plot area: default to pyqtgraph. Force pyqtgraph backend to keep
        # native desktop interactivity and avoid webengine-related issues.
        # pyqtgraph-only mode
        self._plotly_mode = False
        self.web_view = None

        if not self._plotly_mode:
            # PyQtGraph layout with a single plot and a right-side axis for Power
            self.plot_layout = GraphicsLayoutWidget()

            # Main plot shows I-V on left y-axis
            self.plot = self.plot_layout.addPlot(row=0, col=0, title="I-V / P-V")
            self.plot.showGrid(x=True, y=True)
            self.plot.setLabel('bottom', 'Voltage (V)')
            self.plot.setLabel('left', 'Current (A)')

            # Create a right-side axis and a linked ViewBox for the P-V curve
            self.plot.showAxis('right')
            self.plot.getAxis('right').setLabel('Power (W)')
            self.pv_vb = pg.ViewBox()
            self.plot.scene().addItem(self.pv_vb)
            self.plot.getAxis('right').linkToView(self.pv_vb)
            self.pv_vb.setXLink(self.plot)

            # Keep handles for lines
            self._iv_curve = None
            self._pv_curve = None
            self._vmpp_line = None

            # hover/crosshair: single vertical line on main plot, horizontal on pv ViewBox
            self._vline_hover = InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y'))
            self.plot.addItem(self._vline_hover)
            self._pv_hline_hover = InfiniteLine(angle=0, movable=False, pen=pg.mkPen('y'))
            self.pv_vb.addItem(self._pv_hline_hover)

            # connect scene-level mouse move to pv-only handler; we trace PV only
            try:
                self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
            except Exception:
                pass

            # keep views synced when resized
            def _update_views():
                self.pv_vb.setGeometry(self.plot.getViewBox().sceneBoundingRect())
                self.pv_vb.linkedViewChanged(self.plot.getViewBox(), self.pv_vb.XAxis)
            try:
                self.plot.getViewBox().sigResized.connect(_update_views)
            except Exception:
                pass

            # Build side panels: telemetry (left) and MPPT log (right)
            self.telemetry_group = QGroupBox("Operating Point", self)
            telemetry_layout = QVBoxLayout(self.telemetry_group)
            self.v_label = QLabel("V: -- V", self)
            self.i_label = QLabel("I: -- A", self)
            self.p_label = QLabel("P: -- W", self)
            self.g_label = QLabel("G: -- W/m²", self)
            self.t_label = QLabel("T: -- °C", self)
            for w in (self.v_label, self.i_label, self.p_label, self.g_label, self.t_label):
                telemetry_layout.addWidget(w)
            telemetry_layout.addStretch(1)

            self.mppt_group = QGroupBox("MPPT Status", self)
            mppt_layout = QVBoxLayout(self.mppt_group)
            self.mppt_log = QTextEdit(self)
            self.mppt_log.setReadOnly(True)
            self.mppt_log.setPlaceholderText("MPPT state transitions will appear here...")
            mppt_layout.addWidget(self.mppt_log)

            # Assemble top row: telemetry | plot | MPPT log
            top_row.addWidget(self.telemetry_group)
            top_row.addWidget(self.plot_layout, 1)
            top_row.addWidget(self.mppt_group)

        # Last-seen arrays for hover lookup (always present)
        self._last_V = None
        self._last_I = None
        # Ensure optional attributes exist even in Plotly mode
        if not hasattr(self, '_iv_curve'):
            self._iv_curve = None
        if not hasattr(self, '_pv_curve'):
            self._pv_curve = None
        if not hasattr(self, '_vmpp_line'):
            self._vmpp_line = None
        if not hasattr(self, '_vline_hover'):
            self._vline_hover = None
        if not hasattr(self, '_pv_hline_hover'):
            self._pv_hline_hover = None
        if not hasattr(self, '_iv_marker'):
            self._iv_marker = None

        # Controls row: existing buttons
        ctrl_layout = QHBoxLayout()
        root.addLayout(ctrl_layout)

        ctrl_layout.addWidget(QLabel("Points:"))
        self.points_spin = QSpinBox()
        self.points_spin.setRange(10, 2000)
        self.points_spin.setValue(400)
        ctrl_layout.addWidget(self.points_spin)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._on_refresh)
        ctrl_layout.addWidget(self.refresh_btn)

        self.save_btn = QPushButton("Save Figure")
        self.save_btn.clicked.connect(self._on_save)
        ctrl_layout.addWidget(self.save_btn)

        self.live_btn = QPushButton("Live: Off")
        self.live_btn.setCheckable(True)
        self.live_btn.clicked.connect(self._on_toggle_live)
        ctrl_layout.addWidget(self.live_btn)

        self.load_cfg_btn = QPushButton("Load Config")
        self.load_cfg_btn.clicked.connect(self._on_load_config)
        ctrl_layout.addWidget(self.load_cfg_btn)

        # Environment & shading controls
        self.env_group = QGroupBox("Environment & Shading", self)
        env_layout = QHBoxLayout(self.env_group)

        self.partial_check = QCheckBox("Enable partial shading", self)
        env_layout.addWidget(self.partial_check)

        # Global irradiance slider + value readout
        env_layout.addWidget(QLabel("G (W/m²):", self))
        self.g_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.g_slider.setRange(0, 1500)
        self.g_slider.setValue(1000)
        self.g_slider.setTracking(True)
        env_layout.addWidget(self.g_slider)
        self.g_value_label = QLabel(f"{self.g_slider.value()}", self)
        env_layout.addWidget(self.g_value_label)

        # Global temperature slider + value readout
        env_layout.addWidget(QLabel("T (°C):", self))
        self.t_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.t_slider.setRange(-20, 80)
        self.t_slider.setValue(25)
        self.t_slider.setTracking(True)
        env_layout.addWidget(self.t_slider)
        self.t_value_label = QLabel(f"{self.t_slider.value()}", self)
        env_layout.addWidget(self.t_value_label)

        env_layout.addStretch(1)
        root.addWidget(self.env_group)

        # Status label
        self.status = QLabel("")
        root.addWidget(self.status)

        # Connect env controls
        self.partial_check.toggled.connect(self._on_partial_toggled)
        self.g_slider.valueChanged.connect(self._on_env_changed)
        self.t_slider.valueChanged.connect(self._on_env_changed)

        # Timer for live updates
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._plot)
        self._live_interval_ms = 500  # default

    def _on_toggle_live(self, checked: bool):
        """Toggle live update mode on/off."""
        if checked:
            self.live_btn.setText("Live: On")
            self._timer.start(self._live_interval_ms)
        else:
            self.live_btn.setText("Live: Off")
            self._timer.stop()

    def _on_load_config(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load config", "", "JSON Files (*.json);;All Files (*)")
        if fname:
            self.load_and_apply_config(fname)

    def load_and_apply_config(self, path: str) -> None:
        """Load a JSON config file and apply settings.

        Supported keys: "pickle" (path), "points" (int), "live" (bool), "live_interval_ms" (int), "title" (str)
        """
        import json
        from pathlib import Path
        try:
            with open(path, "r") as fh:
                cfg = json.load(fh)
        except Exception as e:
            self.status.setText(f"Failed to load config: {e}")
            return

        # Apply title
        if isinstance(cfg.get("title"), str):
            self.setWindowTitle(cfg["title"])

        # Points
        if isinstance(cfg.get("points"), int):
            try:
                self.points_spin.setValue(int(cfg["points"]))
            except Exception:
                pass

        # Live interval
        if isinstance(cfg.get("live_interval_ms"), int):
            self._live_interval_ms = int(cfg["live_interval_ms"])
            if self._timer.isActive():
                self._timer.start(self._live_interval_ms)

        # Live on/off
        if isinstance(cfg.get("live"), bool):
            if cfg["live"] and not self._timer.isActive():
                self.live_btn.setChecked(True)
                self._on_toggle_live(True)
            elif not cfg["live"] and self._timer.isActive():
                self.live_btn.setChecked(False)
                self._on_toggle_live(False)

        # Note: Pickle loading support removed for security. Use a JSON/YAML
        # layout/config (see `layout` key) to build arrays instead.

        # Layout-based array construction from JSON
        if isinstance(cfg.get("layout"), dict):
            try:
                arr = self._build_array_from_layout(cfg["layout"])
                self.set_array(arr)
                self.status.setText("Built Array from layout config")
            except Exception as e:
                self.status.setText(f"Failed to build array from layout: {e}")

    def _build_array_from_layout(self, layout: dict):
        """Construct an Array from a layout dict.

        layout schema (example):
        {
          "strings": [
            { "substrings": [ { "cells": [ {cell_params}, ... ], "bypass": {..} }, ... ] },
            ...
          ]
        }

        cell_params supports keys used by core.src.cell.Cell constructor (isc_ref, voc_ref, diode_ideality,
        r_s, r_sh, vmpp, impp, irradiance, temperature_c). Missing values use sensible defaults.
        """
        # dynamic imports to avoid module import issues at top-level
        import importlib
        core_cell = importlib.import_module("core.src.cell")
        core_sub = importlib.import_module("core.src.substring")
        core_string = importlib.import_module("core.src.string")
        core_array = importlib.import_module("core.src.array")
        bypass_mod = None
        try:
            bypass_mod = importlib.import_module("core.src.bypassdiode")
        except Exception:
            bypass_mod = None

        strings = []
        for sidx, sconf in enumerate(layout.get("strings", [])):
            substrings = []
            for subidx, subconf in enumerate(sconf.get("substrings", [])):
                cells = []
                for cidx, cconf in enumerate(subconf.get("cells", [])):
                    # defaults
                    params = {
                        "isc_ref": cconf.get("isc_ref", 5.84),
                        "voc_ref": cconf.get("voc_ref", 0.621),
                        "diode_ideality": cconf.get("diode_ideality", 1.3),
                        "r_s": cconf.get("r_s", 0.02),
                        "r_sh": cconf.get("r_sh", 800.0),
                        "vmpp": cconf.get("vmpp", 0.521),
                        "impp": cconf.get("impp", 5.4),
                        "irradiance": cconf.get("irradiance", 1000.0),
                        "temperature_c": cconf.get("temperature_c", 25.0),
                        "autofit": cconf.get("autofit", False),
                    }
                    cell = core_cell.Cell(**params)
                    cells.append(cell)

                bypass = None
                if "bypass" in subconf and bypass_mod is not None:
                    bd = subconf.get("bypass", {})
                    bypass = bypass_mod.Bypass_Diode(**bd)

                substring = core_sub.Substring(cells, bypass=bypass)
                substrings.append(substring)

            pvstring = core_string.PVString(substrings)
            strings.append(pvstring)

        arr = core_array.Array(strings)
        return arr

    def set_array(self, arr: Any) -> None:
        """Attach an Array-like object. The object should implement one of:

        - iv_curve(points) -> (V, I) or (I, V)
        - pv_curve(points) -> (V, I, P)
        - mpp() or get_mpp()
        """
        self.array = arr
        self.status.setText("Array attached")
        self._plot()

    def _fetch_iv(self, points: int = 400):
        if self.array is None:
            return None, None
        # Prefer iv_curve
        if hasattr(self.array, "iv_curve"):
            try:
                a, b = self.array.iv_curve(points)
                V, I = _normalize_iv_arrays(a, b)
                # Resample onto a uniform voltage grid to avoid clustering when
                # underlying implementations sweep current non-linearly.
                try:
                    V = np.asarray(V, dtype=float)
                    I = np.asarray(I, dtype=float)
                    if V.size >= 2 and I.size == V.size:
                        # sort by voltage ascending for interpolation
                        order = np.argsort(V)
                        V_sorted = V[order]
                        I_sorted = I[order]
                        vmin = max(0.0, float(np.nanmin(V_sorted)))
                        vmax = float(np.nanmax(V_sorted)) if V_sorted.size else 1.0
                        V_uniform = np.linspace(vmin, vmax, int(points))
                        I_uniform = np.interp(V_uniform, V_sorted, I_sorted)
                        return V_uniform, I_uniform
                except Exception:
                    pass
                return V, I
            except Exception:
                pass
        # Try pv_curve
        if hasattr(self.array, "pv_curve"):
            try:
                V, I, P = self.array.pv_curve(points)
                return np.asarray(V), np.asarray(I)
            except Exception:
                pass

        # Last resort: try mpp or scalar functions
        raise RuntimeError("Unable to fetch IV from attached array; object does not implement expected API")

    def _plot(self):
        try:
            points = int(self.points_spin.value())
            V, I = self._fetch_iv(points=points)
            if V is None or I is None:
                self.status.setText("No array attached")
                return

            # pyqtgraph rendering

            # update IV plot (pyqtgraph) - IV line in light blue
            if self._iv_curve is None:
                # light blue RGB (173,216,230)
                self._iv_curve = self.plot.plot(V, I, pen=pg.mkPen(color=(173,216,230), width=2))
            else:
                self._iv_curve.setData(V, I)

            # PV curve is plotted on the right-axis ViewBox
            P = V * I
            if self._pv_curve is None:
                self._pv_curve = pg.PlotDataItem(V, P, pen=pg.mkPen(color='orange'))
                try:
                    self.pv_vb.addItem(self._pv_curve)
                except Exception:
                    # fallback: add to main plot if pv_vb unavailable
                    self.plot.addItem(self._pv_curve)
            else:
                self._pv_curve.setData(V, P)

            # create an IV marker (single point) if missing
            if self._iv_marker is None:
                try:
                    self._iv_marker = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(50,150,255), pen=pg.mkPen(None))
                    self.plot.addItem(self._iv_marker)
                    # initialize as invisible (empty data)
                    self._iv_marker.setData([], [])
                except Exception:
                    self._iv_marker = None

            # mark MPP if available
            try:
                if hasattr(self.array, "mpp"):
                    vm, im, pm = self.array.mpp()
                elif hasattr(self.array, "get_mpp"):
                    vm, im, pm = self.array.get_mpp()
                else:
                    vm = im = pm = None
                if vm is not None:
                    # remove old line
                    if self._vmpp_line is not None:
                        try:
                            self.plot.removeItem(self._vmpp_line)
                        except Exception:
                            pass
                    self._vmpp_line = InfiniteLine(pos=vm, angle=90, pen=pg.mkPen('k', style=Qt.PenStyle.DashLine))
                    self.plot.addItem(self._vmpp_line)
            except Exception:
                pass

            # enforce left bound 0 on x-axis for the combined plot
            try:
                xmax = float(np.nanmax(V)) if np.size(V) else 1.0
                self.plot.setXRange(0.0, xmax, padding=0.0)
                # also set minimum limit to 0 so user can't pan left of 0
                self.plot.setLimits(xMin=0.0)
            except Exception:
                pass

            # store last arrays for hover lookup
            try:
                self._last_V = np.asarray(V)
                self._last_I = np.asarray(I)
            except Exception:
                self._last_V = None
                self._last_I = None

            self.status.setText("Plotted: {} points".format(len(V)))
        except Exception as e:
            self.status.setText(f"Error plotting: {e}")

    def set_operating_point(self, v: float, i: float) -> None:
        """Update the IV marker to reflect the current operating point.

        This is intended to be called from a simulation loop (e.g. from
        SimulationEngine.on_sample) so that the front-end can show where
        the controller is operating on the IV curve in real time.
        """
        # If we haven't plotted yet, the marker might not exist; it's created
        # the first time _plot() runs, so we just no-op here instead of
        # forcing a plot on every call.
        if self._iv_marker is None:
            return

        try:
            v_f = float(v)
            i_f = float(i)
        except Exception:
            return

        try:
            # Update marker position
            self._iv_marker.setData([v_f], [i_f])
            # Also update status with a small SIM readout
            self.status.setText(f"[SIM] V={v_f:.4f} V, I={i_f:.4f} A, P={v_f * i_f:.4f} W")
        except Exception:
            # Swallow any drawing errors to avoid breaking the simulation loop
            pass

    def update_telemetry(self, rec: dict) -> None:
        """Update the left-hand telemetry panel from a simulation record.

        Expected keys in ``rec`` (all optional):
        - ``v``: operating voltage (V)
        - ``i``: operating current (A)
        - ``g``: irradiance (W/m²)
        - ``t_mod``: module temperature (°C)
        """
        if not hasattr(self, "v_label"):
            return
        if not isinstance(rec, dict):
            return

        try:
            v = rec.get("v")
            i = rec.get("i")
            g = rec.get("g")
            t_mod = rec.get("t_mod")

            v_f = float(v) if v is not None else None
            i_f = float(i) if i is not None else None

            if v_f is not None:
                self.v_label.setText(f"V: {v_f:.4f} V")
            if i_f is not None:
                self.i_label.setText(f"I: {i_f:.44f} A")
            if v_f is not None and i_f is not None:
                self.p_label.setText(f"P: {v_f * i_f:.4f} W")

            if g is not None:
                g_f = float(g)
                self.g_label.setText(f"G: {g_f:.1f} W/m²")

            if t_mod is not None:
                t_f = float(t_mod)
                self.t_label.setText(f"T: {t_f:.1f} °C")
        except Exception:
            # avoid breaking the sim loop on bad records
            pass

    def append_mppt_log(self, rec: dict) -> None:
        """Append a single MPPT state line to the right-hand log.

        Expects ``rec`` to contain:
        - ``t``: time (s)
        - ``mppt_state`` or ``state``: current controller state name
        - optional ``mppt_detail``: extra info to display
        """
        if not hasattr(self, "mppt_log"):
            return
        if not isinstance(rec, dict):
            return

        t = rec.get("t")
        state = rec.get("mppt_state") or rec.get("state")
        detail = rec.get("mppt_detail")

        if t is None or state is None:
            return

        try:
            t_f = float(t)
            line = f"[t = {t_f:.4f} s] state = {state}"
        except Exception:
            line = f"[t = {t}] state = {state}"

        if detail:
            line += f" | {detail}"

        try:
            self.mppt_log.append(line)
        except Exception:
            pass

    def _set_cell_conditions(self, cell: Any, irradiance: float, temperature_c: float) -> None:
        """Helper to set per-cell irradiance/temperature in a robust way."""
        try:
            if hasattr(cell, "set_conditions"):
                cell.set_conditions(irradiance, temperature_c)
                return
            if hasattr(cell, "irradiance"):
                cell.irradiance = irradiance
            if hasattr(cell, "temperature_c"):
                cell.temperature_c = temperature_c
        except Exception:
            pass

    def _on_env_changed(self) -> None:
        """Handle changes to the global irradiance/temperature sliders.

        When a shared EnvironmentState is attached, we update that state and
        optionally invoke the shared env_func to apply the new conditions to
        the Array. If no env_state is present, we fall back to directly
        broadcasting conditions to the cells as before.
        """
        g = float(self.g_slider.value())
        t_c = float(self.t_slider.value())

        # Update slider value readouts, if present
        try:
            if hasattr(self, "g_value_label"):
                self.g_value_label.setText(f"{g:.0f}")
            if hasattr(self, "t_value_label"):
                self.t_value_label.setText(f"{t_c:.0f}")
        except Exception:
            pass

        # If no array is attached yet, we stop after updating the labels.
        if self.array is None:
            return

        # Preferred path: update shared env_state and apply via env_func
        if self._env_state is not None:
            try:
                self._env_state.global_g = g
                self._env_state.global_t = t_c
            except Exception:
                pass

            # If we have an env_func, apply it once at t=0 for static curves.
            if self._env_func is not None:
                try:
                    self._env_func(self.array, 0.0)
                except Exception:
                    pass

            self._plot()
            return

        # Fallback: direct broadcast to the Array (legacy behavior)
        # Prefer a simple broadcast if the Array exposes set_conditions,
        # unless partial shading is currently enabled.
        if hasattr(self.array, "set_conditions") and not self.partial_check.isChecked():
            try:
                self.array.set_conditions(g, t_c)
            except Exception:
                pass
        else:
            strings = getattr(self.array, "strings", [])
            for string in strings:
                substrings = getattr(string, "substrings", [])
                for substring in substrings:
                    cells = getattr(substring, "cells", [])
                    for cell in cells:
                        self._set_cell_conditions(cell, g, t_c)

        self._plot()

    def _on_partial_toggled(self, enabled: bool) -> None:
        """Enable/disable a simple partial shading pattern.

        When a shared EnvironmentState is present, this simply flips its
        ``partial_enabled`` flag and optionally invokes env_func to apply the
        new pattern. Otherwise, it falls back to directly shading cells.
        """
        if self.array is None:
            return

        # Preferred path: shared env_state drives shading
        if self._env_state is not None:
            try:
                self._env_state.partial_enabled = bool(enabled)
            except Exception:
                pass

            # Apply via env_func if available, then replot
            if self._env_func is not None:
                try:
                    self._env_func(self.array, 0.0)
                except Exception:
                    pass

            self._plot()
            return

        # Fallback: legacy behavior that directly shades the first substring
        if not enabled:
            # Reset to uniform conditions from the sliders
            self._on_env_changed()
            return

        g_base = float(self.g_slider.value())
        t_c = float(self.t_slider.value())
        g_shaded = g_base * 0.4  # 40% of base as an example

        strings = getattr(self.array, "strings", [])
        for s_idx, string in enumerate(strings):
            substrings = getattr(string, "substrings", [])
            for sub_idx, substring in enumerate(substrings):
                cells = getattr(substring, "cells", [])
                for cell in cells:
                    g_use = g_shaded if (s_idx == 0 and sub_idx == 0) else g_base
                    self._set_cell_conditions(cell, g_use, t_c)

        self._plot()

    def _on_refresh(self):
        self._plot()

    def _on_mouse_moved(self, pos):
        """Handle mouse move over the IV plot and display nearest point values."""
        try:
            if self._last_V is None or self._last_I is None:
                return
            # map to main plot coordinates (voltage x) using its ViewBox
            try:
                vb = self.plot.getViewBox()
            except Exception:
                vb = getattr(self.plot, 'vb', None)
            if vb is None:
                return
            # only respond when mouse is inside this plot's scene rect
            try:
                rect = vb.sceneBoundingRect()
                if not rect.contains(pos):
                    return
            except Exception:
                pass
            mouse_point = vb.mapSceneToView(pos)
            mx = float(mouse_point.x())

            V = np.asarray(self._last_V, dtype=float)
            I = np.asarray(self._last_I, dtype=float)
            if V.size == 0:
                return

            # PV-only tracing: use mouse x (voltage) to determine P and I
            try:
                Vv = np.asarray(V, dtype=float)
                Iv = np.asarray(I, dtype=float)
                Pv = Vv * Iv
                # clamp mx to available voltage range
                vmin = float(np.nanmin(Vv))
                vmax = float(np.nanmax(Vv))
                v_val = float(np.clip(mx, vmin, vmax))
                # interpolate power at this voltage
                p_val = float(np.interp(v_val, Vv, Pv))
                if abs(v_val) > 1e-12:
                    i_val = float(p_val / v_val)
                else:
                    # fallback to nearest sample
                    idx = int(np.nanargmin((Vv - v_val)**2 + (Pv - p_val)**2))
                    i_val = float(Iv[idx])
            except Exception:
                return

            # move PV crosshair lines only (vertical on main plot, horizontal on PV ViewBox)
            try:
                self._vline_hover.setPos(v_val)
                self._pv_hline_hover.setPos(p_val)
                # update IV marker position so it lies on the IV curve at this V
                try:
                    if self._iv_marker is not None:
                        self._iv_marker.setData([v_val], [i_val])
                except Exception:
                    pass
            except Exception:
                pass

            # update status text with PV info
            self.status.setText(f"[PV] V={v_val:.4f} V, I={i_val:.4f} A, P={p_val:.4f} W")
        except Exception:
            pass

    

    def _project_point_to_polyline(self, mx: float, my: float, X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
        """Project point (mx,my) onto the polyline defined by (X[i], Y[i]).

        Returns the projected point (px, py). Uses orthogonal projection onto
        each segment and picks the closest projection. This yields a smooth,
        continuous mapping as the mouse moves.
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        n = X.size
        if n == 0:
            return float(mx), float(my)
        if n == 1:
            return float(X[0]), float(Y[0])

        best_d2 = float('inf')
        best_px = X[0]
        best_py = Y[0]

        for i in range(n - 1):
            x0 = X[i]; y0 = Y[i]
            x1 = X[i+1]; y1 = Y[i+1]
            dx = x1 - x0; dy = y1 - y0
            seg_len2 = dx*dx + dy*dy
            if seg_len2 == 0.0:
                # degenerate segment
                proj_x = x0; proj_y = y0
            else:
                t = ((mx - x0) * dx + (my - y0) * dy) / seg_len2
                if t <= 0.0:
                    proj_x = x0; proj_y = y0
                elif t >= 1.0:
                    proj_x = x1; proj_y = y1
                else:
                    proj_x = x0 + t * dx
                    proj_y = y0 + t * dy

            d2 = (proj_x - mx)**2 + (proj_y - my)**2
            if d2 < best_d2:
                best_d2 = d2
                best_px = proj_x
                best_py = proj_y

        return float(best_px), float(best_py)

    def _on_save(self):
        # Save the current widget view as an image using QWidget.grab
        fname, _ = QFileDialog.getSaveFileName(self, "Save figure", "array_plot.png", "PNG Files (*.png);;All Files (*)")
        if fname:
            try:
                # Grab the pyqtgraph layout and save
                pix = self.plot_layout.grab()
                pix.save(fname)
                self.status.setText(f"Saved figure to {fname}")
            except Exception as e:
                self.status.setText(f"Save failed: {e}")


if __name__ == "__main__":
    # Demo: run a small example if executed directly
    if not PYQT_AVAILABLE:
        print("PyQt6 is not available. Install it with 'pip install PyQt6' (or install a Qt binding)")
        raise SystemExit(1)
    if not PG_AVAILABLE:
        print("pyqtgraph is not available. Install it with 'pip install pyqtgraph'")
        raise SystemExit(1)

    # Build a tiny demo Array using simple substrings
    class SimpleSub:
        def __init__(self, voc, isc):
            self._voc = float(voc)
            self._isc = float(isc)

        def set_conditions(self, irradiance, temperature):
            pass

        def v_at_i(self, current: float) -> float:
            # simple linear model
            if current <= 0:
                return self._voc
            if current >= self._isc:
                return 0.0
            return self._voc * (1.0 - current / self._isc)

        def voc(self):
            return self._voc

        def isc(self):
            return self._isc

    from PyQt6.QtWidgets import QApplication

    # Optionally accept a pickled Array file as first CLI arg. If provided, load and plot that.
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    app = QApplication(sys.argv)
    w = ArrayPlotterWindow()

    # CLI arg handling: accept a pickle (.pkl) or a JSON config (.json).
    # Support passing flags (e.g. --plotly) and a positional file; find the
    # first existing non-flag argument and treat it as the file to load.
    file_arg = None
    for a in sys.argv[1:]:
        # skip flags
        if a.startswith('-'):
            continue
        p = Path(a)
        if p.exists():
            file_arg = p
            break

    if file_arg is not None:
        if file_arg.suffix.lower() in (".json", ".cfg", ".yaml", ".yml"):
            w.load_and_apply_config(str(file_arg))
        else:
            # Do not attempt to unpickle arbitrary files for security reasons.
            print(f"Unsupported file type for positional arg: {file_arg}.\n" \
                  "Only JSON/YAML config files are accepted. Use a layout JSON or run the GUI and use 'Load Config'.")
    else:
        # Fallback demo Array: one string with two SimpleSub substrings
        from core.src.string import PVString
        from core.src.array import Array

        sub1 = SimpleSub(0.6, 5.0)
        sub2 = SimpleSub(0.6, 5.0)
        pv = PVString([sub1, sub2])
        arr = Array([pv])
        w.set_array(arr)

    w.show()
    # PyQt6 uses exec()
    app.exec()
