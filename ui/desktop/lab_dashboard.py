"""ui.desktop.lab_dashboard

Single-page 'lab bench' dashboard that combines:
- Source sliders (irradiance + temperature) wired to LiveOverrides (Option A)
- MPPT run controls (algo/profile/time/dt/csv)
- 4 plots (G, T, V, P) with state shading

This is intentionally a focused MVP: it prioritizes live interaction.
Saved runs are written to Aurora/data/runs and can be reopened.
"""

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QCheckBox,
)

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover
    pg = None

from simulators.engine import LiveOverrides, SimulationConfig, SimulationEngine
from core.controller.hybrid_controller import HybridConfig
from simulators.mppt_sim import get_profile

from ui.desktop.terminal_panel import TerminalPanel
from ui.desktop.profile_editor import ProfileEditorDialog


# ---------------------------
# Minimal run-data structure
# ---------------------------
@dataclass
class RunData:
    path: Path
    t: List[float]
    v: List[float]
    i: List[float]
    p: List[float]
    g: List[float]
    t_mod: List[float]
    state: List[str]
    reason: List[Optional[str]]


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _parse_action_debug(s: str) -> Dict[str, Any]:
    try:
        obj = ast.literal_eval(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def build_state_segments(t: List[float], state: List[str]) -> List[Tuple[str, float, float]]:
    if not t or not state:
        return []
    segs: List[Tuple[str, float, float]] = []
    cur = state[0] if state else "UNKNOWN"
    t0 = t[0]
    for k in range(1, len(t)):
        st = state[k] if k < len(state) else cur
        if st != cur:
            segs.append((cur, t0, t[k]))
            cur = st
            t0 = t[k]
    segs.append((cur, t0, t[-1]))
    return segs



def load_run_csv(path: Path) -> RunData:
    t: List[float] = []
    v: List[float] = []
    i: List[float] = []
    p: List[float] = []
    g: List[float] = []
    t_mod: List[float] = []
    state: List[str] = []
    reason: List[Optional[str]] = []

    last_state = "UNKNOWN"

    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            tt = _safe_float(row.get("t"))
            if tt is None:
                continue
            t.append(float(tt))
            v.append(float(_safe_float(row.get("v")) or float("nan")))
            i.append(float(_safe_float(row.get("i")) or float("nan")))
            p.append(float(_safe_float(row.get("p")) or float("nan")))
            g.append(float(_safe_float(row.get("g")) or float("nan")))
            t_mod.append(float(_safe_float(row.get("t_mod")) or float("nan")))

            ad = row.get("action_debug")
            st = None
            rsn = None
            if isinstance(ad, str) and ad.strip():
                dbg = _parse_action_debug(ad)
                st = dbg.get("state")
                rsn = dbg.get("reason")
            if isinstance(st, str) and st:
                last_state = st
            state.append(last_state)
            reason.append(rsn if isinstance(rsn, str) and rsn else None)

    return RunData(path=path, t=t, v=v, i=i, p=p, g=g, t_mod=t_mod, state=state, reason=reason)


def load_env_profile_csv(path: Path):
    """Load a stepwise environment profile from a CSV with header: t,g,t_c.

    Returns a callable env_profile(t) -> (g, t_c).
    """
    rows: List[Tuple[float, float, float]] = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for rr in r:
            tt = rr.get("t") or rr.get("time")
            gg = rr.get("g") or rr.get("irradiance")
            tc = rr.get("t_c") or rr.get("temperature_c") or rr.get("temp_c")
            if tt is None or gg is None or tc is None:
                continue
            try:
                rows.append((float(tt), float(gg), float(tc)))
            except Exception:
                continue
    rows.sort(key=lambda x: x[0])
    if not rows:
        raise ValueError(f"CSV profile has no valid rows: {path}")

    times = [r0 for (r0, _, _) in rows]
    gs = [g0 for (_, g0, _) in rows]
    tcs = [tc0 for (_, _, tc0) in rows]

    def env_at(t: float) -> Tuple[float, float]:
        # Last-value-hold step profile
        idx = 0
        for k in range(len(times)):
            if times[k] <= t:
                idx = k
            else:
                break
        return float(gs[idx]), float(tcs[idx])

    return env_at


def resolve_repo_path(repo_root: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (repo_root / pp)


class _MPPTWorker(QThread):
    """Run MPPT simulation in-process (Option A) and stream CSV-row-shaped samples."""

    sample = pyqtSignal(dict)
    failed = pyqtSignal(str)
    done = pyqtSignal(str)

    def __init__(
        self,
        *,
        out_path: Path,
        algo: str,
        profile_name: str,
        use_csv_profile: bool,
        csv_profile_path: Optional[Path],
        total_time: float,
        dt: float,
        overrides: Optional[LiveOverrides],
    ) -> None:
        super().__init__()
        self.out_path = out_path
        self.algo = algo
        self.profile_name = profile_name
        self.use_csv_profile = use_csv_profile
        self.csv_profile_path = csv_profile_path
        self.total_time = total_time
        self.dt = dt
        self.overrides = overrides
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            if self.use_csv_profile:
                if self.csv_profile_path is None:
                    raise ValueError("CSV profile mode enabled but no csv_profile_path provided")
                profile = load_env_profile_csv(self.csv_profile_path)
            else:
                profile = get_profile(self.profile_name)
            hcfg = HybridConfig(normal_name=self.algo)

            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = [
                "t",
                "dt",
                "v",
                "i",
                "p",
                "g",
                "t_mod",
                "v_ref",
                "action_debug",
            ]

            with self.out_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                def _on_sample(rec: Dict[str, Any]) -> None:
                    action = rec.get("action") if isinstance(rec.get("action"), dict) else {}
                    debug = action.get("debug") if isinstance(action.get("debug"), dict) else {}
                    v_ref = action.get("v_ref")

                    row = {
                        "t": rec.get("t"),
                        "dt": rec.get("dt"),
                        "v": rec.get("v"),
                        "i": rec.get("i"),
                        "p": rec.get("p"),
                        "g": rec.get("g"),
                        "t_mod": rec.get("t_mod"),
                        "v_ref": v_ref,
                        "action_debug": repr(debug),
                    }
                    writer.writerow(row)
                    f.flush()
                    self.sample.emit(row)

                cfg = SimulationConfig(
                    total_time=self.total_time,
                    dt=self.dt,
                    env_profile=profile,
                    controller_cfg=hcfg,
                    overrides=self.overrides,
                    on_sample=_on_sample,
                )

                eng = SimulationEngine(cfg)
                for _ in eng.run():
                    if self._stop:
                        break

            self.done.emit(str(self.out_path))

        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class LabDashboard(QWidget):
    def __init__(
        self,
        *,
        terminal: Optional[TerminalPanel] = None,
        overrides: Optional[LiveOverrides] = None,
    ) -> None:
        super().__init__()

        self.terminal = terminal
        self.overrides = overrides

        # Repo root (Aurora/ui/desktop/lab_dashboard.py)
        self._repo_root = Path(__file__).resolve().parents[2]
        self._runs_root = self._repo_root / "data" / "runs"
        self._runs_root.mkdir(parents=True, exist_ok=True)

        self._worker: Optional[_MPPTWorker] = None
        self._poll: Optional[QTimer] = None
        self._rd: Optional[RunData] = None
        self._state_regions: List[Any] = []

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)

        # ---------------------------
        # Left: controls + saved runs
        # ---------------------------
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)

        left_l.addWidget(QLabel("Live Bench"))

        # Source sliders
        left_l.addWidget(QLabel("Source controls"))

        g_row = QHBoxLayout()
        g_row.addWidget(QLabel("Irradiance"))
        self.g_slider = QSlider(Qt.Orientation.Horizontal)
        self.g_slider.setRange(0, 1400)
        self.g_slider.setSingleStep(10)
        self.g_slider.setPageStep(50)
        self.g_slider.setValue(1000)
        self.g_val = QLabel("1000 W/m²")
        self.g_val.setMinimumWidth(90)
        g_row.addWidget(self.g_slider, 1)
        g_row.addWidget(self.g_val)
        left_l.addLayout(g_row)

        t_row = QHBoxLayout()
        t_row.addWidget(QLabel("Temp"))
        self.t_slider = QSlider(Qt.Orientation.Horizontal)
        self.t_slider.setRange(-20, 100)
        self.t_slider.setSingleStep(1)
        self.t_slider.setPageStep(5)
        self.t_slider.setValue(25)
        self.t_val = QLabel("25 °C")
        self.t_val.setMinimumWidth(60)
        t_row.addWidget(self.t_slider, 1)
        t_row.addWidget(self.t_val)
        left_l.addLayout(t_row)

        # MPPT controls
        left_l.addWidget(QLabel("MPPT controls"))
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Algo"))
        self.algo_edit = QLineEdit("ruca")
        self.algo_edit.setFixedWidth(120)
        row1.addWidget(self.algo_edit)
        row1.addWidget(QLabel("Profile"))
        self.profile_edit = QLineEdit("cloud")
        self.profile_edit.setFixedWidth(120)
        row1.addWidget(self.profile_edit)
        left_l.addLayout(row1)

        # --- CSV profile mode controls ---
        self.use_csv_chk = QCheckBox("Use CSV profile")
        left_l.addWidget(self.use_csv_chk)

        csv_row = QHBoxLayout()
        csv_row.addWidget(QLabel("CSV path"))
        self.csv_path_edit = QLineEdit("profiles/cloud_late.csv")
        csv_row.addWidget(self.csv_path_edit, 1)
        self.btn_browse_profile = QPushButton("Browse…")
        csv_row.addWidget(self.btn_browse_profile)
        left_l.addLayout(csv_row)

        prof_btn_row = QHBoxLayout()
        self.btn_profile_editor = QPushButton("Profile Editor…")
        prof_btn_row.addWidget(self.btn_profile_editor)
        prof_btn_row.addStretch(1)
        left_l.addLayout(prof_btn_row)

        self.csv_path_edit.setEnabled(False)
        self.btn_browse_profile.setEnabled(False)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Time"))
        self.time_edit = QLineEdit("0.5")
        self.time_edit.setFixedWidth(80)
        row2.addWidget(self.time_edit)
        row2.addWidget(QLabel("dt"))
        self.dt_edit = QLineEdit("0.001")
        self.dt_edit.setFixedWidth(80)
        row2.addWidget(self.dt_edit)
        left_l.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("CSV"))
        self.out_edit = QLineEdit("mppt_run.csv")
        row3.addWidget(self.out_edit, 1)
        left_l.addLayout(row3)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Run")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_stop)
        left_l.addLayout(btn_row)

        # Saved runs
        left_l.addWidget(QLabel("Saved runs (data/runs)"))
        self.run_list = QListWidget()
        left_l.addWidget(self.run_list, 1)

        btn_row2 = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_open = QPushButton("Open CSV…")
        btn_row2.addWidget(self.btn_refresh)
        btn_row2.addWidget(self.btn_open)
        left_l.addLayout(btn_row2)

        split.addWidget(left)

        # ---------------------------
        # Right: 4 plots
        # ---------------------------
        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)

        if pg is None:
            right_l.addWidget(QLabel("pyqtgraph not available"))
        else:
            self.g_plot = pg.PlotWidget(title="Irradiance (W/m²)")
            self.t_plot = pg.PlotWidget(title="Temperature (°C)")
            self.v_plot = pg.PlotWidget(title="Voltage (V)")
            self.p_plot = pg.PlotWidget(title="Power (W)")

            # Keep X linked for easier inspection
            self.t_plot.setXLink(self.g_plot)
            self.v_plot.setXLink(self.g_plot)
            self.p_plot.setXLink(self.g_plot)

            right_l.addWidget(self.g_plot, 1)
            right_l.addWidget(self.t_plot, 1)
            right_l.addWidget(self.v_plot, 1)
            right_l.addWidget(self.p_plot, 1)

        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)

        # Wire events
        self.btn_refresh.clicked.connect(self.refresh_runs)
        self.btn_open.clicked.connect(self.open_csv)
        self.run_list.itemSelectionChanged.connect(self._on_select)
        self.btn_run.clicked.connect(self.run_sim)
        self.btn_stop.clicked.connect(self.stop_sim)
        self.g_slider.valueChanged.connect(self._on_override_changed)
        self.t_slider.valueChanged.connect(self._on_override_changed)
        self.use_csv_chk.stateChanged.connect(self._on_profile_mode_changed)
        self.btn_browse_profile.clicked.connect(self.browse_profile_csv)
        self.btn_profile_editor.clicked.connect(self.open_profile_editor)

        self.refresh_runs()
        self._on_override_changed()
        self._on_profile_mode_changed()
    def _on_profile_mode_changed(self) -> None:
        use_csv = self.use_csv_chk.isChecked()
        self.profile_edit.setEnabled(not use_csv)
        self.csv_path_edit.setEnabled(use_csv)
        self.btn_browse_profile.setEnabled(use_csv)

        # Recommendation: when running a CSV profile, disable live overrides so the
        # environment comes from the profile (not from the sliders).
        if use_csv and self.overrides is not None:
            self.overrides.irradiance = None
            self.overrides.temperature_c = None

    def browse_profile_csv(self) -> None:
        start_dir = str(self._repo_root / "profiles")
        (self._repo_root / "profiles").mkdir(parents=True, exist_ok=True)
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select profile CSV",
            start_dir,
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path_str:
            return
        # store as repo-relative when possible
        p = Path(path_str)
        try:
            rel = p.relative_to(self._repo_root)
            self.csv_path_edit.setText(str(rel))
        except Exception:
            self.csv_path_edit.setText(str(p))

    def open_profile_editor(self) -> None:
        dlg = ProfileEditorDialog(parent=self)

        def _on_saved(p: str) -> None:
            self._log(f"[ui] profile saved -> {p}")
            # auto-select CSV mode and point to saved file
            self.use_csv_chk.setChecked(True)
            try:
                rp = Path(p)
                rel = rp.relative_to(self._repo_root)
                self.csv_path_edit.setText(str(rel))
            except Exception:
                self.csv_path_edit.setText(p)

        dlg.editor.profile_saved.connect(_on_saved)
        dlg.exec()

    # ---------------------------
    # Helpers
    # ---------------------------
    def _log(self, msg: str) -> None:
        if self.terminal is not None:
            self.terminal.append_line(msg)

    def refresh_runs(self) -> None:
        self.run_list.clear()
        for p in sorted(self._runs_root.glob("*.csv")):
            self.run_list.addItem(p.name)

    def open_csv(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open run CSV",
            str(self._runs_root),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path_str:
            return
        self._load_and_plot(Path(path_str))

    def _on_select(self) -> None:
        items = self.run_list.selectedItems()
        if not items:
            return
        path = self._runs_root / items[0].text()
        self._load_and_plot(path)

    def _load_and_plot(self, path: Path) -> None:
        try:
            rd = load_run_csv(path)
            self._rd = rd
            # Seed sliders from first sample if present
            if rd.g and rd.g[0] == rd.g[0]:
                self.g_slider.blockSignals(True)
                self.g_slider.setValue(int(round(rd.g[0])))
                self.g_slider.blockSignals(False)
            if rd.t_mod and rd.t_mod[0] == rd.t_mod[0]:
                self.t_slider.blockSignals(True)
                self.t_slider.setValue(int(round(rd.t_mod[0])))
                self.t_slider.blockSignals(False)
            self._on_override_changed()
            self._plot(rd)
        except Exception as e:
            QMessageBox.warning(self, "Failed to load", f"{type(e).__name__}: {e}")

    # ---------------------------
    # Live overrides
    # ---------------------------
    def _on_override_changed(self) -> None:
        g = int(self.g_slider.value())
        t = int(self.t_slider.value())
        self.g_val.setText(f"{g} W/m²")
        self.t_val.setText(f"{t} °C")

        use_csv = hasattr(self, "use_csv_chk") and self.use_csv_chk.isChecked()

        # Option A: mutate shared overrides (only when NOT running a CSV profile)
        if (not use_csv) and self.overrides is not None:
            self.overrides.irradiance = float(g)
            self.overrides.temperature_c = float(t)

        # Also update plots immediately if we have data
        if self._rd is not None:
            self._plot(self._rd)

    # ---------------------------
    # Run/Stop
    # ---------------------------
    def run_sim(self) -> None:
        try:
            sim_time = float(self.time_edit.text().strip())
            dt = float(self.dt_edit.text().strip())
            if sim_time <= 0 or dt <= 0:
                raise ValueError
        except Exception:
            QMessageBox.warning(self, "Invalid settings", "Time and dt must be positive numbers.")
            return

        algo = self.algo_edit.text().strip() or "ruca"
        profile = self.profile_edit.text().strip() or "cloud"

        use_csv = bool(self.use_csv_chk.isChecked())
        csv_profile_path: Optional[Path] = None
        if use_csv:
            raw = self.csv_path_edit.text().strip()
            if not raw:
                QMessageBox.warning(self, "CSV profile", "CSV profile mode is enabled but no CSV path was provided.")
                return
            csv_profile_path = resolve_repo_path(self._repo_root, raw)
            if not csv_profile_path.exists():
                QMessageBox.warning(self, "CSV profile", f"CSV profile not found:\n{csv_profile_path}")
                return

        out_name = self.out_edit.text().strip() or "mppt_run.csv"
        if not out_name.lower().endswith(".csv"):
            out_name += ".csv"
        out_path = self._runs_root / out_name

        # Stop existing
        if self._worker is not None:
            self._worker.request_stop()
            self._worker.wait(1000)
            self._worker = None

        # Reset run data
        self._rd = RunData(path=out_path, t=[], v=[], i=[], p=[], g=[], t_mod=[], state=[], reason=[])

        self._log(f"[ui] Starting MPPT -> {out_path}")
        if use_csv:
            self._log(f"[ui] algo={algo} profile=csv ({csv_profile_path}) time={sim_time} dt={dt}")
        else:
            self._log(f"[ui] algo={algo} profile={profile} time={sim_time} dt={dt}")

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

        w = _MPPTWorker(
            out_path=out_path,
            algo=algo,
            profile_name=profile,
            use_csv_profile=use_csv,
            csv_profile_path=csv_profile_path,
            total_time=sim_time,
            dt=dt,
            overrides=self.overrides,
        )
        w.sample.connect(self._on_live_sample)
        w.failed.connect(self._on_worker_failed)
        w.done.connect(self._on_worker_done)
        self._worker = w
        w.start()

        if self._poll is not None:
            self._poll.stop()
        self._poll = QTimer(self)
        self._poll.setInterval(200)
        self._poll.timeout.connect(self._refresh_live_plot)
        self._poll.start()

    def stop_sim(self) -> None:
        if self._worker is None:
            return
        self._log("[ui] Stopping MPPT…")
        self._worker.request_stop()

    def _on_worker_failed(self, msg: str) -> None:
        self._log(f"[ui] MPPT failed: {msg}")
        self._finish_run_ui()

    def _on_worker_done(self, out_path_str: str) -> None:
        self._log(f"[ui] MPPT done -> {out_path_str}")
        self._finish_run_ui()
        self.refresh_runs()

    def _finish_run_ui(self) -> None:
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self._poll is not None:
            self._poll.stop()
            self._poll = None
        self._worker = None

    def _on_live_sample(self, row: Dict[str, Any]) -> None:
        rd = self._rd
        if rd is None:
            return

        tt = _safe_float(row.get("t"))
        if tt is None:
            return
        rd.t.append(float(tt))
        rd.v.append(float(_safe_float(row.get("v")) or float("nan")))
        rd.i.append(float(_safe_float(row.get("i")) or float("nan")))
        rd.p.append(float(_safe_float(row.get("p")) or float("nan")))
        rd.g.append(float(_safe_float(row.get("g")) or float("nan")))
        rd.t_mod.append(float(_safe_float(row.get("t_mod")) or float("nan")))

        ad = row.get("action_debug")
        st = None
        rsn = None
        if isinstance(ad, str) and ad.strip():
            dbg = _parse_action_debug(ad)
            st = dbg.get("state")
            rsn = dbg.get("reason")

        last_state = rd.state[-1] if rd.state else "UNKNOWN"
        if isinstance(st, str) and st:
            last_state = st
        rd.state.append(last_state)
        rd.reason.append(rsn if isinstance(rsn, str) and rsn else None)

    def _refresh_live_plot(self) -> None:
        if self._rd is None or len(self._rd.t) < 2:
            return
        self._plot(self._rd)

    # ---------------------------
    # Plotting
    # ---------------------------
    def _clear_regions(self) -> None:
        if pg is None:
            return
        for r in self._state_regions:
            try:
                self.g_plot.removeItem(r)
            except Exception:
                pass
            try:
                self.t_plot.removeItem(r)
            except Exception:
                pass
            try:
                self.v_plot.removeItem(r)
            except Exception:
                pass
            try:
                self.p_plot.removeItem(r)
            except Exception:
                pass
        self._state_regions = []

    def _plot(self, rd: RunData) -> None:
        if pg is None:
            return
        if not hasattr(self, "g_plot"):
            return

        # If live overrides are set, show them as flat lines for G and T.
        # But if the user is running a CSV profile, preserve the profile shape in the plot.
        g_series = rd.g
        t_series = rd.t_mod
        use_csv = hasattr(self, "use_csv_chk") and self.use_csv_chk.isChecked()
        if (not use_csv) and self.overrides is not None:
            if self.overrides.irradiance is not None:
                g_series = [float(self.overrides.irradiance)] * len(rd.t)
            if self.overrides.temperature_c is not None:
                t_series = [float(self.overrides.temperature_c)] * len(rd.t)

        self.g_plot.clear()
        self.t_plot.clear()
        self.v_plot.clear()
        self.p_plot.clear()
        self._clear_regions()

        segs = build_state_segments(rd.t, rd.state)
        brushes = {
            "NORMAL": (0, 160, 80, 40),
            "GLOBAL_SEARCH": (245, 158, 11, 45),
            "LOCK_HOLD": (59, 130, 246, 40),
            "UNKNOWN": (120, 120, 120, 25),
        }
        for st, t0, t1 in segs:
            rgba = brushes.get(st, brushes["UNKNOWN"])
            region = pg.LinearRegionItem(values=(t0, t1), brush=pg.mkBrush(*rgba), movable=False)
            region.setZValue(-10)
            self.g_plot.addItem(region)
            self.t_plot.addItem(region)
            self.v_plot.addItem(region)
            self.p_plot.addItem(region)
            self._state_regions.append(region)

        self.g_plot.plot(rd.t, g_series)
        self.t_plot.plot(rd.t, t_series)
        self.v_plot.plot(rd.t, rd.v)
        self.p_plot.plot(rd.t, rd.p)

        if rd.t:
            self.g_plot.setXRange(rd.t[0], rd.t[-1], padding=0.01)
