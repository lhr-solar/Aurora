

"""ui.desktop.mppt_dashboard

MPPT Dashboard (initial scaffold).

This widget is designed to be embedded in `ui.desktop.main_window.MainWindow`.
For now it provides:
- a run picker for CSVs under `data/runs/`
- a file chooser to load any CSV
- a lightweight summary panel (row counts, time range, state transitions)

Plots/state shading will be added next.
"""

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QComboBox,
    QLineEdit,
    QMessageBox,
)

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover
    pg = None  # type: ignore

# Simulation engine (Option A)
from simulators.engine import SimulationConfig, SimulationEngine, LiveOverrides
from core.controller.hybrid_controller import HybridConfig
from simulators.mppt_sim import get_profile

from ui.desktop.terminal_panel import TerminalPanel


@dataclass
class RunSummary:
    path: Path
    n_rows: int
    t_start: float
    t_end: float
    dt_mode: Optional[float]
    g_min: Optional[float]
    g_max: Optional[float]
    p_min: Optional[float]
    p_max: Optional[float]
    states: Dict[str, int]
    transitions: List[Tuple[str, str, float]]


@dataclass
class RunData:
    path: Path
    t: List[float]
    v: List[float]
    i: List[float]
    p: List[float]
    g: List[float]
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
    """Parse the `action_debug` column (stringified dict) safely."""
    try:
        d = ast.literal_eval(s)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def summarize_run_csv(path: Path) -> RunSummary:
    """Load a run CSV and compute a lightweight summary.

    Uses only stdlib csv + ast (no pandas dependency in the UI layer).
    """
    n_rows = 0
    t_vals: List[float] = []
    dt_vals: List[float] = []
    g_vals: List[float] = []
    p_vals: List[float] = []

    states: List[str] = []
    state_counts: Dict[str, int] = {}

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_rows += 1

            t = _safe_float(row.get("t"))
            if t is not None:
                t_vals.append(t)

            dt = _safe_float(row.get("dt"))
            if dt is not None:
                dt_vals.append(dt)

            g = _safe_float(row.get("g"))
            if g is not None:
                g_vals.append(g)

            p = _safe_float(row.get("p"))
            if p is not None:
                p_vals.append(p)

            ad = row.get("action_debug")
            if isinstance(ad, str) and ad.strip():
                dbg = _parse_action_debug(ad)
                st = dbg.get("state")
                if isinstance(st, str) and st:
                    states.append(st)
                    state_counts[st] = state_counts.get(st, 0) + 1

    t_start = t_vals[0] if t_vals else 0.0
    t_end = t_vals[-1] if t_vals else 0.0

    # crude mode for dt (most runs have constant dt)
    dt_mode: Optional[float] = None
    if dt_vals:
        # round to reduce float noise
        rounded = [round(x, 9) for x in dt_vals]
        # simple frequency count
        freq: Dict[float, int] = {}
        for x in rounded:
            freq[x] = freq.get(x, 0) + 1
        dt_mode = max(freq.items(), key=lambda kv: kv[1])[0]

    g_min = min(g_vals) if g_vals else None
    g_max = max(g_vals) if g_vals else None
    p_min = min(p_vals) if p_vals else None
    p_max = max(p_vals) if p_vals else None

    transitions: List[Tuple[str, str, float]] = []
    if t_vals and states:
        # states list may be shorter if some rows lack action_debug; treat missing as last known
        # We'll reconstruct transitions using rows that *do* have states.
        last = states[0]
        # We don't have per-row time for each state entry unless we re-parse; keep it simple:
        # derive transitions by a second pass that collects (t,state) pairs.
        ts: List[Tuple[float, str]] = []
        with path.open("r", newline="") as f2:
            reader2 = csv.DictReader(f2)
            for row in reader2:
                ad = row.get("action_debug")
                t = _safe_float(row.get("t"))
                if not (isinstance(ad, str) and ad.strip() and t is not None):
                    continue
                st = _parse_action_debug(ad).get("state")
                if isinstance(st, str) and st:
                    ts.append((t, st))
        if ts:
            last_state = ts[0][1]
            for (tt, st) in ts[1:]:
                if st != last_state:
                    transitions.append((last_state, st, float(tt)))
                    last_state = st

    return RunSummary(
        path=path,
        n_rows=n_rows,
        t_start=float(t_start),
        t_end=float(t_end),
        dt_mode=dt_mode,
        g_min=float(g_min) if g_min is not None else None,
        g_max=float(g_max) if g_max is not None else None,
        p_min=float(p_min) if p_min is not None else None,
        p_max=float(p_max) if p_max is not None else None,
        states=state_counts,
        transitions=transitions,
    )


def load_run_csv(path: Path) -> RunData:
    """Load a run CSV into arrays for plotting."""
    t: List[float] = []
    v: List[float] = []
    i_: List[float] = []
    p: List[float] = []
    g: List[float] = []
    state: List[str] = []
    reason: List[Optional[str]] = []

    last_state: str = "UNKNOWN"

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tt = _safe_float(row.get("t"))
            vv = _safe_float(row.get("v"))
            ii = _safe_float(row.get("i"))
            pp = _safe_float(row.get("p"))
            gg = _safe_float(row.get("g"))

            if tt is None:
                continue

            t.append(float(tt))
            v.append(float(vv) if vv is not None else float("nan"))
            i_.append(float(ii) if ii is not None else float("nan"))
            p.append(float(pp) if pp is not None else float("nan"))
            g.append(float(gg) if gg is not None else float("nan"))

            st = None
            rsn = None
            ad = row.get("action_debug")
            if isinstance(ad, str) and ad.strip():
                dbg = _parse_action_debug(ad)
                st = dbg.get("state")
                rsn = dbg.get("reason")

            if isinstance(st, str) and st:
                last_state = st
            state.append(last_state)
            reason.append(rsn if isinstance(rsn, str) and rsn else None)

    return RunData(path=path, t=t, v=v, i=i_, p=p, g=g, state=state, reason=reason)


def build_state_segments(t: List[float], state: List[str]) -> List[Tuple[str, float, float]]:
    """Return contiguous (state, t_start, t_end) segments."""
    if not t or not state or len(t) != len(state):
        return []

    segs: List[Tuple[str, float, float]] = []
    cur = state[0]
    t0 = t[0]
    for k in range(1, len(t)):
        if state[k] != cur:
            segs.append((cur, float(t0), float(t[k - 1])))
            cur = state[k]
            t0 = t[k]
    segs.append((cur, float(t0), float(t[-1])))
    return segs


class _MPPTWorker(QThread):
    """Run MPPT simulation in-process (Option A) and stream samples."""

    sample = pyqtSignal(dict)
    failed = pyqtSignal(str)
    done = pyqtSignal(str)

    def __init__(
        self,
        *,
        out_path: Path,
        algo: str,
        profile_name: str,
        total_time: float,
        dt: float,
        overrides: Optional[LiveOverrides],
    ) -> None:
        super().__init__()
        self.out_path = out_path
        self.algo = algo
        self.profile_name = profile_name
        self.total_time = total_time
        self.dt = dt
        self.overrides = overrides
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            profile = get_profile(self.profile_name)

            # Force the NORMAL (local tracker) algorithm while keeping the hybrid structure.
            hcfg = HybridConfig(normal_name=self.algo)

            # Ensure output directory exists
            self.out_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare CSV writer. We keep the existing UI expectations:
            # - action_debug is a dict-like debug payload containing state/reason, etc.
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
                    # `rec` is JSON-friendly from SimulationEngine: includes `action` dict.
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
                        # store as a Python-literal string so the UI can parse with ast.literal_eval
                        "action_debug": repr(debug),
                    }
                    writer.writerow(row)
                    # Flush frequently so the run appears on disk and can be reloaded
                    f.flush()

                    self.sample.emit(row)

                cfg = SimulationConfig(
                    total_time=self.total_time,
                    dt=self.dt,
                    start_v=18.0,
                    array_kwargs={"n_strings": 2, "substrings_per_string": 3, "cells_per_substring": 18},
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


class MPPTDashboard(QWidget):
    """MPPT dashboard."""

    def __init__(self, *, terminal: Optional[TerminalPanel] = None, overrides: Optional[LiveOverrides] = None) -> None:
        super().__init__()
        self.terminal: Optional[TerminalPanel] = terminal
        self.overrides: Optional[LiveOverrides] = overrides

        # Resolve Aurora repo root regardless of where the UI is launched from.
        # This file lives at: Aurora/ui/desktop/mppt_dashboard.py
        self._repo_root = Path(__file__).resolve().parents[2]

        root = QVBoxLayout(self)

        # Header
        header_row = QHBoxLayout()
        title = QLabel("MPPT Dashboard")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        header_row.addWidget(title)
        header_row.addStretch(1)

        self.btn_refresh = QPushButton("Refresh list")
        self.btn_load_latest = QPushButton("Load latest")
        self.btn_open = QPushButton("Open CSV…")

        header_row.addWidget(self.btn_refresh)
        header_row.addWidget(self.btn_load_latest)
        header_row.addWidget(self.btn_open)

        root.addLayout(header_row)

        # Run controls (start/stop a new MPPT sim)
        controls = QHBoxLayout()

        controls.addWidget(QLabel("Algo"))
        self.algo_box = QComboBox()
        self.algo_box.setEditable(True)
        # Defaults (editable; user can type any string)
        # Keep in sync with simulators.mppt_sim (it prints the available list on error)
        self.algo_box.addItems(["ruca", "mepo", "nl_esc", "pando", "pso"])
        self.algo_box.setCurrentText("ruca")
        controls.addWidget(self.algo_box)

        controls.addWidget(QLabel("Profile"))
        self.profile_box = QComboBox()
        self.profile_box.setEditable(True)
        self.profile_box.addItems(["cloud", "stc", "csv"])
        self.profile_box.setCurrentText("cloud")
        controls.addWidget(self.profile_box)

        controls.addWidget(QLabel("Time"))
        self.time_edit = QLineEdit("0.5")
        self.time_edit.setFixedWidth(80)
        controls.addWidget(self.time_edit)

        controls.addWidget(QLabel("dt"))
        self.dt_edit = QLineEdit("0.001")
        self.dt_edit.setFixedWidth(80)
        controls.addWidget(self.dt_edit)

        controls.addWidget(QLabel("CSV name"))
        self.out_edit = QLineEdit("mppt_run.csv")
        self.out_edit.setFixedWidth(180)
        controls.addWidget(self.out_edit)

        self.btn_run = QPushButton("Run")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        controls.addWidget(self.btn_run)
        controls.addWidget(self.btn_stop)

        controls.addStretch(1)
        root.addLayout(controls)

        # Main split: left run list, right summary/plots
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.run_list = QListWidget()
        self.run_list.setAlternatingRowColors(True)
        left_layout.addWidget(QLabel("Runs (data/runs/*.csv)") )
        left_layout.addWidget(self.run_list)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # --- New: splitter with summary (top), plots (middle), log (bottom) ---
        right_split = QSplitter(Qt.Orientation.Vertical)

        # --- Summary (top) ---
        summary_box = QWidget()
        summary_layout = QVBoxLayout(summary_box)
        summary_layout.setContentsMargins(0, 0, 0, 0)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlaceholderText("Select a run to see a summary…")

        summary_layout.addWidget(QLabel("Summary"))
        summary_layout.addWidget(self.summary)

        # --- Plots (middle) ---
        plots_box = QWidget()
        plots_layout = QVBoxLayout(plots_box)
        plots_layout.setContentsMargins(0, 0, 0, 0)

        plots_layout.addWidget(QLabel("Plots"))

        if pg is None:
            self._plot_note = QLabel("pyqtgraph is not available. Install it to enable plotting.")
            self._plot_note.setStyleSheet("color: #a00;")
            self._plot_note.setWordWrap(True)
            plots_layout.addWidget(self._plot_note)
            self.p_plot = None
            self.v_plot = None
        else:
            # Two stacked plots sharing X (time)
            self.p_plot = pg.PlotWidget()
            self.v_plot = pg.PlotWidget()
            self.p_plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.v_plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

            self.p_plot.setLabel("bottom", "Time", units="s")
            self.p_plot.setLabel("left", "Power", units="W")
            self.p_plot.showGrid(x=True, y=True, alpha=0.25)

            self.v_plot.setLabel("bottom", "Time", units="s")
            self.v_plot.setLabel("left", "Voltage", units="V")
            self.v_plot.showGrid(x=True, y=True, alpha=0.25)

            plots_layout.addWidget(self.p_plot, 2)
            plots_layout.addWidget(self.v_plot, 2)

            # Keep track of region items for easy clearing
            self._state_regions: List[Any] = []

        # --- Terminal / log (bottom) ---
        log_box = QWidget()
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(0, 0, 0, 0)

        # Use the shared terminal dock if provided; otherwise create a local panel.
        if self.terminal is None:
            self.terminal = TerminalPanel(title="Run log")
        log_layout.addWidget(self.terminal)

        right_split.addWidget(summary_box)
        right_split.addWidget(plots_box)
        right_split.addWidget(log_box)
        right_split.setStretchFactor(0, 1)
        right_split.setStretchFactor(1, 2)
        right_split.setStretchFactor(2, 1)

        right_layout.addWidget(right_split)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        root.addWidget(splitter)

        # Wire events
        self.btn_refresh.clicked.connect(self.refresh_runs)
        self.btn_load_latest.clicked.connect(self.load_latest)
        self.btn_open.clicked.connect(self.open_csv)
        self.run_list.itemSelectionChanged.connect(self._on_select)

        self.btn_run.clicked.connect(self.run_sim)
        self.btn_stop.clicked.connect(self.stop_sim)

        self._worker: Optional[_MPPTWorker] = None
        self._poll: Optional[QTimer] = None
        self._live_csv_path: Optional[Path] = None
        self._csv_file_pos: int = 0
        self._csv_header: Optional[List[str]] = None
        self._live_rd: Optional[RunData] = None

        # Initial population
        self.refresh_runs()

    # ---------------------------
    # Run discovery / loading
    # ---------------------------
    def _runs_dir(self) -> Path:
        return self._repo_root / "data" / "runs"

    def refresh_runs(self) -> None:
        self.run_list.clear()
        runs_dir = self._runs_dir()
        runs_dir.mkdir(parents=True, exist_ok=True)

        paths = sorted(runs_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in paths:
            item = QListWidgetItem(p.name)
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            self.run_list.addItem(item)

        if not paths:
            self.summary.setPlainText(
                "No CSV runs found in data/runs.\n\n"
                "Tip: run a simulation with:\n"
                "  python3 -m simulators.source_sim --out-csv my_run.csv\n"
                "or\n"
                "  python3 -m simulators.mppt_sim --csv my_run.csv\n"
            )

    def load_latest(self) -> None:
        runs_dir = self._runs_dir()
        paths = sorted(runs_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not paths:
            self.summary.setPlainText("No runs available in data/runs.")
            return
        self._load_and_show(paths[0])

    def open_csv(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open MPPT run CSV",
            str(self._runs_dir()),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path_str:
            return
        self._load_and_show(Path(path_str))

    def _on_select(self) -> None:
        items = self.run_list.selectedItems()
        if not items:
            return
        p = Path(items[0].data(Qt.ItemDataRole.UserRole))
        self._load_and_show(p)

    def _load_and_show(self, path: Path) -> None:
        try:
            s = summarize_run_csv(path)
        except Exception as e:
            self.summary.setPlainText(f"Failed to load {path}: {type(e).__name__}: {e}")
            return

        lines: List[str] = []
        lines.append(f"File: {s.path}")
        lines.append(f"Rows: {s.n_rows}")
        lines.append(f"Time: {s.t_start:.3f} → {s.t_end:.3f} s")
        if s.dt_mode is not None:
            lines.append(f"dt (mode): {s.dt_mode:g} s")
        if s.g_min is not None and s.g_max is not None:
            lines.append(f"Irradiance: {s.g_min:g} → {s.g_max:g} W/m²")
        if s.p_min is not None and s.p_max is not None:
            lines.append(f"Power: {s.p_min:g} → {s.p_max:g} W")

        if s.states:
            lines.append("\nState counts:")
            for k, v in sorted(s.states.items(), key=lambda kv: kv[0]):
                lines.append(f"  {k}: {v}")

        if s.transitions:
            lines.append("\nTransitions:")
            for a, b, tt in s.transitions:
                lines.append(f"  {a} → {b} @ t={tt:.3f}s")
        else:
            lines.append("\nTransitions: (none detected)")

        self.summary.setPlainText("\n".join(lines))

        # Plot run (if plotting is available)
        try:
            rd = load_run_csv(path)
            self._plot_run(rd)
        except Exception:
            # plotting should never block summary rendering
            pass

        # Also select/highlight it in the list if it's under data/runs
        try:
            runs_dir = self._runs_dir().resolve()
            p_res = path.resolve()
            if str(p_res).startswith(str(runs_dir)):
                for i in range(self.run_list.count()):
                    it = self.run_list.item(i)
                    if Path(it.data(Qt.ItemDataRole.UserRole)).resolve() == p_res:
                        self.run_list.setCurrentItem(it)
                        break
        except Exception:
            pass


    def _plot_run(self, rd: RunData) -> None:
        """Plot P(t) and V(t) and shade by controller state."""
        if pg is None or self.p_plot is None or self.v_plot is None:
            return

        # Clear plots
        self.p_plot.clear()
        self.v_plot.clear()

        # Clear old regions
        for r in getattr(self, "_state_regions", []):
            try:
                self.p_plot.removeItem(r)
            except Exception:
                pass
            try:
                self.v_plot.removeItem(r)
            except Exception:
                pass
        self._state_regions = []

        # Shade by state
        segs = build_state_segments(rd.t, rd.state)

        # Stable, readable brushes per state (light transparency)
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
            self.p_plot.addItem(region)
            self.v_plot.addItem(region)
            self._state_regions.append(region)

        # Basic lines
        self.p_plot.plot(rd.t, rd.p)
        self.v_plot.plot(rd.t, rd.v)

        # Improve view
        if rd.t:
            self.p_plot.setXRange(rd.t[0], rd.t[-1], padding=0.01)
            self.v_plot.setXRange(rd.t[0], rd.t[-1], padding=0.01)


    # ---------------------------
    # Run / Stop (live)
    # ---------------------------
    def _append_log(self, text: str) -> None:
        if self.terminal is not None:
            self.terminal.append_text(text)

    def run_sim(self) -> None:
        """Run a new MPPT simulation in-process (Option A) and plot live."""
        # Validate numeric inputs
        try:
            sim_time = float(self.time_edit.text().strip())
            dt = float(self.dt_edit.text().strip())
            if sim_time <= 0 or dt <= 0:
                raise ValueError
        except Exception:
            QMessageBox.warning(self, "Invalid settings", "Time and dt must be positive numbers.")
            return

        algo = self.algo_box.currentText().strip() or "ruca"
        known_algos = {self.algo_box.itemText(i) for i in range(self.algo_box.count())}
        if algo not in known_algos:
            QMessageBox.warning(
                self,
                "Unknown algorithm",
                "Unknown algorithm name.\n\n"
                f"You entered: {algo}\n"
                f"Known options: {', '.join(sorted(known_algos))}",
            )
            return

        profile = self.profile_box.currentText().strip() or "cloud"
        if profile.lower() == "csv":
            QMessageBox.information(
                self,
                "CSV profile",
                "CSV profiles are supported in source_sim right now.\n"
                "For MPPT live mode, select 'cloud' or 'stc' for now.",
            )
            return

        out_name = self.out_edit.text().strip() or "mppt_run.csv"
        if not out_name.lower().endswith(".csv"):
            out_name += ".csv"

        out_path = self._runs_dir() / out_name
        self._live_csv_path = out_path
        self._live_rd = RunData(path=out_path, t=[], v=[], i=[], p=[], g=[], state=[], reason=[])

        # Clear UI
        self._append_log(f"[ui] Starting MPPT (in-process) -> {out_path}\n")
        self._append_log(f"[ui] algo={algo} profile={profile} time={sim_time} dt={dt}\n")

        if pg is not None and self.p_plot is not None and self.v_plot is not None:
            self.p_plot.clear(); self.v_plot.clear()
            self._state_regions = []

        # Stop an existing run if needed
        if self._worker is not None:
            self._worker.request_stop()
            self._worker.wait(1000)
            self._worker = None

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

        # Start worker
        w = _MPPTWorker(
            out_path=out_path,
            algo=algo,
            profile_name=profile,
            total_time=sim_time,
            dt=dt,
            overrides=self.overrides,
        )
        w.sample.connect(self._on_live_sample)
        w.failed.connect(self._on_worker_failed)
        w.done.connect(self._on_worker_done)
        self._worker = w
        w.start()

        # Plot refresh timer (avoid redrawing for every sample)
        if self._poll is not None:
            self._poll.stop()
        self._poll = QTimer(self)
        self._poll.setInterval(200)
        self._poll.timeout.connect(self._refresh_live_plot)
        self._poll.start()

    def stop_sim(self) -> None:
        if self._worker is None:
            return
        self._append_log("[ui] Stopping MPPT…\n")
        self._worker.request_stop()

    def _start_polling_csv(self, path: Path) -> None:
        if self._poll is not None:
            self._poll.stop()
        self._poll = QTimer(self)
        self._poll.setInterval(200)
        self._poll.timeout.connect(self._poll_csv)
        self._poll.start()

    def _poll_csv(self) -> None:
        """Poll the output CSV for new rows and update plots incrementally."""
        if self._live_csv_path is None:
            return
        path = self._live_csv_path
        if not path.exists() or path.stat().st_size == 0:
            return

        # Read only newly appended bytes
        try:
            with path.open("r", newline="") as f:
                f.seek(self._csv_file_pos)
                chunk = f.read()
                self._csv_file_pos = f.tell()
        except Exception:
            return

        if not chunk.strip():
            return

        lines = chunk.splitlines()

        # If header not captured yet, use the first non-empty line as header
        if self._csv_header is None:
            # If we started reading mid-file, we may not see the header; fallback to full read
            if "," in lines[0] and "t" in lines[0]:
                self._csv_header = [c.strip() for c in lines[0].split(",")]
                data_lines = lines[1:]
            else:
                # fallback: full parse once
                try:
                    rd = load_run_csv(path)
                    self._live_rd = rd
                    self._plot_run(rd)
                except Exception:
                    pass
                return
        else:
            data_lines = lines

        # Parse rows
        if self._live_rd is None:
            self._live_rd = RunData(path=path, t=[], v=[], i=[], p=[], g=[], state=[], reason=[])

        last_state = self._live_rd.state[-1] if self._live_rd.state else "UNKNOWN"

        for line in data_lines:
            if not line.strip():
                continue
            parts = list(csv.reader([line]))[0]
            if self._csv_header is None or len(parts) != len(self._csv_header):
                continue
            row = dict(zip(self._csv_header, parts))

            tt = _safe_float(row.get("t"))
            if tt is None:
                continue

            self._live_rd.t.append(float(tt))
            self._live_rd.v.append(float(_safe_float(row.get("v")) or float("nan")))
            self._live_rd.i.append(float(_safe_float(row.get("i")) or float("nan")))
            self._live_rd.p.append(float(_safe_float(row.get("p")) or float("nan")))
            self._live_rd.g.append(float(_safe_float(row.get("g")) or float("nan")))

            st = None
            rsn = None
            ad = row.get("action_debug")
            if isinstance(ad, str) and ad.strip():
                dbg = _parse_action_debug(ad)
                st = dbg.get("state")
                rsn = dbg.get("reason")
            if isinstance(st, str) and st:
                last_state = st
            self._live_rd.state.append(last_state)
            self._live_rd.reason.append(rsn if isinstance(rsn, str) and rsn else None)

        # Update plots (throttle by only plotting if we have enough points)
        if self._live_rd and len(self._live_rd.t) >= 2:
            self._plot_run(self._live_rd)
    def _on_live_sample(self, row: Dict[str, Any]) -> None:
        """Receive a streamed CSV-row-shaped sample from the worker."""
        rd = self._live_rd
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
        if self._live_rd is None or len(self._live_rd.t) < 2:
            return
        self._plot_run(self._live_rd)

    def _on_worker_failed(self, msg: str) -> None:
        self._append_log(f"[ui] MPPT worker failed: {msg}\n")
        self._finish_run_ui()

    def _on_worker_done(self, out_path_str: str) -> None:
        self._append_log(f"[ui] MPPT worker done -> {out_path_str}\n")
        self._finish_run_ui()
        # Refresh list + load final
        self.refresh_runs()
        try:
            self._load_and_show(Path(out_path_str))
        except Exception:
            pass

    def _finish_run_ui(self) -> None:
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self._poll is not None:
            self._poll.stop()
            self._poll = None
        self._worker = None