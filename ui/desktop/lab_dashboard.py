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
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

from PyQt6.QtCore import Qt, QObject, QThread, QTimer, pyqtSignal
class _TerminalStream(QObject):
    """File-like stream that forwards writes into the UI via a Qt signal."""

    text_written = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._buf = ""

    def write(self, s: str) -> int:
        try:
            if not s:
                return 0
            self._buf += str(s)
            # Emit full lines; keep partials buffered.
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line != "":
                    self.text_written.emit(line)
            return len(s)
        except Exception:
            return 0

    def flush(self) -> None:
        try:
            if self._buf:
                self.text_written.emit(self._buf)
                self._buf = ""
        except Exception:
            pass
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
    QComboBox,
    QSizePolicy,
)

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover
    pg = None

from simulators.engine import LiveOverrides, SimulationConfig, SimulationEngine
from core.controller.hybrid_controller import HybridConfig
from simulators.mppt_sim import get_profile
from core.mppt_algorithms import registry as mppt_registry

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

    # Robust GMPP telemetry (optional; may be NaN for older runs)
    v_gmp_ref: List[float]
    p_gmp_ref: List[float]
    v_best: List[float]
    p_best: List[float]
    eff_best: List[float]
    g_strings: List[Optional[str]]
    
    k: List[float]
    v_cmd: List[float]
    v_true: List[float]
    i_true: List[float]
    p_true: List[float]


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

    v_gmp_ref: List[float] = []
    p_gmp_ref: List[float] = []
    v_best: List[float] = []
    p_best: List[float] = []
    eff_best: List[float] = []
    g_strings: List[Optional[str]] = []
    k: List[float] = []
    v_cmd: List[float] = []
    v_true: List[float] = []
    i_true: List[float] = []
    p_true: List[float] = []

    last_state = "UNKNOWN"

    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            tt = _safe_float(row.get("t"))
            if tt is None:
                continue
            t.append(float(tt))
            k.append(float(_safe_float(row.get("k")) or float("nan")))
            v_cmd.append(float(_safe_float(row.get("v_cmd")) or float("nan")))
            v_true.append(float(_safe_float(row.get("v_true")) or float("nan")))
            i_true.append(float(_safe_float(row.get("i_true")) or float("nan")))
            p_true.append(float(_safe_float(row.get("p_true")) or float("nan")))
            v.append(float(_safe_float(row.get("v")) or float("nan")))
            i.append(float(_safe_float(row.get("i")) or float("nan")))
            p.append(float(_safe_float(row.get("p")) or float("nan")))
            g.append(float(_safe_float(row.get("g")) or float("nan")))
            t_mod.append(float(_safe_float(row.get("t_mod")) or float("nan")))

            # GMPP reference/validator columns
            v_gmp_ref.append(float(_safe_float(row.get("v_gmp_ref")) or float("nan")))
            p_gmp_ref.append(float(_safe_float(row.get("p_gmp_ref")) or float("nan")))
            v_best.append(float(_safe_float(row.get("v_best")) or float("nan")))
            p_best.append(float(_safe_float(row.get("p_best")) or float("nan")))
            eff_best.append(float(_safe_float(row.get("eff_best")) or float("nan")))
            gs = row.get("g_strings")
            g_strings.append(gs if isinstance(gs, str) and gs.strip() else None)

            # Prefer explicit state/reason columns (newer runs); fall back to action_debug (older runs)
            st = row.get("state")
            rsn = row.get("reason")

            if isinstance(st, str) and st.strip():
                last_state = st.strip()
                state.append(last_state)
            else:
                ad = row.get("action_debug")
                st2 = None
                if isinstance(ad, str) and ad.strip():
                    dbg = _parse_action_debug(ad)
                    st2 = dbg.get("state")
                if isinstance(st2, str) and st2:
                    last_state = st2
                state.append(last_state)

            if isinstance(rsn, str) and rsn.strip():
                reason.append(rsn.strip())
            else:
                ad = row.get("action_debug")
                rsn2 = None
                if isinstance(ad, str) and ad.strip():
                    dbg = _parse_action_debug(ad)
                    rsn2 = dbg.get("reason")
                reason.append(rsn2 if isinstance(rsn2, str) and rsn2 else None)

    return RunData(
        path=path,
        t=t,
        v=v,
        i=i,
        p=p,
        g=g,
        t_mod=t_mod,
        state=state,
        reason=reason,
        v_gmp_ref=v_gmp_ref,
        p_gmp_ref=p_gmp_ref,
        v_best=v_best,
        p_best=p_best,
        eff_best=eff_best,
        g_strings=g_strings,
        k=k,
        v_cmd=v_cmd,
        v_true=v_true,
        i_true=i_true,
        p_true=p_true,
    )


def load_env_profile_csv(path: Path) -> List[Tuple[float, float, float]]:
    """Load an environment profile from CSV with header: t,g,t_c.

    IMPORTANT: SimulationConfig.env_profile is expected to be *iterable*.
    The engine supports a legacy tuple format:
        List[(time_s, irradiance_w_m2, temperature_c)]

    This function therefore returns a list of tuples (t, g, t_c), sorted by t.
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

    return rows


def resolve_repo_path(repo_root: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (repo_root / pp)

# ---------------------------
# Shared config helpers
# ---------------------------

RUN_CSV_FIELDNAMES: List[str] = [
    "t",
    "dt",
    "k",
    "v_cmd",
    "v_true",
    "i_true",
    "p_true",
    "v",
    "i",
    "p",
    "g",
    "t_mod",
    "v_ref",
    "state",
    "reason",
    "action_debug",
    "v_gmp_ref",
    "p_gmp_ref",
    "v_best",
    "p_best",
    "eff_best",
    "g_strings",
]


def resolve_env_profile(
    *,
    use_csv_profile: bool,
    csv_profile_path: Optional[Path],
    profile_name: str,
) -> List[Tuple[float, float, float]]:
    """Return the env profile in the engine's expected iterable format."""
    if use_csv_profile:
        if csv_profile_path is None:
            raise ValueError("CSV profile mode enabled but no csv_profile_path provided")
        return load_env_profile_csv(csv_profile_path)

    if isinstance(profile_name, str) and profile_name.strip():
        return get_profile(profile_name.strip())

    # Default baseline (sliders / LiveOverrides typically override this in manual mode).
    return [(0.0, 1000.0, 25.0)]


def resolve_controller_selection(
    selection: str,
) -> Tuple[str, Optional[str], Optional[HybridConfig]]:
    """Return (controller_mode, algo_name, controller_cfg)."""
    sel = (selection or "hybrid").strip()
    sel_l = sel.lower()

    if sel_l in ("hybrid", "hybrid_mppt", "hybridmppt"):
        return ("hybrid", None, HybridConfig())

    if not mppt_registry.is_valid(sel):
        raise SystemExit(
            f"[ui] Unknown algorithm '{sel}'. Available: {', '.join(mppt_registry.ALGORITHMS)}"
        )

    return ("single", mppt_registry.resolve_key(sel), None)


# ---------------------------
# Record -> CSV-row shaping
# ---------------------------

def record_to_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an engine record dict into the CSV row shape used by the UI."""
    action = rec.get("action") if isinstance(rec.get("action"), dict) else {}
    debug = action.get("debug") if isinstance(action.get("debug"), dict) else {}
    gmpp = rec.get("gmpp") if isinstance(rec.get("gmpp"), dict) else {}

    v_ref = action.get("v_ref")
    g_strings = rec.get("g_strings")
    st = debug.get("state")
    rsn = debug.get("reason")

    row = {
        "t": rec.get("t"),
        "dt": rec.get("dt"),
        "k": rec.get("k"),
        "v_cmd": rec.get("v_cmd"),
        "v_true": rec.get("v_true"),
        "i_true": rec.get("i_true"),
        "p_true": rec.get("p_true"),
        "v": rec.get("v"),
        "i": rec.get("i"),
        "p": rec.get("p"),
        "g": rec.get("g"),
        "t_mod": rec.get("t_mod"),
        "v_ref": v_ref,
        "state": st if isinstance(st, str) else "",
        "reason": rsn if isinstance(rsn, str) else "",
        "action_debug": repr(debug),
        "v_gmp_ref": gmpp.get("v_gmp_ref"),
        "p_gmp_ref": gmpp.get("p_gmp_ref"),
        "v_best": gmpp.get("v_best"),
        "p_best": gmpp.get("p_best"),
        "eff_best": gmpp.get("eff_best"),
        "g_strings": repr(g_strings) if g_strings is not None else "",
    }
    return row


class _MPPTWorker(QThread):
    """Run MPPT simulation in-process (Option A) and stream CSV-row-shaped samples."""

    sample = pyqtSignal(dict)
    failed = pyqtSignal(str)
    done = pyqtSignal(str)

    def __init__(
        self,
        *,
        out_path: Path,
        selection: str,
        profile_name: str,
        use_csv_profile: bool,
        csv_profile_path: Optional[Path],
        total_time: float,
        dt: float,
        overrides: Optional[LiveOverrides],
        gmpp_ref: bool,
    ) -> None:
        super().__init__()
        self.out_path = out_path
        self.selection = selection
        self.profile_name = profile_name
        self.use_csv_profile = use_csv_profile
        self.csv_profile_path = csv_profile_path
        self.total_time = total_time
        self.dt = dt
        self.overrides = overrides

        # Enforce CSV profile mode: LiveOverrides always win over env_profile in the engine.
        # If CSV is enabled, do NOT allow overrides to mask it.
        if self.use_csv_profile:
            self.overrides = None

        self.gmpp_ref = bool(gmpp_ref)
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            profile = resolve_env_profile(
                use_csv_profile=self.use_csv_profile,
                csv_profile_path=self.csv_profile_path,
                profile_name=self.profile_name,
            )

            controller_mode, algo_name, controller_cfg = resolve_controller_selection(self.selection)

            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = RUN_CSV_FIELDNAMES

            with self.out_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                _rows_since_flush = 0
                _last_flush_wall = time.monotonic()
                _flush_every_rows = 200
                _flush_every_s = 0.5

                def _on_sample(rec: Dict[str, Any]) -> None:
                    row = record_to_row(rec)
                    writer.writerow(row)

                    nonlocal _rows_since_flush, _last_flush_wall
                    _rows_since_flush += 1
                    now = time.monotonic()
                    if _rows_since_flush >= _flush_every_rows or (now - _last_flush_wall) >= _flush_every_s:
                        f.flush()
                        _rows_since_flush = 0
                        _last_flush_wall = now

                    self.sample.emit(row)

                cfg = SimulationConfig(
                    total_time=self.total_time,
                    dt=self.dt,
                    env_profile=profile,
                    controller_mode=controller_mode,
                    algo_name=algo_name,
                    algo_kwargs={},
                    controller_cfg=controller_cfg,
                    overrides=self.overrides,
                    gmpp_ref=self.gmpp_ref,
                    gmpp_ref_period_s=0.05,
                    gmpp_ref_points=121,
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
        
        # Capture stdout/stderr during runs so engine/worker `print()` output
        # (including end-of-run stats) appears in the in-app terminal.
        self._capturing_stdio = False
        self._stdout_orig = None
        self._stderr_orig = None
        self._stream_out = _TerminalStream()
        self._stream_err = _TerminalStream()

        def _emit_line(line: str) -> None:
            try:
                if line is None:
                    return
                s = str(line)
                if s == "":
                    return
                self._log(s)
            except Exception:
                pass

        # Queue into the GUI thread even if writes occur from a worker thread.
        try:
            self._stream_out.text_written.connect(_emit_line, type=Qt.ConnectionType.QueuedConnection)
        except Exception:
            self._stream_out.text_written.connect(_emit_line)
        try:
            self._stream_err.text_written.connect(_emit_line, type=Qt.ConnectionType.QueuedConnection)
        except Exception:
            self._stream_err.text_written.connect(_emit_line)
        
        # Terminal streaming (per-sample) config.
        # dt can be tiny (e.g. 0.001) so logging EVERY sample can freeze the UI.
        # We default to streaming at a sane rate, but still show the latest state continuously.
        self._term_stream_samples = True
        self._term_period_default_s = 0.05
        self._term_period_s = self._term_period_default_s  # log at most every 50ms of wall time
        self._last_term_wall = 0.0

        # Terminal buffering: when printing every step (period=0), we batch UI updates
        # to avoid starving the Qt event loop. All samples are still printed.
        self._term_buf = deque()  # type: ignore[var-annotated]
        self._term_flush_timer = QTimer(self)
        self._term_flush_timer.setInterval(30)
        self._term_flush_timer.timeout.connect(self._flush_terminal_buffer)

        # Repo root (Aurora/ui/desktop/lab_dashboard.py)
        self._repo_root = Path(__file__).resolve().parents[2]
        self._runs_root = self._repo_root / "data" / "runs"
        self._runs_root.mkdir(parents=True, exist_ok=True)

        self._worker: Optional[_MPPTWorker] = None
        self._poll: Optional[QTimer] = None

        # Continuous-mode runner state (no thread; driven by QTimer)
        self._cont_timer: Optional[QTimer] = None
        self._cont_eng: Optional[SimulationEngine] = None
        self._cont_f = None
        self._cont_writer: Optional[csv.DictWriter] = None
        self._cont_rows_since_flush = 0
        self._cont_last_flush_wall = 0.0
        self._cont_stopping = False
        self._pending_stop_msg: Optional[str] = None
        self._batch_stopping = False
        self._accept_live_samples = True
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
        # Keep the control panel narrow so plots get most of the horizontal space.
        left.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        left.setMinimumWidth(360)
        left.setMaximumWidth(460)

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
        self.algo_box = QComboBox()
        self.algo_box.setEditable(False)
        self.algo_box.setFixedWidth(160)

        self.algo_box.addItem("Hybrid (controller)", "hybrid")
        for key in mppt_registry.ALGORITHMS:
            self.algo_box.addItem(key, key)

        # Default to Hybrid to avoid conflating RUCA with the hybrid controller
        self.algo_box.setCurrentIndex(0)
        row1.addWidget(self.algo_box)

        # Continuous is the default mode for interactive slider use.
        # When CSV profile mode is enabled, we automatically disable continuous.
        self.cont_chk = QCheckBox("Continuous")
        self.cont_chk.setChecked(True)
        row1.addWidget(self.cont_chk)

        row1.addStretch(1)
        left_l.addLayout(row1)

        # --- CSV profile mode controls ---
        self.use_csv_chk = QCheckBox("Use CSV profile")
        left_l.addWidget(self.use_csv_chk)

        csv_row = QHBoxLayout()
        csv_row.addWidget(QLabel("CSV path"))
        self.csv_path_edit = QLineEdit("")
        self.csv_path_edit.setPlaceholderText("profiles/my_profile.csv")
        csv_row.addWidget(self.csv_path_edit, 1)
        self.btn_browse_profile = QPushButton("Browse…")
        csv_row.addWidget(self.btn_browse_profile)
        left_l.addLayout(csv_row)

        prof_btn_row = QHBoxLayout()
        self.btn_profile_editor = QPushButton("Profile Editor…")
        prof_btn_row.addWidget(self.btn_profile_editor)

        self.btn_save_profile = QPushButton("Save as profile…")
        prof_btn_row.addWidget(self.btn_save_profile)

        prof_btn_row.addStretch(1)
        left_l.addLayout(prof_btn_row)

        self.csv_path_edit.setEnabled(False)
        self.btn_browse_profile.setEnabled(False)


        row2 = QHBoxLayout()
        row2.setSpacing(4)

        time_lbl = QLabel("Time")
        time_lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        row2.addWidget(time_lbl)

        self.time_edit = QLineEdit("0.5")
        self.time_edit.setFixedWidth(80)
        self.time_edit.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        row2.addWidget(self.time_edit)

        row2.addSpacing(50)

        dt_lbl = QLabel("dt")
        dt_lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        row2.addWidget(dt_lbl)

        self.dt_edit = QLineEdit("0.001")
        self.dt_edit.setFixedWidth(80)
        self.dt_edit.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        row2.addWidget(self.dt_edit)

        # Keep the fields packed to the left instead of spreading across the row.
        row2.addStretch(1)
        left_l.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("CSV"))
        self.out_edit = QLineEdit("mppt_run.csv")
        row3.addWidget(self.out_edit, 1)
        left_l.addLayout(row3)

        # Robust GMPP validator toggle
        self.gmpp_chk = QCheckBox("Compute GMPP reference")
        self.gmpp_chk.setChecked(True)
        left_l.addWidget(self.gmpp_chk)
        
        # Terminal streaming controls
        self.term_stream_chk = QCheckBox("Terminal: stream samples")
        self.term_stream_chk.setChecked(True)
        left_l.addWidget(self.term_stream_chk)

        term_row = QHBoxLayout()
        term_row.addWidget(QLabel("Terminal period (s)"))
        self.term_period_edit = QLineEdit("0.05")
        self.term_period_edit.setFixedWidth(80)
        term_row.addWidget(self.term_period_edit)
        term_row.addStretch(1)
        left_l.addLayout(term_row)

        # Continuous tick (used only when continuous is enabled)
        cont_row = QHBoxLayout()
        cont_row.addWidget(QLabel("Tick (ms)"))
        self.cont_tick_edit = QLineEdit("33")
        self.cont_tick_edit.setFixedWidth(80)
        cont_row.addWidget(self.cont_tick_edit)
        cont_row.addStretch(1)
        left_l.addLayout(cont_row)

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
        right.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        if pg is None:
            right_l.addWidget(QLabel("pyqtgraph not available"))
        else:
            self.g_plot = pg.PlotWidget(title="Irradiance (W/m²)")
            self.t_plot = pg.PlotWidget(title="Temperature (°C)")
            self.v_plot = pg.PlotWidget(title="Voltage (V)")
            self.p_plot = pg.PlotWidget(title="Power (W)")
            
            # Persistent plot items (don’t recreate every refresh)
            self._g_curve = self.g_plot.plot([], [])
            self._t_curve = self.t_plot.plot([], [])
            self._v_curve = self.v_plot.plot([], [])
            self._p_curve = self.p_plot.plot([], [])
            self._p_gmpp_curve = self.p_plot.plot([], [])

            self._last_state_segs = []
            self._last_state_transitions: List[Tuple[str, float]] = []

            # Debounce plot refresh so sliders don’t trigger heavy work on every tick
            self._plot_debounce = QTimer(self)
            self._plot_debounce.setSingleShot(True)
            self._plot_debounce.setInterval(50)
            self._plot_debounce.timeout.connect(self._on_plot_debounce)

            # Keep X linked for easier inspection
            self.t_plot.setXLink(self.g_plot)
            self.v_plot.setXLink(self.g_plot)
            self.p_plot.setXLink(self.g_plot)

            right_l.addWidget(self.g_plot, 1)
            right_l.addWidget(self.t_plot, 1)
            right_l.addWidget(self.v_plot, 1)
            right_l.addWidget(self.p_plot, 1)

        split.addWidget(right)
        # Left panel is width-constrained (Fixed policy). Give remaining space to plots.
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)

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
        self.csv_path_edit.editingFinished.connect(self._sync_time_to_csv_profile)
        self.btn_profile_editor.clicked.connect(self.open_profile_editor)
        self.btn_save_profile.clicked.connect(self.save_current_as_profile)
        self.term_stream_chk.stateChanged.connect(self._on_terminal_settings_changed)
        self.term_period_edit.editingFinished.connect(self._on_terminal_settings_changed)
        self.cont_chk.stateChanged.connect(lambda _=None: self._on_continuous_mode_changed())
        self.cont_chk.stateChanged.connect(lambda _=None: self.cont_tick_edit.setEnabled(self._continuous_enabled()))
        # If a profile is saved/selected programmatically, also resync Time.
        self.use_csv_chk.stateChanged.connect(lambda _=None: self._sync_time_to_csv_profile())

        self.refresh_runs()
        self._on_override_changed()
        self._on_profile_mode_changed()
        self._on_terminal_settings_changed()
        # Ensure tick edit reflects initial continuous/csv state
        self.cont_tick_edit.setEnabled(self._continuous_enabled())
        # Apply continuous-mode mutual exclusivity + restore terminal period default if Continuous is on.
        self._on_continuous_mode_changed()
        
    def _on_continuous_mode_changed(self) -> None:
        """Keep Continuous and CSV modes mutually exclusive."""
        cont = bool(self.cont_chk.isChecked())

        # If user turns OFF continuous, force CSV mode ON.
        if not cont:
            if hasattr(self, "use_csv_chk") and not self.use_csv_chk.isChecked():
                self.use_csv_chk.blockSignals(True)
                self.use_csv_chk.setChecked(True)
                self.use_csv_chk.blockSignals(False)
            self._on_profile_mode_changed()
            return

        # If user turns ON continuous, force CSV mode OFF.
        if hasattr(self, "use_csv_chk") and self.use_csv_chk.isChecked():
            self.use_csv_chk.blockSignals(True)
            self.use_csv_chk.setChecked(False)
            self.use_csv_chk.blockSignals(False)

        # Continuous is interactive: restore terminal period to the default (rate-limited)
        # so the terminal doesn’t spam unless explicitly set to 0.
        try:
            self.term_period_edit.setText(f"{getattr(self, '_term_period_default_s', 0.05):g}")
        except Exception:
            try:
                self.term_period_edit.setText("0.05")
            except Exception:
                pass
        try:
            self._on_terminal_settings_changed()
        except Exception:
            pass

        self._on_profile_mode_changed()

    
    def _on_profile_mode_changed(self) -> None:
        use_csv = self.use_csv_chk.isChecked()

        # If using CSV, disable manual sliders (CSV drives the environment).
        self.g_slider.setEnabled(not use_csv)
        self.t_slider.setEnabled(not use_csv)

        # CSV widgets are only relevant when using CSV.
        self.csv_path_edit.setEnabled(use_csv)
        self.btn_browse_profile.setEnabled(use_csv)

        # When running a CSV profile, ensure live overrides are not applied.
        if use_csv and self.overrides is not None:
            self.overrides.irradiance = None
            self.overrides.temperature_c = None

        # Continuous and CSV modes are mutually exclusive.
        # When CSV is enabled, continuous must be OFF (and disabled).
        # When CSV is disabled, continuous must be ON.
        if hasattr(self, "cont_chk"):
            if use_csv:
                self.cont_chk.blockSignals(True)
                self.cont_chk.setChecked(False)
                self.cont_chk.setEnabled(False)
                self.cont_chk.blockSignals(False)
            else:
                self.cont_chk.setEnabled(True)
                self.cont_chk.blockSignals(True)
                self.cont_chk.setChecked(True)
                self.cont_chk.blockSignals(False)
                # Leaving CSV mode auto-enables Continuous, but we block signals above,
                # so `_on_continuous_mode_changed()` won't run. Restore the default
                # terminal period here so the terminal doesn't stay in "period=0" spam mode.
                try:
                    default_p = float(getattr(self, "_term_period_default_s", 0.05))
                except Exception:
                    default_p = 0.05
                try:
                    self.term_period_edit.setText(f"{default_p:g}")
                except Exception:
                    pass
                try:
                    self._term_period_s = default_p
                    self._last_term_wall = 0.0
                except Exception:
                    pass
                try:
                    self._on_terminal_settings_changed()
                except Exception:
                    pass

        if hasattr(self, "cont_tick_edit"):
            self.cont_tick_edit.setEnabled((not use_csv) and bool(getattr(self, "cont_chk", None) and self.cont_chk.isChecked()))

        # Saving a profile only makes sense in manual mode (sliders drive the environment).
        if hasattr(self, "btn_save_profile"):
            self.btn_save_profile.setEnabled(not use_csv)

        # When exiting CSV mode, re-apply the current slider values as overrides.
        if not use_csv:
            self._on_override_changed()
        # When CSV mode is enabled, keep Time in sync with the selected profile.
        if use_csv:
            self._sync_time_to_csv_profile()
            
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
        try:
            self._sync_time_to_csv_profile()
        except Exception:
            pass

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
            try:
                self._sync_time_to_csv_profile()
            except Exception:
                pass

        dlg.editor.profile_saved.connect(_on_saved)
        dlg.exec()

    def save_current_as_profile(self) -> None:
        """Save the current irradiance/temperature as a new environment profile CSV (t,g,t_c).

        Preference order:
        1) If we have streamed run data in `self._rd`, save the full time series (exact reproduction).
        2) Otherwise, save a flat profile from the current slider values across [0, Time].

        After saving, auto-enable CSV mode and point the CSV path to the saved file.
        """
        # If we're already in CSV mode, saving from sliders is ambiguous.
        if hasattr(self, "use_csv_chk") and self.use_csv_chk.isChecked():
            QMessageBox.information(self, "Save profile", "Disable 'Use CSV profile' to save from manual sliders.")
            return

        profiles_dir = self._repo_root / "profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)

        default_path = str(profiles_dir / "manual_profile.csv")
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save environment profile CSV",
            default_path,
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path_str:
            return

        out_path = Path(path_str)
        if out_path.suffix.lower() != ".csv":
            out_path = out_path.with_suffix(".csv")

        # Build rows (t,g,t_c)
        rows: List[Tuple[float, float, float]] = []

        rd = getattr(self, "_rd", None)
        if (
            rd is not None
            and getattr(rd, "t", None)
            and getattr(rd, "g", None)
            and getattr(rd, "t_mod", None)
            and len(rd.t) >= 2
            and len(rd.g) == len(rd.t)
            and len(rd.t_mod) == len(rd.t)
        ):
            # Save the exact time series used in the run.
            for tt, gg, tc in zip(rd.t, rd.g, rd.t_mod):
                try:
                    ttf = float(tt)
                    ggf = float(gg)
                    tcf = float(tc)
                except Exception:
                    continue
                # filter NaNs
                if ttf != ttf or ggf != ggf or tcf != tcf:
                    continue
                rows.append((ttf, ggf, tcf))
        else:
            # Fall back: flat profile from current sliders over [0, sim_time]
            try:
                sim_time = float(self.time_edit.text().strip())
                if sim_time <= 0:
                    sim_time = 1.0
            except Exception:
                sim_time = 1.0

            g = float(self.g_slider.value())
            t = float(self.t_slider.value())
            rows = [(0.0, g, t), (sim_time, g, t)]

        rows.sort(key=lambda x: x[0])
        if not rows:
            QMessageBox.warning(self, "Save profile", "No valid data available to save as a profile.")
            return

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["t", "g", "t_c"])
                w.writeheader()
                for tt, gg, tc in rows:
                    w.writerow({"t": tt, "g": gg, "t_c": tc})
        except Exception as e:
            QMessageBox.warning(self, "Save profile", f"{type(e).__name__}: {e}")
            return

        self._log(f"[ui] profile saved -> {out_path}")

        # Auto-select CSV mode and point to saved file.
        self.use_csv_chk.setChecked(True)
        try:
            rel = out_path.relative_to(self._repo_root)
            self.csv_path_edit.setText(str(rel))
        except Exception:
            self.csv_path_edit.setText(str(out_path))

    def _on_plot_debounce(self) -> None:
        if self._rd is not None:
            self._plot(self._rd)

    # ---------------------------
    # Helpers
    # ---------------------------
    def _sync_time_to_csv_profile(self) -> None:
        """If CSV profile mode is active and a valid CSV is selected, set Time to the profile end time."""
        try:
            if not (hasattr(self, "use_csv_chk") and self.use_csv_chk.isChecked()):
                return
        except Exception:
            return

        try:
            raw = self.csv_path_edit.text().strip()
        except Exception:
            raw = ""
        if not raw:
            return

        try:
            p = resolve_repo_path(self._repo_root, raw)
        except Exception:
            p = Path(raw)

        if not p.exists():
            return

        try:
            prof = load_env_profile_csv(p)
            if not prof:
                return
            t_last = float(prof[-1][0])
        except Exception:
            return

        # Update the Time UI field to match the profile duration.
        try:
            cur = float(self.time_edit.text().strip())
        except Exception:
            cur = None

        # Only touch the UI if it actually needs updating (avoids annoying cursor jumps).
        if cur is None or abs(cur - t_last) > 1e-9:
            try:
                self.time_edit.setText(f"{t_last:g}")
            except Exception:
                pass
            try:
                self._log(f"[ui] Time set from CSV profile: {t_last:g}s")
            except Exception:
                pass
    def _log(self, msg: str) -> None:
        """Append a line to the in-app terminal panel (if present)."""
        if self.terminal is not None:
            try:
                self.terminal.append_line(msg)
            except Exception:
                pass

    def _install_stdio_capture(self) -> None:
        if getattr(self, "_capturing_stdio", False):
            return
        try:
            self._stdout_orig = sys.stdout
            self._stderr_orig = sys.stderr
            sys.stdout = self._stream_out  # type: ignore[assignment]
            sys.stderr = self._stream_err  # type: ignore[assignment]
            self._capturing_stdio = True
        except Exception:
            pass

    def _restore_stdio_capture(self) -> None:
        if not getattr(self, "_capturing_stdio", False):
            return
        try:
            try:
                self._stream_out.flush()
            except Exception:
                pass
            try:
                self._stream_err.flush()
            except Exception:
                pass
            if self._stdout_orig is not None:
                sys.stdout = self._stdout_orig
            if self._stderr_orig is not None:
                sys.stderr = self._stderr_orig
        except Exception:
            pass
        finally:
            self._capturing_stdio = False
    

    def _flush_terminal_buffer(self) -> None:
        """Flush buffered terminal lines in small batches to keep UI responsive."""
        if self.terminal is None:
            self._term_buf.clear()
            return
        if not self._term_buf:
            return

        # Flush up to N lines per tick to avoid long UI stalls.
        n = 200
        for _ in range(min(n, len(self._term_buf))):
            try:
                self.terminal.append_line(self._term_buf.popleft())
            except Exception:
                # If terminal fails, drop the rest to prevent infinite loops.
                self._term_buf.clear()
                break

        # Stop the flush timer when there's nothing left.
        if not self._term_buf and hasattr(self, "_term_flush_timer") and self._term_flush_timer.isActive():
            try:
                self._term_flush_timer.stop()
            except Exception:
                pass
    
    def _drain_terminal_buffer(self) -> None:
        """Flush ALL buffered terminal lines (used on Stop / end-of-run)."""
        if self.terminal is None:
            self._term_buf.clear()
            return
        while self._term_buf:
            try:
                self.terminal.append_line(self._term_buf.popleft())
            except Exception:
                self._term_buf.clear()
                break

        # Stop the flush timer when drained.
        if hasattr(self, "_term_flush_timer") and self._term_flush_timer.isActive():
            try:
                self._term_flush_timer.stop()
            except Exception:
                pass
    
    def _on_terminal_settings_changed(self) -> None:
        self._term_stream_samples = bool(self.term_stream_chk.isChecked())
        try:
            val = float(self.term_period_edit.text().strip())
            self._term_period_s = 0.0 if val <= 0 else val
        except Exception:
            self._term_period_s = 0.05
            self.term_period_edit.setText("0.05")

    def _fmt(self, x: Any, nd: int = 4) -> str:
        try:
            if x is None:
                return "None"
            xf = float(x)
            if xf != xf:  # NaN
                return "nan"
            return f"{xf:.{nd}f}"
        except Exception:
            return str(x)

    def _maybe_log_sample(self, row: Dict[str, Any], *, state: str, reason: Optional[str]) -> None:
        if self.terminal is None or not self._term_stream_samples:
            return

        now = time.monotonic()

        # If period > 0, rate-limit by wall time.
        if self._term_period_s > 0 and (now - self._last_term_wall) < self._term_period_s:
            return
        self._last_term_wall = now

        # Row already contains the flattened “CSV shape” fields.
        msg = (
            "[sim] "
            f"k={self._fmt(row.get('k'), 0)} "
            f"t={self._fmt(row.get('t'), 4)} dt={self._fmt(row.get('dt'), 6)} | "
            f"g={self._fmt(row.get('g'), 1)} t_mod={self._fmt(row.get('t_mod'), 2)} | "
            f"v_cmd={self._fmt(row.get('v_cmd'), 4)} v_ref={self._fmt(row.get('v_ref'), 4)} | "
            f"meas(v,i,p)=({self._fmt(row.get('v'), 4)},{self._fmt(row.get('i'), 4)},{self._fmt(row.get('p'), 4)}) | "
            f"true(v,i,p)=({self._fmt(row.get('v_true'), 4)},{self._fmt(row.get('i_true'), 4)},{self._fmt(row.get('p_true'), 4)}) | "
            f"state={state}"
        )
        if reason:
            msg += f" reason={reason}"

        # Always include action_debug if present
        ad = row.get("action_debug")
        if isinstance(ad, str) and ad.strip():
            msg += f" | action_debug={ad}"

        # Robust GMPP validator fields (if enabled / present)
        if row.get("v_gmp_ref") is not None or row.get("p_gmp_ref") is not None:
            msg += (
                " | gmpp:"
                f" v_gmp_ref={self._fmt(row.get('v_gmp_ref'), 4)}"
                f" p_gmp_ref={self._fmt(row.get('p_gmp_ref'), 4)}"
                f" v_best={self._fmt(row.get('v_best'), 4)}"
                f" p_best={self._fmt(row.get('p_best'), 4)}"
                f" eff_best={self._fmt(row.get('eff_best'), 4)}"
            )

        gs = row.get("g_strings")
        if isinstance(gs, str) and gs.strip():
            msg += f" | g_strings={gs}"

        # If period==0, we still print EVERY step, but buffer terminal updates and flush on a timer
        # to avoid starving the Qt event loop (which would prevent live plotting).
        if self._term_period_s == 0.0:
            self._term_buf.append(msg)
            if not self._term_flush_timer.isActive():
                self._term_flush_timer.start()
        else:
            self._log(msg)

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

        # mutate shared overrides (only when NOT running a CSV profile)
        if (not use_csv) and self.overrides is not None:
            self.overrides.irradiance = float(g)
            self.overrides.temperature_c = float(t)

        if self._rd is not None and hasattr(self, "_plot_debounce"):
            if not self._plot_debounce.isActive():
                self._plot_debounce.start()

    # ---------------------------
    # Run/Stop
    # ---------------------------

    def _continuous_enabled(self) -> bool:
        # By default, continuous is for interactive/manual mode. CSV profiles run in batch mode.
        use_csv = bool(getattr(self, "use_csv_chk", None) and self.use_csv_chk.isChecked())
        if use_csv:
            return False
        return bool(getattr(self, "cont_chk", None) and self.cont_chk.isChecked())

    def _continuous_tick_ms(self) -> int:
        try:
            ms = int(float(self.cont_tick_edit.text().strip()))
        except Exception:
            ms = 33
        return max(5, ms)

    def _stop_continuous(self) -> None:
        if self._cont_timer is not None:
            try:
                self._cont_timer.stop()
            except Exception:
                pass
            self._cont_timer = None

        self._cont_eng = None

        if self._cont_f is not None:
            try:
                self._cont_f.flush()
            except Exception:
                pass
            try:
                self._cont_f.close()
            except Exception:
                pass
            self._cont_f = None

        self._cont_writer = None
        self._cont_rows_since_flush = 0
        self._cont_last_flush_wall = 0.0

        # Stop terminal flush timer and flush ALL remaining lines
        try:
            self._drain_terminal_buffer()
        except Exception:
            pass
        if hasattr(self, "_term_flush_timer") and self._term_flush_timer.isActive():
            self._term_flush_timer.stop()
        self._cont_stopping = False

    def _start_continuous(
        self,
        *,
        out_path: Path,
        selection: str,
        profile_name: str,
        use_csv_profile: bool,
        csv_profile_path: Optional[Path],
        total_time: float,
        dt: float,
        overrides: Optional[LiveOverrides],
        gmpp_ref: bool,
    ) -> None:
        """Run the sim in wall-clock time: one engine step per QTimer tick."""
        profile = resolve_env_profile(
            use_csv_profile=use_csv_profile,
            csv_profile_path=csv_profile_path,
            profile_name=profile_name,
        )

        controller_mode, algo_name, controller_cfg = resolve_controller_selection(selection)

        # Enforce CSV profile mode: LiveOverrides always win over env_profile in the engine.
        # If CSV is enabled, do NOT allow overrides to mask it.
        if use_csv_profile:
            overrides = None

        # Prepare CSV output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = RUN_CSV_FIELDNAMES

        self._cont_f = out_path.open("w", newline="")
        self._cont_writer = csv.DictWriter(self._cont_f, fieldnames=fieldnames)
        self._cont_writer.writeheader()
        self._cont_rows_since_flush = 0
        self._cont_last_flush_wall = time.monotonic()
        flush_every_rows = 200
        flush_every_s = 0.5

        cfg = SimulationConfig(
            total_time=total_time,
            dt=dt,
            env_profile=profile,
            controller_mode=controller_mode,
            algo_name=algo_name,
            algo_kwargs={},
            controller_cfg=controller_cfg,
            overrides=overrides,
            gmpp_ref=bool(gmpp_ref),
            gmpp_ref_period_s=0.05,
            gmpp_ref_points=121,
            on_sample=None,
        )

        self._cont_eng = SimulationEngine(cfg)

        # Emit initial record (t=0)
        rec0 = self._cont_eng.reset()
        row0 = record_to_row(rec0)
        if self._cont_writer is not None:
            self._cont_writer.writerow(row0)
        self._cont_rows_since_flush += 1
        self._on_live_sample(row0)
        if hasattr(self, "_plot_debounce"):
            if not self._plot_debounce.isActive():
                self._plot_debounce.start()

        # Start ticking
        self._cont_timer = QTimer(self)
        self._cont_timer.setInterval(self._continuous_tick_ms())
        # Continuous mode uses the current terminal streaming settings (rate-limited by default).
        # Do NOT force period=0 here; Continuous should remain usable without spamming the UI.
        try:
            self._on_terminal_settings_changed()
        except Exception:
            pass

        def _tick() -> None:
            if getattr(self, "_cont_stopping", False):
                return
            if self._cont_eng is None or self._cont_writer is None or self._cont_f is None:
                return

            rec = self._cont_eng.step_once()
            if rec is None:
                path_msg = f"[ui] MPPT output -> {out_path}"
                done_msg = "[ui] MPPT Simulation Complete"

                # Stop stepping first, then clean up, then print completion on next event loop turn.
                self._cont_stopping = True
                self._accept_live_samples = False
                try:
                    if self._cont_timer is not None:
                        self._cont_timer.stop()
                except Exception:
                    pass

                self._stop_continuous()
                self._cont_stopping = False
                self._finish_run_ui()
                self.refresh_runs()

                def _final_log() -> None:
                    try:
                        self._drain_terminal_buffer()
                    except Exception:
                        pass
                    try:
                        self._log(path_msg)
                    except Exception:
                        pass
                    try:
                        self._log(done_msg)
                    except Exception:
                        pass
                    try:
                        if self.terminal is not None:
                            self.terminal.flush()
                    except Exception:
                        pass
                    try:
                        self._restore_stdio_capture()
                    except Exception:
                        pass

                QTimer.singleShot(0, _final_log)
                return

            row = record_to_row(rec)
            self._cont_writer.writerow(row)
            self._on_live_sample(row)

            # Periodic flush
            self._cont_rows_since_flush += 1
            now = time.monotonic()
            if self._cont_rows_since_flush >= flush_every_rows or (now - self._cont_last_flush_wall) >= flush_every_s:
                try:
                    self._cont_f.flush()
                except Exception:
                    pass
                self._cont_rows_since_flush = 0
                self._cont_last_flush_wall = now

        self._cont_timer.timeout.connect(_tick)
        self._cont_timer.start()
    def run_sim(self) -> None:
        try:
            sim_time = float(self.time_edit.text().strip())
            dt = float(self.dt_edit.text().strip())
            if sim_time <= 0 or dt <= 0:
                raise ValueError
        except Exception:
            QMessageBox.warning(self, "Invalid settings", "Time and dt must be positive numbers.")
            return

        # Reset any stale stop state from a prior run so completion messages fire normally.
        self._pending_stop_msg = None
        self._batch_stopping = False
        self._cont_stopping = False
        self._accept_live_samples = True

        # Apply terminal streaming settings at run start so batch/profile runs respect them.
        # (Continuous mode already forces period=0.)
        self._on_terminal_settings_changed()

        algo = str(getattr(self, "algo_box", None) and self.algo_box.currentData() or "hybrid")
        profile = ""

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
            # Debug visibility: log profile bounds and clamp Time to the profile end.
            try:
                _prof = load_env_profile_csv(csv_profile_path)
                if _prof:
                    t_last = float(_prof[-1][0])
                    self._log(f"[ui] CSV profile rows={len(_prof)} t0={_prof[0][0]:.4f} t_last={t_last:.4f}")
                    if t_last + 1e-12 < sim_time:
                        self._log(f"[ui] CSV profile ends at t={t_last:.4f}; clamping Time from {sim_time:.4f} -> {t_last:.4f}")
                        sim_time = t_last
                        try:
                            self.time_edit.setText(f"{sim_time:g}")
                        except Exception:
                            pass
            except Exception:
                pass

        # CSV/profile runs are analysis-oriented; default to printing every step.
        if use_csv and self._term_stream_samples:
            self._term_period_s = 0.0
            self._last_term_wall = 0.0
            if hasattr(self, "term_period_edit"):
                self.term_period_edit.setText("0")

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
        self._rd = RunData(
            path=out_path,
            t=[], v=[], i=[], p=[], g=[], t_mod=[],
            state=[], reason=[],
            v_gmp_ref=[], p_gmp_ref=[], v_best=[], p_best=[], eff_best=[], g_strings=[],
            k=[],
            v_cmd=[],
            v_true=[],
            i_true=[],
            p_true=[],
        )

        self._log(f"[ui] Starting MPPT -> {out_path}")
        if use_csv:
            self._log(f"[ui] selection={algo} profile=csv ({csv_profile_path}) time={sim_time} dt={dt}")
        else:
            self._log(f"[ui] selection={algo} profile=(manual sliders) time={sim_time} dt={dt}")
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

        # CSV mode must fully drive the environment; don't let sliders override it.
        overrides_for_run = None if use_csv else self.overrides

        # Continuous mode runs on a QTimer and steps the engine once per tick.
        # Batch mode keeps the existing worker thread behavior.
        # Route any engine/worker prints into the in-app terminal while the run is active.
        self._install_stdio_capture()
        if self._continuous_enabled():
            # Ensure any prior continuous run is stopped
            self._stop_continuous()

            self._start_continuous(
                out_path=out_path,
                selection=algo,
                profile_name=profile,
                use_csv_profile=use_csv,
                csv_profile_path=csv_profile_path,
                total_time=1e9,
                dt=dt,
                overrides=overrides_for_run,
                gmpp_ref=bool(getattr(self, "gmpp_chk", None) and self.gmpp_chk.isChecked()),
            )
        else:
            w = _MPPTWorker(
                out_path=out_path,
                selection=algo,
                profile_name=profile,
                use_csv_profile=use_csv,
                csv_profile_path=csv_profile_path,
                total_time=sim_time,
                dt=dt,
                overrides=overrides_for_run,
                gmpp_ref=bool(getattr(self, "gmpp_chk", None) and self.gmpp_chk.isChecked()),
            )
            
            # If user requested per-step terminal output (period==0), ensure the flush timer is running
            # so buffered lines appear during the run.
            if getattr(self, "_term_period_s", 0.05) == 0.0 and hasattr(self, "_term_flush_timer"):
                if not self._term_flush_timer.isActive():
                    self._term_flush_timer.start()
            
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

        return


    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Ensure background simulation threads/timers are stopped before the widget is destroyed.

        Prevents: QThread: Destroyed while thread is still running.
        """
        # Stop continuous mode if active.
        try:
            if self._cont_timer is not None:
                try:
                    self._cont_timer.stop()
                except Exception:
                    pass
                self._stop_continuous()
        except Exception:
            pass

        # Stop worker thread if active.
        w = getattr(self, "_worker", None)
        if w is not None:
            try:
                w.request_stop()
            except Exception:
                pass
            try:
                w.wait(3000)
            except Exception:
                pass
            try:
                if w.isRunning():
                    w.terminate()
                    w.wait(1000)
            except Exception:
                pass
            try:
                self._worker = None
            except Exception:
                pass

        # Drain any pending terminal output.
        try:
            self._drain_terminal_buffer()
        except Exception:
            pass

        try:
            self._restore_stdio_capture()
        except Exception:
            pass

        try:
            super().closeEvent(event)
        except Exception:
            try:
                event.accept()
            except Exception:
                pass


    def stop_sim(self) -> None:
        # Stop continuous timer if active
        if self._cont_timer is not None:
            stop_msg = "[ui] MPPT Simulation Stopped (continuous)"

            # Prevent further stepping/logging while we stop.
            self._cont_stopping = True
            self._accept_live_samples = False

            # Stop the timer immediately so `_tick` cannot enqueue more samples.
            try:
                self._cont_timer.stop()
            except Exception:
                pass

            # Tear down continuous run (this drains any remaining buffered output).
            self._stop_continuous()
            self._cont_stopping = False
            self._finish_run_ui()
            self.refresh_runs()

            # Ensure absolutely everything is flushed BEFORE printing the stop message.
            try:
                self._drain_terminal_buffer()
            except Exception:
                pass
            
            try:
                self._log(stop_msg)
            except Exception:
                pass
            try:
                if self.terminal is not None:
                    self.terminal.flush()
            except Exception:
                pass
            try:
                self._restore_stdio_capture()
            except Exception:
                pass
            return

        # Stop batch worker if active
        if self._worker is None:
            return
        
        # Batch stop is asynchronous; remember that Stop was user-initiated.
        self._pending_stop_msg = "[ui] MPPT Simulation Stopped (batch)"
        self._batch_stopping = True
        self._accept_live_samples = False
        self._worker.request_stop()

    def _on_worker_failed(self, msg: str) -> None:
        # If Stop was pressed, treat completion as a user stop and print stop message last.
        if getattr(self, "_pending_stop_msg", None):
            stop_msg = self._pending_stop_msg
            self._pending_stop_msg = None
            self._batch_stopping = False
            self._accept_live_samples = False
            self._finish_run_ui()
            self.refresh_runs()

            def _final_log() -> None:
                try:
                    self._drain_terminal_buffer()
                except Exception:
                    pass
                try:
                    self._log(stop_msg)
                except Exception:
                    pass
                try:
                    if self.terminal is not None:
                        self.terminal.flush()
                except Exception:
                    pass
                try:
                    self._restore_stdio_capture()
                except Exception:
                    pass

            QTimer.singleShot(0, _final_log)
            return

        self._batch_stopping = False
        self._accept_live_samples = False
        fail_msg = f"[ui] MPPT failed: {msg}"

        self._finish_run_ui()

        def _final_log() -> None:
            try:
                self._drain_terminal_buffer()
            except Exception:
                pass
            try:
                self._log(fail_msg)
            except Exception:
                pass
            try:
                if self.terminal is not None:
                    self.terminal.flush()
            except Exception:
                pass
            try:
                self._restore_stdio_capture()
            except Exception:
                pass

        QTimer.singleShot(0, _final_log)

    def _on_worker_done(self, out_path_str: str) -> None:
        # If Stop was pressed, print the stop message last and suppress the "done" log.
        if getattr(self, "_pending_stop_msg", None):
            stop_msg = self._pending_stop_msg
            self._pending_stop_msg = None
            self._batch_stopping = False
            self._accept_live_samples = False
            self._finish_run_ui()
            self.refresh_runs()

            def _final_log() -> None:
                try:
                    self._drain_terminal_buffer()
                except Exception:
                    pass
                try:
                    self._log(stop_msg)
                except Exception:
                    pass
                try:
                    if self.terminal is not None:
                        self.terminal.flush()
                except Exception:
                    pass
                try:
                    self._restore_stdio_capture()
                except Exception:
                    pass

            QTimer.singleShot(0, _final_log)
            return

        self._batch_stopping = False
        self._accept_live_samples = False
        path_msg = f"[ui] MPPT output -> {out_path_str}"
        done_msg = "[ui] MPPT Simulation Complete"

        self._finish_run_ui()
        self.refresh_runs()

        def _final_log() -> None:
            try:
                self._drain_terminal_buffer()
            except Exception:
                pass
            try:
                self._log(path_msg)
            except Exception:
                pass
            try:
                self._log(done_msg)
            except Exception:
                pass
            try:
                if self.terminal is not None:
                    self.terminal.flush()
            except Exception:
                pass
            try:
                self._restore_stdio_capture()
            except Exception:
                pass

        QTimer.singleShot(0, _final_log)

    def _finish_run_ui(self) -> None:
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self._poll is not None:
            self._poll.stop()
            self._poll = None
        # Clear any continuous state
        if self._cont_timer is not None:
            self._stop_continuous()
        self._worker = None

        # Flush ALL remaining terminal output
        try:
            self._drain_terminal_buffer()
        except Exception:
            pass
        if hasattr(self, "_term_flush_timer") and self._term_flush_timer.isActive():
            self._term_flush_timer.stop()
        try:
            if self.terminal is not None:
                self.terminal.flush()
        except Exception:
            pass

        # Reset batch-stopping flag. Do NOT clear `_pending_stop_msg` here; it is
        # consumed and cleared in `_on_worker_done` / `_on_worker_failed` so that
        # the stop message is reliably printed last.
        self._batch_stopping = False

    def _on_live_sample(self, row: Dict[str, Any]) -> None:
        if not getattr(self, "_accept_live_samples", True):
            return
        rd = self._rd
        if rd is None:
            return

        tt = _safe_float(row.get("t"))
        if tt is None:
            return
        rd.t.append(float(tt))
        rd.k.append(float(_safe_float(row.get("k")) or float("nan")))
        rd.v_cmd.append(float(_safe_float(row.get("v_cmd")) or float("nan")))
        rd.v_true.append(float(_safe_float(row.get("v_true")) or float("nan")))
        rd.i_true.append(float(_safe_float(row.get("i_true")) or float("nan")))
        rd.p_true.append(float(_safe_float(row.get("p_true")) or float("nan")))
        rd.v.append(float(_safe_float(row.get("v")) or float("nan")))
        rd.i.append(float(_safe_float(row.get("i")) or float("nan")))
        rd.p.append(float(_safe_float(row.get("p")) or float("nan")))
        rd.g.append(float(_safe_float(row.get("g")) or float("nan")))
        rd.t_mod.append(float(_safe_float(row.get("t_mod")) or float("nan")))

        # Append GMPP reference/validator fields
        rd.v_gmp_ref.append(float(_safe_float(row.get("v_gmp_ref")) or float("nan")))
        rd.p_gmp_ref.append(float(_safe_float(row.get("p_gmp_ref")) or float("nan")))
        rd.v_best.append(float(_safe_float(row.get("v_best")) or float("nan")))
        rd.p_best.append(float(_safe_float(row.get("p_best")) or float("nan")))
        rd.eff_best.append(float(_safe_float(row.get("eff_best")) or float("nan")))
        gs = row.get("g_strings")
        rd.g_strings.append(gs if isinstance(gs, str) and gs.strip() else None)

        # Prefer explicit state/reason columns; fall back to parsing action_debug for older rows
        st = row.get("state")
        rsn = row.get("reason")

        if not (isinstance(st, str) and st.strip()):
            ad = row.get("action_debug")
            if isinstance(ad, str) and ad.strip():
                dbg = _parse_action_debug(ad)
                st = dbg.get("state")
                rsn = dbg.get("reason")

        last_state = rd.state[-1] if rd.state else "UNKNOWN"
        if isinstance(st, str) and st:
            last_state = st
        rd.state.append(last_state)
        rd.reason.append(rsn if isinstance(rsn, str) and rsn else None)
        if (not getattr(self, "_cont_stopping", False)) and (not getattr(self, "_batch_stopping", False)):
            self._maybe_log_sample(row, state=last_state, reason=(rsn if isinstance(rsn, str) and rsn else None))
        # Refresh plots in both batch and continuous modes.
        # Batch mode also has `_poll`, but continuous mode relies on this debounce.
        if self._rd is not None and hasattr(self, "_plot_debounce"):
            if not self._plot_debounce.isActive():
                self._plot_debounce.start()

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

        # ---------------------------
        # Sticky-axis baseline config
        # ---------------------------
        BASE_G = 1000.0     # "normal" irradiance baseline
        BASE_T = 25.0       # "normal" temperature baseline
        HALFSPAN_G = 300.0  # default visible window: 1000 ± 300
        HALFSPAN_T = 20.0   # default visible window: 25 ± 20
        PAD_FRAC = 0.05     # extra padding when expanding axis

        def _finite_minmax(xs: List[float]) -> Optional[Tuple[float, float]]:
            mn = None
            mx = None
            for x in xs:
                try:
                    xf = float(x)
                except Exception:
                    continue
                if xf != xf:  # NaN
                    continue
                if mn is None or xf < mn:
                    mn = xf
                if mx is None or xf > mx:
                    mx = xf
            if mn is None or mx is None:
                return None
            return (mn, mx)

        def _apply_sticky_y(plot: Any, *, base: float, halfspan: float, series: List[float]) -> None:
            mm = _finite_minmax(series)
            if mm is None:
                # Fall back to baseline window
                plot.setYRange(base - halfspan, base + halfspan, padding=0.0)
                return

            smin, smax = mm
            base_min = base - halfspan
            base_max = base + halfspan

            # If data fits in baseline window, keep it stable
            if smin >= base_min and smax <= base_max:
                plot.setYRange(base_min, base_max, padding=0.0)
                return

            # Otherwise expand just enough to include data (with a small pad)
            span = max(1e-9, (smax - smin))
            pad = PAD_FRAC * span
            plot.setYRange(smin - pad, smax + pad, padding=0.0)

        # ---------------------------
        # Build series
        # ---------------------------
        g_series = rd.g[:]       # preserve profile shape
        t_series = rd.t_mod[:]   # preserve profile shape

        use_csv = hasattr(self, "use_csv_chk") and self.use_csv_chk.isChecked()
        
        # ---------------------------
        # Clear + state shading
        # ---------------------------

        # Build state transition list: (state, start_time). This is stable as time advances
        # and only changes when the controller state changes.
        transitions: List[Tuple[str, float]] = []
        if rd.t and rd.state:
            cur = rd.state[0] if rd.state else "UNKNOWN"
            t0 = rd.t[0]
            transitions.append((cur, t0))
            for idx in range(1, min(len(rd.t), len(rd.state))):
                st = rd.state[idx]
                if st != cur:
                    cur = st
                    t0 = rd.t[idx]
                    transitions.append((cur, t0))

        t_end = rd.t[-1] if rd.t else 0.0

        # Only rebuild regions if the transition structure changed.
        if transitions != getattr(self, "_last_state_transitions", []):
            self._clear_regions()
            self._last_state_transitions = transitions

            brushes = {
                "NORMAL": (0, 160, 80, 40),
                "GLOBAL_SEARCH": (245, 158, 11, 45),
                "LOCK_HOLD": (59, 130, 246, 40),
                "UNKNOWN": (120, 120, 120, 25),
            }

            # Create one region per transition, ending at the next transition (or current t_end).
            for i, (st, t0) in enumerate(transitions):
                t1 = transitions[i + 1][1] if (i + 1) < len(transitions) else t_end
                rgba = brushes.get(st, brushes["UNKNOWN"])
                region = pg.LinearRegionItem(values=(t0, t1), brush=pg.mkBrush(*rgba), movable=False)
                region.setZValue(-10)
                self.g_plot.addItem(region)
                self.t_plot.addItem(region)
                self.v_plot.addItem(region)
                self.p_plot.addItem(region)
                self._state_regions.append(region)
        else:
            # Fast path: just extend the last region to the new end time.
            if self._state_regions:
                try:
                    last_region = self._state_regions[-1]
                    # Keep the start as-is, update only the end.
                    r0, _ = last_region.getRegion()
                    last_region.setRegion((r0, t_end))
                except Exception:
                    pass

        # ---------------------------
        # Plot lines
        # ---------------------------
        self._g_curve.setData(rd.t, g_series)
        self._t_curve.setData(rd.t, t_series)
        self._v_curve.setData(rd.t, rd.v)
        self._p_curve.setData(rd.t, rd.p)

        if rd.p_gmp_ref and any(x == x for x in rd.p_gmp_ref):
            self._p_gmpp_curve.setData(rd.t, rd.p_gmp_ref)
        else:
            self._p_gmpp_curve.setData([], [])

        # ---------------------------
        # Sticky Y axes for G/T
        # ---------------------------
        _apply_sticky_y(self.g_plot, base=BASE_G, halfspan=HALFSPAN_G, series=g_series)
        _apply_sticky_y(self.t_plot, base=BASE_T, halfspan=HALFSPAN_T, series=t_series)

        # X range
        if rd.t:
            self.g_plot.setXRange(rd.t[0], rd.t[-1], padding=0.01)  