

"""ui.desktop.source_dashboard

Source Dashboard (initial implementation).

This widget is designed to be embedded in `ui.desktop.main_window.MainWindow`.

Focus:
- Inspect *source/environment* behavior from run CSVs under `data/runs/`.
- Plot irradiance G(t) and module temperature T(t).
- Shade regions by controller state (NORMAL / GLOBAL_SEARCH / LOCK_HOLD) when present.

Notes:
- This dashboard reads the *run output CSV* (e.g. produced by simulators/source_sim.py).
- Profile-authoring UI (editing the input profiles/*.csv) can be added later.
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from simulators.engine import LiveOverrides
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QSlider,
)

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover
    pg = None  # type: ignore


@dataclass
class RunSummary:
    path: Path
    n_rows: int
    t_start: float
    t_end: float
    dt_mode: Optional[float]
    g_min: Optional[float]
    g_max: Optional[float]
    t_min: Optional[float]
    t_max: Optional[float]
    p_min: Optional[float]
    p_max: Optional[float]
    states: Dict[str, int]
    transitions: List[Tuple[str, str, float]]


@dataclass
class RunData:
    path: Path
    t: List[float]
    g: List[float]
    t_mod: List[float]
    v: List[float]
    i: List[float]
    p: List[float]
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
    """Compute a lightweight summary for a run CSV."""
    n_rows = 0
    t_vals: List[float] = []
    dt_vals: List[float] = []
    g_vals: List[float] = []
    tmod_vals: List[float] = []
    p_vals: List[float] = []

    state_counts: Dict[str, int] = {}

    # For transitions we need time+state pairs
    ts: List[Tuple[float, str]] = []

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

            tmod = _safe_float(row.get("t_mod"))
            if tmod is not None:
                tmod_vals.append(tmod)

            p = _safe_float(row.get("p"))
            if p is not None:
                p_vals.append(p)

            ad = row.get("action_debug")
            if isinstance(ad, str) and ad.strip() and t is not None:
                dbg = _parse_action_debug(ad)
                st = dbg.get("state")
                if isinstance(st, str) and st:
                    state_counts[st] = state_counts.get(st, 0) + 1
                    ts.append((float(t), st))

    t_start = float(t_vals[0]) if t_vals else 0.0
    t_end = float(t_vals[-1]) if t_vals else 0.0

    dt_mode: Optional[float] = None
    if dt_vals:
        rounded = [round(x, 9) for x in dt_vals]
        freq: Dict[float, int] = {}
        for x in rounded:
            freq[x] = freq.get(x, 0) + 1
        dt_mode = max(freq.items(), key=lambda kv: kv[1])[0]

    transitions: List[Tuple[str, str, float]] = []
    if ts:
        last = ts[0][1]
        for tt, st in ts[1:]:
            if st != last:
                transitions.append((last, st, float(tt)))
                last = st

    return RunSummary(
        path=path,
        n_rows=n_rows,
        t_start=t_start,
        t_end=t_end,
        dt_mode=dt_mode,
        g_min=min(g_vals) if g_vals else None,
        g_max=max(g_vals) if g_vals else None,
        t_min=min(tmod_vals) if tmod_vals else None,
        t_max=max(tmod_vals) if tmod_vals else None,
        p_min=min(p_vals) if p_vals else None,
        p_max=max(p_vals) if p_vals else None,
        states=state_counts,
        transitions=transitions,
    )


def load_run_csv(path: Path) -> RunData:
    """Load arrays for plotting G(t), T(t), with state shading."""
    t: List[float] = []
    g: List[float] = []
    t_mod: List[float] = []
    v: List[float] = []
    i_: List[float] = []
    p: List[float] = []
    state: List[str] = []
    reason: List[Optional[str]] = []

    last_state: str = "UNKNOWN"

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tt = _safe_float(row.get("t"))
            if tt is None:
                continue

            t.append(float(tt))
            g.append(float(_safe_float(row.get("g")) or float("nan")))
            t_mod.append(float(_safe_float(row.get("t_mod")) or float("nan")))
            v.append(float(_safe_float(row.get("v")) or float("nan")))
            i_.append(float(_safe_float(row.get("i")) or float("nan")))
            p.append(float(_safe_float(row.get("p")) or float("nan")))

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

    return RunData(path=path, t=t, g=g, t_mod=t_mod, v=v, i=i_, p=p, state=state, reason=reason)


def build_state_segments(t: List[float], state: List[str]) -> List[Tuple[str, float, float]]:
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


class SourceDashboard(QWidget):
    """Source/environment dashboard."""

    def __init__(self, *, overrides: Optional[LiveOverrides] = None, terminal: Any = None) -> None:
        super().__init__()
        self.overrides: Optional[LiveOverrides] = overrides
        self._current_rd: Optional[RunData] = None
        self._g_baseline: float = 1000.0
        self._t_baseline: float = 25.0

        root = QVBoxLayout(self)

        header_row = QHBoxLayout()
        title = QLabel("Source Dashboard")
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

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: run list
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.run_list = QListWidget()
        self.run_list.setAlternatingRowColors(True)
        left_layout.addWidget(QLabel("Runs (data/runs/*.csv)"))
        left_layout.addWidget(self.run_list)

        # Right: summary (top) + plots (bottom)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_split = QSplitter(Qt.Orientation.Vertical)

        summary_box = QWidget()
        summary_layout = QVBoxLayout(summary_box)
        summary_layout.setContentsMargins(0, 0, 0, 0)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlaceholderText("Select a run to see a summary…")

        summary_layout.addWidget(QLabel("Summary"))
        summary_layout.addWidget(self.summary)

        plots_box = QWidget()
        plots_layout = QVBoxLayout(plots_box)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.addWidget(QLabel("Plots"))

        # --- Live override sliders (visualization only for now) ---
        slider_row = QHBoxLayout()

        # Irradiance override
        slider_row.addWidget(QLabel("Irradiance"))
        self.g_slider = QSlider(Qt.Orientation.Horizontal)
        self.g_slider.setRange(0, 1400)  # W/m^2
        self.g_slider.setSingleStep(10)
        self.g_slider.setPageStep(50)
        self.g_slider.setValue(1000)
        self.g_val = QLabel("1000 W/m²")
        self.g_val.setMinimumWidth(90)
        slider_row.addWidget(self.g_slider, 1)
        slider_row.addWidget(self.g_val)

        # Temperature override
        slider_row.addWidget(QLabel("Temp"))
        self.t_slider = QSlider(Qt.Orientation.Horizontal)
        self.t_slider.setRange(-20, 100)  # °C
        self.t_slider.setSingleStep(1)
        self.t_slider.setPageStep(5)
        self.t_slider.setValue(25)
        self.t_val = QLabel("25 °C")
        self.t_val.setMinimumWidth(60)
        slider_row.addWidget(self.t_slider, 1)
        slider_row.addWidget(self.t_val)

        plots_layout.addLayout(slider_row)

        self._override_note = QLabel(
            "Overrides update the plots. If a live simulation is running in-process (Option A), they also override env_at(t)."
        )
        self._override_note.setStyleSheet("color: #666;")
        self._override_note.setWordWrap(True)
        plots_layout.addWidget(self._override_note)

        if pg is None:
            self._plot_note = QLabel("pyqtgraph is not available. Install it to enable plotting.")
            self._plot_note.setStyleSheet("color: #a00;")
            self._plot_note.setWordWrap(True)
            plots_layout.addWidget(self._plot_note)
            self.g_plot = None
            self.t_plot = None
        else:
            # Two stacked plots sharing X (time)
            self.g_plot = pg.PlotWidget()
            self.t_plot = pg.PlotWidget()
            self.g_plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.t_plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

            self.g_plot.setLabel("bottom", "Time", units="s")
            self.g_plot.setLabel("left", "Irradiance", units="W/m²")
            self.g_plot.showGrid(x=True, y=True, alpha=0.25)

            self.t_plot.setLabel("bottom", "Time", units="s")
            self.t_plot.setLabel("left", "Module temp", units="°C")
            self.t_plot.showGrid(x=True, y=True, alpha=0.25)

            plots_layout.addWidget(self.g_plot, 2)
            plots_layout.addWidget(self.t_plot, 2)

            self._state_regions: List[Any] = []

        right_split.addWidget(summary_box)
        right_split.addWidget(plots_box)
        right_split.setStretchFactor(0, 1)
        right_split.setStretchFactor(1, 2)

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
        self.g_slider.valueChanged.connect(self._on_override_changed)
        self.t_slider.valueChanged.connect(self._on_override_changed)

        # Resolve Aurora repo root regardless of where the UI is launched from.
        # This file lives at: Aurora/ui/desktop/source_dashboard.py
        self._repo_root = Path(__file__).resolve().parents[2]

        self.refresh_runs()

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
                "Tip: run a source simulation with:\n"
                "  python3 -m simulators.source_sim --out-csv my_run.csv\n"
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
            "Open source run CSV",
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
        if s.t_min is not None and s.t_max is not None:
            lines.append(f"Module temp: {s.t_min:g} → {s.t_max:g} °C")
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

        self.summary.setPlainText("\n".join(lines))

        # Plot (if available)
        try:
            rd = load_run_csv(path)
            self._current_rd = rd
            # No longer overwrite baselines from the run; use fixed defaults.

            # Seed sliders from fixed "normal" baselines
            self.g_slider.blockSignals(True)
            self.g_slider.setValue(int(round(self._g_baseline)))
            self.g_slider.blockSignals(False)

            self.t_slider.blockSignals(True)
            self.t_slider.setValue(int(round(self._t_baseline)))
            self.t_slider.blockSignals(False)

            self._on_override_changed()
        except Exception:
            pass

        # Highlight selection when applicable
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
        if pg is None or self.g_plot is None or self.t_plot is None:
            return

        self.g_plot.clear()
        self.t_plot.clear()

        # Force-disable autorange before adding items
        self.g_plot.getViewBox().enableAutoRange(axis='y', enable=False)
        self.t_plot.getViewBox().enableAutoRange(axis='y', enable=False)

        # Remove existing regions
        for r in getattr(self, "_state_regions", []):
            try:
                self.g_plot.removeItem(r)
            except Exception:
                pass
            try:
                self.t_plot.removeItem(r)
            except Exception:
                pass
        self._state_regions = []

        # Plot lines
        self.g_plot.plot(rd.t, rd.g)
        self.t_plot.plot(rd.t, rd.t_mod)

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
            self._state_regions.append(region)

        if rd.t:
            self.g_plot.setXRange(rd.t[0], rd.t[-1], padding=0.01)
            self.t_plot.setXRange(rd.t[0], rd.t[-1], padding=0.01)
    def _on_override_changed(self) -> None:
        # Update labels
        g = int(self.g_slider.value())
        t = int(self.t_slider.value())
        self.g_val.setText(f"{g} W/m²")
        self.t_val.setText(f"{t} °C")

        # Live override channel (Option A): mutate the shared overrides object.
        if self.overrides is not None:
            self.overrides.irradiance = float(g)
            self.overrides.temperature_c = float(t)

        # Replot using the current run (if any)
        if self._current_rd is not None:
            self._plot_run_with_overrides(self._current_rd)

    def _plot_run_with_overrides(self, rd: RunData) -> None:
        """Plot with slider overrides as *deltas* relative to the run's baseline.

        Instead of plotting a constant line (which makes the Y axis rescale and hides movement),
        we shift the original run series by the slider delta so the curve visibly moves up/down.

        The Y axis stays anchored to the baseline range unless the delta is dramatic.
        """
        if pg is None or self.g_plot is None or self.t_plot is None:
            return

        g_override = float(self.g_slider.value())
        t_override = float(self.t_slider.value())

        def _finite(vals: list[float]) -> list[float]:
            return [float(x) for x in vals if x == x]  # filter NaNs

        # Baselines: use fixed defaults
        g_base = self._g_baseline
        t_base = self._t_baseline

        dg = g_override - g_base
        dt = t_override - t_base

        # Shift the original series so the curve moves up/down relative to the baseline.
        g_series = [(float(x) + dg) if (x == x) else float("nan") for x in rd.g]
        t_series = [(float(x) + dt) if (x == x) else float("nan") for x in rd.t_mod]

        self.g_plot.clear()
        self.t_plot.clear()
        # Force-disable autorange before adding items (prevents axis "chasing" on some pyqtgraph versions)
        try:
            self.g_plot.getViewBox().enableAutoRange(axis='y', enable=False)
            self.t_plot.getViewBox().enableAutoRange(axis='y', enable=False)
        except Exception:
            pass

        # Remove existing regions
        for r in getattr(self, "_state_regions", []):
            try:
                self.g_plot.removeItem(r)
            except Exception:
                pass
            try:
                self.t_plot.removeItem(r)
            except Exception:
                pass
        self._state_regions = []

        # Shade by state first
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
            self._state_regions.append(region)

        # Plot shifted lines
        self.g_plot.plot(rd.t, g_series)
        self.t_plot.plot(rd.t, t_series)

        # X range follows the run.
        if rd.t:
            self.g_plot.setXRange(rd.t[0], rd.t[-1], padding=0.01)
            self.t_plot.setXRange(rd.t[0], rd.t[-1], padding=0.01)

        # Disable Y auto-range so the axis doesn't constantly chase the line.
        try:
            self.g_plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
            self.t_plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        except Exception:
            try:
                self.g_plot.enableAutoRange('y', False)
                self.t_plot.enableAutoRange('y', False)
            except Exception:
                pass

        # Baseline-anchored Y ranges (do NOT follow CSV min/max).
        # Default windows:
        #   - Irradiance: 1000 ± 200 W/m²
        #   - Temperature: 25 ± 10 °C
        # The axis only expands if the shifted curve would clip outside the window by a meaningful amount.
        g_window = 200.0
        t_window = 10.0

        g_base_min = g_base - g_window
        g_base_max = g_base + g_window
        t_base_min = t_base - t_window
        t_base_max = t_base + t_window

        g_shifted = _finite(g_series)
        t_shifted = _finite(t_series)

        if g_shifted:
            g_smin, g_smax = min(g_shifted), max(g_shifted)
            # Allow some headroom before expanding (prevents frequent axis changes).
            g_margin = 0.25 * g_window
            y0, y1 = g_base_min, g_base_max
            if g_smin < (g_base_min - g_margin) or g_smax > (g_base_max + g_margin):
                y0 = min(g_base_min, g_smin)
                y1 = max(g_base_max, g_smax)
            self.g_plot.setYRange(y0, y1, padding=0.0)
        else:
            self.g_plot.setYRange(g_base_min, g_base_max, padding=0.0)

        if t_shifted:
            t_smin, t_smax = min(t_shifted), max(t_shifted)
            t_margin = 0.25 * t_window
            y0, y1 = t_base_min, t_base_max
            if t_smin < (t_base_min - t_margin) or t_smax > (t_base_max + t_margin):
                y0 = min(t_base_min, t_smin)
                y1 = max(t_base_max, t_smax)
            self.t_plot.setYRange(y0, y1, padding=0.0)
        else:
            self.t_plot.setYRange(t_base_min, t_base_max, padding=0.0)