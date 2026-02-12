"""ui.desktop.benchmarks_dashboard

Benchmarks tab for Aurora.

This dashboard provides a simple UI to:
- select MPPT algorithms (single-controller runs from `core.mppt_algorithms.registry`) or the special `hybrid` controller
- select benchmark scenarios (from `benchmarks.scenarios`)
- select budgets (from `benchmarks.runner.default_budgets()`)
- run the benchmark suite in a background thread
- render a sortable results table from `summaries.jsonl`

Notes:
- This runs benchmarks in-process (recommended for an MVP).
- Output is written under `Aurora/data/benchmarks/` by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.desktop.terminal_panel import TerminalPanel

# Session path utility for new session-based output
from benchmarks.session import read_latest_session_path


def _repo_root() -> Path:
    # file is Aurora/ui/desktop/benchmarks_dashboard.py
    return Path(__file__).resolve().parents[2]


def _data_dir() -> Path:
    return _repo_root() / "data" / "benchmarks"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    return rows


def _flatten_metrics(extra: Dict[str, Any]) -> Dict[str, Any]:
    """Pull commonly-used metric keys out of the nested `extra` dict."""
    out: Dict[str, Any] = {}
    if not isinstance(extra, dict):
        return out

    # Preferred: `extra["metrics"]` from metrics.py integration
    m = extra.get("metrics")
    if isinstance(m, dict):
        out.update(m)

    # Convenience copies may also exist
    for k in (
        "energy_true_ratio",
        "energy_meas_ratio",
        "settle_time_s_true",
        "settle_time_s_meas",
        "t_disturb",
        "settle_time_s_true_post",
        "settle_time_s_meas_post",
        "recovery_settle_s_true",
        "recovery_settle_s_meas",
        "ripple_rms_true",
        "ripple_rms_meas",
        "tracking_error_area_true",
        "tracking_error_area_meas",
        "score",
    ):
        if k in extra and k not in out:
            out[k] = extra[k]
    return out


class _BenchWorker(QThread):
    """Run the benchmark suite in a background thread."""

    log = pyqtSignal(str)
    done = pyqtSignal(str)  # run_dir
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        algo_specs: Sequence[str],
        scenario_names: Sequence[str],
        budget_names: Sequence[str],
        out_dir: Path,
        total_time: float,
        gmpp_ref: bool,
        gmpp_period: float,
        gmpp_points: int,
        save_records: bool,
        log_every_s: float,
        keep_records: bool,
    ) -> None:
        super().__init__()
        self.algo_specs = list(algo_specs)
        self.scenario_names = list(scenario_names)
        self.budget_names = list(budget_names)
        self.out_dir = out_dir
        self.total_time = float(total_time)
        self.gmpp_ref = bool(gmpp_ref)
        self.gmpp_period = float(gmpp_period)
        self.gmpp_points = int(gmpp_points)
        self.save_records = bool(save_records)
        self.log_every_s = float(log_every_s)
        self.keep_records = bool(keep_records)

    def run(self) -> None:
        try:
            from benchmarks.runner import (
                default_budgets,
                default_scenarios,
                resolve_algorithms_from_registry,
                run_suite,
            )

            # Resolve selections
            algos = resolve_algorithms_from_registry(self.algo_specs)
            scenarios_all = default_scenarios()
            budgets_all = default_budgets()

            scenarios = [s for s in scenarios_all if s.name in set(self.scenario_names)]
            budgets = [b for b in budgets_all if b.name in set(self.budget_names)]

            if not algos:
                raise ValueError("No algorithms selected")
            if not scenarios:
                raise ValueError("No scenarios selected")
            if not budgets:
                raise ValueError("No budgets selected")

            self.log.emit(
                f"[bench] running: algos={len(algos)}, scenarios={len(scenarios)}, budgets={len(budgets)}"
            )
            self.log.emit(f"[bench] out_dir: {self.out_dir}")
            if self.isInterruptionRequested():
                raise RuntimeError("Cancelled")
            # Run suite (writes into a persistent session folder under out_dir)
            run_suite(
                algorithms=algos,
                scenarios=scenarios,
                budgets=budgets,
                total_time=self.total_time,
                gmpp_ref=self.gmpp_ref,
                gmpp_ref_period_s=self.gmpp_period,
                gmpp_ref_points=self.gmpp_points,
                out_dir=self.out_dir,
                save_records=self.save_records,
                log=self.log.emit,
                log_every_s=self.log_every_s,
                keep_records=self.keep_records,
                cancel=self.isInterruptionRequested,
            )

            # Resolve the active session directory written by the runner
            run_dir = read_latest_session_path(self.out_dir)
            if run_dir is None or not run_dir.exists():
                raise RuntimeError(
                    "Benchmark run completed but no active session folder was found. "
                    "Expected latest_session_path.txt to exist under out_dir."
                )

            self.log.emit(f"[bench] complete: {run_dir}")
            self.done.emit(str(run_dir))

        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class BenchmarksDashboard(QWidget):
    """Benchmarks dashboard widget."""

    def __init__(
        self,
        *,
        terminal: Optional[TerminalPanel] = None,
        overrides: Any = None,  # kept for consistent DI signature
    ) -> None:
        super().__init__()
        self.terminal = terminal
        self._worker: Optional[_BenchWorker] = None

        root = QVBoxLayout(self)

        # --------------------
        # Create top control widgets (must exist before layout wiring)
        # --------------------
        self.out_path = QLineEdit(str(_data_dir()))
        self.out_path.setMinimumWidth(480)
        self.out_path.setPlaceholderText("Output directory (benchmarks)")
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._pick_out_dir)

        # Session folder controls (read-only)
        self.session_path = QLineEdit("")
        self.session_path.setReadOnly(True)
        self.session_path.setPlaceholderText("Session folder (auto-filled after run)")

        btn_session_browse = QPushButton("Browse session…")
        btn_session_browse.clicked.connect(self._pick_session_dir)

        btn_session_load = QPushButton("Load session")
        btn_session_load.clicked.connect(self._load_session_from_path)

        self.chk_gmpp = QCheckBox("GMPP reference")
        self.chk_gmpp.setChecked(True)

        self.sp_total_time = QDoubleSpinBox()
        self.sp_total_time.setRange(0.1, 60.0)
        self.sp_total_time.setValue(1.0)
        self.sp_total_time.setSuffix(" s")

        self.sp_gmpp_period = QDoubleSpinBox()
        self.sp_gmpp_period.setRange(0.01, 10.0)
        self.sp_gmpp_period.setValue(0.25)
        self.sp_gmpp_period.setSuffix(" s")

        self.sp_gmpp_points = QSpinBox()
        self.sp_gmpp_points.setRange(50, 5000)
        self.sp_gmpp_points.setValue(300)

        self.sp_log_every = QDoubleSpinBox()
        self.sp_log_every.setRange(0.0, 5.0)
        self.sp_log_every.setSingleStep(0.05)
        self.sp_log_every.setValue(0.25)
        self.sp_log_every.setSuffix(" s")

        self.chk_no_keep_records = QCheckBox("Don't keep records in memory")
        self.chk_no_keep_records.setChecked(False)
        self.chk_no_keep_records.setToolTip(
            "If enabled, runner will not store per-tick records in RAM. For metrics, enable 'Save per-tick records'."
        )

        self.chk_save_records = QCheckBox("Save per-tick records")
        self.chk_save_records.setChecked(False)

        self.btn_run = QPushButton("Run Benchmarks")
        self.btn_run.clicked.connect(self._run)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)

        self.btn_open = QPushButton("Open summaries…")
        self.btn_open.clicked.connect(self._open_summaries)

        # --------------------
        # Top controls
        # --------------------
        top = QVBoxLayout()
        top_row1 = QHBoxLayout()
        top_row2 = QHBoxLayout()
        top_row3 = QHBoxLayout()

        # Row 1: output + run controls
        top_row1.addWidget(QLabel("Out:"))
        top_row1.addWidget(self.out_path, 1)
        top_row1.addWidget(btn_browse)
        top_row1.addSpacing(10)
        top_row1.addStretch(1)
        top_row1.addWidget(self.btn_open)
        top_row1.addWidget(self.btn_run)
        top_row1.addWidget(self.btn_stop)

        # Row 2: benchmark parameters
        top_row2.addWidget(self.chk_gmpp)
        top_row2.addWidget(QLabel("Total:"))
        top_row2.addWidget(self.sp_total_time)
        top_row2.addWidget(QLabel("GMPP period:"))
        top_row2.addWidget(self.sp_gmpp_period)
        top_row2.addWidget(QLabel("Pts:"))
        top_row2.addWidget(self.sp_gmpp_points)
        top_row2.addWidget(QLabel("Log every:"))
        top_row2.addWidget(self.sp_log_every)
        top_row2.addWidget(self.chk_no_keep_records)
        top_row2.addWidget(self.chk_save_records)
        top_row2.addStretch(1)

        # Row 3: session folder (leaderboard source)
        top_row3.addWidget(QLabel("Session:"))
        top_row3.addWidget(self.session_path, 1)
        top_row3.addWidget(btn_session_browse)
        top_row3.addWidget(btn_session_load)

        top.addLayout(top_row1)
        top.addLayout(top_row2)
        top.addLayout(top_row3)

        root.addLayout(top)

        # --------------------
        # Left: selectors
        # Right: results table
        # --------------------
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)

        self.list_algos = QListWidget()
        self.list_algos.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list_algos.setToolTip(
            "Select 'Hybrid (controller)' to run the HybridMPPT state machine, or select registry algorithms (ruca, pando, etc.) to run that single algorithm for the entire run."
        )

        self.list_scenarios = QListWidget()
        self.list_scenarios.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

        self.list_budgets = QListWidget()
        self.list_budgets.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

        btn_refresh = QPushButton("Refresh lists")
        btn_refresh.clicked.connect(self._refresh_lists)

        left_layout.addWidget(QLabel("Algorithms / Controller"))
        left_layout.addWidget(self.list_algos, 1)
        left_layout.addWidget(QLabel("Scenarios"))
        left_layout.addWidget(self.list_scenarios, 1)
        left_layout.addWidget(QLabel("Budgets"))
        left_layout.addWidget(self.list_budgets, 1)
        left_layout.addWidget(btn_refresh)

        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.lbl_status = QLabel("Select algorithms/scenarios/budgets, then run.")
        self.lbl_status.setStyleSheet("color: #666;")
        right_layout.addWidget(self.lbl_status)

        self.table = QTableWidget(0, 0)
        self.table.setSortingEnabled(True)
        right_layout.addWidget(self.table, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # populate lists
        self._refresh_lists(select_all=True)

    # --------------------
    # UI actions
    # --------------------

    def _log(self, msg: str) -> None:
        if self.terminal is not None:
            try:
                self.terminal.append_line(msg)
            except Exception:
                pass

    def _pick_out_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose output directory", self.out_path.text())
        if path:
            self.out_path.setText(path)

    def _pick_session_dir(self) -> None:
        start = self.session_path.text().strip() or str(Path(self.out_path.text()).expanduser())
        path = QFileDialog.getExistingDirectory(self, "Choose session folder", start)
        if path:
            self.session_path.setText(path)
            # Auto-load on pick
            self._load_session_from_path()

    def _load_session_from_path(self) -> None:
        p = Path(self.session_path.text().strip()).expanduser()
        if not p.exists() or not p.is_dir():
            QMessageBox.warning(self, "Benchmarks", "Session path is not a valid folder.")
            return
        summaries = p / "summaries.jsonl"
        if not summaries.exists():
            QMessageBox.warning(self, "Benchmarks", f"No summaries.jsonl found in {p}")
            return
        self._load_summaries(summaries)

    def _refresh_lists(self, *, select_all: bool = False) -> None:
        # Algorithms
        self.list_algos.clear()
        try:
            from benchmarks.runner import list_registry_algorithms

            algos = list_registry_algorithms()
        except Exception as e:
            algos = []
            self._log(f"[ui] failed to load algorithms: {type(e).__name__}: {e}")

        for a in algos:
            a_str = str(a)
            item = QListWidgetItem("Hybrid (controller)" if a_str == "hybrid" else a_str)
            # Store the actual spec string separately so selection returns canonical values
            item.setData(Qt.ItemDataRole.UserRole, a_str)
            self.list_algos.addItem(item)

        # Scenarios
        self.list_scenarios.clear()
        try:
            from benchmarks.scenarios import list_scenarios

            scs = list_scenarios()
        except Exception as e:
            scs = ["steady"]
            self._log(f"[ui] failed to load scenarios: {type(e).__name__}: {e}")

        for s in scs:
            self.list_scenarios.addItem(QListWidgetItem(str(s)))

        # Budgets
        self.list_budgets.clear()
        try:
            from benchmarks.runner import default_budgets

            bds = default_budgets()
            bd_names = [b.name for b in bds]
        except Exception as e:
            bd_names = []
            self._log(f"[ui] failed to load budgets: {type(e).__name__}: {e}")

        for b in bd_names:
            self.list_budgets.addItem(QListWidgetItem(str(b)))

        if select_all:
            for lw in (self.list_algos, self.list_scenarios, self.list_budgets):
                for i in range(lw.count()):
                    lw.item(i).setSelected(True)

    def _selected_texts(self, lw: QListWidget) -> List[str]:
        out: List[str] = []
        for i in lw.selectedItems():
            v = i.data(Qt.ItemDataRole.UserRole)
            out.append(str(v) if v is not None else i.text())
        return out

    def _run(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Benchmarks", "A benchmark run is already in progress.")
            return

        algos = self._selected_texts(self.list_algos)
        scenarios = self._selected_texts(self.list_scenarios)
        budgets = self._selected_texts(self.list_budgets)

        if not algos or not scenarios or not budgets:
            QMessageBox.warning(
                self,
                "Benchmarks",
                "Select at least one algorithm, one scenario, and one budget.",
            )
            return

        out_dir = Path(self.out_path.text()).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.chk_no_keep_records.isChecked() and not self.chk_save_records.isChecked():
            QMessageBox.warning(
                self,
                "Benchmarks",
                "If you disable keeping records in memory, enable 'Save per-tick records' so metrics can be computed.",
            )
            return

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("Running benchmarks…")
        self._log(f"[ui] benchmarks starting (out={out_dir})")

        self._worker = _BenchWorker(
            algo_specs=algos,
            scenario_names=scenarios,
            budget_names=budgets,
            out_dir=out_dir,
            total_time=float(self.sp_total_time.value()),
            gmpp_ref=bool(self.chk_gmpp.isChecked()),
            gmpp_period=float(self.sp_gmpp_period.value()),
            gmpp_points=int(self.sp_gmpp_points.value()),
            save_records=bool(self.chk_save_records.isChecked()),
            log_every_s=float(self.sp_log_every.value()),
            keep_records=not bool(self.chk_no_keep_records.isChecked()),
        )
        self._worker.log.connect(self._log)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _stop(self) -> None:
        if self._worker is None:
            return
        self._log("[ui] stopping benchmark…")
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Stopping…")
        try:
            self._worker.requestInterruption()
        except Exception:
            pass

    def _on_failed(self, msg: str) -> None:
        self.btn_stop.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.lbl_status.setText("Benchmark failed")
        self._log(f"[ui] benchmark failed: {msg}")
        if "Cancelled" in msg or "cancelled" in msg or "canceled" in msg:
            self.btn_run.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.lbl_status.setText("Benchmark stopped")
            self._log(f"[ui] benchmark stopped: {msg}")
            return
        QMessageBox.critical(self, "Benchmarks", msg)

    def _on_done(self, run_dir_str: str) -> None:
        self.btn_stop.setEnabled(False)
        self.btn_run.setEnabled(True)
        run_dir = Path(run_dir_str)
        try:
            self.session_path.setText(str(run_dir))
        except Exception:
            pass
        self.lbl_status.setText(f"Session: {run_dir}")
        self._log(f"[ui] benchmark complete: {run_dir}")

        summaries = run_dir / "summaries.jsonl"
        if not summaries.exists():
            QMessageBox.warning(self, "Benchmarks", f"No summaries.jsonl found in {run_dir}")
            return

        self._load_summaries(summaries)

    def _open_summaries(self) -> None:
        start = self.session_path.text().strip() or str(Path(self.out_path.text()).expanduser())
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open summaries.jsonl",
            start,
            "JSONL Files (*.jsonl);;All Files (*)",
        )
        if not path_str:
            return
        self._load_summaries(Path(path_str))

    def _load_summaries(self, path: Path) -> None:
        try:
            rows = _read_jsonl(path)
        except Exception as e:
            QMessageBox.critical(self, "Benchmarks", f"Failed to read file: {type(e).__name__}: {e}")
            return

        self._render_table(rows)
        self.lbl_status.setText(f"Loaded session: {path}")
        self._log(f"[ui] loaded session summaries: {path} ({len(rows)} rows)")

    # --------------------
    # Results table
    # --------------------

    def _render_table(self, rows: List[Dict[str, Any]]) -> None:
        # Define columns (base + metrics)
        def _score_of(row: Dict[str, Any]) -> float:
            extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
            metrics_flat = _flatten_metrics(extra) if isinstance(extra, dict) else {}
            v = None
            if isinstance(extra, dict) and "score" in extra:
                v = extra.get("score")
            if v is None and "score" in metrics_flat:
                v = metrics_flat.get("score")
            try:
                return float(v)
            except Exception:
                return float("-inf")

        rows = sorted(list(rows), key=_score_of, reverse=True)

        cols = [
            ("rank", "Rank"),
            ("score", "Score"),
            ("algo", "Algo"),
            ("scenario", "Scenario"),
            ("budget", "Budget"),
            ("energy_meas_ratio", "Energy (meas)"),
            ("recovery_settle_s_meas", "Recovery (s)"),
            ("ripple_rms_meas", "Ripple (meas)"),
            ("tracking_error_area_meas", "Track area (meas)"),
            ("ctrl_us_p95", "Ctrl µs p95"),
            ("budget_violations", "Over budget"),
            ("eff_meas_final", "Eff Meas (final)"),
            ("eff_true_final", "Eff True (final)"),
        ]

        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setRowCount(len(rows))
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels([c[1] for c in cols])

        for r_i, row in enumerate(rows):
            extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
            metrics_flat = _flatten_metrics(extra) if isinstance(extra, dict) else {}
            rank_val = r_i + 1

            def get_val(key: str) -> Any:
                if key == "rank":
                    return rank_val
                if key in row:
                    return row.get(key)
                if key in metrics_flat:
                    return metrics_flat.get(key)
                if isinstance(extra, dict) and key in extra:
                    return extra.get(key)
                return None

            for c_i, (key, _) in enumerate(cols):
                v = get_val(key)
                item = QTableWidgetItem()
                if v is None:
                    item.setText("")
                else:
                    # Use EditRole for numeric values so Qt sorts numerically.
                    try:
                        fv = float(v)
                        item.setData(Qt.ItemDataRole.EditRole, fv)
                    except Exception:
                        item.setText(str(v))

                is_num = False
                try:
                    float(v)
                    is_num = True
                except Exception:
                    is_num = False

                if is_num:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

                self.table.setItem(r_i, c_i, item)

        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)
        # Default leaderboard sort: Score descending
        self.table.sortItems(1, Qt.SortOrder.DescendingOrder)


# For MainWindow dynamic loader
__all__ = ["BenchmarksDashboard"]
