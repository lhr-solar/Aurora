

"""ui.desktop.profile_editor

CSV profile editor for Aurora environment profiles.

Profile CSV schema (header required):
    t,g,t_c

Where:
  - t: time (seconds)
  - g: irradiance (W/m^2)
  - t_c: module temperature (°C)

This editor provides:
- Table-based editing
- Load / Save / Browse
- Optional live preview (pyqtgraph, if installed)
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover
    pg = None


@dataclass
class ProfileRow:
    t: float
    g: float
    t_c: float


def _repo_root() -> Path:
    # Aurora/ui/desktop/profile_editor.py
    return Path(__file__).resolve().parents[2]


def profiles_dir() -> Path:
    p = _repo_root() / "profiles"
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_profile_csv(path: Path) -> List[ProfileRow]:
    rows: List[ProfileRow] = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for rr in r:
            t = rr.get("t") or rr.get("time")
            g = rr.get("g") or rr.get("irradiance")
            tc = rr.get("t_c") or rr.get("temperature_c") or rr.get("temp_c")
            if t is None or g is None or tc is None:
                continue
            try:
                rows.append(ProfileRow(float(t), float(g), float(tc)))
            except Exception:
                continue
    rows.sort(key=lambda x: x.t)
    return rows


def write_profile_csv(path: Path, rows: List[ProfileRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda x: x.t)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "g", "t_c"])
        w.writeheader()
        for r in rows_sorted:
            w.writerow({"t": r.t, "g": r.g, "t_c": r.t_c})


class ProfileEditor(QWidget):
    """Table-based CSV profile editor."""

    profile_saved = pyqtSignal(str)  # emits saved absolute path

    def __init__(self, *, initial_path: Optional[Path] = None) -> None:
        super().__init__()
        self._path: Optional[Path] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # Header: path + load/save
        top = QHBoxLayout()
        top.addWidget(QLabel("Profile file"))
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("profiles/my_profile.csv")
        top.addWidget(self.path_edit, 1)

        self.btn_browse = QPushButton("Browse…")
        self.btn_load = QPushButton("Load")
        self.btn_save = QPushButton("Save")
        top.addWidget(self.btn_browse)
        top.addWidget(self.btn_load)
        top.addWidget(self.btn_save)
        root.addLayout(top)

        # Table
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["t (s)", "g (W/m²)", "T (°C)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.table, 1)

        # Row controls
        row_btns = QHBoxLayout()
        self.btn_add = QPushButton("Add row")
        self.btn_del = QPushButton("Delete row")
        self.btn_clear = QPushButton("Clear")
        row_btns.addWidget(self.btn_add)
        row_btns.addWidget(self.btn_del)
        row_btns.addWidget(self.btn_clear)
        row_btns.addStretch(1)
        root.addLayout(row_btns)

        # Preview
        root.addWidget(QLabel("Preview"))
        if pg is None:
            root.addWidget(QLabel("pyqtgraph not available (preview disabled)."))
            self.g_plot = None
            self.t_plot = None
        else:
            self.g_plot = pg.PlotWidget(title="Irradiance")
            self.t_plot = pg.PlotWidget(title="Temperature")
            self.t_plot.setXLink(self.g_plot)
            root.addWidget(self.g_plot, 1)
            root.addWidget(self.t_plot, 1)

        # Wiring
        self.btn_add.clicked.connect(self.add_row)
        self.btn_del.clicked.connect(self.delete_selected_rows)
        self.btn_clear.clicked.connect(self.clear)
        self.btn_browse.clicked.connect(self.browse)
        self.btn_load.clicked.connect(self.load)
        self.btn_save.clicked.connect(self.save)
        self.table.itemChanged.connect(lambda _: self._refresh_preview())

        # Seed
        if initial_path is not None:
            self.set_path(initial_path)
            self.load()
        else:
            self.set_path(profiles_dir() / "cloud_late.csv")
            if self.table.rowCount() == 0:
                self._seed_default()

    # ---------------------------
    # Path helpers
    # ---------------------------
    def set_path(self, path: Path) -> None:
        self._path = path
        try:
            rel = path.relative_to(_repo_root())
            self.path_edit.setText(str(rel))
        except Exception:
            self.path_edit.setText(str(path))

    def resolved_path(self) -> Optional[Path]:
        s = self.path_edit.text().strip()
        if not s:
            return None
        p = Path(s)
        if not p.is_absolute():
            p = _repo_root() / p
        return p

    # ---------------------------
    # Table helpers
    # ---------------------------
    def rows(self) -> List[ProfileRow]:
        out: List[ProfileRow] = []
        for r in range(self.table.rowCount()):
            try:
                t = float(self._cell_text(r, 0))
                g = float(self._cell_text(r, 1))
                tc = float(self._cell_text(r, 2))
                out.append(ProfileRow(t, g, tc))
            except Exception:
                continue
        out.sort(key=lambda x: x.t)
        return out

    def _cell_text(self, r: int, c: int) -> str:
        it = self.table.item(r, c)
        return it.text().strip() if it is not None else ""

    def _set_row(self, r: int, row: ProfileRow) -> None:
        self.table.setItem(r, 0, QTableWidgetItem(f"{row.t:g}"))
        self.table.setItem(r, 1, QTableWidgetItem(f"{row.g:g}"))
        self.table.setItem(r, 2, QTableWidgetItem(f"{row.t_c:g}"))

    def _seed_default(self) -> None:
        self.table.setRowCount(0)
        for rr in [
            ProfileRow(0.0, 1000.0, 25.0),
            ProfileRow(0.18, 600.0, 25.0),
            ProfileRow(0.28, 950.0, 25.0),
            ProfileRow(0.38, 500.0, 25.0),
            ProfileRow(0.50, 900.0, 25.0),
        ]:
            self.add_row(rr)

    # ---------------------------
    # Actions
    # ---------------------------
    def add_row(self, row: Optional[ProfileRow] = None) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        if row is None:
            t_last = 0.0
            if r > 0:
                try:
                    t_last = float(self._cell_text(r - 1, 0))
                except Exception:
                    t_last = 0.0
            row = ProfileRow(t_last + 0.05, 1000.0, 25.0)
        self._set_row(r, row)
        self._refresh_preview()

    def delete_selected_rows(self) -> None:
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return
        for idx in sorted([s.row() for s in sel], reverse=True):
            self.table.removeRow(idx)
        self._refresh_preview()

    def clear(self) -> None:
        self.table.setRowCount(0)
        self._refresh_preview()

    def browse(self) -> None:
        start = str(profiles_dir())
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open profile CSV",
            start,
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path_str:
            return
        self.set_path(Path(path_str))

    def load(self) -> None:
        p = self.resolved_path()
        if p is None:
            QMessageBox.warning(self, "No file", "Please enter a profile CSV path.")
            return
        if not p.exists():
            QMessageBox.warning(self, "Not found", f"File does not exist:\n{p}")
            return
        rows = read_profile_csv(p)
        self.table.setRowCount(0)
        for rr in rows:
            self.add_row(rr)
        self._refresh_preview()

    def save(self) -> None:
        p = self.resolved_path()
        if p is None:
            QMessageBox.warning(self, "No file", "Please enter a profile CSV path.")
            return
        if p.suffix.lower() != ".csv":
            p = p.with_suffix(".csv")
        rows = self.rows()
        if not rows:
            QMessageBox.warning(self, "Empty", "Profile has no valid rows to save.")
            return
        try:
            write_profile_csv(p, rows)
            self.set_path(p)
            self.profile_saved.emit(str(p))
        except Exception as e:
            QMessageBox.warning(self, "Save failed", f"{type(e).__name__}: {e}")

    # ---------------------------
    # Preview
    # ---------------------------
    def _refresh_preview(self) -> None:
        if pg is None or self.g_plot is None or self.t_plot is None:
            return
        rows = self.rows()
        if not rows:
            self.g_plot.clear()
            self.t_plot.clear()
            return
        t = [r.t for r in rows]
        g = [r.g for r in rows]
        tc = [r.t_c for r in rows]
        self.g_plot.clear()
        self.t_plot.clear()
        self.g_plot.plot(t, g)
        self.t_plot.plot(t, tc)
        self.g_plot.setXRange(t[0], t[-1], padding=0.02)


class ProfileEditorDialog(QDialog):
    """Modal dialog wrapper for ProfileEditor."""

    def __init__(
        self,
        *,
        initial_path: Optional[Path] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Profile Editor")

        root = QVBoxLayout(self)
        self.editor = ProfileEditor(initial_path=initial_path)
        root.addWidget(self.editor, 1)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_close = QPushButton("Close")
        btns.addWidget(self.btn_close)
        root.addLayout(btns)

        self.btn_close.clicked.connect(self.accept)