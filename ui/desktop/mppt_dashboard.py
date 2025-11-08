"""MPPT Dashboard

High-level overview UI for Aurora's MPPT algorithm stack.

This window is intentionally *lightweight* and focuses on:

- Discovering which MPPT algorithms are registered.
- Inspecting their metadata (labels, parameters, notes).
- Providing a placeholder panel for runtime telemetry that can be wired
  to a simulator/engine later on.

It does **not** attempt to run full simulations by itself; instead,
other parts of the application (e.g. `simulators/engine.py`) should:

- Construct and step algorithms via `core.mppt_algorithms.registry.build(...)`.
- Stream telemetry (measurements, actions, state) into this UI
  if you choose to extend it.

The goal is to keep this file purely in the "desktop frontend" layer,
without introducing heavy dependencies beyond PyQt and the registry
introspection functions.
"""

from pathlib import Path
from typing import Any, Dict, Optional

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QApplication,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except Exception as e:  # pragma: no cover - import guard
    raise RuntimeError(
        "PyQt6 must be installed to use the Aurora MPPT dashboard. "
        "Install it with `pip install PyQt6`."
    ) from e


# We only depend on the stable, introspection-style pieces of the MPPT stack.
try:
    from core.mppt_algorithms import registry as mppt_registry
except Exception:  # pragma: no cover - helpful when running from different roots
    # Fallback for running this file directly from random working directories.
    # We avoid crashing here, and instead show an error in the UI if needed.
    mppt_registry = None  # type: ignore[assignment]


class MPPTDashboardWindow(QMainWindow):
    """Desktop dashboard for exploring MPPT algorithms.

    This window is safe to run even if no simulators exist yet. It can be
    launched on its own, or embedded/owned by a higher-level main window.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Aurora – MPPT Dashboard")

        self._catalog: Dict[str, Dict[str, Any]] = {}

        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        # ------------------------------------------------------------------
        # Left: algorithm list
        # ------------------------------------------------------------------
        left_panel = QVBoxLayout()

        alg_group = QGroupBox("Available MPPT Algorithms", self)
        alg_layout = QVBoxLayout(alg_group)

        self.alg_list = QListWidget(self)
        self.alg_list.currentItemChanged.connect(self._on_algo_selected)

        refresh_btn = QPushButton("Refresh Catalog", self)
        refresh_btn.clicked.connect(self._refresh_catalog)

        alg_layout.addWidget(self.alg_list, 1)
        alg_layout.addWidget(refresh_btn, 0)

        left_panel.addWidget(alg_group, 1)

        root_layout.addLayout(left_panel, 1)

        # ------------------------------------------------------------------
        # Right: details + telemetry placeholder
        # ------------------------------------------------------------------
        right_panel = QVBoxLayout()

        # Details group -----------------------------------------------------
        details_group = QGroupBox("Algorithm Details", self)
        details_layout = QVBoxLayout(details_group)

        self.detail_label = QLabel("Select an algorithm to see its details.", self)
        self.detail_label.setWordWrap(True)

        self.param_table = QTableWidget(self)
        self.param_table.setColumnCount(4)
        self.param_table.setHorizontalHeaderLabels([
            "Name",
            "Default",
            "Range",
            "Description",
        ])
        self.param_table.horizontalHeader().setStretchLastSection(True)
        self.param_table.verticalHeader().setVisible(False)
        self.param_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.param_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        self.notes_edit = QTextEdit(self)
        self.notes_edit.setReadOnly(True)
        self.notes_edit.setPlaceholderText("Algorithm notes and documentation will appear here.")

        details_layout.addWidget(self.detail_label)
        details_layout.addWidget(self.param_table, 2)
        details_layout.addWidget(QLabel("Notes:", self))
        details_layout.addWidget(self.notes_edit, 3)

        right_panel.addWidget(details_group, 2)

        # Telemetry / runtime panel ----------------------------------------
        telemetry_group = QGroupBox("Runtime Telemetry (hooked in by simulators)", self)
        telemetry_layout = QVBoxLayout(telemetry_group)

        self.telemetry_view = QTextEdit(self)
        self.telemetry_view.setReadOnly(True)
        self.telemetry_view.setPlaceholderText(
            "When you connect this dashboard to a running simulator, "
            "you can stream measurements, controller state, and actions "
            "into this panel (e.g., via a simple signal/slot API).\n\n"
            "For now, this acts as a scratch area for logging and notes."
        )

        telemetry_layout.addWidget(self.telemetry_view)

        right_panel.addWidget(telemetry_group, 1)

        root_layout.addLayout(right_panel, 2)

        # Status bar --------------------------------------------------------
        self.statusBar().showMessage("Loading MPPT catalog...")

        # Initial load
        self._refresh_catalog()

    # ------------------------------------------------------------------
    # Catalog loading / selection
    # ------------------------------------------------------------------

    def _refresh_catalog(self) -> None:
        """(Re)load the registry catalog into the list widget."""
        self.alg_list.clear()
        self._catalog.clear()

        if mppt_registry is None:
            self.statusBar().showMessage("MPPT registry not importable – check PYTHONPATH.")
            QMessageBox.warning(
                self,
                "MPPT registry unavailable",
                "Could not import `core.mppt_algorithms.registry`.\n\n"
                "Make sure you are running the dashboard from the Aurora repo "
                "(or that the package is installed).",
            )
            return

        try:
            # catalog() is expected to return a mapping from key -> metadata dict
            catalog = mppt_registry.catalog()  # type: ignore[attr-defined]
        except Exception as e:  # pragma: no cover - defensive
            self.statusBar().showMessage("Failed to query MPPT catalog.")
            QMessageBox.critical(
                self,
                "Catalog error",
                f"Error while querying mppt_registry.catalog():\n{e}",
            )
            return

        # Keep a local copy so we don't have to hit the registry every time.
        self._catalog = dict(catalog)

        # Populate list widget with human-friendly labels
        for key in sorted(self._catalog.keys()):
            meta = self._catalog.get(key, {})
            label = meta.get("label", key.upper())
            item = QListWidgetItem(label, self.alg_list)
            item.setData(Qt.ItemDataRole.UserRole, key)

        self.statusBar().showMessage(f"Loaded {len(self._catalog)} MPPT algorithms.")

        # Auto-select the first algorithm if available
        if self.alg_list.count() > 0 and self.alg_list.currentRow() < 0:
            self.alg_list.setCurrentRow(0)

    def _on_algo_selected(
        self,
        current: Optional[QListWidgetItem],
        _previous: Optional[QListWidgetItem],
    ) -> None:
        """Update details when the user selects a different algorithm."""
        if current is None:
            self._clear_details()
            return

        key = current.data(Qt.ItemDataRole.UserRole)
        if not isinstance(key, str) or key not in self._catalog:
            self._clear_details()
            return

        meta = self._catalog.get(key, {})

        # Top label ---------------------------------------------------------
        label = meta.get("label", key.upper())
        factory = None
        try:
            if mppt_registry is not None:
                available = mppt_registry.available()  # type: ignore[attr-defined]
                factory = available.get(key)
        except Exception:
            factory = None

        label_text = f"Key: {key}\nLabel: {label}"
        if factory:
            label_text += f"\nFactory: {factory}"
        self.detail_label.setText(label_text)

        # Parameters table --------------------------------------------------
        params = meta.get("params", []) or []
        self._populate_param_table(params)

        # Notes / docstring -------------------------------------------------
        notes = meta.get("notes") or meta.get("description") or ""
        if not notes:
            notes = (
                "No notes/description provided for this algorithm in the registry.\n\n"
                "You can add a `notes` or `description` field to its registry "
                "metadata to surface richer documentation here."
            )
        self.notes_edit.setPlainText(str(notes))

        # Helpful status message
        self.statusBar().showMessage(f"Selected algorithm: {key}")

    def _populate_param_table(self, params: Any) -> None:
        """Fill the parameters table from the registry metadata.

        The registry is expected to return a list of objects that can be
        treated like dictionaries, with some of the following keys:

            - name (str)
            - default (Any)
            - min / max (numeric, optional)
            - help / description / label (str, optional)

        This function is intentionally tolerant: if a field is missing,
        we simply leave the cell blank.
        """
        rows = 0
        try:
            rows = len(params)
        except TypeError:
            params = []

        self.param_table.setRowCount(rows)

        for row, raw in enumerate(params):
            if isinstance(raw, dict):
                p = raw
            else:
                # Best-effort: try to access __dict__ or treat as generic object
                p = getattr(raw, "__dict__", {})

            name = str(p.get("name", ""))
            default = p.get("default")
            if isinstance(default, float):
                default_str = f"{default:g}"
            else:
                default_str = "" if default is None else str(default)

            min_v = p.get("min")
            max_v = p.get("max")
            if min_v is not None or max_v is not None:
                range_str = f"{min_v if min_v is not None else ''} .. {max_v if max_v is not None else ''}"
            else:
                range_str = ""

            desc = (
                p.get("help")
                or p.get("description")
                or p.get("label")
                or ""
            )

            self.param_table.setItem(row, 0, QTableWidgetItem(name))
            self.param_table.setItem(row, 1, QTableWidgetItem(default_str))
            self.param_table.setItem(row, 2, QTableWidgetItem(range_str))
            self.param_table.setItem(row, 3, QTableWidgetItem(str(desc)))

        self.param_table.resizeColumnsToContents()

    def _clear_details(self) -> None:
        """Reset the details panel to an empty state."""
        self.detail_label.setText("Select an algorithm to see its details.")
        self.param_table.setRowCount(0)
        self.notes_edit.clear()

    # ------------------------------------------------------------------
    # Public helpers for future integration
    # ------------------------------------------------------------------

    def update_from_record(self, rec: Dict[str, Any]) -> None:
        """Update the dashboard from a SimulationEngine record.

        This is a convenience helper for live simulation.
        A typical record is a dict with keys like::

            {"t", "v", "i", "p", "g", "t_mod", "action", ...}

        where ``action`` is itself a dict that may contain a ``state`` or
        ``mode`` field. This method formats a human-readable line and
        forwards it to :meth:`append_telemetry_line`.
        """
        # Extract core scalars, falling back safely when fields are missing.
        try:
            t = float(rec.get("t", 0.0))
            v = float(rec.get("v", 0.0))
            i = float(rec.get("i", 0.0))
        except Exception:
            # If we can't parse the basic numeric fields, just dump the dict.
            self.append_telemetry_line(str(rec))
            return

        # Derive power if not explicitly given.
        p_val = rec.get("p")
        try:
            p = float(p_val) if p_val is not None else v * i
        except Exception:
            p = v * i

        g = rec.get("g")
        t_mod = rec.get("t_mod")
        action = rec.get("action") or {}
        state = None
        if isinstance(action, dict):
            state = action.get("state") or action.get("mode")

        parts = [
            f"t={t:.4f}s",
            f"V={v:.3f} V",
            f"I={i:.3f} A",
            f"P={p:.3f} W",
        ]

        if g is not None:
            try:
                parts.append(f"G={float(g):.1f} W/m²")
            except Exception:
                parts.append(f"G={g}")

        if t_mod is not None:
            try:
                parts.append(f"T={float(t_mod):.1f} °C")
            except Exception:
                parts.append(f"T={t_mod}")

        if state is not None:
            parts.append(f"state={state}")

        line = " | ".join(parts)
        self.append_telemetry_line(line)

    def append_telemetry_line(self, text: str) -> None:
        """Append a line of text to the telemetry panel.

        This is a convenience method that simulators can call once you
        wire them in. Example:

            dashboard.append_telemetry_line(
                f"t={t:.3f}s | V={m.v:.2f} V, I={m.i:.2f} A, P={m.p:.2f} W"
            )
        """
        self.telemetry_view.append(text)
        self.telemetry_view.moveCursor(self.telemetry_view.textCursor().MoveOperation.End)


def main() -> int:
    """Run the MPPT dashboard as a standalone window.

    Useful while developing algorithms or when you just want to inspect
    what is registered without launching the full Aurora frontend.
    """
    import sys

    app = QApplication(sys.argv)

    win = MPPTDashboardWindow()
    win.resize(1000, 600)
    win.show()

    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())