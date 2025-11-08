"""Source Dashboard

Frontend window for inspecting and managing PV array / source configurations
in Aurora.

This UI is intentionally conservative:

- It focuses on **viewing** configuration/layout files (typically JSON) that
  describe the PV array hierarchy and conditions.
- It provides a live, hierarchical view (Array → Strings → Substrings → Cells)
  when the schema matches what the core models expect.
- It falls back gracefully to a generic JSON browser if the structure is
  slightly different.
- It exposes a telemetry pane that simulators can target later for logging.

It does *not* modify or persist configs yet; edits should be done in code or
via dedicated tooling. The goal is to keep this firmly in the "desktop
frontend" layer, without making assumptions about how the backend stores
or version-controls configuration.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QApplication,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QStatusBar,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except Exception as e:  # pragma: no cover - import guard
    raise RuntimeError(
        "PyQt6 must be installed to use the Aurora source dashboard. "
        "Install it with `pip install PyQt6`."
    ) from e


# Optional imports from the core modeling layer. We don't strictly need them to
# *view* configs, but they are nice to have for future integration.
try:  # pragma: no cover - optional
    from core.src.array import Array  # type: ignore
except Exception:  # pragma: no cover - best-effort only
    Array = None  # type: ignore[assignment]


class SourceDashboardWindow(QMainWindow):
    """Desktop dashboard for PV array / source configurations."""

    def __init__(self, parent: Optional[Widget] = None) -> None:  # type: ignore[name-defined]
        super().__init__(parent)
        self.setWindowTitle("Aurora – Source Dashboard")

        self._config_path: Optional[Path] = None
        self._raw_config: Optional[Dict[str, Any]] = None
        self._layout_root: Optional[Dict[str, Any]] = None

        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        # ------------------------------------------------------------------
        # Top: config file controls
        # ------------------------------------------------------------------
        config_group = QGroupBox("Configuration File", self)
        cfg_layout = QHBoxLayout(config_group)

        self.path_edit = QLineEdit(self)
        self.path_edit.setPlaceholderText("No config selected (JSON layout)...")

        browse_btn = QPushButton("Browse…", self)
        browse_btn.clicked.connect(self._on_browse)

        reload_btn = QPushButton("Reload", self)
        reload_btn.clicked.connect(self._on_reload)

        cfg_layout.addWidget(QLabel("Config:", self))
        cfg_layout.addWidget(self.path_edit, 1)
        cfg_layout.addWidget(browse_btn)
        cfg_layout.addWidget(reload_btn)

        root_layout.addWidget(config_group)

        # ------------------------------------------------------------------
        # Middle: split – tree on the left, details on the right
        # ------------------------------------------------------------------
        mid_layout = QHBoxLayout()

        # Left: hierarchy tree
        tree_group = QGroupBox("Array Hierarchy", self)
        tree_layout = QVBoxLayout(tree_group)

        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Element", "Type"])
        self.tree.itemSelectionChanged.connect(self._on_tree_selection_changed)

        tree_layout.addWidget(self.tree)

        mid_layout.addWidget(tree_group, 1)

        # Right: details + raw snippet
        detail_group = QGroupBox("Details", self)
        detail_layout = QVBoxLayout(detail_group)

        self.summary_label = QLabel("Load a config file to inspect the array.", self)
        self.summary_label.setWordWrap(True)

        self.attr_table = QTableWidget(self)
        self.attr_table.setColumnCount(2)
        self.attr_table.setHorizontalHeaderLabels(["Key", "Value"])
        self.attr_table.verticalHeader().setVisible(False)
        self.attr_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.attr_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.attr_table.horizontalHeader().setStretchLastSection(True)

        self.raw_view = QTextEdit(self)
        self.raw_view.setReadOnly(True)
        self.raw_view.setPlaceholderText(
            "Raw JSON snippet for the selected node will appear here. "
            "This is useful for debugging and for understanding the exact "
            "schema used by the configuration."
        )

        detail_layout.addWidget(self.summary_label)
        detail_layout.addWidget(self.attr_table, 2)
        detail_layout.addWidget(QLabel("Raw snippet:", self))
        detail_layout.addWidget(self.raw_view, 3)

        mid_layout.addWidget(detail_group, 2)

        root_layout.addLayout(mid_layout, 1)

        # ------------------------------------------------------------------
        # Bottom: telemetry / notes
        # ------------------------------------------------------------------
        telemetry_group = QGroupBox("Telemetry / Notes", self)
        telemetry_layout = QVBoxLayout(telemetry_group)

        self.telemetry_view = QTextEdit(self)
        self.telemetry_view.setReadOnly(True)
        self.telemetry_view.setPlaceholderText(
            "Simulators or external tools can write log lines here to "
            "capture how a given configuration behaves under different "
            "conditions (irradiance, temperature, partial shading, etc.).\n\n"
            "For now, you can also use it as a scratch pad while inspecting "
            "configs."
        )

        telemetry_layout.addWidget(self.telemetry_view)

        root_layout.addWidget(telemetry_group)

        # Status bar
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _on_browse(self) -> None:
        dlg = QFileDialog(self, "Select config file")
        dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
        dlg.setNameFilters([
            "JSON files (*.json)",
            "Config files (*.json *.cfg)",
            "All files (*)",
        ])

        if not dlg.exec():
            return

        files = dlg.selectedFiles()
        if not files:
            return

        path = Path(files[0])
        self._load_config_path(path)

    def _on_reload(self) -> None:
        if self._config_path is None:
            QMessageBox.information(
                self,
                "No config",
                "No configuration file is currently selected.",
            )
            return
        if not self._config_path.exists():
            QMessageBox.warning(
                self,
                "Missing file",
                f"The file '{self._config_path}' no longer exists.",
            )
            return
        self._load_config_path(self._config_path)

    def _load_config_path(self, path: Path) -> None:
        """Load and parse the given config file, updating the UI."""
        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Config error",
                f"Failed to read or parse JSON from '{path}':\n{e}",
            )
            self.statusBar().showMessage("Failed to load config")
            return

        self._config_path = path
        self._raw_config = data if isinstance(data, dict) else {"_root": data}
        self.path_edit.setText(str(path))

        # Heuristic: many Aurora configs keep the layout under "layout".
        layout = self._raw_config.get("layout") if isinstance(self._raw_config, dict) else None
        if layout is None:
            layout = self._raw_config

        if not isinstance(layout, dict):
            # Fallback: treat entire config as generic root
            layout = {"root": layout}

        self._layout_root = layout

        self._populate_tree(layout)
        self._update_summary_for_layout(layout)

        self.statusBar().showMessage(f"Loaded config: {path}")

    # ------------------------------------------------------------------
    # Tree population and selection handling
    # ------------------------------------------------------------------

    def _populate_tree(self, layout: Dict[str, Any]) -> None:
        self.tree.clear()

        # Try a structured view first (Array → Strings → Substrings → Cells)
        root_item = QTreeWidgetItem(["Array", "array"])
        root_item.setData(0, Qt.ItemDataRole.UserRole, layout)
        self.tree.addTopLevelItem(root_item)

        strings = layout.get("strings") if isinstance(layout, dict) else None
        if isinstance(strings, list) and strings:
            for i, s in enumerate(strings):
                s_item = QTreeWidgetItem([f"String {i}", "string"])
                s_item.setData(0, Qt.ItemDataRole.UserRole, s)
                root_item.addChild(s_item)

                substrings = None
                if isinstance(s, dict):
                    substrings = s.get("substrings") or s.get("sub_strings")
                if isinstance(substrings, list) and substrings:
                    for j, sub in enumerate(substrings):
                        sub_item = QTreeWidgetItem([f"Substring {j}", "substring"])
                        sub_item.setData(0, Qt.ItemDataRole.UserRole, sub)
                        s_item.addChild(sub_item)

                        cells = None
                        if isinstance(sub, dict):
                            cells = sub.get("cells") or sub.get("cell_list")
                        if isinstance(cells, list) and cells:
                            for k, cell in enumerate(cells):
                                cell_item = QTreeWidgetItem([f"Cell {k}", "cell"])
                                cell_item.setData(0, Qt.ItemDataRole.UserRole, cell)
                                sub_item.addChild(cell_item)
        else:
            # Generic fallback: show top-level keys of layout
            if isinstance(layout, dict):
                for key, val in layout.items():
                    child = QTreeWidgetItem([str(key), type(val).__name__])
                    child.setData(0, Qt.ItemDataRole.UserRole, val)
                    root_item.addChild(child)

        self.tree.expandAll()
        self.tree.resizeColumnToContents(0)

    def _on_tree_selection_changed(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            self._clear_details()
            return

        item = items[0]
        data = item.data(0, Qt.ItemDataRole.UserRole)
        self._update_details_for_node(item.text(0), item.text(1), data)

    # ------------------------------------------------------------------
    # Detail panel helpers
    # ------------------------------------------------------------------

    def _update_summary_for_layout(self, layout: Dict[str, Any]) -> None:
        if not isinstance(layout, dict):
            self.summary_label.setText("Loaded layout (non-dict root).")
            return

        strings = layout.get("strings")
        n_strings = len(strings) if isinstance(strings, list) else 0

        n_substrings = 0
        n_cells = 0
        if isinstance(strings, list):
            for s in strings:
                subs = None
                if isinstance(s, dict):
                    subs = s.get("substrings") or s.get("sub_strings")
                if isinstance(subs, list):
                    n_substrings += len(subs)
                    for sub in subs:
                        cells = None
                        if isinstance(sub, dict):
                            cells = sub.get("cells") or sub.get("cell_list")
                        if isinstance(cells, list):
                            n_cells += len(cells)

        text = f"Strings: {n_strings} | Substrings: {n_substrings} | Cells: {n_cells}"

        # Try to surface any obvious global conditions
        cond = layout.get("conditions") if isinstance(layout, dict) else None
        if isinstance(cond, dict):
            irr = cond.get("irradiance") or cond.get("G")
            temp = cond.get("temperature") or cond.get("T")
            extra_bits = []
            if irr is not None:
                extra_bits.append(f"G={irr}")
            if temp is not None:
                extra_bits.append(f"T={temp}")
            if extra_bits:
                text += " | " + ", ".join(extra_bits)

        self.summary_label.setText(text)

    def _update_details_for_node(self, name: str, kind: str, data: Any) -> None:
        # Attribute table
        attrs: Dict[str, Any] = {}
        if isinstance(data, dict):
            attrs = data
        elif isinstance(data, (list, tuple)):
            attrs = {f"[{i}]": v for i, v in enumerate(data)}
        else:
            attrs = {"value": data}

        keys = list(attrs.keys())
        self.attr_table.setRowCount(len(keys))

        for row, key in enumerate(keys):
            val = attrs[key]
            if isinstance(val, (dict, list, tuple)):
                display = "<nested>"
            else:
                display = str(val)

            self.attr_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.attr_table.setItem(row, 1, QTableWidgetItem(display))

        self.attr_table.resizeColumnsToContents()

        # Raw JSON snippet
        try:
            pretty = json.dumps(attrs, indent=2, sort_keys=True)
        except Exception:
            pretty = str(attrs)

        header = f"Node: {name} ({kind})\n\n"
        self.raw_view.setPlainText(header + pretty)

    def _clear_details(self) -> None:
        self.summary_label.setText("Load a config file to inspect the array.")
        self.attr_table.setRowCount(0)
        self.raw_view.clear()

    # ------------------------------------------------------------------
    # Public helper for simulators
    # ------------------------------------------------------------------

    def update_from_record(self, rec: Dict[str, Any]) -> None:
        """Update the telemetry panel from a SimulationEngine record.

        This helper is intended for live simulations where `rec` is a dict
        with keys such as::

            {"t", "g", "t_mod", ...}

        It focuses on environment / configuration context (irradiance,
        temperature, etc.) rather than controller internals.
        """
        parts = []

        # Time
        t = rec.get("t")
        if t is not None:
            try:
                parts.append(f"t={float(t):.4f}s")
            except Exception:
                parts.append(f"t={t}")

        # Irradiance / temperature
        g = rec.get("g")
        if g is not None:
            try:
                parts.append(f"G={float(g):.1f} W/m²")
            except Exception:
                parts.append(f"G={g}")

        t_mod = rec.get("t_mod")
        if t_mod is not None:
            try:
                parts.append(f"T={float(t_mod):.1f} °C")
            except Exception:
                parts.append(f"T={t_mod}")

        # Optional extra fields (e.g. shading pattern, config name)
        cfg_name = None
        if self._config_path is not None:
            cfg_name = self._config_path.name
        if cfg_name:
            parts.append(f"config={cfg_name}")

        # Fallback: if we gathered nothing, just dump the dict.
        if not parts:
            self.append_telemetry_line(str(rec))
            return

        line = " | ".join(parts)
        self.append_telemetry_line(line)

    def append_telemetry_line(self, text: str) -> None:
        """Append a line of text to the telemetry panel.

        Simulators can call this to log how a given configuration evolves.
        """
        self.telemetry_view.append(text)
        self.telemetry_view.moveCursor(self.telemetry_view.textCursor().MoveOperation.End)


def main() -> int:
    """Run the Source Dashboard as a standalone window."""
    import sys

    app = QApplication(sys.argv)

    win = SourceDashboardWindow()
    win.resize(1100, 700)
    win.show()

    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())