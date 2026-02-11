from __future__ import annotations

"""ui.desktop.device_model_dashboard

Device / PV model configuration panel.

Purpose
- Let users swap PV *device model* parameters (cell characteristics) without
  cluttering the Lab Dashboard.
- Persist these parameters across app restarts (configs/cell_conf.json).
- Emit a signal when parameters are applied so dashboards can inject them into
  SimulationConfig.cell_params.

Notes
- This dashboard is intentionally conservative: it edits *device parameters*
  (Isc/Voc, Rs/Rsh, diode ideality, temperature coefficients, Vmpp/Impp, etc.).
  Environment values (irradiance/temperature) are not set here.
- Parameter key names must match Cell.__init__ kwargs.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QDoubleSpinBox,
    QCheckBox,
    QMessageBox,
    QInputDialog,
    QGroupBox,
    QSizePolicy,
    QMenu,
)

# Root of the Aurora repo (…/Aurora)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONF_PATH = REPO_ROOT / "configs" / "cell_conf.json"


def _default_cell_params() -> Dict[str, Any]:
    """Default device parameters used by Aurora if nothing is configured.

    IMPORTANT: Key names must match Cell.__init__.
    """
    return {
        # STC reference points
        "isc_ref": 5.84,
        "voc_ref": 0.621,

        # Diode model / parasitics
        "diode_ideality": 1.30,
        "r_s": 0.02,
        "r_sh": 200.0,

        # Temperature coefficients
        "voc_temp_coeff": -0.0023,
        "isc_temp_coeff": 0.00035,

        # Optional MPP point (if your Cell model uses it)
        "vmpp": 0.50,
        "impp": 5.50,

        # Fit behavior
        "autofit": False,
    }


class DeviceModelDashboard(QWidget):
    """Tab for editing PV device/cell model parameters."""

    # Emits a dict of cell parameters to be used for subsequent runs.
    cell_params_applied = pyqtSignal(object)  # object => dict

    def __init__(self, parent: Optional[QWidget] = None, *, conf_path: Optional[Path] = None) -> None:
        super().__init__(parent)
        self.setObjectName("DeviceModelDashboard")

        self._conf_path = Path(conf_path) if conf_path is not None else DEFAULT_CONF_PATH
        self._dirty = False
        self._presets: Dict[str, Dict[str, Any]] = {"Default": _default_cell_params()}
        self._active_preset: str = "Default"

        self._build_ui()
        self._load_from_disk_or_defaults()
        self._refresh_preview()

    # ---------------- UI ---------------- #

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        header = QLabel("Aurora — Device Model")
        header.setStyleSheet("font-size: 16px; font-weight: 600;")
        root.addWidget(header)

        subtitle = QLabel(
            "Edit PV cell/device characteristics used to build the simulated array. "
            "These are saved to configs/cell_conf.json and applied to new runs."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #666;")
        root.addWidget(subtitle)

        # ---- Presets row (top bar) ----
        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)

        preset_label = QLabel("Preset")
        preset_label.setMinimumWidth(60)
        preset_row.addWidget(preset_label)

        self.preset_combo = QComboBox()
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self.preset_combo, 1)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self._on_reset_clicked)
        preset_row.addWidget(self.btn_reset)

        self.btn_new = QPushButton("New")
        self.btn_new.setToolTip("Create a new preset initialized to defaults")
        self.btn_new.clicked.connect(self._on_new_clicked)
        preset_row.addWidget(self.btn_new)

        self.btn_save_as = QPushButton("Duplicate")
        self.btn_save_as.setToolTip("Duplicate the current field values into a new preset")
        self.btn_save_as.clicked.connect(self._on_save_as_clicked)
        preset_row.addWidget(self.btn_save_as)

        # Less-frequent actions go into a dropdown to keep the top bar clean.
        self.btn_manage = QPushButton("Manage")
        self.btn_manage.setToolTip("Preset management actions")

        manage_menu = QMenu(self)
        act_rename = manage_menu.addAction("Rename…")
        act_delete = manage_menu.addAction("Delete…")
        manage_menu.addSeparator()
        act_reload = manage_menu.addAction("Reload from disk")

        act_rename.triggered.connect(self._on_rename_clicked)
        act_delete.triggered.connect(self._on_delete_clicked)
        act_reload.triggered.connect(self._on_load_clicked)

        self.btn_manage.setMenu(manage_menu)
        preset_row.addWidget(self.btn_manage)

        root.addLayout(preset_row)

        # ---- Parameters form ----
        group = QGroupBox("Cell parameters")
        g_layout = QHBoxLayout(group)
        g_layout.setContentsMargins(8, 8, 8, 8)
        g_layout.setSpacing(6)

        # Left: parameter form
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        left_layout.addLayout(form)

        # Right: info / preview panel
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        info_title = QLabel("Preset summary")
        info_title.setStyleSheet("font-weight: 600;")
        right_layout.addWidget(info_title)

        self.preview = QLabel("")
        self.preview.setWordWrap(True)
        self.preview.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.preview.setStyleSheet("color: #E6E6E6; font-weight: 500;")
        right_layout.addWidget(self.preview)

        hint = QLabel(
            "Tip: ‘Apply’ affects new runs. Use ‘Save’ to persist presets to disk. "
            "Use ‘Reload’ to re-read configs/cell_conf.json."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #A8A8A8;")
        right_layout.addWidget(hint)

        # Apply lives in the summary panel to reduce top-bar clutter.
        self.btn_apply_right = QPushButton("Apply")
        self.btn_apply_right.setToolTip("Apply this preset to new runs (also saves)")
        self.btn_apply_right.clicked.connect(self._on_apply_clicked)
        self.btn_apply_right.setMinimumHeight(34)
        self.btn_apply_right.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        right_layout.addWidget(self.btn_apply_right)

        # Save/Reload buttons styled like Apply (full width)
        self.btn_save = QPushButton("Save")
        self.btn_save.setToolTip("Save presets to disk (no apply)")
        self.btn_save.clicked.connect(self._on_save_clicked)
        self.btn_save.setMinimumHeight(34)
        self.btn_save.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        right_layout.addWidget(self.btn_save)

        self.btn_reload = QPushButton("Reload")
        self.btn_reload.setToolTip("Reload presets from disk")
        self.btn_reload.clicked.connect(self._on_load_clicked)
        self.btn_reload.setMinimumHeight(34)
        self.btn_reload.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        right_layout.addWidget(self.btn_reload)

        right_layout.addStretch(1)

        g_layout.addWidget(left, 3)
        g_layout.addWidget(right, 2)

        self._fields: Dict[str, Any] = {}

        # Helper to create consistent spin boxes
        def _spin(
            *,
            decimals: int,
            minimum: float,
            maximum: float,
            step: float,
            suffix: str = "",
        ) -> QDoubleSpinBox:
            w = QDoubleSpinBox()
            w.setDecimals(decimals)
            w.setMinimum(minimum)
            w.setMaximum(maximum)
            w.setSingleStep(step)
            if suffix:
                w.setSuffix(suffix)
            w.setMinimumWidth(180)
            w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            w.valueChanged.connect(self._mark_dirty)
            return w

        # STC reference points
        self._fields["isc_ref"] = _spin(decimals=5, minimum=0.0, maximum=1000.0, step=0.01, suffix=" A")
        form.addRow("Isc_ref", self._fields["isc_ref"])

        self._fields["voc_ref"] = _spin(decimals=6, minimum=0.0, maximum=10.0, step=0.001, suffix=" V")
        form.addRow("Voc_ref", self._fields["voc_ref"])

        # Diode model / parasitics
        self._fields["diode_ideality"] = _spin(decimals=4, minimum=0.5, maximum=5.0, step=0.01)
        form.addRow("Diode ideality (n)", self._fields["diode_ideality"])

        self._fields["r_s"] = _spin(decimals=6, minimum=0.0, maximum=10.0, step=0.001, suffix=" Ω")
        form.addRow("Series resistance (Rs)", self._fields["r_s"])

        self._fields["r_sh"] = _spin(decimals=3, minimum=0.0, maximum=1_000_000.0, step=1.0, suffix=" Ω")
        form.addRow("Shunt resistance (Rsh)", self._fields["r_sh"])

        # Temp coefficients
        self._fields["voc_temp_coeff"] = _spin(decimals=7, minimum=-1.0, maximum=1.0, step=0.0001, suffix=" V/°C")
        form.addRow("Voc temp coeff", self._fields["voc_temp_coeff"])

        self._fields["isc_temp_coeff"] = _spin(decimals=7, minimum=-1.0, maximum=1.0, step=0.0001, suffix=" A/°C")
        form.addRow("Isc temp coeff", self._fields["isc_temp_coeff"])

        # Optional MPP point
        self._fields["vmpp"] = _spin(decimals=6, minimum=0.0, maximum=10.0, step=0.001, suffix=" V")
        form.addRow("Vmpp", self._fields["vmpp"])

        self._fields["impp"] = _spin(decimals=5, minimum=0.0, maximum=1000.0, step=0.01, suffix=" A")
        form.addRow("Impp", self._fields["impp"])

        # Fit mode
        self._fields["autofit"] = QCheckBox("Autofit diode params from STC points")
        self._fields["autofit"].stateChanged.connect(self._mark_dirty)
        form.addRow("", self._fields["autofit"])

        root.addWidget(group, 1)

    # ---------------- Data I/O ---------------- #

    def _load_from_disk_or_defaults(self) -> None:
        """Load presets + active preset from disk (or fallback to defaults).

        Supports:
        - Legacy format: a flat dict of params
        - New format: {"active_preset": str, "presets": {name: params_dict}}
        """
        cfg: Optional[Dict[str, Any]] = None
        try:
            if self._conf_path.exists():
                cfg = json.loads(self._conf_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = None

        presets: Dict[str, Dict[str, Any]] = {"Default": _default_cell_params()}
        active = "Default"

        if isinstance(cfg, dict) and "presets" in cfg:
            raw_presets = cfg.get("presets")
            raw_active = cfg.get("active_preset")
            if isinstance(raw_presets, dict):
                for name, params in raw_presets.items():
                    if isinstance(name, str) and isinstance(params, dict):
                        presets[name] = params
            if isinstance(raw_active, str) and raw_active in presets:
                active = raw_active
        elif isinstance(cfg, dict):
            # Legacy schema: treat as Default preset
            presets["Default"] = cfg
            active = "Default"

        self._presets = presets
        self._active_preset = active

        self._rebuild_preset_combo(select_name=self._active_preset)

        # Load active preset into fields
        self.set_params(self._presets.get(self._active_preset, _default_cell_params()))
        self._dirty = False

    def _save_to_disk(self, params: Dict[str, Any]) -> None:
        """Persist current params into the active preset and write preset schema."""
        self._conf_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure in-memory structures
        if not isinstance(getattr(self, "_presets", None), dict):
            self._presets = {"Default": _default_cell_params()}
        if not isinstance(getattr(self, "_active_preset", None), str) or not self._active_preset:
            self._active_preset = "Default"

        self._presets[self._active_preset] = dict(params)

        payload = {
            "active_preset": self._active_preset,
            "presets": self._presets,
        }
        self._conf_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    # ---------------- Public helpers ---------------- #
    def _rebuild_preset_combo(self, select_name: Optional[str] = None) -> None:
        """Rebuild the preset dropdown from current presets and select `select_name` if present."""
        if not isinstance(getattr(self, "_presets", None), dict):
            self._presets = {"Default": _default_cell_params()}

        names = sorted([n for n in self._presets.keys() if isinstance(n, str)])
        # Ensure Default exists and is visible
        if "Default" in self._presets and "Default" not in names:
            names.insert(0, "Default")

        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        for n in names:
            self.preset_combo.addItem(n)

        target = (select_name or self._active_preset or "Default").strip() or "Default"
        idx = self.preset_combo.findText(target)
        if idx < 0:
            idx = self.preset_combo.findText("Default")
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)
        self.preset_combo.blockSignals(False)

    def get_params(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, w in self._fields.items():
            if isinstance(w, QDoubleSpinBox):
                out[k] = float(w.value())
            elif isinstance(w, QCheckBox):
                out[k] = bool(w.isChecked())
        return out

    def set_params(self, params: Dict[str, Any]) -> None:
        # Fill known fields; ignore unknown keys so we can extend later.
        for k, w in self._fields.items():
            if k not in params:
                continue
            v = params[k]
            try:
                if isinstance(w, QDoubleSpinBox):
                    w.blockSignals(True)
                    w.setValue(float(v))
                    w.blockSignals(False)
                elif isinstance(w, QCheckBox):
                    w.blockSignals(True)
                    w.setChecked(bool(v))
                    w.blockSignals(False)
            except Exception:
                # Ignore bad values; keep whatever is already shown.
                try:
                    w.blockSignals(False)
                except Exception:
                    pass

        self._refresh_preview()

    # ---------------- Handlers ---------------- #

    def _mark_dirty(self, *_args) -> None:
        self._dirty = True
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        p = self.get_params()
        try:
            voc = float(p.get("voc_ref", 0.0))
            isc = float(p.get("isc_ref", 0.0))
            vmpp = float(p.get("vmpp", 0.0))
            impp = float(p.get("impp", 0.0))
            pmp = vmpp * impp
            self.preview.setText(
                f"STC: Voc={voc:.4f} V, Isc={isc:.3f} A | Vmpp={vmpp:.4f} V, Impp={impp:.3f} A | Pmpp≈{pmp:.2f} W"
                + ("  (modified)" if self._dirty else "")
            )
        except Exception:
            self.preview.setText("(preview unavailable)")

    def _on_preset_changed(self, _idx: int) -> None:
        name = self.preset_combo.currentText().strip() or "Default"
        if not isinstance(getattr(self, "_presets", None), dict):
            self._presets = {"Default": _default_cell_params()}
        if name not in self._presets:
            self._presets[name] = _default_cell_params()
        self._active_preset = name
        self.set_params(self._presets[name])
        self._dirty = False
        self._refresh_preview()

    def _on_reset_clicked(self) -> None:
        self.set_params(_default_cell_params())
        self._dirty = True

    def _on_new_clicked(self) -> None:
        """Create a new preset from the default field values."""
        name, ok = QInputDialog.getText(self, "New preset", "Preset name:")
        if not ok:
            return
        name = (name or "").strip()
        if not name:
            QMessageBox.warning(self, "Invalid name", "Preset name cannot be empty.")
            return

        if not isinstance(getattr(self, "_presets", None), dict):
            self._presets = {"Default": _default_cell_params()}

        if name in self._presets:
            QMessageBox.warning(self, "Preset exists", f"A preset named '{name}' already exists.")
            return

        # Use default cell params for new preset
        self._presets[name] = _default_cell_params()
        self._active_preset = name

        self._rebuild_preset_combo(select_name=name)
        self.set_params(self._presets[name])

        self._dirty = True
        self._refresh_preview()

    def _on_save_as_clicked(self) -> None:
        """Create a new preset from current field values (does not overwrite existing presets)."""
        name, ok = QInputDialog.getText(self, "Save As…", "New preset name:")
        if not ok:
            return
        name = (name or "").strip()
        if not name:
            QMessageBox.warning(self, "Invalid name", "Preset name cannot be empty.")
            return

        if not isinstance(getattr(self, "_presets", None), dict):
            self._presets = {"Default": _default_cell_params()}

        if name in self._presets:
            QMessageBox.warning(self, "Preset exists", f"A preset named '{name}' already exists.")
            return

        self._presets[name] = self.get_params()
        self._active_preset = name
        self._rebuild_preset_combo(select_name=name)
        self._dirty = True
        self._refresh_preview()

    def _on_rename_clicked(self) -> None:
        """Rename the currently selected preset."""
        old = (self.preset_combo.currentText() or "").strip() or "Default"
        if old == "Default":
            QMessageBox.warning(self, "Not allowed", "The 'Default' preset cannot be renamed.")
            return

        name, ok = QInputDialog.getText(self, "Rename preset", "New name:", text=old)
        if not ok:
            return
        name = (name or "").strip()
        if not name:
            QMessageBox.warning(self, "Invalid name", "Preset name cannot be empty.")
            return

        if not isinstance(getattr(self, "_presets", None), dict):
            self._presets = {"Default": _default_cell_params()}

        if name == old:
            return
        if name in self._presets:
            QMessageBox.warning(self, "Preset exists", f"A preset named '{name}' already exists.")
            return

        # Move preset under new key
        self._presets[name] = self._presets.pop(old)
        if self._active_preset == old:
            self._active_preset = name

        self._rebuild_preset_combo(select_name=name)
        self._dirty = True
        self._refresh_preview()

    def _on_delete_clicked(self) -> None:
        """Delete the currently selected preset."""
        name = (self.preset_combo.currentText() or "").strip() or "Default"
        if name == "Default":
            QMessageBox.warning(self, "Not allowed", "The 'Default' preset cannot be deleted.")
            return

        if not isinstance(getattr(self, "_presets", None), dict):
            self._presets = {"Default": _default_cell_params()}

        if name not in self._presets:
            return

        resp = QMessageBox.question(
            self,
            "Delete preset",
            f"Delete preset '{name}'? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return

        self._presets.pop(name, None)

        # Select a safe active preset after deletion
        self._active_preset = "Default" if "Default" in self._presets else next(iter(self._presets.keys()), "Default")
        self._rebuild_preset_combo(select_name=self._active_preset)
        self.set_params(self._presets.get(self._active_preset, _default_cell_params()))
        self._dirty = True
        self._refresh_preview()

    def _on_load_clicked(self) -> None:
        self._load_from_disk_or_defaults()

    def _on_save_clicked(self) -> None:
        params = self.get_params()
        self._active_preset = self.preset_combo.currentText().strip() or "Default"
        try:
            self._save_to_disk(params)
            self._dirty = False
            self._refresh_preview()
            QMessageBox.information(self, "Saved", f"Saved cell parameters to:\n{self._conf_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"Failed to save cell parameters:\n{type(e).__name__}: {e}")

    def _validate(self, params: Dict[str, Any]) -> Optional[str]:
        """Return an error string if invalid, else None."""
        try:
            voc = float(params.get("voc_ref", 0.0))
            isc = float(params.get("isc_ref", 0.0))
            vmpp = float(params.get("vmpp", 0.0))
            impp = float(params.get("impp", 0.0))
        except Exception:
            return "One or more parameters are not numeric."

        if voc <= 0 or isc <= 0:
            return "Voc_ref and Isc_ref must be > 0."
        if vmpp < 0 or impp < 0:
            return "Vmpp and Impp must be ≥ 0."
        if vmpp > voc + 1e-12:
            return "Vmpp should not exceed Voc_ref."
        if impp > isc + 1e-12:
            return "Impp should not exceed Isc_ref."

        # Basic resistance sanity
        try:
            if float(params.get("r_s", 0.0)) < 0:
                return "Series resistance (r_s) must be ≥ 0."
            if float(params.get("r_sh", 0.0)) < 0:
                return "Shunt resistance (r_sh) must be ≥ 0."
        except Exception:
            return "Resistance parameters must be numeric."

        return None

    def _on_apply_clicked(self) -> None:
        params = self.get_params()
        self._active_preset = self.preset_combo.currentText().strip() or "Default"
        err = self._validate(params)
        if err is not None:
            QMessageBox.warning(self, "Invalid parameters", err)
            return

        # Persist and emit.
        try:
            self._save_to_disk(params)
        except Exception:
            # If save fails, still apply in-memory so the user isn't blocked.
            pass

        self._dirty = False
        self._refresh_preview()
        self.cell_params_applied.emit(params)
        QMessageBox.information(self, "Applied", "Cell parameters applied to new runs.")