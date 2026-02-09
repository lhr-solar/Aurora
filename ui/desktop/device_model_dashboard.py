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
    QGroupBox,
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

        # ---- Presets row (v1: only Default, but keeps UI extensible) ----
        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)

        preset_label = QLabel("Preset")
        preset_label.setMinimumWidth(60)
        preset_row.addWidget(preset_label)

        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Default")
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self.preset_combo, 1)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self._on_reset_clicked)
        preset_row.addWidget(self.btn_reset)

        self.btn_load = QPushButton("Load")
        self.btn_load.clicked.connect(self._on_load_clicked)
        preset_row.addWidget(self.btn_load)

        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self._on_save_clicked)
        preset_row.addWidget(self.btn_save)

        root.addLayout(preset_row)

        # ---- Parameters form ----
        group = QGroupBox("Cell parameters")
        g_layout = QVBoxLayout(group)
        g_layout.setContentsMargins(8, 8, 8, 8)
        g_layout.setSpacing(6)

        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        g_layout.addLayout(form)

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

        # ---- Preview + apply ----
        footer = QHBoxLayout()
        footer.setSpacing(8)

        self.preview = QLabel("")
        self.preview.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.preview.setStyleSheet("color: #444;")
        footer.addWidget(self.preview, 1)

        self.btn_apply = QPushButton("Apply")
        self.btn_apply.clicked.connect(self._on_apply_clicked)
        footer.addWidget(self.btn_apply)

        root.addLayout(footer)

    # ---------------- Data I/O ---------------- #

    def _load_from_disk_or_defaults(self) -> None:
        params = None
        try:
            if self._conf_path.exists():
                params = json.loads(self._conf_path.read_text(encoding="utf-8"))
        except Exception:
            params = None

        if not isinstance(params, dict):
            params = _default_cell_params()

        self.set_params(params)
        self._dirty = False

    def _save_to_disk(self, params: Dict[str, Any]) -> None:
        self._conf_path.parent.mkdir(parents=True, exist_ok=True)
        self._conf_path.write_text(json.dumps(params, indent=2, sort_keys=True), encoding="utf-8")

    # ---------------- Public helpers ---------------- #

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
        # v1 only: Default preset resets to default params.
        if self.preset_combo.currentText() == "Default":
            self.set_params(_default_cell_params())
            self._dirty = True

    def _on_reset_clicked(self) -> None:
        self.set_params(_default_cell_params())
        self._dirty = True

    def _on_load_clicked(self) -> None:
        self._load_from_disk_or_defaults()

    def _on_save_clicked(self) -> None:
        params = self.get_params()
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