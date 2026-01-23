

"""ui.desktop.terminal_panel

Reusable terminal/log panel for Aurora.

This is intentionally pragmatic (not a full PTY terminal):
- Displays streaming text output (stdout/stderr merged or separate)
- Provides Clear + Save buttons
- Optional input line + Send button (for future command routing)
- Can attach to a QProcess to mirror subprocess output into the panel

Use:
    panel = TerminalPanel(title="Run log")
    panel.append_line("hello")
    panel.attach_process(proc)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QProcess, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)


@dataclass
class TerminalSaveOptions:
    default_dir: Path = Path(".")
    default_name: str = "aurora_log.txt"


class TerminalPanel(QWidget):
    """A lightweight terminal/log panel."""

    # Emitted when the user presses Send (input is enabled)
    send_text = pyqtSignal(str)

    def __init__(
        self,
        *,
        title: str = "Terminal",
        placeholder: str = "Output will appear here…",
        max_blocks: int = 5000,
        enable_input: bool = False,
        save_opts: Optional[TerminalSaveOptions] = None,
    ) -> None:
        super().__init__()
        self._save_opts = save_opts or TerminalSaveOptions()
        self._proc: Optional[QProcess] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # Header row
        header = QHBoxLayout()
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: 600;")
        header.addWidget(self.title_label)
        header.addStretch(1)

        self.btn_clear = QPushButton("Clear")
        self.btn_save = QPushButton("Save…")
        header.addWidget(self.btn_clear)
        header.addWidget(self.btn_save)

        root.addLayout(header)

        # Output
        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setMaximumBlockCount(max_blocks)
        self.out.setPlaceholderText(placeholder)
        # Use a monospace-ish look without relying on system fonts
        self.out.setStyleSheet("font-family: Menlo, Monaco, Consolas, 'Courier New', monospace;")
        root.addWidget(self.out, 1)

        # Optional input
        self._input_enabled = enable_input
        self.input_row: Optional[QHBoxLayout] = None
        self.inp: Optional[QLineEdit] = None
        self.btn_send: Optional[QPushButton] = None

        if enable_input:
            row = QHBoxLayout()
            inp = QLineEdit()
            inp.setPlaceholderText("Type here…")
            btn = QPushButton("Send")
            row.addWidget(inp, 1)
            row.addWidget(btn)
            root.addLayout(row)
            self.input_row = row
            self.inp = inp
            self.btn_send = btn

            btn.clicked.connect(self._on_send)
            inp.returnPressed.connect(self._on_send)

        # Wire buttons
        self.btn_clear.clicked.connect(self.clear)
        self.btn_save.clicked.connect(self.save)

    # ---------------------------
    # Public API
    # ---------------------------
    def append_text(self, text: str) -> None:
        """Append raw text (no newline normalization)."""
        if not text:
            return
        # Preserve existing newlines; QPlainTextEdit appends at the end
        cursor = self.out.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.out.setTextCursor(cursor)
        self.out.ensureCursorVisible()

    def append_line(self, line: str) -> None:
        """Append a single line with newline handling."""
        if line is None:
            return
        s = str(line)
        if not s.endswith("\n"):
            s += "\n"
        self.append_text(s)

    def clear(self) -> None:
        self.out.clear()

    def set_title(self, title: str) -> None:
        self.title_label.setText(title)

    def attach_process(self, proc: QProcess, *, merge_channels: bool = True) -> None:
        """Attach to a QProcess and mirror its output into the panel.

        If merge_channels is True, the caller should set QProcess to MergedChannels,
        or we will attempt to set it here.
        """
        self.detach_process()
        self._proc = proc

        if merge_channels:
            try:
                proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            except Exception:
                pass

        proc.readyReadStandardOutput.connect(self._on_proc_stdout)
        proc.readyReadStandardError.connect(self._on_proc_stderr)
        proc.started.connect(lambda: self.append_line("[ui] process started"))
        proc.finished.connect(lambda code, status: self.append_line(f"[ui] process finished (code={code}, status={status.name})"))
        proc.errorOccurred.connect(lambda _: self.append_line("[ui] process error occurred"))

    def detach_process(self) -> None:
        """Detach from any attached QProcess."""
        if self._proc is None:
            return
        try:
            self._proc.readyReadStandardOutput.disconnect(self._on_proc_stdout)
        except Exception:
            pass
        try:
            self._proc.readyReadStandardError.disconnect(self._on_proc_stderr)
        except Exception:
            pass
        self._proc = None

    def save(self) -> None:
        """Save the current terminal buffer to a text file."""
        start_dir = str(self._save_opts.default_dir)
        default_name = self._save_opts.default_name
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save log",
            str(Path(start_dir) / default_name),
            "Text Files (*.txt);;All Files (*)",
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.out.toPlainText(), encoding="utf-8")
            self.append_line(f"[ui] log saved to {path}")
        except Exception as e:
            self.append_line(f"[ui] failed to save log: {type(e).__name__}: {e}")

    # ---------------------------
    # Internals
    # ---------------------------
    def _on_proc_stdout(self) -> None:
        if self._proc is None:
            return
        data = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self.append_text(data)

    def _on_proc_stderr(self) -> None:
        if self._proc is None:
            return
        data = bytes(self._proc.readAllStandardError()).decode("utf-8", errors="replace")
        if data:
            self.append_text(data)

    def _on_send(self) -> None:
        if not self._input_enabled or self.inp is None:
            return
        txt = self.inp.text().rstrip("\n")
        if not txt:
            return
        self.send_text.emit(txt)
        # Echo locally like a terminal would
        self.append_line(f"> {txt}")
        self.inp.clear()