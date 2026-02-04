

from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QListWidget,
    QListWidgetItem,
    QTextBrowser,
    QLineEdit,
    QLabel,
    QSplitter,
)

# Root of the Aurora repo (…/Aurora)
REPO_ROOT = Path(__file__).resolve().parents[2]

DOC_SOURCES = [
    ("Overview", REPO_ROOT / "README.md"),
    ("How to Run", REPO_ROOT / "docs" / "usage.md"),
    ("Architecture", REPO_ROOT / "docs" / "architecture.md"),
    ("MPPT Algorithms", REPO_ROOT / "docs" / "api.md"),
    ("Output Interpretations", REPO_ROOT / "docs" / "outputs.md"),
    ("Glossary", REPO_ROOT / "docs" / "glossary.md"),
]


class GlossaryDashboard(QWidget):
    """
    Documentation / Glossary dashboard.

    Goal:
    Someone with *zero* Aurora context should be able to:
    - Run the program
    - Understand the architecture
    - Modify or add algorithms
    - Extend the UI / benchmarks
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("GlossaryDashboard")

        self._build_ui()
        self._load_index()
        self._select_default()

    # ---------------- UI ---------------- #

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        header = QLabel("Aurora — Documentation & Glossary")
        header.setStyleSheet("font-size: 16px; font-weight: 600;")
        root.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        # ---- Left panel (index + search) ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(4)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search documentation…")
        self.search.textChanged.connect(self._filter_index)
        left_layout.addWidget(self.search)

        self.index = QListWidget()
        self.index.itemClicked.connect(self._on_item_selected)
        left_layout.addWidget(self.index, 1)

        splitter.addWidget(left)

        # ---- Right panel (content) ----
        self.viewer = QTextBrowser()
        self.viewer.setOpenExternalLinks(True)
        self.viewer.setMarkdown("")
        splitter.addWidget(self.viewer)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # ---------------- Index ---------------- #

    def _load_index(self):
        self.index.clear()
        self._items: list[tuple[str, Path]] = []

        for title, path in DOC_SOURCES:
            item = QListWidgetItem(title)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.index.addItem(item)
            self._items.append((title.lower(), path))

    def _filter_index(self, text: str):
        query = text.lower().strip()
        for i in range(self.index.count()):
            item = self.index.item(i)
            visible = query in item.text().lower()
            item.setHidden(not visible)

    def _select_default(self):
        if self.index.count() > 0:
            self.index.setCurrentRow(0)
            self._load_doc(self.index.item(0))

    # ---------------- Loading ---------------- #

    def _on_item_selected(self, item: QListWidgetItem):
        self._load_doc(item)

    def _load_doc(self, item: QListWidgetItem):
        path: Path = item.data(Qt.ItemDataRole.UserRole)

        if not path.exists():
            self.viewer.setMarkdown(self._missing_doc_md(path))
            return

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            self.viewer.setMarkdown(
                f"### Error loading document\n\n```\n{e}\n```"
            )
            return

        # QTextBrowser.setSource expects a QUrl in PyQt6
        self.viewer.setSource(QUrl.fromLocalFile(str(path)))
        self.viewer.setMarkdown(content)

    # ---------------- Helpers ---------------- #

    @staticmethod
    def _missing_doc_md(path: Path) -> str:
        return f"""
### Missing documentation file

The file below does not exist yet:

```
{path}
```

Recommended action:
- Create this file
- Add onboarding, explanations, and examples
- Keep it Markdown so it renders cleanly here

This page is intentionally part of the UI so documentation
evolves *with* the codebase.
"""