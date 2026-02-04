#!/usr/bin/env python3
"""
Aurora – MPPT Simulation & Analysis Platform
Primary application entry point.
"""

import sys
from PyQt6.QtWidgets import QApplication

def main():
    # Import here so PYTHONPATH is already correct
    from ui.desktop.main_window import MainWindow

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()