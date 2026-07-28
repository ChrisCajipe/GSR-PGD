"""
main.py
-------
Entry point for the GSR-PGD Adversarial Robustness Evaluation Tool GUI.

Run with:
    python main.py
"""

import sys

from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow
from ui.theme import GLOBAL_QSS
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(GLOBAL_QSS)
    
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
