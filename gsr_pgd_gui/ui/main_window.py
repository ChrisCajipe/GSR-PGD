"""
main_window.py
---------------
Top-level QMainWindow: hosts the NavBar and a QStackedWidget with the
three pages (Upload, Results, About).
"""

from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QStackedWidget

from ui.nav_bar import NavBar
from ui.upload_panel import UploadPanel
from ui.results_panel import ResultsPanel
from ui.about_panel import AboutPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GSR-PGD \u2013 Adversarial Robustness Evaluation Tool")
        self.resize(1400, 860)
        self.setFixedSize(1400, 860)

        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(16)

        # ---- Nav bar --------------------------------------------------
        self.nav_bar = NavBar()
        self.nav_bar.pageRequested.connect(self.switch_page)
        root_layout.addWidget(self.nav_bar)

        # ---- Pages ------------------------------------------------------
        self.stacked_widget = QStackedWidget()
        root_layout.addWidget(self.stacked_widget, stretch=1)

        self.upload_panel = UploadPanel()
        self.results_panel = ResultsPanel()
        self.about_panel = AboutPanel()

        self.stacked_widget.addWidget(self.upload_panel)   # index 0
        self.stacked_widget.addWidget(self.results_panel)  # index 1
        self.stacked_widget.addWidget(self.about_panel)    # index 2

        # Once an attack is generated on the Upload page, jump to Results.
        self.upload_panel.attackFinished.connect(self.show_results)

    def switch_page(self, page_index: int):
        self.stacked_widget.setCurrentIndex(page_index)
        self.nav_bar.set_active_page(page_index)

    def show_results(self, results):
        self.results_panel.load_results(results)
        self.switch_page(NavBar.PAGE_RESULTS)