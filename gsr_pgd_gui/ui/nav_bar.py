"""
nav_bar.py
----------
Top navigation bar: application title on the left, three page
navigation buttons (Upload / Results / About) on the right.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QButtonGroup


class NavBar(QWidget):
    """Emits `pageRequested(int)` whenever the user selects a page."""

    pageRequested = Signal(int)

    PAGE_UPLOAD = 0
    PAGE_RESULTS = 1
    PAGE_ABOUT = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("NavBar")
        self.setFixedHeight(64)
        # A plain QWidget does not paint its QSS background/border by
        # default (only QFrame/QLabel/etc. do). This attribute tells Qt
        # to route painting through the stylesheet so the gradient
        # background actually shows up.
        self.setAttribute(Qt.WA_StyledBackground, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(28, 10, 28, 10)

        # --- Title -----------------------------------------------------
        title = QLabel("GSR-PGD")
        title.setObjectName("AppTitle")

        layout.addWidget(title)
        layout.addStretch(1)

        # --- Nav buttons -------------------------------------------------
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)

        self.uploadNavButton = self._make_button("UPLOAD", self.PAGE_UPLOAD)
        self.resultsNavButton = self._make_button("RESULTS", self.PAGE_RESULTS)
        self.aboutNavButton = self._make_button("ABOUT", self.PAGE_ABOUT)

        for btn in (self.uploadNavButton, self.resultsNavButton, self.aboutNavButton):
            layout.addWidget(btn)

        self.uploadNavButton.setChecked(True)

    def _make_button(self, text: str, page_index: int) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("NavButton")
        btn.setCheckable(True)
        btn.setCursor(Qt.PointingHandCursor)
        self.button_group.addButton(btn)
        btn.clicked.connect(lambda: self.pageRequested.emit(page_index))
        return btn

    def set_active_page(self, page_index: int):
        """Keep the nav buttons in sync when the page changes programmatically."""
        mapping = {
            self.PAGE_UPLOAD: self.uploadNavButton,
            self.PAGE_RESULTS: self.resultsNavButton,
            self.PAGE_ABOUT: self.aboutNavButton,
        }
        button = mapping.get(page_index)
        if button:
            button.setChecked(True)
