"""
page_header.py
--------------
A small reusable widget used at the top of every page: a colored
circular icon badge, a big title, an optional subtitle, and a
decorative grid of dots on the right (purely cosmetic, mirrors the
mock-ups).
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QGridLayout


class DotGrid(QWidget):
    """Purely decorative grid of small circle outlines (top-right flourish)."""

    def __init__(self, rows: int = 2, cols: int = 4, parent=None):
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(0, 0, 0, 0)
        for r in range(rows):
            for c in range(cols):
                dot = QLabel()
                dot.setFixedSize(10, 10)
                dot.setStyleSheet(
                    "border: 1.5px solid #d99a6c; border-radius: 5px; background: transparent;"
                )
                layout.addWidget(dot, r, c)


class PageHeader(QWidget):
    """
    Icon badge + Title (+ optional Subtitle) on the left, decorative
    dot-grid on the right. Used consistently across Upload / Results /
    About pages.
    """

    def __init__(self, icon_text: str, title: str, subtitle: str = "",
                 blue: bool = False, parent=None):
        super().__init__(parent)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Icon badge
        badge = QLabel(icon_text)
        badge.setObjectName("IconBadgeBlue" if blue else "IconBadge")
        badge.setFixedSize(40, 40)
        badge.setAlignment(Qt.AlignCenter)

        # Title block
        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("PageTitleBlue" if blue else "PageTitle")
        title_block.addWidget(title_label)

        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setObjectName("PageSubtitle")
            title_block.addWidget(subtitle_label)

        outer.addWidget(badge)
        outer.addSpacing(12)
        outer.addLayout(title_block)
        outer.addStretch(1)
        outer.addWidget(DotGrid())
