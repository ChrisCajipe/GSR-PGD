"""
about_panel.py
---------------
Static "ABOUT" page: project description, usage instructions and
author cards. No backend interaction is required here.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QLabel, QSizePolicy
)

from ui.page_header import PageHeader

PLACEHOLDER_DESCRIPTION = (
    "GSR-PGD (Gaussian Spectral-Regularized Projected Gradient Descent) is an "
    "adversarial attack generation method designed to produce imperceptible, "
    "spectrally-regularized perturbations that evade forensic detectors such "
    "as LightShed and TruFor while preserving high visual fidelity to the "
    "original image."
)

INSTRUCTIONS = [
    "Upload a 512x512 image that belongs to a supported ImageNet class.",
    "Select a target class and generate both PGD and GSR-PGD adversarial images.",
    "Review the results, evaluation metrics and detector outputs on the Results page.",
]

AUTHORS = [
    ("ALON, NOEMI MANLISES", "Contributor \u2013 research & development."),
    ("VICU\u00d1A, SOFIA ALEZANDRA GROSPE", "Contributor \u2013 research & development."),
    ("CAJIPE, CHRIS CAGUIN", "Contributor \u2013 research & development."),
    ("PACAON, MARK REDEN SUCANO", "Contributor \u2013 research & development."),
]


class AboutPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(20)

        header = PageHeader(icon_text="\U0001F4C4", title="ABOUT GSR-PGD", blue=False)
        root.addWidget(header)

        body = QHBoxLayout()
        body.setSpacing(20)
        root.addLayout(body, stretch=1)

        body.addWidget(self._build_description_card(), stretch=1)
        body.addWidget(self._build_authors_card(), stretch=2)

    # ------------------------------------------------------------------
    def _build_description_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        desc_pill = QLabel("DESCRIPTION")
        desc_pill.setObjectName("SectionPill")
        layout.addWidget(desc_pill)

        desc_text = QLabel(PLACEHOLDER_DESCRIPTION)
        desc_text.setWordWrap(True)
        desc_text.setStyleSheet("color: #1f2440; font-size: 13px;")
        layout.addWidget(desc_text)

        instructions_pill = QLabel("INSTRUCTIONS")
        instructions_pill.setObjectName("SectionPill")
        layout.addWidget(instructions_pill)

        for idx, text in enumerate(INSTRUCTIONS, start=1):
            row = QHBoxLayout()
            number = QLabel(str(idx))
            number.setFixedSize(26, 26)
            number.setAlignment(Qt.AlignCenter)
            number.setStyleSheet(
                "background: #c0392b; color: white; border-radius: 13px; font-weight: 800;"
            )
            desc = QLabel(text)
            desc.setWordWrap(True)
            desc.setStyleSheet("color: #1f2440; font-size: 12px;")
            row.addWidget(number)
            row.addWidget(desc, stretch=1)
            layout.addLayout(row)

        layout.addStretch(1)
        return card

    def _build_authors_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        pill = QLabel("MEET THE AUTHORS")
        pill.setObjectName("SectionPill")
        layout.addWidget(pill)

        grid = QGridLayout()
        grid.setSpacing(16)
        layout.addLayout(grid)

        for i, (name, bio) in enumerate(AUTHORS):
            row, col = divmod(i, 2)
            grid.addLayout(self._author_tile(name, bio), row, col)

        layout.addStretch(1)
        return card

    def _author_tile(self, name: str, bio: str) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(12)

        photo = QLabel("Photo")
        photo.setObjectName("ImagePlaceholder")
        photo.setFixedSize(90, 90)
        photo.setAlignment(Qt.AlignCenter)

        text_col = QVBoxLayout()
        name_label = QLabel(name)
        name_label.setWordWrap(True)
        name_label.setStyleSheet("font-weight: 800; color: #1f2440; font-size: 12px;")
        bio_label = QLabel(bio)
        bio_label.setWordWrap(True)
        bio_label.setStyleSheet("color: #5b6180; font-size: 11px;")
        text_col.addWidget(name_label)
        text_col.addWidget(bio_label)
        text_col.addStretch(1)

        layout.addWidget(photo)
        layout.addLayout(text_col, stretch=1)
        return layout
