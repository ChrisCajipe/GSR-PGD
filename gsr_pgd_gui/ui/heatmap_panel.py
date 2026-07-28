"""
heatmap_panel.py
-----------------
Displays LightShed and TruFor detector heatmaps for both the
standard-PGD and GSR-PGD adversarial images. Used inside the
"Heatmaps" tab of the Results page.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QGridLayout, QVBoxLayout, QLabel, QSizePolicy


class HeatmapTile(QWidget):
    """
    A single labeled heatmap cell.

    Note: the bordered/shaded "container" box wraps ONLY the image
    placeholder. The caption is deliberately placed outside of that
    container (as plain text below it) rather than inside the same
    bordered frame.
    """

    def __init__(self, caption: str):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # --- Image container (the only element with a border/background) ---
        self.imageLabel = QLabel("No Heatmap")
        self.imageLabel.setObjectName("ImagePlaceholder")
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setMinimumSize(200, 200)
        self.imageLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.imageLabel)

        # --- Caption, outside the image container ---
        caption_label = QLabel(caption)
        caption_label.setObjectName("TileCaption")
        caption_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(caption_label)


class HeatmapPanel(QWidget):
    """
    Grid of 4 heatmap placeholders:
        LightShed x (PGD, GSR-PGD)
        TruFor    x (PGD, GSR-PGD)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setSpacing(16)

        lightshed_pgd = HeatmapTile("LightShed Heatmap \u2013 PGD")
        lightshed_gsr = HeatmapTile("LightShed Heatmap \u2013 GSR-PGD")
        trufor_pgd = HeatmapTile("TruFor Heatmap \u2013 PGD")
        trufor_gsr = HeatmapTile("TruFor Heatmap \u2013 GSR-PGD")

        self.lightshedHeatmapPgdLabel = lightshed_pgd.imageLabel
        self.lightshedHeatmapGsrLabel = lightshed_gsr.imageLabel
        self.truforHeatmapPgdLabel = trufor_pgd.imageLabel
        self.truforHeatmapGsrLabel = trufor_gsr.imageLabel

        grid.addWidget(lightshed_pgd, 0, 0)
        grid.addWidget(lightshed_gsr, 0, 1)
        grid.addWidget(trufor_pgd, 1, 0)
        grid.addWidget(trufor_gsr, 1, 1)
