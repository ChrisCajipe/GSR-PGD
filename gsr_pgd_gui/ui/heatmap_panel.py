"""
heatmap_panel.py
-----------------
Displays LightShed and TruFor detector heatmaps for both the
standard-PGD and GSR-PGD adversarial images. Used inside the
"Heatmaps" tab of the Results page.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QGridLayout, QVBoxLayout, QLabel, QSizePolicy


class HeatmapTile(QVBoxLayout):
    """A single labeled image placeholder used as a heatmap cell."""

    def __init__(self, caption: str):
        super().__init__()
        self.setSpacing(6)

        self.imageLabel = QLabel("No Heatmap")
        self.imageLabel.setObjectName("ImagePlaceholder")
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setMinimumSize(220, 220)
        self.imageLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        caption_label = QLabel(caption)
        caption_label.setObjectName("TileCaption")
        caption_label.setAlignment(Qt.AlignCenter)

        self.addWidget(self.imageLabel)
        self.addWidget(caption_label)


class HeatmapPanel(QWidget):
    """
    Grid of 4 heatmap placeholders:
        LightShed x (PGD, GSR-PGD)
        TruFor    x (PGD, GSR-PGD)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setSpacing(20)

        lightshed_pgd = HeatmapTile("LightShed Heatmap \u2013 PGD")
        lightshed_gsr = HeatmapTile("LightShed Heatmap \u2013 GSR-PGD")
        trufor_pgd = HeatmapTile("TruFor Heatmap \u2013 PGD")
        trufor_gsr = HeatmapTile("TruFor Heatmap \u2013 GSR-PGD")

        # Expose as attributes for backend access.
        self.lightshedHeatmapPgdLabel = lightshed_pgd.imageLabel
        self.lightshedHeatmapGsrLabel = lightshed_gsr.imageLabel
        self.truforHeatmapPgdLabel = trufor_pgd.imageLabel
        self.truforHeatmapGsrLabel = trufor_gsr.imageLabel

        grid.addLayout(lightshed_pgd, 0, 0)
        grid.addLayout(lightshed_gsr, 0, 1)
        grid.addLayout(trufor_pgd, 1, 0)
        grid.addLayout(trufor_gsr, 1, 1)
