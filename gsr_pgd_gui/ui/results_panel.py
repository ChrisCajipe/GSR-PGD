"""
results_panel.py
-----------------
"IMAGE & EVALUATION RESULTS" page.

Left side: a card with a row of toggle buttons (Adversarial / Original /
Perturbation / Heatmaps / Overall) that switch between different image
comparison layouts via a QStackedWidget.

Right side: MetricsPanel with General / LightShed / TruFor tables.

Backend hook points (see TODOs):
    - compare_results()
    - save_results()
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton,
    QButtonGroup, QStackedWidget, QSizePolicy, QGridLayout, QFileDialog
)

from ui.page_header import PageHeader
from ui.metrics_panel import MetricsPanel
from ui.heatmap_panel import HeatmapPanel


def make_image_placeholder(text: str, min_size=(220, 220)) -> QLabel:
    label = QLabel(text)
    label.setObjectName("ImagePlaceholder")
    label.setAlignment(Qt.AlignCenter)
    label.setMinimumSize(*min_size)
    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return label


def captioned(image_label: QLabel, caption_text: str) -> QVBoxLayout:
    layout = QVBoxLayout()
    layout.setSpacing(6)
    layout.addWidget(image_label)
    caption = QLabel(caption_text)
    caption.setObjectName("TileCaption")
    caption.setAlignment(Qt.AlignCenter)
    layout.addWidget(caption)
    return layout


class ResultsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(20)

        # ---- Header -----------------------------------------------------
        header = PageHeader(icon_text="\U0001F4C4", title="IMAGE & EVALUATION RESULTS")
        root.addWidget(header)

        # ---- Body: images card (left) + metrics card (right) ------------
        body = QHBoxLayout()
        body.setSpacing(20)
        root.addLayout(body, stretch=1)

        body.addWidget(self._build_images_card(), stretch=3)

        self.metricsPanel = MetricsPanel()
        body.addWidget(self.metricsPanel, stretch=2)

        # ---- Footer: save results -----------------------------------------
        footer = QHBoxLayout()
        footer.addStretch(1)
        self.saveResultsButton = QPushButton("Save Results")
        self.saveResultsButton.setObjectName("PrimaryButton")
        self.saveResultsButton.setCursor(Qt.PointingHandCursor)
        self.saveResultsButton.clicked.connect(self.save_results)
        footer.addWidget(self.saveResultsButton)
        root.addLayout(footer)

    # ------------------------------------------------------------------
    def _build_images_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Toggle button bar
        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(10)
        self.toggleButtonGroup = QButtonGroup(self)
        self.toggleButtonGroup.setExclusive(True)

        self.adversarialToggleButton = self._make_toggle("Adversarial")
        self.originalToggleButton = self._make_toggle("Original")
        self.perturbationToggleButton = self._make_toggle("Perturbation")
        self.heatmapsToggleButton = self._make_toggle("Heatmaps")
        self.overallToggleButton = self._make_toggle("Overall")

        for btn in (
            self.adversarialToggleButton, self.originalToggleButton,
            self.perturbationToggleButton, self.heatmapsToggleButton,
            self.overallToggleButton,
        ):
            toggle_row.addWidget(btn)
        toggle_row.addStretch(1)
        layout.addLayout(toggle_row)

        # Stacked content
        self.viewStack = QStackedWidget()
        layout.addWidget(self.viewStack, stretch=1)

        self.viewStack.addWidget(self._build_adversarial_view())
        self.viewStack.addWidget(self._build_original_view())
        self.viewStack.addWidget(self._build_perturbation_view())
        self.viewStack.addWidget(HeatmapPanel())
        self.viewStack.addWidget(self._build_overall_view())

        self.adversarialToggleButton.clicked.connect(lambda: self._switch_view(0))
        self.originalToggleButton.clicked.connect(lambda: self._switch_view(1))
        self.perturbationToggleButton.clicked.connect(lambda: self._switch_view(2))
        self.heatmapsToggleButton.clicked.connect(lambda: self._switch_view(3))
        self.overallToggleButton.clicked.connect(lambda: self._switch_view(4))

        self.adversarialToggleButton.setChecked(True)

        return card

    def _make_toggle(self, text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("OutlineButton")
        btn.setCheckable(True)
        btn.setCursor(Qt.PointingHandCursor)

        btn.setMinimumHeight(36)
        btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.toggleButtonGroup.addButton(btn)
        return btn

    def _switch_view(self, index: int):
        self.viewStack.setCurrentIndex(index)
        # Notify backend so it can lazily populate the newly shown view.
        self.compare_results(index)

    # ------------------------------------------------------------------
    # Individual view builders
    # ------------------------------------------------------------------
    def _build_adversarial_view(self) -> QWidget:
        """Side-by-side: GSR-PGD adversarial vs Standard PGD adversarial."""
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setSpacing(20)

        gsr_col = QVBoxLayout()
        gsr_title = QLabel("GSR-PGD")
        gsr_title.setObjectName("GroupCaption")
        self.gsrImageLabel = make_image_placeholder("Adversarial Image (GSR-PGD)", (300, 300))
        gsr_col.addWidget(gsr_title)
        gsr_col.addWidget(self.gsrImageLabel, stretch=1)

        pgd_col = QVBoxLayout()
        pgd_title = QLabel("STANDARD")
        pgd_title.setObjectName("GroupCaption")
        self.pgdImageLabel = make_image_placeholder("Adversarial Image (PGD)", (300, 300))
        pgd_col.addWidget(pgd_title)
        pgd_col.addWidget(self.pgdImageLabel, stretch=1)

        row.addLayout(gsr_col, stretch=1)
        row.addLayout(pgd_col, stretch=1)
        return widget

    def _build_original_view(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignCenter)
        title = QLabel("ORIGINAL")
        title.setObjectName("GroupCaption")
        title.setAlignment(Qt.AlignCenter)
        self.originalResultImageLabel = make_image_placeholder("Original Image", (340, 340))
        layout.addWidget(title)
        layout.addWidget(self.originalResultImageLabel)
        return widget

    def _build_perturbation_view(self) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setSpacing(20)

        gsr_col = QVBoxLayout()
        gsr_title = QLabel("GSR-PGD")
        gsr_title.setObjectName("GroupCaption")
        self.gsrPerturbationLabel = make_image_placeholder("Perturbation (GSR-PGD)", (300, 300))
        gsr_col.addWidget(gsr_title)
        gsr_col.addWidget(self.gsrPerturbationLabel, stretch=1)

        pgd_col = QVBoxLayout()
        pgd_title = QLabel("STANDARD")
        pgd_title.setObjectName("GroupCaption")
        self.pgdPerturbationLabel = make_image_placeholder("Perturbation (PGD)", (300, 300))
        pgd_col.addWidget(pgd_title)
        pgd_col.addWidget(self.pgdPerturbationLabel, stretch=1)

        row.addLayout(gsr_col, stretch=1)
        row.addLayout(pgd_col, stretch=1)
        return widget

    def _build_overall_view(self) -> QWidget:
        """Full grid overview mirroring the reference mock-up."""
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setSpacing(14)

        # Column / row captions
        grid.addWidget(self._caption_label("ORIGINAL"), 0, 0)
        grid.addWidget(self._caption_label("GSR-PGD"), 0, 1)
        grid.addWidget(self._caption_label(""), 0, 2)

        self.overallOriginalImageLabel = make_image_placeholder("Original", (180, 180))
        grid.addLayout(captioned(self.overallOriginalImageLabel, ""), 1, 0)

        self.overallGsrAdversarialLabel = make_image_placeholder("Adversarial Image", (180, 180))
        grid.addLayout(captioned(self.overallGsrAdversarialLabel, "Adversarial Image"), 1, 1)

        self.overallGsrPerturbationLabel = make_image_placeholder("Adversarial Perturbation", (180, 180))
        grid.addLayout(captioned(self.overallGsrPerturbationLabel, "Adversarial Perturbation"), 1, 2)

        grid.addWidget(self._caption_label("STANDARD"), 2, 0)

        self.overallPgdAdversarialLabel = make_image_placeholder("Adversarial Image", (180, 180))
        grid.addLayout(captioned(self.overallPgdAdversarialLabel, "Adversarial Image"), 3, 0)

        self.overallPgdPerturbationLabel = make_image_placeholder("Adversarial Perturbation", (180, 180))
        grid.addLayout(captioned(self.overallPgdPerturbationLabel, "Adversarial Perturbation"), 3, 1)

        self.overallLightshedExtractedLabel = make_image_placeholder("LightShed Extracted Perturbation", (180, 180))
        grid.addLayout(captioned(self.overallLightshedExtractedLabel, "LightShed Extracted Perturbation"), 3, 2)

        return widget

    def _caption_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("GroupCaption")
        return label

    # ------------------------------------------------------------------
    # Backend hook placeholders
    # ------------------------------------------------------------------
    def compare_results(self, view_index: int = None):
        """
        TODO: Connect to backend `compare_results()`.
        Should:
            1. Pull latest original / PGD / GSR-PGD images and perturbations.
            2. Populate the QLabel placeholders (originalResultImageLabel,
               pgdImageLabel, gsrImageLabel, *PerturbationLabel, etc.)
               with actual QPixmap data.
            3. Update MetricsPanel labels (PSNR, SSIM, prediction, status).
            4. Update HeatmapPanel labels with LightShed/TruFor heatmaps.
        `view_index` indicates which toggle tab is now visible, useful for
        lazy-loading only what's needed.
        """
        pass

    def save_results(self):
        """
        TODO: Connect to backend `save_results()`.
        Should export the currently displayed images/metrics to disk.
        A QFileDialog.getExistingDirectory() or getSaveFileName() call is
        a reasonable starting point.
        """
        # Example scaffold (left inactive until backend is wired up):
        # directory = QFileDialog.getExistingDirectory(self, "Save Results To")
        pass
