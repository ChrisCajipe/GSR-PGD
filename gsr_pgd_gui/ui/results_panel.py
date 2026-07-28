"""
results_panel.py
-----------------
"IMAGE & EVALUATION RESULTS" page.

Left side: a card with a row of toggle buttons (Adversarial / Original /
Perturbation / Heatmaps / Overall) that switch between different image
comparison layouts via a QStackedWidget.

Right side: MetricsPanel with General / LightShed / TruFor tables.

Backend hook points (see TODOs):
    - save_results()
"""
from PIL import Image

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton,
    QButtonGroup, QStackedWidget, QSizePolicy, QGridLayout, QFileDialog
)

from PySide6.QtGui import (
    QPainter,
    QPainterPath,
    QPixmap,
)

from ui.page_header import PageHeader
from ui.metrics_panel import MetricsPanel
from ui.heatmap_panel import HeatmapPanel
from backend.pixmap_util import tensor_to_pixmap
from backend.classifier import predict_tensor
from backend.metrics import (
    compute_psnr,
    compute_ssim,
)


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
        self.heatmapsToggleButton = self._make_toggle("Evaluation Output")
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
        self.heatmapPanel = HeatmapPanel()
        self.viewStack.addWidget(self.heatmapPanel)
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
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setSpacing(14)

        # ---------------- GSR-PGD ----------------
        grid.addWidget(self._caption_label("GSR-PGD"), 0, 0)

        self.overallGsrAdversarialLabel = make_image_placeholder(
            "Adversarial Image", (170, 170)
        )
        grid.addLayout(
            captioned(
                self.overallGsrAdversarialLabel,
                "Adversarial Image"
            ),
            1, 0
        )

        self.overallGsrPerturbationLabel = make_image_placeholder(
            "Perturbation", (170, 170)
        )
        grid.addLayout(
            captioned(
                self.overallGsrPerturbationLabel,
                "Adversarial Perturbation"
            ),
            1, 1
        )

        self.overallGsrLightShedLabel = make_image_placeholder(
            "LightShed", (170, 170)
        )
        grid.addLayout(
            captioned(
                self.overallGsrLightShedLabel,
                "LightShed Extracted Perturbation"
            ),
            1, 2
        )

        self.overallGsrTruForLabel = make_image_placeholder(
            "TruFor", (170, 170)
        )
        grid.addLayout(
            captioned(
                self.overallGsrTruForLabel,
                "TruFor Heatmap"
            ),
            1, 3
        )

        # ---------------- STANDARD PGD ----------------
        grid.addWidget(self._caption_label("STANDARD PGD"), 2, 0)

        self.overallPgdAdversarialLabel = make_image_placeholder(
            "Adversarial Image", (170, 170)
        )
        grid.addLayout(
            captioned(
                self.overallPgdAdversarialLabel,
                "Adversarial Image"
            ),
            3, 0
        )

        self.overallPgdPerturbationLabel = make_image_placeholder(
            "Perturbation", (170, 170)
        )
        grid.addLayout(
            captioned(
                self.overallPgdPerturbationLabel,
                "Adversarial Perturbation"
            ),
            3, 1
        )

        self.overallPgdLightShedLabel = make_image_placeholder(
            "LightShed", (170, 170)
        )
        grid.addLayout(
            captioned(
                self.overallPgdLightShedLabel,
                "LightShed Extracted Perturbation"
            ),
            3, 2
        )

        self.overallPgdTruForLabel = make_image_placeholder(
            "TruFor", (170, 170)
        )
        grid.addLayout(
            captioned(
                self.overallPgdTruForLabel,
                "TruFor Heatmap"
            ),
            3, 3
        )

        return widget

    def _caption_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("GroupCaption")
        return label

    # ------------------------------------------------------------------
    # Backend hook placeholders
    # ------------------------------------------------------------------
    def _path_to_pixmap(self, path):
        if path is None:
            return None

        pixmap = QPixmap(str(path))

        if pixmap.isNull():
            return None

        return pixmap
    
    def load_results(self, results):
        """
        Receive attack results from the backend and display them.
        """
        print("Results received!")
        self.results = results

        # update every view
        self.compare_results()
    
    def compare_results(self, view_index: int = None):

        if not hasattr(self, "results"):
            return

        # Tensor Images
            # ACTUAL IMAGE TENSORS
        pgd_pix = tensor_to_pixmap(self.results["pgd"])
        gsr_pix = tensor_to_pixmap(self.results["gsr"])
        orig_pix = tensor_to_pixmap(self.results["original"])

        # ----------------------------
        # ResNet-50 predictions
        # ----------------------------

        pgd_label, pgd_conf = predict_tensor(
            self.results["pgd"]
        )

        gsr_label, gsr_conf = predict_tensor(
            self.results["gsr"]
)


        self.metricsPanel.predictionPgdLabel.setText(
            f"{pgd_label}\n({pgd_conf:.2%})"
        )

        self.metricsPanel.predictionGsrLabel.setText(
            f"{gsr_label}\n({gsr_conf:.2%})"
        )

        # ----------------------------
        # PSNR / SSIM
        # ----------------------------

        original = self.results["original"]
        pgd = self.results["pgd"]
        gsr = self.results["gsr"]


        pgd_psnr = compute_psnr(
            original,
            pgd
        )

        gsr_psnr = compute_psnr(
            original,
            gsr
        )


        pgd_ssim = compute_ssim(
            original,
            pgd
        )

        gsr_ssim = compute_ssim(
            original,
            gsr
        )


        self.metricsPanel.psnrPgdLabel.setText(
            f"{pgd_psnr:.2f} dB"
        )

        self.metricsPanel.psnrGsrLabel.setText(
            f"{gsr_psnr:.2f} dB"
        )


        self.metricsPanel.ssimPgdLabel.setText(
            f"{pgd_ssim:.4f}"
        )

        self.metricsPanel.ssimGsrLabel.setText(
            f"{gsr_ssim:.4f}"
        )

            # PERTURBATION TENSORS
        gsr_pert = tensor_to_pixmap(self.results["gsr_perturbation"])
        pgd_pert = tensor_to_pixmap(self.results["pgd_perturbation"])

        # Main Views
        self._set_pixmap(self.gsrImageLabel, gsr_pix)
        self._set_pixmap(self.pgdImageLabel, pgd_pix)
        self._set_pixmap(self.originalResultImageLabel, orig_pix)

        # Perturbation
        self._set_pixmap(self.gsrPerturbationLabel, gsr_pert)
        self._set_pixmap(self.pgdPerturbationLabel, pgd_pert)

        # LIGHTSHED EXTRACTED PERTURBATIONS
        gsr_ls = self.results.get("gsr_lightshed")
        pgd_ls = self.results.get("pgd_lightshed")

        if gsr_ls:
            lightshed_data = gsr_ls.get("extracted_perturbation")

            if isinstance(lightshed_data, dict):
                path = lightshed_data.get("extracted_perturbation")
            else:
                path = lightshed_data

            pix = self._path_to_pixmap(path)

            if pix:
                self._set_pixmap(self.heatmapPanel.lightshedHeatmapGsrLabel,pix)
                self._set_pixmap(self.overallGsrLightShedLabel,pix)

        if pgd_ls:
            lightshed_data = pgd_ls.get("extracted_perturbation")

            if isinstance(lightshed_data, dict):
                path = lightshed_data.get("extracted_perturbation")
            else:
                path = lightshed_data

            pix = self._path_to_pixmap(path)

            if pix:
                self._set_pixmap(self.heatmapPanel.lightshedHeatmapPgdLabel, pix)
                self._set_pixmap(self.overallPgdLightShedLabel, pix)

        # TRUFOR HEATMAPS AYOKO NAAAAAAAA
        gsr_tf = self.results.get("gsr_trufor")
        pgd_tf = self.results.get("pgd_trufor")

        if gsr_tf:
            pix = self._path_to_pixmap(gsr_tf.get("heatmap"))
            if pix:
                self._set_pixmap(self.heatmapPanel.truforHeatmapGsrLabel, pix)
                self._set_pixmap(self.overallGsrTruForLabel, pix)
        if pgd_tf:
            pix = self._path_to_pixmap(pgd_tf.get("heatmap"))
            if pix:
                self._set_pixmap(self.heatmapPanel.truforHeatmapPgdLabel, pix)
                self._set_pixmap(self.overallPgdTruForLabel, pix)

        # -----------------------------
        # Detection Metrics
        # -----------------------------

        # LightShed
        pgd_ls = self.results.get("pgd_lightshed")
        gsr_ls = self.results.get("gsr_lightshed")

        if pgd_ls:
            lightshed_detected = pgd_ls["extracted_perturbation"]["detected"]

            self.metricsPanel.set_detection_status(
                self.metricsPanel.lightshedStatusPgdLabel,
                lightshed_detected
            )


        if gsr_ls:
            lightshed_detected = gsr_ls["extracted_perturbation"]["detected"]

            self.metricsPanel.set_detection_status(
                self.metricsPanel.lightshedStatusGsrLabel,
                lightshed_detected
            )


        # TruFor
        pgd_tf = self.results.get("pgd_trufor")
        gsr_tf = self.results.get("gsr_trufor")


        if pgd_tf:
            trufor_detected = pgd_tf["prediction"]

            self.metricsPanel.set_detection_status(
                self.metricsPanel.truforStatusPgdLabel,
                trufor_detected
            )


        if gsr_tf:
            trufor_detected = gsr_tf["prediction"]

            self.metricsPanel.set_detection_status(
                self.metricsPanel.truforStatusGsrLabel,
                trufor_detected
            )

        # Overall
        self._set_pixmap(self.overallGsrAdversarialLabel, gsr_pix)
        self._set_pixmap(self.overallPgdAdversarialLabel, pgd_pix)
        self._set_pixmap(self.overallGsrPerturbationLabel, gsr_pert)
        self._set_pixmap(self.overallPgdPerturbationLabel, pgd_pert)

    def _set_pixmap(self, label, pixmap):
        # Scale while preserving aspect ratio
        scaled = pixmap.scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        # Create transparent pixmap
        rounded = QPixmap(scaled.size())
        rounded.fill(Qt.transparent)

        # Draw rounded image
        painter = QPainter(rounded)
        painter.setRenderHint(QPainter.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(
            rounded.rect(),
            18,   # corner radius
            18
        )

        painter.setClipPath(path)
        painter.drawPixmap(0, 0, scaled)
        painter.end()

        label.setPixmap(rounded)

        # Remove placeholder appearance
        label.setText("")
        label.setStyleSheet("""
            QLabel {
                background: transparent;
                border: 2px solid #D8DCEB;
                border-radius: 18px;
            }
        """)
