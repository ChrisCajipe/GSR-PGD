"""
upload_panel.py
----------------
"UPLOAD" page: lets the user pick a 512x512 ImageNet image, preview
it, choose a target class for the attack, verify it with ResNet-50
and finally trigger adversarial generation.
"""
import os
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton,
    QComboBox, QSizePolicy, QFileDialog
)
from PIL.ImageQt import ImageQt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import Qt, Signal
from PySide6.QtCore import QThread


from ui.page_header import PageHeader

from backend.image_loader import load_image
from backend.classifier import predict_image
from backend.ui_config import TARGET_CLASSES
from backend.attack_worker import AttackWorker

TARGET_CLASS_NAMES = list(TARGET_CLASSES.keys())


class UploadPanel(QWidget):
    attackFinished = Signal(dict)
    def __init__(self, parent=None):
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(20)

        # ---- Header ---------------------------------------------------
        header = PageHeader(
            icon_text="\U0001F4C4",
            title="UPLOAD IMAGE",
            subtitle="Please upload a 512x512 image belonging to an ImageNet class.",
            blue=True,
        )
        root.addWidget(header)

        # ---- Body: dropzone (left) + preview/target (right) -----------
        body = QHBoxLayout()
        body.setSpacing(20)
        root.addLayout(body, stretch=1)

        body.addWidget(self._build_dropzone(), stretch=1)
        body.addWidget(self._build_preview_and_target(), stretch=2)

        # ---- Footer: generate action ------------------------------------
        # Verify Image + predicted-class now live inside the Target Class
        # column (see _build_preview_and_target), so the footer only holds
        # the final "Generate Adversarial Image" call-to-action.
        self.generateButton = QPushButton("Generate Adversarial Image")
        self.generateButton.setObjectName("PrimaryButton")
        self.generateButton.setCursor(Qt.PointingHandCursor)
        self.generateButton.clicked.connect(self.generate_attack)
        root.addWidget(self.generateButton)
        self._build_loading_overlay()

    # ------------------------------------------------------------------
    # UI builders
    # ------------------------------------------------------------------
    def _build_dropzone(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("DropZone")
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(frame)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(14)

        cloud_icon = QLabel("\u2601")
        cloud_icon.setAlignment(Qt.AlignCenter)
        cloud_icon.setStyleSheet("font-size: 56px; color: #4a54e1;")

        drop_text = QLabel("Drag & drop image here")
        drop_text.setAlignment(Qt.AlignCenter)
        drop_text.setStyleSheet("font-size: 15px; font-weight: 600; color: #1f2440;")

        or_text = QLabel("or")
        or_text.setAlignment(Qt.AlignCenter)
        or_text.setStyleSheet("color: #5b6180;")

        self.uploadButton = QPushButton("Browse Files")
        self.uploadButton.setObjectName("PrimaryButton")
        self.uploadButton.setCursor(Qt.PointingHandCursor)
        self.uploadButton.setFixedWidth(220)
        self.uploadButton.clicked.connect(lambda checked=False: self.upload_image())

        layout.addWidget(cloud_icon)
        layout.addWidget(drop_text)
        layout.addWidget(or_text)
        layout.addWidget(self.uploadButton, alignment=Qt.AlignCenter)

        # Allow drag-and-drop of image files directly onto this frame.
        frame.setAcceptDrops(True)
        frame.dragEnterEvent = self._drag_enter_event
        frame.dropEvent = self._drop_event

        return frame

    def _build_preview_and_target(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("Card")
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(16)

        row = QHBoxLayout()
        row.setSpacing(20)
        outer.addLayout(row, stretch=1)

        # --- Preview column ---
        preview_col = QVBoxLayout()
        preview_caption = QLabel("\u2713  IMAGE PREVIEW")
        preview_caption.setObjectName("GroupCaption")
        preview_caption.setStyleSheet("color: #1d3195;")
        preview_col.addWidget(preview_caption)

        self.originalImageLabel = QLabel("No Image")
        self.originalImageLabel.setObjectName("ImagePlaceholder")
        self.originalImageLabel.setAlignment(Qt.AlignCenter)
        self.originalImageLabel.setMinimumSize(260, 260)
        self.originalImageLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_col.addWidget(self.originalImageLabel, stretch=1)

        meta_row = QHBoxLayout()
        self.fileNameLabel = QLabel("File Name: --")
        self.fileNameLabel.setStyleSheet("font-size: 11px; font-weight: 700; color: #616a96;")
        self.fileSizeLabel = QLabel("-- MB")
        self.fileSizeLabel.setStyleSheet("font-size: 11px; color: #616a96;")
        meta_row.addWidget(self.fileNameLabel)
        meta_row.addStretch(1)
        meta_row.addWidget(self.fileSizeLabel)
        preview_col.addLayout(meta_row)

        row.addLayout(preview_col, stretch=3)

        # --- Target class column ---
        target_col = QVBoxLayout()
        target_col.setSpacing(10)
        target_caption = QLabel("TARGET CLASS")
        target_caption.setObjectName("GroupCaption")
        target_caption.setStyleSheet("color: #1d3195;")
        target_col.addWidget(target_caption)

        # Real dropdown menu (click to expand) instead of an always-open list.
        self.targetClassCombo = QComboBox()
        self.targetClassCombo.addItems(TARGET_CLASS_NAMES)
        self.targetClassCombo.setCurrentText("Persian Cat")
        self.targetClassCombo.setCursor(Qt.PointingHandCursor)
        target_col.addWidget(self.targetClassCombo)

        target_col.addSpacing(8)
        target_col.addWidget(self._divider())
        target_col.addSpacing(8)

        # The space freed up by removing the expanded list is now used for
        # the "Verify Image" action and the resulting predicted class.
        verify_caption = QLabel("VERIFICATION")
        verify_caption.setObjectName("GroupCaption")
        verify_caption.setStyleSheet("color: #1d3195;")
        target_col.addWidget(verify_caption)

        self.originalPredictionLabel = QLabel("Waiting for image...")
        self.originalPredictionLabel.setAlignment(Qt.AlignCenter)
        self.originalPredictionLabel.setStyleSheet("""
            font-size: 20px;
            font-weight: 700;
            color: #616a96;
            padding: 10px;
        """)
        target_col.addWidget(self.originalPredictionLabel)

        disclaimer = QLabel(
            "If the predicted label does not match your uploaded image,\n"
            "please upload a different ImageNet image before proceeding."
        )
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet("""
            font-size: 11px;
            color: #616a96;
        """)
        target_col.addWidget(disclaimer)

        prediction_col = QVBoxLayout()
        prediction_col.setSpacing(2)

        # Push everything above upward and let this stretch absorb any
        # remaining vertical space so the column fills the card evenly.
        target_col.addStretch(1)

        row.addLayout(target_col, stretch=2)

        return frame

    def _divider(self):
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet("""
            QFrame {
                background-color: #D9D9D9;
                border: 0;
            }
        """)
        return line

    def _build_loading_overlay(self):
        self.loadingOverlay = QFrame(self)

        self.loadingOverlay.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 245);
            }
        """)

        overlay_layout = QVBoxLayout(self.loadingOverlay)
        overlay_layout.setAlignment(Qt.AlignCenter)

        card = QFrame()
        card.setFixedWidth(500)

        card.setStyleSheet("""
            QFrame {
                background: #f6f6f6;
                border: 1px solid #616a96;
                border-radius: 16px;
            }
        """)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(40, 35, 40, 35)
        card_layout.setSpacing(18)

        title = QLabel("PROCESSING IMAGE")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 22px;
            font-weight: 900;
            color: #1d3195;
            border: none;
        """)

        self.loadingStatusLabel = QLabel("Preparing...")
        self.loadingStatusLabel.setAlignment(Qt.AlignCenter)
        self.loadingStatusLabel.setStyleSheet("""
            font-size: 13px;
            color: #616a96;
            border: none;
        """)

        card_layout.addWidget(title)
        card_layout.addWidget(self.loadingStatusLabel)
        card_layout.addSpacing(10)

        self.loadingStageLabels = {}

        stages = [
            ("adversarial_generation", "Adversarial Generation"),
            ("resnet_classification", "ResNet-50 Classification"),
            ("lightshed_simulation", "LightShed Simulation"),
            ("trufor_simulation", "TruFor Simulation"),
        ]

        for key, text in stages:

            label = QLabel(f"○   {text}")

            label.setStyleSheet("""
                font-size: 14px;
                font-weight: 700;
                color: #9a9db0;
                padding: 8px;
                border: none;
            """)

            self.loadingStageLabels[key] = label
            card_layout.addWidget(label)

        overlay_layout.addWidget(card)

        self.loadingOverlay.hide()

    # ------------------------------------------------------------------
    # Drag & drop plumbing (UI only -- delegates to upload_image)
    # ------------------------------------------------------------------
    def _drag_enter_event(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def _drop_event(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.upload_image(file_path)

    # ------------------------------------------------------------------
    # Backend hook placeholders
    # ------------------------------------------------------------------

    def _display_image(self, image):
        qt_image = ImageQt(image)
        pixmap = QPixmap.fromImage(qt_image)

        self.originalImageLabel.setPixmap(
            pixmap.scaled(
                self.originalImageLabel.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def upload_image(self, file_path=None):
        if isinstance(file_path, bool):
            file_path = None

        if file_path is None:
            dialog = QFileDialog(self)
            dialog.setFileMode(QFileDialog.ExistingFile)
            dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
            dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)

            if dialog.exec():
                file_path = dialog.selectedFiles()[0]
            else:
                return

        # Store the uploaded image information
        self.imagePath = file_path
        loaded = load_image(file_path)

        self.uploadedImage = loaded["pil"]
        self.uploadedTensor = loaded["tensor"]

        self._display_image(self.uploadedImage)
        self.verify_image()

        self.fileNameLabel.setText(
            f"File Name: {os.path.basename(file_path)}"
        )

        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        self.fileSizeLabel.setText(f"{size_mb:.2f} MB")

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if hasattr(self, "uploadedImage"):
            self._display_image(self.uploadedImage)

        if hasattr(self, "loadingOverlay"):
            self.loadingOverlay.setGeometry(self.rect())

    def verify_image(self):

        self.originalPredictionLabel.setText("Verifying...")
        if not hasattr(self, "uploadedImage"):
            QMessageBox.warning(
                self,
                "No Image",
                "Please upload an image first."
            )
            return

        class_name, confidence = predict_image(self.uploadedImage)

        self.originalPredictionLabel.setText(
            f"{class_name}\n({confidence*100:.2f}%)"
        )

    def _format_time(self, elapsed_ms):
        if elapsed_ms >= 1000:
            return f"{elapsed_ms / 1000:.2f} s"

        return f"{elapsed_ms:.2f} ms"

    def update_loading_stage(self, current_stage):

        stages = [
            "adversarial_generation",
            "resnet_classification",
            "lightshed_simulation",
            "trufor_simulation",
        ]

        stage_names = {
            "adversarial_generation": "Generating adversarial images...",
            "resnet_classification": "Classifying with ResNet-50...",
            "lightshed_simulation": "Running LightShed simulation...",
            "trufor_simulation": "Running TruFor simulation...",
        }

        display_names = {
            "adversarial_generation": "Adversarial Generation",
            "resnet_classification": "ResNet-50 Classification",
            "lightshed_simulation": "LightShed Simulation",
            "trufor_simulation": "TruFor Simulation",
        }

        current_index = stages.index(current_stage)

        self.loadingStatusLabel.setText(
            stage_names[current_stage]
        )

        for index, stage in enumerate(stages):

            label = self.loadingStageLabels[stage]
            name = display_names[stage]

            # Previous stages = completed
            if index < current_index:

                timings = self.stageElapsed.get(stage, {})

                pgd_time = timings.get("pgd")
                gsr_time = timings.get("gsr")

                if pgd_time is not None and gsr_time is not None:

                    label.setText(
                        f"✓   {name}\n"
                        f"     PGD: {self._format_time(pgd_time)}   |   "
                        f"GSR-PGD: {self._format_time(gsr_time)}"
                    )

                else:
                    label.setText(f"✓   {name}")

                label.setStyleSheet("""
                    font-size: 14px;
                    font-weight: 700;
                    color: #2e8b57;
                    padding: 8px;
                    border: none;
                """)

            # Current stage
            elif index == current_index:

                label.setText(f"●   {name}")

                label.setStyleSheet("""
                    font-size: 14px;
                    font-weight: 800;
                    color: #1d3195;
                    padding: 8px;
                    border: none;
                """)

            # Future stages
            else:

                label.setText(f"○   {name}")

                label.setStyleSheet("""
                    font-size: 14px;
                    font-weight: 700;
                    color: #9a9db0;
                    padding: 8px;
                    border: none;
                """)

    def generate_attack(self):
        if not hasattr(self, "uploadedImage"):
            QMessageBox.warning(
                self,
                "No Image",
                "Please upload an image first."
            )
            return

        self.stageElapsed = {}

        # Update UI immediately
        self.generateButton.setText(
            "Generating... Please Wait..."
        )
        self.generateButton.setEnabled(False)
        self.loadingOverlay.setGeometry(self.rect())
        self.loadingOverlay.show()
        self.loadingOverlay.raise_()

        self.loadingStatusLabel.setText("Preparing algorithm...")

        for stage, label in self.loadingStageLabels.items():
            name = {
                "adversarial_generation": "Adversarial Generation",
                "resnet_classification": "ResNet-50 Classification",
                "lightshed_simulation": "LightShed Simulation",
                "trufor_simulation": "TruFor Simulation",
            }[stage]

            label.setText(f"○   {name}")
            label.setStyleSheet("""
                font-size: 14px;
                font-weight: 700;
                color: #9a9db0;
                padding: 8px;
                border: none;
            """)


        target_name = self.targetClassCombo.currentText()
        target_label = TARGET_CLASSES[target_name]


        # Create thread
        self.thread = QThread()

        self.worker = AttackWorker(
            self.uploadedTensor,
            target_label
        )

        self.worker.moveToThread(self.thread)


        # Start worker
        self.thread.started.connect(
            self.worker.run
        )

        self.worker.stageChanged.connect(
            self.update_loading_stage
        )

        self.worker.stageCompleted.connect(
            self.complete_loading_stage
        )

        # When finished
        self.worker.finished.connect(
            self.attack_finished
        )

        self.worker.error.connect(
            self.attack_error
        )


        # Cleanup
        self.worker.finished.connect(
            self.thread.quit
        )

        self.worker.finished.connect(
            self.worker.deleteLater
        )

        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)

        self.thread.finished.connect(
            self.thread.deleteLater
        )


        self.thread.start()

    def attack_finished(self, results):
        self.loadingOverlay.hide()
        self.pgdImage = results["pgd"]
        self.gsrImage = results["gsr"]

        self.pgdIterations = results["pgd_iterations"]
        self.gsrIterations = results["gsr_iterations"]

        print("PGD iterations:", self.pgdIterations)
        print("GSR-PGD iterations:", self.gsrIterations)


        results["original"] = self.uploadedTensor


        self.attackFinished.emit(results)


        self.generateButton.setText(
            "Generate Adversarial Image"
        )

        self.generateButton.setEnabled(True)

    def complete_loading_stage(
        self,
        stage,
        elapsed_ms,
        variant
    ):

        # Create dictionary for stage if it doesn't exist
        if stage not in self.stageElapsed:
            self.stageElapsed[stage] = {}

        # Store PGD or GSR timing
        self.stageElapsed[stage][variant] = elapsed_ms

        names = {
            "adversarial_generation": "Adversarial Generation",
            "resnet_classification": "ResNet-50 Classification",
            "lightshed_simulation": "LightShed Simulation",
            "trufor_simulation": "TruFor Simulation",
        }

        label = self.loadingStageLabels[stage]

        pgd_time = self.stageElapsed[stage].get("pgd")
        gsr_time = self.stageElapsed[stage].get("gsr")

        # Both PGD and GSR-PGD finished
        if pgd_time is not None and gsr_time is not None:

            label.setText(
                f"✓   {names[stage]}\n"
                f"     PGD: {self._format_time(pgd_time)}   |   "
                f"GSR-PGD: {self._format_time(gsr_time)}"
            )

            label.setStyleSheet("""
                font-size: 14px;
                font-weight: 700;
                color: #2e8b57;
                padding: 8px;
                border: none;
            """)

        # Only one of the two has finished
        else:

            pgd_text = (
                self._format_time(pgd_time)
                if pgd_time is not None
                else "..."
            )

            gsr_text = (
                self._format_time(gsr_time)
                if gsr_time is not None
                else "..."
            )

            label.setText(
                f"●   {names[stage]}\n"
                f"     PGD: {pgd_text}   |   "
                f"GSR-PGD: {gsr_text}"
            )

            label.setStyleSheet("""
                font-size: 14px;
                font-weight: 800;
                color: #1d3195;
                padding: 8px;
                border: none;
            """)
        
    def attack_error(self, error):
        self.loadingOverlay.hide()
        QMessageBox.critical(
            self,
            "Attack Failed",
            error
        )

        self.generateButton.setText(
            "Generate Adversarial Image"
        )

        self.generateButton.setEnabled(True)