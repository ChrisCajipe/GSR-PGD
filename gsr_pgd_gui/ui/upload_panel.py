"""
upload_panel.py
----------------
"UPLOAD" page: lets the user pick a 512x512 ImageNet image, preview
it, choose a target class for the attack, verify it with ResNet-50
and finally trigger adversarial generation.

Backend hook points (see TODOs):
    - upload_image()
    - verify_image()
    - generate_attack()
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

from ui.page_header import PageHeader

from backend.image_loader import load_image

# Placeholder ImageNet target classes shown in the mock-up.
TARGET_CLASSES = [
    "Electric Cray", "Brambling", "Goldfish", "Water Ouzel",
    "Quail", "Sea Slug", "Persian Cat", "Piggy Bank",
]


class UploadPanel(QWidget):
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
        self.targetClassCombo.addItems(TARGET_CLASSES)
        self.targetClassCombo.setCurrentText("Sea Slug")
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

    def upload_image(self, file_path: str = None):
        if file_path is None:
            dialog = QFileDialog(self)
            dialog.setFileMode(QFileDialog.ExistingFile)
            dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
            dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)

            result = dialog.exec()

            if result:
                file_path = dialog.selectedFiles()[0]
            else:
                return
        
        self.uploadedImage = load_image(file_path)
        self._display_image(self.uploadedImage)

        self.fileNameLabel.setText(
            f"File Name: {os.path.basename(file_path)}"
        )

        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        self.fileSizeLabel.setText(f"{size_mb:.2f} MB")

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if hasattr(self, "uploadedImage"):
            self._display_image(self.uploadedImage)

    def verify_image(self):
        """
        TODO: Connect to backend `verify_image()` (ResNet-50 classifier).
        Should:
            1. Run the currently loaded image through ResNet-50.
            2. Update `self.originalPredictionLabel` with the predicted class.
        """
        pass

    def generate_attack(self):
        """
        TODO: Connect to backend `generate_attack()`.
        Should:
            1. Read the selected target class from `self.targetClassCombo`.
            2. Run both standard PGD and GSR-PGD attack generation.
            3. Trigger navigation to the Results page and populate it.
        """
        pass
