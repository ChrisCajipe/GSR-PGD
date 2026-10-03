"""
metrics_panel.py
-----------------
Right-hand "EVALUATION METRICS" card shown on the Results page.
Contains three sub-tables:
    - General Metrics   (prediction, PSNR, SSIM)
    - LightShed Detector (detection status)
    - TruFor Detector    (detection status)

All values are initialized to placeholder text ("--") and exposed as
attributes so the backend can update them directly, e.g.:

    metrics_panel.psnrPgdLabel.setText("28.61 dB")
    metrics_panel.lightshedStatusGsrLabel.setText("No")
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QSizePolicy


class MetricsPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(18)

        # ---- Section pill header --------------------------------------
        pill = QLabel("\U0001F6E1  EVALUATION METRICS")
        pill.setObjectName("SectionPill")
        outer.addWidget(pill)

        # ---- General metrics table -------------------------------------
        outer.addWidget(self._section_caption("GENERAL METRICS"))
        general_frame, general_rows = self._build_table(
            row_labels=["PREDICTION", "PSNR", "SSIM"]
        )
        outer.addWidget(general_frame)

        self.predictionPgdLabel, self.predictionGsrLabel = general_rows["PREDICTION"]
        self.psnrPgdLabel, self.psnrGsrLabel = general_rows["PSNR"]
        self.ssimPgdLabel, self.ssimGsrLabel = general_rows["SSIM"]

        # Convenience aliases matching the requested generic names.
        self.psnrLabel = self.psnrGsrLabel
        self.ssimLabel = self.ssimGsrLabel

        self.predictionPgdLabel.setText("--")
        self.predictionGsrLabel.setText("--")
        self.psnrPgdLabel.setText("--")
        self.psnrGsrLabel.setText("--")
        self.ssimPgdLabel.setText("--")
        self.ssimGsrLabel.setText("--")

        # ---- LightShed detector table ------------------------------------
        outer.addWidget(self._section_caption("LIGHTSHED DETECTOR"))
        lightshed_frame, lightshed_rows = self._build_table(
            row_labels=["DETECTED"], status=True
        )
        outer.addWidget(lightshed_frame)
        self.lightshedStatusPgdLabel, self.lightshedStatusGsrLabel = lightshed_rows["DETECTED"]
        self.lightshedStatusLabel = self.lightshedStatusGsrLabel  # generic alias

        # ---- TruFor detector table ---------------------------------------
        outer.addWidget(self._section_caption("TRUFOR DETECTOR"))
        trufor_frame, trufor_rows = self._build_table(
            row_labels=["DETECTED"], status=True
        )
        outer.addWidget(trufor_frame)
        self.truforStatusPgdLabel, self.truforStatusGsrLabel = trufor_rows["DETECTED"]
        self.truforStatusLabel = self.truforStatusGsrLabel  # generic alias

        outer.addStretch(1)
       
        # ---- Processing Time ---------------------------------------
        outer.addWidget(self._section_caption("PROCESSING TIME"))

        timing_frame, timing_rows = self._build_table(
            row_labels=[
                "ADVERSARIAL GENERATION",
                "RESNET-50 CLASSIFICATION",
                "LIGHTSHED SIMULATION",
                "TRUFOR SIMULATION",
            ]
        )

        outer.addWidget(timing_frame)

        self.adversarialTimePgdLabel, self.adversarialTimeGsrLabel = (
            timing_rows["ADVERSARIAL GENERATION"]
        )

        self.resnetTimePgdLabel, self.resnetTimeGsrLabel = (
            timing_rows["RESNET-50 CLASSIFICATION"]
        )

        self.lightshedTimePgdLabel, self.lightshedTimeGsrLabel = (
            timing_rows["LIGHTSHED SIMULATION"]
        )

        self.truforTimePgdLabel, self.truforTimeGsrLabel = (
            timing_rows["TRUFOR SIMULATION"]
        )

        outer.addStretch(1)

    # ------------------------------------------------------------------
    def _section_caption(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("GroupCaption")
        return label

    def _build_table(self, row_labels, status: bool = False):
        """
        Builds a small 3-column table: METRICS | PGD | GSR-PGD
        Returns (frame_widget, {row_label: (pgd_value_label, gsr_value_label)})
        """
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Header row
        header_row = QHBoxLayout()
        header_row.addWidget(self._col_label("METRICS", flex=2), stretch=2)
        header_row.addWidget(self._col_label("PGD", flex=1), stretch=1)
        header_row.addWidget(self._col_label("GSR-PGD", flex=1), stretch=1)
        layout.addLayout(header_row)
        layout.addWidget(self._divider())

        row_values = {}
        for name in row_labels:
            row_layout = QHBoxLayout()
            name_label = QLabel(name)
            name_label.setObjectName("MetricsRowLabel")
            row_layout.addWidget(name_label, stretch=2)

            pgd_value = QLabel("--")
            gsr_value = QLabel("--")
            for value_label in (pgd_value, gsr_value):
                value_label.setObjectName("StatusYes" if status else "MetricsValue")
                value_label.setAlignment(Qt.AlignLeft)
                row_layout.addWidget(value_label, stretch=1)

            layout.addLayout(row_layout)
            layout.addWidget(self._divider())
            row_values[name] = (pgd_value, gsr_value)

        return frame, row_values
    def _build_timing_table(self, row_labels):
        """
        Builds a 2-column table:

            STAGE | ELAPSED TIME
        """

        frame = QFrame()

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Header
        header_row = QHBoxLayout()

        header_row.addWidget(
            self._col_label("STAGE", flex=2),
            stretch=2
        )

        header_row.addWidget(
            self._col_label("ELAPSED", flex=1),
            stretch=1
        )

        layout.addLayout(header_row)
        layout.addWidget(self._divider())

        row_values = {}

        for name in row_labels:

            row = QHBoxLayout()

            name_label = QLabel(name)
            name_label.setObjectName("MetricsRowLabel")

            value_label = QLabel("--")
            value_label.setObjectName("MetricsValue")

            row.addWidget(name_label, stretch=2)
            row.addWidget(value_label, stretch=1)

            layout.addLayout(row)
            layout.addWidget(self._divider())

            row_values[name] = value_label

        return frame, row_values
    
    def _col_label(self, text: str, flex: int) -> QLabel:
        label = QLabel(text)
        label.setObjectName("MetricsColHeader")
        label.setAlignment(Qt.AlignLeft)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        return label

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
    # Helpers the backend can call to color status labels
    # ------------------------------------------------------------------
    def set_detection_status(self, label: QLabel, detected: bool):
        """TODO: call this from backend after running LightShed / TruFor."""
        label.setText("Yes" if detected else "No")
        label.setObjectName("StatusYes" if detected else "StatusNo")
        label.setStyle(label.style())  # force QSS re-polish
