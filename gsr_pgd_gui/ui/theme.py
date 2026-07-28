"""
theme.py
--------
Central place for colors, fonts and the global QSS stylesheet.
Keeping this in one module makes it easy to re-skin the whole
application without touching individual widget files.
"""

# ---- Palette -----------------------------------------------------------
COLOR_ORANGE = "#dd7e28"
COLOR_RED = "#be282b"
COLOR_DARK_RED = "#8b0d0f"
COLOR_BLUE = "#1d3195"
COLOR_TEXT_DARK = "#101d60"
COLOR_TEXT_MUTED = "#616a96"
COLOR_CARD_BG = "#f6f6f6"
COLOR_WHITE = "#ffffff"
COLOR_BORDER = "#616a96"

HEADER_GRADIENT = (
    f"qlineargradient(x1:0, y1:0, x2:1, y2:0, "
    f"stop:0 {COLOR_ORANGE}, stop:0.55 {COLOR_RED}, stop:1 {COLOR_BLUE})"
)

PILL_GRADIENT = (
    f"qlineargradient(x1:0, y1:0, x2:1, y2:0, "
    f"stop:0 {COLOR_ORANGE}, stop:1 {COLOR_RED})"
)

BLUE_GRADIENT = (
    "qlineargradient(x1:0, y1:0, x2:1, y2:0, "
    "stop:0 #8ab6f9, stop:1 #4a54e1)"
)

GLOBAL_QSS = f"""
QWidget {{
    font-family: 'Arial Nova', 'Arial';
    color: {COLOR_TEXT_DARK};
}}

QMainWindow {{
    background-color: {COLOR_WHITE};
}}

/* ---------------- Navigation bar ---------------- */
#NavBar {{
    background: {HEADER_GRADIENT};
    border-radius: 10px;
}}

#AppTitle {{
    color: white;
    font: Arial;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -1.0px;
}}

#AppTitleSuffix {{
    color: rgba(255,255,255,0.85);
    font-size: 22px;
    font-weight: 400;
}}

QPushButton#NavButton {{
    background: rgba(255,255,255,0.16);
    color: white;
    border: none;
    border-radius: 16px;
    padding: 8px 22px;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 1px;
        min-height: 34px;
    border-radius: 17px;
    border: none;
    padding: 2px 22px;
}}

QPushButton#NavButton:hover {{
    background: rgba(255,255,255,0.28);
}}

QPushButton#NavButton:checked {{
    background: {COLOR_DARK_RED};
    color: white;
}}

/* ---------------- Cards / Frames ---------------- */
QFrame#Card {{
    background-color: {COLOR_CARD_BG};
    border: 1px solid {COLOR_BORDER};
    border-radius: 10px;
}}

QFrame#DropZone {{
    background-color: white;
    border: 2px dashed {COLOR_BORDER};
    border-radius: 12px;
}}

QFrame#ImageTile {{
    background-color: white;
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
}}

QFrame#ImageTileSelected {{
    background-color: white;
    border: 2px solid {COLOR_RED};
    border-radius: 8px;
}}

/* ---------------- Section header pill (e.g. EVALUATION METRICS) -- */
QLabel#SectionPill {{
    background: {PILL_GRADIENT};
    color: white;
    font-weight: 800;
    font-size: 13px;
    padding: 10px 18px;
    border-radius: 18px;
}}

/* ---------------- Page title -------------------- */
QLabel#PageTitle {{
    color: {COLOR_RED};
    font-size: 30px;
    font-weight: 900;
}}

QLabel#PageTitleBlue {{
    color: {COLOR_BLUE};
    font-size: 30px;
    font-weight: 900;
}}

QLabel#PageSubtitle {{
    color: {COLOR_TEXT_MUTED};
    font-size: 13px;
}}

QLabel#IconBadge {{
    background: {PILL_GRADIENT};
    border-radius: 20px;
    color: white;
    font-weight: 900;
    font-size: 16px;
}}

QLabel#IconBadgeBlue {{
    background: {BLUE_GRADIENT};
    border-radius: 20px;
    color: white;
    font-weight: 900;
    font-size: 16px;
}}

/* ---------------- Group captions ---------------- */
QLabel#GroupCaption {{
    color: {COLOR_RED};
    font-weight: 800;
    font-size: 14px;
}}

QLabel#TileCaption {{
    color: {COLOR_TEXT_MUTED};
    font-size: 11px;
    font-style: italic;
}}

/* ---------------- Image placeholders ------------ */
QLabel#ImagePlaceholder {{
    background-color: #eceef5;
    border-radius: 6px;
    color: {COLOR_TEXT_MUTED};
    font-size: 12px;
    font-weight: 600;
}}

/* ---------------- Buttons ------------------------ */
QPushButton#PrimaryButton {{
    background: {BLUE_GRADIENT};
    color: white;
    border: none;
    border-radius: 20px;
    padding: 12px 24px;
    font-weight: 700;
    font-size: 13px;
}}
QPushButton#PrimaryButton:hover {{ background: #3d46c9; }}
QPushButton#PrimaryButton:pressed {{ background: #2f3799; }}

QPushButton#OutlineButton {{
    background: white;
    color: {COLOR_TEXT_DARK};
    border: 1px solid {COLOR_BORDER};
    border-radius: 16px;
    padding: 8px 18px;
    font-weight: 700;
    font-size: 12px;
}}
QPushButton#OutlineButton:hover {{
    border: 1px solid {COLOR_BLUE};
}}

QPushButton#OutlineButton:checked {{
    background: {PILL_GRADIENT};
    color: white;
    border: none;
    border-radius: 16px;
    padding: 8px 18px;
}}

QPushButton#SecondaryActionButton {{
    background: {COLOR_CARD_BG};
    color: {COLOR_TEXT_DARK};
    border: 1px solid {COLOR_BORDER};
    border-radius: 14px;
    padding: 8px 16px;
    font-weight: 700;
    font-size: 12px;
}}
QPushButton#SecondaryActionButton:hover {{
    border-color: {COLOR_RED};
    color: {COLOR_RED};
}}

/* ---------------- Inputs ------------------------- */
QComboBox {{
    background: white;
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 8px 12px;
    font-weight: 700;
    color: {COLOR_BLUE};
}}
QComboBox::drop-down {{
    border: none;
    width: 26px;
}}

QComboBox QAbstractItemView {{
    background: white;
    color: #101d60;
    border: 1px solid #616a96;
    selection-background-color: #be282b;
    selection-color: white;
    outline: none;
}}

QComboBox QAbstractItemView::item {{
    background: white;
    color: #101d60;
    min-height: 28px;
}}

QComboBox QAbstractItemView::item:selected {{
    background: #be282b;
    color: white;
}}

QListWidget {{
    background: white;
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    outline: none;
}}
QListWidget::item {{
    padding: 8px 12px;
    border-bottom: 1px solid #eef0f7;
}}
QListWidget::item:selected {{
    background: #fdeee9;
    color: {COLOR_RED};
    font-weight: 700;
}}

/* ---------------- Metrics table ------------------ */
QFrame#MetricsHeaderRow {{
    background: transparent;
}}
QLabel#MetricsColHeader {{
    color: {COLOR_TEXT_DARK};
    font-weight: 800;
    font-size: 12px;
}}
QLabel#MetricsRowLabel {{
    color: {COLOR_TEXT_MUTED};
    font-size: 12px;
    font-weight: 600;
}}
QLabel#MetricsValue {{
    color: {COLOR_TEXT_DARK};
    font-size: 12px;
    font-weight: 700;
}}
QLabel#StatusYes {{
    color: {COLOR_RED};
    font-weight: 800;
}}
QLabel#StatusNo {{
    color: #2e8b57;
    font-weight: 800;
}}
QFrame#HDivider {{
    background-color: {COLOR_BORDER};
    max-height: 1px;
    min-height: 1px;
}}

QScrollArea {{
    border: none;
    background: transparent;
}}
"""
