"""PyQt6 GUI adapter for the existing cross-event verification pipeline.

This module is deliberately a presentation-layer adapter.  It owns Qt widgets,
QSS and the main-thread polling timer, but it does not implement detection,
tracking, identity, registration, storage or runtime-parameter validation.
Those responsibilities remain behind :class:`FrameWorker` and
:class:`VideoVerifierPipeline`, exactly as they are for the Tk adapter.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import queue
import sys
import threading
from typing import Any

import cv2

from .automation import AutomationPolicy
from .engine import CrossEventVerifier
from .gui_theme import (
    ENGINEERING_GRID_MAJOR,
    ENGINEERING_GRID_MINOR,
    INDUSTRIAL_NAVY,
    METALLIC_GRAY,
    MODULE_SUBTITLES,
    MUTED_NAVY,
    NASA_BLUE,
    PANEL_CREAM,
    RETRO_ORANGE,
    RETRO_WHITE,
    SIGNAL_RED,
    SOLAR_YELLOW,
    STATUS_GREEN,
    STATUS_PENDING,
    VINTAGE_PAPER,
    WARM_IVORY,
)
from .media import (
    FrameMessage,
    FrameWorker,
    ParameterUpdateMessage,
    RegistrationMessage,
    SourceSpec,
    StatusMessage,
)
from .pipeline import FrameResult, VideoVerifierPipeline
from .runtime_parameters import RuntimeParameterSpec, RuntimeParameterState
from .storage import SqliteStore
from .vision_factory import build_vision_adapter


try:  # Keep the import failure local to the optional Qt adapter.
    from PyQt6.QtCore import QLineF, QTimer, Qt, QRectF
    from PyQt6.QtGui import (
        QColor,
        QFont,
        QImage,
        QPainter,
        QPainterPath,
        QPen,
        QPixmap,
        QBrush,
    )
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QCheckBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QRadioButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
except ImportError as error:  # pragma: no cover - exercised by dependency checks
    raise ImportError(
        "PyQt6 is required for the Qt GUI. Install the optional GUI dependency "
        "with `pip install -e .[gui-qt]`."
    ) from error


GUI_POLL_INTERVAL_MS = 16
VIDEO_STANDBY_TEXT = "◎\n\n没有画面\nNO VISUAL FEED\nCAMERA CHANNEL STANDBY"


def _named(widget: QWidget, name: str) -> QWidget:
    """Set a QSS object name without changing the widget's functional type."""

    widget.setObjectName(name)
    return widget


QT_STYLE = f"""
QWidget {{
    background: transparent;
    color: {INDUSTRIAL_NAVY};
}}
QMainWindow, QWidget#missionSurface {{
    background: {RETRO_WHITE};
    color: {INDUSTRIAL_NAVY};
    font-family: "Segoe UI";
    font-size: 10pt;
}}
QFrame#headerPanel {{
    background: transparent;
    border: none;
    padding: 0;
}}
QFrame#missionPanelFrame {{
    background: transparent;
    border: none;
    padding: 0;
}}
QLabel#headerTitle {{
    color: {RETRO_WHITE};
    font-size: 16pt;
    font-weight: 700;
    letter-spacing: 2px;
}}
QLabel#headerSubtitle, QLabel#moduleMarker, QLabel#rangeLabel {{
    color: {SOLAR_YELLOW};
    font-family: Consolas;
    font-size: 8pt;
    font-weight: 700;
}}
QLabel#statusBadge {{
    background: {NASA_BLUE};
    color: {RETRO_WHITE};
    border: 1px solid {SOLAR_YELLOW};
    border-radius: 12px;
    padding: 7px 12px;
    min-width: 132px;
    font-family: Consolas;
    font-weight: 700;
}}
QGroupBox {{
    background: transparent;
    border: none;
    border-radius: 0;
    margin-top: 12px;
    padding: 8px 8px 6px 8px;
    font-weight: 700;
}}
QGroupBox#majorPanel, QGroupBox#modulePanel {{
    border: none;
    background: transparent;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 5px;
    color: {INDUSTRIAL_NAVY};
    background: {WARM_IVORY};
}}
QLabel#mutedLabel, QLabel#statusLine, QLabel#parameterDescription {{
    color: {MUTED_NAVY};
}}
QLabel#instrumentLabel {{
    color: {INDUSTRIAL_NAVY};
    background: {WARM_IVORY};
    padding: 4px 6px;
    font-family: Consolas;
}}
QLineEdit {{
    background: #FFF7E6;
    color: {INDUSTRIAL_NAVY};
    border: 1px solid {INDUSTRIAL_NAVY};
    border-radius: 2px;
    padding: 4px 6px;
    selection-background-color: {NASA_BLUE};
    selection-color: {RETRO_WHITE};
}}
QLineEdit:focus {{ border: 1px solid {NASA_BLUE}; }}
QLineEdit:disabled {{ color: {MUTED_NAVY}; background: {PANEL_CREAM}; }}
QAbstractScrollArea, QScrollArea {{
    background: transparent;
    border: none;
}}
QAbstractScrollArea::viewport, QScrollArea::viewport {{
    background: {WARM_IVORY};
    border: none;
}}
QWidget#sideSurface {{ background: {WARM_IVORY}; }}
QStackedWidget {{ background: {WARM_IVORY}; border: none; }}
QTextEdit, QPlainTextEdit {{
    background: #FFF7E6;
    color: {INDUSTRIAL_NAVY};
    border: 1px solid {INDUSTRIAL_NAVY};
    selection-background-color: {NASA_BLUE};
    selection-color: {RETRO_WHITE};
}}
QComboBox, QSpinBox, QDoubleSpinBox {{
    background: #FFF7E6;
    color: {INDUSTRIAL_NAVY};
    border: 1px solid {INDUSTRIAL_NAVY};
    border-radius: 2px;
    padding: 4px 6px;
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {NASA_BLUE}; }}
QComboBox QAbstractItemView {{
    background: #FFF7E6;
    color: {INDUSTRIAL_NAVY};
    border: 1px solid {INDUSTRIAL_NAVY};
    selection-background-color: {NASA_BLUE};
    selection-color: {RETRO_WHITE};
}}
QPushButton {{
    background: {WARM_IVORY};
    color: {INDUSTRIAL_NAVY};
    border: 1px solid {INDUSTRIAL_NAVY};
    border-radius: 3px;
    padding: 6px 10px;
    font-weight: 700;
}}
QPushButton:hover {{ background: {SOLAR_YELLOW}; }}
QPushButton:pressed {{ background: {RETRO_ORANGE}; padding-top: 7px; padding-bottom: 5px; }}
QPushButton#primaryButton {{ background: {NASA_BLUE}; color: {RETRO_WHITE}; }}
QPushButton#primaryButton:hover {{ background: #2A87B8; }}
QPushButton#dangerButton {{ background: {SIGNAL_RED}; color: {RETRO_WHITE}; }}
QPushButton#dangerButton:hover {{ background: #E4513E; }}
QTabWidget::pane {{
    border: 1px solid {INDUSTRIAL_NAVY};
    background: transparent;
    top: -1px;
}}
QTabBar::tab {{
    background: {PANEL_CREAM};
    color: {INDUSTRIAL_NAVY};
    border: 1px solid {INDUSTRIAL_NAVY};
    border-bottom: 2px solid {INDUSTRIAL_NAVY};
    padding: 7px 14px;
    margin-right: 3px;
    font-weight: 700;
}}
QTabBar::tab:selected {{
    background: {NASA_BLUE};
    color: {RETRO_WHITE};
    border-bottom: 3px solid {SOLAR_YELLOW};
}}
QTabBar::tab:hover:!selected {{ background: {SOLAR_YELLOW}; }}
QSlider::groove:horizontal {{
    height: 6px;
    background: {VINTAGE_PAPER};
    border: 1px solid {INDUSTRIAL_NAVY};
}}
QSlider::sub-page:horizontal {{ background: {NASA_BLUE}; }}
QSlider::handle:horizontal {{
    width: 16px;
    margin: -6px 0;
    background: {NASA_BLUE};
    border: 1px solid {INDUSTRIAL_NAVY};
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{ background: {RETRO_ORANGE}; }}
QSlider::handle:horizontal:pressed {{ background: {RETRO_ORANGE}; }}
QScrollBar:vertical {{
    background: #E7DCC0;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{ background: {INDUSTRIAL_NAVY}; min-height: 28px; border-radius: 3px; }}
QScrollBar::handle:vertical:hover {{ background: {NASA_BLUE}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: #E7DCC0; }}
QAbstractItemView {{
    background: #FFF7E6;
    color: {INDUSTRIAL_NAVY};
    selection-background-color: {NASA_BLUE};
    selection-color: {RETRO_WHITE};
}}
QTableWidget, QTreeView, QTableView {{
    background: #FFF7E6;
    alternate-background-color: {WARM_IVORY};
    color: {INDUSTRIAL_NAVY};
    border: 1px solid {INDUSTRIAL_NAVY};
    gridline-color: {VINTAGE_PAPER};
    font-family: Consolas;
    font-size: 9pt;
    selection-background-color: {NASA_BLUE};
    selection-color: {RETRO_WHITE};
}}
QTableWidget::viewport, QTreeView::viewport, QTableView::viewport {{
    background: #FFF7E6;
}}
QHeaderView::section {{
    background: {INDUSTRIAL_NAVY};
    color: {RETRO_WHITE};
    border: none;
    padding: 6px 4px;
    font-weight: 700;
}}
QCheckBox, QRadioButton {{ color: {INDUSTRIAL_NAVY}; spacing: 5px; }}
QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    border: 2px solid {INDUSTRIAL_NAVY};
    border-radius: 7px;
    background: {WARM_IVORY};
}}
QRadioButton::indicator:checked {{
    background: {NASA_BLUE};
    border: 2px solid {SOLAR_YELLOW};
}}
QRadioButton::indicator:checked:hover {{ background: {RETRO_ORANGE}; }}
QRadioButton::indicator:disabled {{
    background: {PANEL_CREAM};
    border: 2px solid {METALLIC_GRAY};
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {INDUSTRIAL_NAVY};
    background: {WARM_IVORY};
}}
QCheckBox::indicator:checked {{
    background: {NASA_BLUE};
    border: 2px solid {SOLAR_YELLOW};
}}
QProgressBar {{
    background: {WARM_IVORY};
    border: 1px solid {INDUSTRIAL_NAVY};
    text-align: center;
    height: 12px;
}}
QProgressBar::chunk {{ background: {NASA_BLUE}; }}
"""


class EngineeringSurface(QWidget):
    """Static NASA engineering-paper background, painted below child widgets."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("missionSurface")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def paintEvent(self, event: Any) -> None:  # pragma: no cover - visual drawing
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(RETRO_WHITE))
        minor_pen = QPen(QColor(ENGINEERING_GRID_MINOR), 1)
        major_pen = QPen(QColor(ENGINEERING_GRID_MAJOR), 1)
        spacing = 28
        major_every = 5
        width = self.width()
        height = self.height()
        for index, x in enumerate(range(0, width + spacing, spacing)):
            painter.setPen(major_pen if index % major_every == 0 else minor_pen)
            painter.drawLine(x, 0, x, height)
        for index, y in enumerate(range(0, height + spacing, spacing)):
            painter.setPen(major_pen if index % major_every == 0 else minor_pen)
            painter.drawLine(0, y, width, y)
        marker_pen = QPen(QColor("#CFC1AA"), 1)
        painter.setPen(marker_pen)
        major_spacing = spacing * major_every
        for x in range(major_spacing, width, major_spacing * 2):
            for y in range(major_spacing, height, major_spacing * 2):
                painter.drawLine(x - 3, y, x + 3, y)
                painter.drawLine(x, y - 3, x, y + 3)
        painter.end()
        super().paintEvent(event)


def _cut_corner_path(rect: QRectF, cut: float = 8.0) -> QPainterPath:
    """Return a restrained Googie-style chamfered panel outline."""

    corner = min(cut, rect.width() / 4.0, rect.height() / 4.0)
    path = QPainterPath()
    path.moveTo(rect.left() + corner, rect.top())
    path.lineTo(rect.right() - corner, rect.top())
    path.lineTo(rect.right(), rect.top() + corner)
    path.lineTo(rect.right(), rect.bottom() - corner)
    path.lineTo(rect.right() - corner, rect.bottom())
    path.lineTo(rect.left() + corner, rect.bottom())
    path.lineTo(rect.left(), rect.bottom() - corner)
    path.lineTo(rect.left(), rect.top() + corner)
    path.closeSubpath()
    return path


def _draw_panel_surface(
    painter: QPainter,
    rect: QRectF,
    *,
    fill: str,
    accent: str,
    border: str = INDUSTRIAL_NAVY,
    cut_corner: float = 8.0,
) -> None:
    """Draw the static, non-interactive chrome shared by mission panels."""

    outer = rect.adjusted(1.0, 1.0, -1.0, -1.0)
    inner = rect.adjusted(5.0, 5.0, -5.0, -5.0)
    painter.setBrush(QBrush(QColor(fill)))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawPath(_cut_corner_path(outer, cut_corner))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(QColor(border), 2))
    painter.drawPath(_cut_corner_path(outer, cut_corner))
    painter.setPen(QPen(QColor(METALLIC_GRAY), 1))
    painter.drawPath(_cut_corner_path(inner, max(cut_corner - 3.0, 2.0)))


def _draw_panel_details(
    painter: QPainter,
    rect: QRectF,
    *,
    accent: str,
    cut_corner: float = 8.0,
) -> None:
    """Add a small accent rail and a few low-weight instrument rivets."""

    outer = rect.adjusted(2.0, 2.0, -2.0, -2.0)
    painter.setPen(QPen(QColor(accent), 3))
    painter.drawLine(
        QLineF(
            outer.left(),
            outer.top() + cut_corner + 4.0,
            outer.left(),
            outer.bottom() - cut_corner - 4.0,
        )
    )
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(METALLIC_GRAY)))
    for x, y in (
        (outer.left() + 8.0, outer.top() + 8.0),
        (outer.right() - 8.0, outer.top() + 8.0),
        (outer.right() - 8.0, outer.bottom() - 8.0),
    ):
        painter.drawEllipse(QRectF(x - 2.0, y - 2.0, 4.0, 4.0))


class MissionPanelFrame(QFrame):
    """A visual-only cut-corner frame for large mission surfaces."""

    def __init__(
        self,
        *,
        fill: str = WARM_IVORY,
        accent: str = NASA_BLUE,
        border: str = INDUSTRIAL_NAVY,
        cut_corner: float = 8.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._fill = fill
        self._accent = accent
        self._border = border
        self._cut_corner = cut_corner
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def paintEvent(self, event: Any) -> None:  # pragma: no cover - visual drawing
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect())
        _draw_panel_surface(
            painter,
            rect,
            fill=self._fill,
            accent=self._accent,
            border=self._border,
            cut_corner=self._cut_corner,
        )
        _draw_panel_details(painter, rect, accent=self._accent, cut_corner=self._cut_corner)
        painter.end()


class MissionGroupBox(QGroupBox):
    """A visual-only QGroupBox with Apollo-era panel hierarchy."""

    def __init__(
        self,
        title: str,
        *,
        fill: str = WARM_IVORY,
        accent: str = NASA_BLUE,
        border: str = INDUSTRIAL_NAVY,
        cut_corner: float = 7.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(title, parent)
        self._fill = fill
        self._accent = accent
        self._border = border
        self._cut_corner = cut_corner
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def paintEvent(self, event: Any) -> None:  # pragma: no cover - visual drawing
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect())
        _draw_panel_surface(
            painter,
            rect,
            fill=self._fill,
            accent=self._accent,
            border=self._border,
            cut_corner=self._cut_corner,
        )
        painter.end()
        # Let QGroupBox render the existing title and child layout normally.
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        _draw_panel_details(painter, rect, accent=self._accent, cut_corner=self._cut_corner)
        painter.end()


class InstrumentSlider(QSlider):
    """Keep QSlider semantics while adding static mechanical calibration ticks."""

    def paintEvent(self, event: Any) -> None:  # pragma: no cover - visual drawing
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rail = QRectF(self.rect()).adjusted(10.0, 0.0, -10.0, 0.0)
        if rail.width() <= 0:
            painter.end()
            return
        y = rail.center().y()
        for index in range(9):
            x = rail.left() + rail.width() * index / 8.0
            major = index % 2 == 0
            length = 6.0 if major else 3.0
            painter.setPen(QPen(QColor(METALLIC_GRAY), 1))
            painter.drawLine(QLineF(x, y - length, x, y + length))
        span = max(self.maximum() - self.minimum(), 1)
        ratio = (self.value() - self.minimum()) / span
        marker_x = rail.left() + rail.width() * ratio
        painter.setPen(QPen(QColor(SOLAR_YELLOW), 2))
        painter.drawLine(QLineF(marker_x, y - 10.0, marker_x, y + 10.0))
        marker = QPainterPath()
        marker.moveTo(marker_x - 4.0, y + 12.0)
        marker.lineTo(marker_x + 4.0, y + 12.0)
        marker.lineTo(marker_x, y + 17.0)
        marker.closeSubpath()
        painter.setBrush(QBrush(QColor(SOLAR_YELLOW)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(marker)
        painter.end()


class OrbitWidget(QWidget):
    """Small header-only orbit marker with no application state."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(112, 56)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event: Any) -> None:  # pragma: no cover - visual drawing
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(INDUSTRIAL_NAVY))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        center = self.rect().center()
        painter.setPen(QPen(QColor(NASA_BLUE), 2))
        painter.drawEllipse(QRectF(center.x() - 24, center.y() - 13, 48, 26))
        painter.setPen(QPen(QColor(SOLAR_YELLOW), 1))
        painter.drawEllipse(QRectF(center.x() - 34, center.y() - 7, 68, 14))
        painter.setPen(QPen(QColor(RETRO_ORANGE), 1))
        painter.drawArc(QRectF(center.x() - 30, center.y() - 18, 60, 36), 205 * 16, 130 * 16)
        painter.setBrush(QBrush(QColor(RETRO_ORANGE)))
        painter.setPen(QPen(QColor(RETRO_WHITE), 1))
        painter.drawEllipse(QRectF(center.x() - 5, center.y() - 5, 10, 10))
        painter.setBrush(QBrush(QColor(SOLAR_YELLOW)))
        painter.setPen(Qt.PenStyle.NoPen)
        for x, y in ((center.x() - 29, center.y() - 15), (center.x() + 29, center.y() + 15)):
            painter.drawEllipse(QRectF(x - 2, y - 2, 4, 4))
        painter.setPen(QPen(QColor(SOLAR_YELLOW), 1))
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.drawText(4, self.height() - 5, "SYS / 01")
        painter.end()


class StandbyWidget(QWidget):
    """Low-contrast radar/target illustration shown before the first frame."""

    def paintEvent(self, event: Any) -> None:  # pragma: no cover - visual drawing
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(WARM_IVORY))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        center = self.rect().center()
        radius = max(min(self.width() // 6, self.height() // 5), 42)
        muted = QColor("#8D918F")
        painter.setPen(QPen(muted, 1))
        painter.drawEllipse(QRectF(center.x() - radius, center.y() - radius // 2, radius * 2, radius))
        painter.drawEllipse(QRectF(center.x() - radius // 2, center.y() - radius // 2, radius, radius))
        painter.drawLine(center.x() - radius - 18, center.y(), center.x() + radius + 18, center.y())
        painter.drawLine(center.x(), center.y() - radius // 2 - 18, center.x(), center.y() + radius // 2 + 18)
        painter.setBrush(QBrush(QColor(SOLAR_YELLOW)))
        painter.setPen(Qt.PenStyle.NoPen)
        for x, y in (
            (center.x() - radius - 18, center.y()),
            (center.x() + radius + 18, center.y()),
            (center.x(), center.y() - radius // 2 - 18),
        ):
            painter.drawEllipse(QRectF(x - 3, y - 3, 6, 6))
        painter.setPen(QPen(QColor(INDUSTRIAL_NAVY), 1))
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.drawText(QRectF(0, center.y() + radius // 2 + 32, self.width(), 24), Qt.AlignmentFlag.AlignCenter, "没有画面")
        painter.setPen(QPen(QColor(MUTED_NAVY), 1))
        painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        painter.drawText(QRectF(0, center.y() + radius // 2 + 54, self.width(), 22), Qt.AlignmentFlag.AlignCenter, "NO VISUAL FEED")
        painter.setPen(QPen(QColor(RETRO_ORANGE), 1))
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(QRectF(0, center.y() + radius // 2 + 76, self.width(), 20), Qt.AlignmentFlag.AlignCenter, "CAMERA CHANNEL / STANDBY")
        painter.end()


class VerifierWindow(QMainWindow):
    """Qt presentation adapter over the existing worker and verification pipeline."""

    def __init__(
        self,
        database_path: str = "data/verifier-production-v1.sqlite3",
        vision_backend: str = "production",
    ) -> None:
        super().__init__()
        self.setWindowTitle("Cross-event Prototype Verifier")
        self.resize(1440, 860)
        self.setMinimumSize(980, 640)
        self.setStyleSheet(QT_STYLE)

        self._closed = False
        self._preload_cancel = threading.Event()
        self._preload_messages: queue.Queue[tuple[str, str, float, object | None]] = queue.Queue()
        self._preload_thread: threading.Thread | None = None
        self.vision = None
        self.pipeline: VideoVerifierPipeline | None = None
        self.worker: FrameWorker | None = None
        self._photo: QPixmap | None = None
        self._last_frame: object | None = None
        self._parameter_syncing = False
        self._runtime_parameter_state: RuntimeParameterState | None = None
        self.parameter_vars: dict[str, QLineEdit] = {}
        self.parameter_entries: dict[str, QLineEdit] = {}
        self.parameter_scales: dict[str, QSlider] = {}
        self.parameter_specs: dict[str, RuntimeParameterSpec] = {}
        self._scale_steps: dict[str, int] = {}
        self._table_rows: dict[str, int] = {}

        database = Path(database_path)
        if str(database) != ":memory:":
            database.parent.mkdir(parents=True, exist_ok=True)
        self.verifier = CrossEventVerifier(store=SqliteStore(str(database)))
        self._build_layout()
        self._message_timer = QTimer(self)
        self._message_timer.timeout.connect(self._poll_messages)
        selected = vision_backend.strip().lower()
        if selected == "demo":
            self.vision = build_vision_adapter(vision_backend)
            self._finish_runtime_initialization()
        else:
            self._preload_timer = QTimer(self)
            self._preload_timer.timeout.connect(self._poll_preload)
            self._preload_timer.start(50)
            self._preload_thread = threading.Thread(
                target=self._preload_backend,
                args=(vision_backend,),
                name="cross-event-model-preload",
                daemon=True,
            )
            self._preload_thread.start()

    def _build_layout(self) -> None:
        surface = EngineeringSurface()
        self.setCentralWidget(surface)
        outer = QVBoxLayout(surface)
        outer.setContentsMargins(24, 12, 24, 14)
        outer.setSpacing(10)

        header = _named(
            MissionPanelFrame(
                fill=INDUSTRIAL_NAVY,
                accent=SOLAR_YELLOW,
                border=SOLAR_YELLOW,
                cut_corner=8.0,
            ),
            "headerPanel",
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 5, 8, 5)
        orbit = OrbitWidget()
        header_layout.addWidget(orbit)
        title_box = QVBoxLayout()
        title = _named(QLabel("CROSS-EVENT PROTOTYPE VERIFIER"), "headerTitle")
        subtitle = _named(
            QLabel("CEPV // GAIT IDENTITY SYSTEM  ·  视觉身份 / 步态验证  ·  MISSION CONTROL"),
            "headerSubtitle",
        )
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box, 1)
        self.system_badge = _named(QLabel("● SYSTEM ONLINE"), "statusBadge")
        header_layout.addWidget(self.system_badge)
        outer.addWidget(header)

        controls = MissionGroupBox("输入源", accent=NASA_BLUE)
        controls.setObjectName("majorPanel")
        controls_layout = QGridLayout(controls)
        self._preload_controls_layout = controls_layout
        self._preload_controls = controls
        controls_layout.setContentsMargins(8, 6, 8, 5)
        controls_layout.setHorizontalSpacing(6)
        controls_layout.setVerticalSpacing(4)
        self.camera_radio = QRadioButton("摄像头")
        self.file_radio = QRadioButton("视频文件")
        self.camera_radio.setChecked(True)
        self.camera_index = QLineEdit("0")
        self.camera_index.setMaximumWidth(64)
        self.video_path = QLineEdit()
        self.camera_id = QLineEdit("camera-1")
        self.camera_id.setMaximumWidth(150)
        self.video_repeat_count = QLineEdit("1")
        self.video_repeat_count.setMaximumWidth(64)
        self.browse_button = QPushButton("浏览…")
        self.start_button = _named(QPushButton("开始"), "primaryButton")
        self.stop_button = _named(QPushButton("停止"), "dangerButton")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.camera_radio, 0, 0)
        controls_layout.addWidget(self.camera_index, 0, 1)
        controls_layout.addWidget(QLabel("设备号"), 0, 2)
        controls_layout.addWidget(self.file_radio, 0, 3)
        controls_layout.addWidget(self.video_path, 0, 4)
        controls_layout.addWidget(self.browse_button, 0, 5)
        controls_layout.addWidget(QLabel("来源 ID"), 0, 6)
        controls_layout.addWidget(self.camera_id, 0, 7)
        controls_layout.addWidget(QLabel("视频重复学习"), 0, 8)
        controls_layout.addWidget(self.video_repeat_count, 0, 9)
        controls_layout.addWidget(self.start_button, 0, 10)
        controls_layout.addWidget(self.stop_button, 0, 11)
        controls_layout.setColumnStretch(4, 1)
        module_marker = _named(QLabel("MODULE / 01  ·  INPUT ARRAY"), "moduleMarker")
        self.backend_status = _named(QLabel("视觉后端：正在后台准备…"), "mutedLabel")
        controls_layout.addWidget(module_marker, 1, 0, 1, 3)
        controls_layout.addWidget(self.backend_status, 1, 3, 1, 9)
        self.preload_phase = _named(QLabel("正在后台准备生产视觉后端…"), "mutedLabel")
        self.preload_progress = QProgressBar()
        self.preload_progress.setRange(0, 100)
        self.preload_progress.setValue(0)
        controls_layout.addWidget(self.preload_phase, 2, 0, 1, 3)
        controls_layout.addWidget(self.preload_progress, 2, 3, 1, 9)
        outer.addWidget(controls)

        self.notebook = QTabWidget()
        self.monitor_page = QWidget()
        self.parameter_page = QWidget()
        self.notebook.addTab(self.monitor_page, "01 / 实时识别")
        self.notebook.addTab(self.parameter_page, "02 / 实时参数")
        outer.addWidget(self.notebook, 1)
        self._build_monitor_page()
        self._build_parameter_page()

        self.status = _named(QLabel("正在后台分阶段加载模型，请稍候…"), "statusLine")
        self.status.setMinimumHeight(24)
        outer.addWidget(self.status)

        self.camera_radio.toggled.connect(self._source_mode_changed)
        self.file_radio.toggled.connect(self._source_mode_changed)
        self.browse_button.clicked.connect(self._browse_video)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self._source_mode_changed()
        self.automatic_registration.toggled.connect(self._toggle_automatic_registration)

    def _build_monitor_page(self) -> None:
        layout = QHBoxLayout(self.monitor_page)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(10)
        video_panel = _named(MissionPanelFrame(accent=NASA_BLUE), "missionPanelFrame")
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(1, 1, 1, 1)
        self.video_stack = QStackedWidget()
        self.video_label = QLabel(VIDEO_STANDBY_TEXT)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(f"background: {WARM_IVORY}; color: {INDUSTRIAL_NAVY};")
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_standby = StandbyWidget()
        self.video_stack.addWidget(self.video_label)
        self.video_stack.addWidget(self.video_standby)
        self.video_stack.setCurrentWidget(self.video_standby)
        video_layout.addWidget(self.video_stack)
        layout.addWidget(video_panel, 1)

        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setMinimumWidth(420)
        side_scroll.setMaximumWidth(560)
        side_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        side_scroll.setFrameShape(QFrame.Shape.NoFrame)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.side_canvas = side_scroll
        self.side_scrollbar = side_scroll.verticalScrollBar()
        side = _named(QWidget(), "sideSurface")
        side.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(10)
        side_scroll.setWidget(side)
        layout.addWidget(side_scroll, 2)
        layout.setStretch(0, 3)
        layout.setStretch(1, 2)

        side_layout.addWidget(_named(QLabel("MODULE / 03  ·  TARGET ACQUISITION"), "moduleMarker"))
        side_layout.addWidget(_named(QLabel("当前目标"), "instrumentLabel"))
        self.track_tree = QTableWidget(0, 5)
        self.track_tree.setHorizontalHeaderLabels(("Track", "身份/候选", "结果", "自动流程", "分数"))
        self.track_tree.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.track_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.track_tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.track_tree.setAlternatingRowColors(True)
        self.track_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.track_tree.verticalHeader().setDefaultSectionSize(24)
        header = self.track_tree.horizontalHeader()
        for column, width in ((0, 58), (1, 105), (2, 125), (4, 62)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            self.track_tree.setColumnWidth(column, width)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        side_layout.addWidget(self.track_tree)

        enrollment = MissionGroupBox("人物注册", accent=SOLAR_YELLOW)
        enrollment_layout = QGridLayout(enrollment)
        self.automatic_registration = QCheckBox("自动注册新人物（默认开启）")
        self.automation_status = _named(QLabel("正在后台加载生产视觉模型…"), "mutedLabel")
        self.automation_status.setWordWrap(True)
        self.identity_id = QLineEdit("P1")
        self.candidate_id = QLineEdit()
        register_button = QPushButton("人工登记选中目标（兜底）")
        register_button.clicked.connect(self.register_selected)
        enrollment_layout.addWidget(self.automatic_registration, 0, 0, 1, 2)
        enrollment_layout.addWidget(self.automation_status, 1, 0, 1, 2)
        enrollment_layout.addWidget(QLabel("人工身份 ID"), 2, 0)
        enrollment_layout.addWidget(self.identity_id, 2, 1)
        enrollment_layout.addWidget(QLabel("跨事件候选键"), 3, 0)
        enrollment_layout.addWidget(self.candidate_id, 3, 1)
        enrollment_note = _named(
            QLabel("视觉身份尚未完成时，临时 Track 换视频/摄像头仍需保持不变；已有 P 会由 OSNet 自动重新绑定。"),
            "mutedLabel",
        )
        enrollment_note.setWordWrap(True)
        enrollment_layout.addWidget(enrollment_note, 4, 0, 1, 2)
        enrollment_layout.addWidget(register_button, 5, 0, 1, 2)
        workflow_note = _named(
            QLabel("当前流程：OSNet 先确认视觉身份并编号；随后按独立步态事件采集 GaitGraph2 原型。"),
            "mutedLabel",
        )
        workflow_note.setWordWrap(True)
        enrollment_layout.addWidget(workflow_note, 6, 0, 1, 2)
        side_layout.addWidget(enrollment)

        request_box = MissionGroupBox("外观吸收（自动；此处为人工兜底）", accent=RETRO_ORANGE)
        request_layout = QGridLayout(request_box)
        self.appearance_request_id = QLineEdit()
        apply_request_button = QPushButton("应用到后续帧")
        apply_request_button.clicked.connect(self.apply_appearance_request)
        self.pending_requests = _named(QLabel("无"), "mutedLabel")
        self.pending_requests.setWordWrap(True)
        request_layout.addWidget(QLabel("请求令牌"), 0, 0)
        request_layout.addWidget(self.appearance_request_id, 0, 1)
        request_layout.addWidget(apply_request_button, 1, 0, 1, 2)
        request_layout.addWidget(QLabel("当前待响应请求："), 2, 0, 1, 2)
        request_layout.addWidget(self.pending_requests, 3, 0, 1, 2)
        side_layout.addWidget(request_box)

        gallery = MissionGroupBox("当前正式身份", accent=NASA_BLUE)
        gallery_layout = QVBoxLayout(gallery)
        self.gallery_label = QLabel("无")
        clear_button = _named(QPushButton("清除现有 ID（重新建库）"), "dangerButton")
        clear_button.clicked.connect(self.clear_existing_ids)
        gallery_layout.addWidget(self.gallery_label)
        gallery_layout.addWidget(clear_button)
        side_layout.addWidget(gallery)
        side_layout.addStretch(1)
        self.side_scrollbar.setStyleSheet("")

    def _build_parameter_page(self) -> None:
        self.parameter_layout = QVBoxLayout(self.parameter_page)
        self.parameter_layout.setContentsMargins(24, 14, 24, 14)
        self.parameter_layout.setSpacing(10)
        self._rebuild_parameter_page()

    def _rebuild_parameter_page(self) -> None:
        self._clear_layout(self.parameter_layout)
        self.parameter_vars.clear()
        self.parameter_entries.clear()
        self.parameter_scales.clear()
        self.parameter_specs.clear()
        self._scale_steps.clear()
        if self.pipeline is None:
            self.parameter_layout.addWidget(_named(QLabel("生产视觉模型加载完成后，这里会显示实时参数。"), "mutedLabel"))
            self.parameter_layout.addStretch(1)
            return

        toolbar = QHBoxLayout()
        apply_button = _named(QPushButton("应用到运行中"), "primaryButton")
        reload_button = QPushButton("重新读取当前值")
        defaults_button = QPushButton("填入默认值（未应用）")
        apply_button.clicked.connect(self.apply_runtime_parameters)
        reload_button.clicked.connect(self.reload_runtime_parameters)
        defaults_button.clicked.connect(self.restore_default_parameters)
        toolbar.addWidget(apply_button)
        toolbar.addWidget(reload_button)
        toolbar.addWidget(defaults_button)
        self.parameter_status = _named(QLabel("参数尚未修改"), "mutedLabel")
        toolbar.addWidget(self.parameter_status)
        toolbar.addStretch(1)
        self.parameter_layout.addLayout(toolbar)
        description = _named(
            QLabel(
                "所有输入会先整组校验，再在采集线程的帧边界一次性生效。"
                "可通过滑条调节，也可直接在输入框中精确编辑。"
                "参数仅影响本次程序运行；生产阈值应以目标摄像头验证集为依据。"
            ),
            "parameterDescription",
        )
        description.setWordWrap(True)
        self.parameter_layout.addWidget(description)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.parameter_canvas = scroll
        inner = _named(QWidget(), "parameterSurface")
        grid = QGridLayout(inner)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        scroll.setWidget(inner)
        self.parameter_layout.addWidget(scroll, 1)

        sections: list[str] = []
        for spec in self.pipeline.runtime_parameter_specs:
            if spec.section not in sections:
                sections.append(spec.section)
        for section_index, section in enumerate(sections):
            accent = RETRO_ORANGE if section_index % 2 == 0 else NASA_BLUE
            group = MissionGroupBox(section, accent=accent)
            group.setObjectName("modulePanel")
            group_layout = QGridLayout(group)
            group_layout.setContentsMargins(7, 8, 7, 6)
            group_layout.setHorizontalSpacing(5)
            group_layout.setVerticalSpacing(5)
            group_layout.setColumnStretch(2, 1)
            group_layout.setColumnStretch(3, 2)
            grid.addWidget(
                group,
                section_index // 2,
                section_index % 2,
                Qt.AlignmentFlag.AlignTop,
            )
            module_label = _named(
                QLabel(
                    f"MODULE / {section_index + 4:02d}  ·  "
                    f"{MODULE_SUBTITLES.get(section, 'RUNTIME CONTROL')}"
                ),
                "moduleMarker",
            )
            group_layout.addWidget(module_label, 0, 0, 1, 4)
            rows = [item for item in self.pipeline.runtime_parameter_specs if item.section == section]
            for row_index, spec in enumerate(rows, start=1):
                variable = QLineEdit()
                variable.setAlignment(Qt.AlignmentFlag.AlignCenter)
                variable.setMaximumWidth(92)
                scale = InstrumentSlider(Qt.Orientation.Horizontal)
                scale.setRange(0, 1000)
                scale.setMinimumWidth(150)
                detail = QWidget()
                detail_layout = QVBoxLayout(detail)
                detail_layout.setContentsMargins(0, 0, 0, 0)
                detail_layout.setSpacing(1)
                range_label = _named(QLabel(f"RANGE  {spec.minimum:g}—{spec.maximum:g}"), "rangeLabel")
                detail_label = _named(QLabel(spec.description), "mutedLabel")
                detail_label.setWordWrap(True)
                detail_layout.addWidget(range_label)
                detail_layout.addWidget(detail_label)
                self.parameter_vars[spec.key] = variable
                self.parameter_entries[spec.key] = variable
                self.parameter_scales[spec.key] = scale
                self.parameter_specs[spec.key] = spec
                self._scale_steps[spec.key] = 1000
                variable.textChanged.connect(lambda _text, key=spec.key: self._parameter_var_changed(key))
                scale.valueChanged.connect(lambda value, key=spec.key: self._parameter_scale_changed(key, value))
                group_layout.addWidget(QLabel(spec.label), row_index, 0)
                group_layout.addWidget(variable, row_index, 1)
                group_layout.addWidget(scale, row_index, 2)
                group_layout.addWidget(detail, row_index, 3)
        self._load_runtime_parameter_state(self._runtime_parameter_state)

    @staticmethod
    def _clear_layout(layout: QVBoxLayout | QHBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                VerifierWindow._clear_layout(child_layout)  # type: ignore[arg-type]

    def _set_preload_ui_visible(self, visible: bool) -> None:
        """Show only while the existing backend preload lifecycle is active.

        The progress widgets remain ordinary presentation widgets.  Removing
        them from the grid after completion is intentional: hiding a child
        alone would leave an empty row in the source-control panel.
        """

        layout = getattr(self, "_preload_controls_layout", None)
        if layout is None:
            return
        if visible:
            if layout.indexOf(self.preload_phase) < 0:
                layout.addWidget(self.preload_phase, 2, 0, 1, 3)
            if layout.indexOf(self.preload_progress) < 0:
                layout.addWidget(self.preload_progress, 2, 3, 1, 9)
            self.preload_phase.show()
            self.preload_progress.show()
            return

        layout.removeWidget(self.preload_phase)
        layout.removeWidget(self.preload_progress)
        self.preload_phase.hide()
        self.preload_progress.hide()
        layout.setRowMinimumHeight(2, 0)
        layout.setRowStretch(2, 0)
        self._preload_controls.updateGeometry()

    def _publish_preload(self, text: str, progress: float) -> None:
        if not self._preload_cancel.is_set():
            self._preload_messages.put(("stage", text, progress, None))

    def _preload_backend(self, backend: str) -> None:
        try:
            vision = build_vision_adapter(backend, preload=True, on_stage=self._publish_preload)
        except Exception as error:
            self._preload_messages.put(("error", f"视觉后端加载失败：{error}", 1.0, error))
            return
        self._preload_messages.put(("done", "模型预加载完成，正在打开界面…", 1.0, vision))

    def _poll_preload(self) -> None:
        if self._closed:
            return
        while True:
            try:
                kind, text, progress, payload = self._preload_messages.get_nowait()
            except queue.Empty:
                break
            self.preload_progress.setValue(int(progress * 100))
            self.preload_phase.setText(text)
            self.status.setText(text)
            if kind == "done":
                self.vision = payload
                self._preload_timer.stop()
                self._finish_runtime_initialization()
                return
            if kind == "error":
                self._preload_timer.stop()
                self._preload_failed(text)
                return

    def _preload_failed(self, text: str) -> None:
        self.preload_phase.setText("模型加载失败")
        self.status.setText(text)
        QMessageBox.critical(self, "视觉后端加载失败", text)

    def _finish_runtime_initialization(self) -> None:
        if self._closed or self.vision is None:
            return
        automatic_capable = bool(getattr(self.vision, "supports_automatic_registration", False))
        self.pipeline = VideoVerifierPipeline(
            self.verifier,
            self.vision,
            automation_policy=AutomationPolicy(enabled=automatic_capable),
            appearance_first=True,
        )
        self.worker = FrameWorker(self.pipeline)
        self._runtime_parameter_state = self.pipeline.runtime_parameter_state()
        self.automatic_registration.setChecked(automatic_capable)
        self.automation_status.setText(
            "自动注册：开启，等待人物进入画面"
            if automatic_capable
            else "诊断后端：自动注册已安全关闭"
        )
        self.backend_status.setText(
            f"视觉后端：{getattr(self.vision, 'backend_status', type(self.vision).__name__)}"
        )
        self.status.setText("请选择摄像头或视频文件，然后点击“开始”")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self._set_preload_ui_visible(False)
        self._rebuild_parameter_page()
        warning = getattr(self.vision, "startup_warning", None)
        if warning:
            QTimer.singleShot(100, lambda: self._show_startup_warning(warning))
        self._message_timer.start(GUI_POLL_INTERVAL_MS)

    def _show_startup_warning(self, warning: str) -> None:
        if not self._closed:
            QMessageBox.warning(self, "视觉后端降级", warning)

    def _show_video_standby(self) -> None:
        self._photo = None
        self.video_label.clear()
        self.video_label.setText(VIDEO_STANDBY_TEXT)
        self.video_stack.setCurrentWidget(self.video_standby)

    def _hide_video_standby(self) -> None:
        self.video_stack.setCurrentWidget(self.video_label)
        self.video_label.setText("")

    def _render_video_frame(self) -> None:
        if self._last_frame is None:
            return
        frame = self._last_frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()
        self._photo = QPixmap.fromImage(image)
        scaled = self._photo.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        if self._last_frame is not None and self.video_stack.currentWidget() is self.video_label:
            self._render_video_frame()

    def _source_mode_changed(self) -> None:
        enabled = self.file_radio.isChecked()
        self.video_path.setEnabled(enabled)
        self.browse_button.setEnabled(enabled)
        self.video_repeat_count.setEnabled(enabled)

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv);;所有文件 (*.*)",
        )
        if path:
            self.video_path.setText(path)

    def _source_spec(self) -> SourceSpec:
        candidate_id = self.candidate_id.text().strip() or None
        if self.file_radio.isChecked():
            path = self.video_path.text().strip()
            if not path:
                raise ValueError("请先选择视频文件")
            try:
                repeat_count = int(self.video_repeat_count.text().strip())
            except ValueError as error:
                raise ValueError("视频重复学习次数必须是正整数") from error
            if repeat_count < 1:
                raise ValueError("视频重复学习次数必须是正整数")
            return SourceSpec("file", path, Path(path).name, candidate_id, repeat_count)
        try:
            index = int(self.camera_index.text().strip())
        except ValueError as error:
            raise ValueError("摄像头设备号必须是整数") from error
        label = self.camera_id.text().strip() or f"camera-{index}"
        return SourceSpec("camera", index, label, candidate_id)

    def start(self) -> None:
        if self.worker is None:
            self.status.setText("模型仍在后台加载，请稍候")
            return
        try:
            spec = self._source_spec()
            self.worker.start(spec)
            self._show_video_standby()
            repeat_text = (
                f"（自动重复学习 {spec.repeat_count} 次）"
                if spec.kind == "file" and spec.repeat_count > 1
                else ""
            )
            self.status.setText(f"正在打开：{spec.label}{repeat_text}")
        except Exception as error:
            QMessageBox.critical(self, "无法开始", str(error))

    def stop(self) -> None:
        if self.worker is None:
            self.status.setText("模型仍在后台加载，请稍候")
            return
        self.worker.stop()
        self.status.setText("已停止")

    def _toggle_automatic_registration(self, checked: bool) -> None:
        if self.worker is None:
            self.automatic_registration.setChecked(False)
            self.status.setText("模型仍在后台加载，请稍候")
            return
        if checked and not bool(getattr(self.vision, "supports_automatic_registration", False)):
            self.automatic_registration.setChecked(False)
            QMessageBox.warning(self, "自动注册已阻止", "HOG 诊断后端不能提供强步态证据，请使用 production 后端。")
            return
        self.worker.set_automatic_registration(checked)
        self.automation_status.setText(
            "自动注册：开启，OSNet 将先确认视觉身份，随后学习步态原型"
            if checked
            else "自动注册：关闭；识别和外观令牌响应仍继续"
        )

    def register_selected(self) -> None:
        if self.worker is None:
            self.status.setText("模型仍在后台加载，请稍候")
            return
        identity_id = self.identity_id.text().strip()
        if not identity_id:
            QMessageBox.warning(self, "缺少身份 ID", "请输入身份 ID，例如 P001")
            return
        selected = self.track_tree.selectionModel().selectedRows()
        track_id = int(self.track_tree.item(selected[0].row(), 0).text()) if selected else None
        self.worker.register_identity(identity_id, track_id)
        self.status.setText("登记请求已排队，将在采集线程安全执行")

    def clear_existing_ids(self) -> None:
        if self.worker is None or self.pipeline is None:
            self.status.setText("模型仍在后台加载，请稍候")
            return
        answer = QMessageBox.question(
            self,
            "确认清除身份",
            "将清除全部视觉身份、步态原型、事件和审计记录。\n清除前会自动创建数据库备份，是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.worker.stop()
        store = self.verifier.store
        backup_path: Path | None = None
        if store.path != ":memory:":
            database = Path(store.path)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_path = database.with_name(f"{database.stem}-before-clear-{stamp}{database.suffix}")
            store.backup_to(str(backup_path))
        try:
            self.pipeline.clear_gallery()
        except Exception as error:
            QMessageBox.critical(self, "清除失败", f"身份数据未完成清除：{error}")
            return
        self.track_tree.setRowCount(0)
        self._table_rows.clear()
        self.gallery_label.setText("无")
        self.pending_requests.setText("无")
        self.automation_status.setText("自动注册：开启，等待人物进入画面")
        self.status.setText("已清除全部身份数据" + (f"；备份：{backup_path.name}" if backup_path is not None else ""))

    def apply_appearance_request(self) -> None:
        if self.worker is None:
            self.status.setText("模型仍在后台加载，请稍候")
            return
        request_id = self.appearance_request_id.text().strip()
        selected = self.track_tree.selectionModel().selectedRows()
        track_id = int(self.track_tree.item(selected[0].row(), 0).text()) if selected else None
        self.worker.set_appearance_request(request_id or None, track_id)
        self.status.setText(
            "已清除外观响应令牌"
            if not request_id
            else (f"外观响应令牌已绑定到 Track {track_id}" if track_id is not None else "外观响应令牌将应用到后续目标")
        )

    @staticmethod
    def _track_visual_tag(kind: str) -> str:
        if kind in {"formal_match", "visual_identity_created", "appearance_response_accepted"}:
            return "verified"
        if kind == "conflict":
            return "conflict"
        if kind in {"unknown", "ambiguous"}:
            return "unknown"
        if kind == "appearance_requested":
            return "waiting"
        if kind in {"deferred", "need_more_data", "candidate_created", "candidate_updated"}:
            return "learning"
        return ""

    def _update_frame(self, result: FrameResult) -> None:
        self.backend_status.setText(
            f"视觉后端：{getattr(self.vision, 'backend_status', type(self.vision).__name__)}"
        )
        try:
            self._last_frame = result.frame_bgr
            self._render_video_frame()
            self._hide_video_standby()
        except Exception as error:
            self.status.setText(f"画面显示失败：{error}")
        existing = set(self._table_rows)
        seen: set[str] = set()
        request_message = ""
        automation_messages: list[str] = []
        for track in result.tracks:
            decision = track.decision
            subject = decision.identity_id or decision.candidate_id or "-"
            score = "-" if decision.score is None else f"{decision.score:.3f}"
            item_id = str(track.track_id)
            row = self._table_rows.get(item_id)
            if row is None:
                row = self.track_tree.rowCount()
                self.track_tree.insertRow(row)
                self._table_rows[item_id] = row
            values = (str(track.track_id), subject, decision.kind.value, track.automation.message, score)
            tag = self._track_visual_tag(decision.kind.value)
            for column, value in enumerate(values):
                item = self.track_tree.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.track_tree.setItem(row, column, item)
                item.setText(value)
                color = {
                    "verified": STATUS_GREEN,
                    "learning": NASA_BLUE,
                    "waiting": RETRO_ORANGE,
                    "unknown": STATUS_PENDING,
                    "conflict": SIGNAL_RED,
                }.get(tag)
                if color:
                    item.setForeground(QBrush(QColor(color)))
            seen.add(item_id)
            automation_messages.append(f"T{track.track_id}: {track.automation.message}")
            if decision.appearance_request_id:
                self.appearance_request_id.setText(decision.appearance_request_id)
                request_message = track.automation.message
            if decision.kind.value == "appearance_response_accepted":
                self.appearance_request_id.clear()
                request_message = track.automation.message
        for item_id in existing - seen:
            row = self._table_rows.pop(item_id, None)
            if row is not None:
                self.track_tree.removeRow(row)
        self._table_rows = {
            self.track_tree.item(row, 0).text(): row
            for row in range(self.track_tree.rowCount())
            if self.track_tree.item(row, 0) is not None
        }
        enabled = self.pipeline is not None and self.pipeline.automatic_registration_enabled
        self.automation_status.setText(
            " | ".join(automation_messages[:2])
            if automation_messages
            else ("自动注册：开启，等待人物进入画面" if enabled else "自动注册：关闭")
        )
        self.gallery_label.setText(", ".join(result.formal_identities) or "无")
        self.pending_requests.setText("\n".join(result.pending_request_ids[:3]) if result.pending_request_ids else "无")
        self.status.setText(
            f"帧 {result.frame_index} | 目标 {len(result.tracks)} | 处理 {result.processing_seconds * 1000:.0f} ms"
        )
        if request_message:
            self.status.setText(request_message)

    def _poll_messages(self) -> None:
        if self._closed or self.worker is None:
            return
        latest_frame: FrameMessage | None = None
        while True:
            try:
                message = self.worker.messages.get_nowait()
            except queue.Empty:
                break
            if isinstance(message, FrameMessage):
                latest_frame = message
            elif isinstance(message, RegistrationMessage):
                self.status.setText(message.text)
                (QMessageBox.information if message.success else QMessageBox.critical)(self, "登记结果", message.text)
            elif isinstance(message, ParameterUpdateMessage):
                self.parameter_status.setText(message.text)
                self.status.setText(message.text)
                if message.success and message.state is not None:
                    self._load_runtime_parameter_state(message.state)
                elif not message.success:
                    QMessageBox.critical(self, "参数应用失败", message.text)
            elif isinstance(message, StatusMessage):
                self.status.setText(message.text)
        if latest_frame is not None:
            self._update_frame(latest_frame.result)

    def _parameter_var_changed(self, key: str) -> None:
        if self._parameter_syncing:
            return
        spec = self.parameter_specs.get(key)
        scale = self.parameter_scales.get(key)
        variable = self.parameter_vars.get(key)
        if spec is None or scale is None or variable is None:
            return
        try:
            value = spec.coerce(variable.text())
        except ValueError:
            return
        self._parameter_syncing = True
        try:
            scale.setValue(self._scale_value(spec, float(value)))
        finally:
            self._parameter_syncing = False
        self.parameter_status.setText("参数已修改；点击“应用到运行中”后生效")

    def _parameter_scale_changed(self, key: str, raw_value: int) -> None:
        if self._parameter_syncing:
            return
        spec = self.parameter_specs.get(key)
        variable = self.parameter_vars.get(key)
        scale = self.parameter_scales.get(key)
        if spec is None or variable is None or scale is None:
            return
        value = float(spec.minimum) + (float(raw_value) / 1000.0) * (float(spec.maximum) - float(spec.minimum))
        if spec.kind == "int":
            value = float(round(value))
        value = min(max(value, float(spec.minimum)), float(spec.maximum))
        self._parameter_syncing = True
        try:
            scale.setValue(self._scale_value(spec, value))
            variable.setText(spec.format(value))
        finally:
            self._parameter_syncing = False
        self.parameter_status.setText("参数已修改；点击“应用到运行中”后生效")

    @staticmethod
    def _scale_value(spec: RuntimeParameterSpec, value: float) -> int:
        span = float(spec.maximum) - float(spec.minimum)
        return 0 if span <= 0 else int(round((value - float(spec.minimum)) / span * 1000))

    def _load_runtime_parameter_state(self, state: RuntimeParameterState | None) -> None:
        if state is None or self.pipeline is None:
            return
        self._runtime_parameter_state = state
        available = set(state.available_keys)
        for spec in self.pipeline.runtime_parameter_specs:
            variable = self.parameter_vars.get(spec.key)
            scale = self.parameter_scales.get(spec.key)
            if variable is None or scale is None:
                continue
            variable.setEnabled(spec.key in available)
            scale.setEnabled(spec.key in available)
            self._parameter_syncing = True
            try:
                if spec.key in available:
                    value = state.values[spec.key]
                    variable.setText(spec.format(value))
                    scale.setValue(self._scale_value(spec, float(value)))
                else:
                    variable.setText("当前后端不可用")
            finally:
                self._parameter_syncing = False
        self.parameter_status.setText(f"当前运行时参数版本：{state.revision}")

    def apply_runtime_parameters(self) -> None:
        if self.worker is None or self._runtime_parameter_state is None:
            self.status.setText("模型仍在后台加载，请稍候")
            return
        available = set(self._runtime_parameter_state.available_keys)
        values = {key: variable.text() for key, variable in self.parameter_vars.items() if key in available}
        self.worker.set_runtime_parameters(values)
        self.parameter_status.setText("参数更新已排队，将在下一帧前整组生效")

    def reload_runtime_parameters(self) -> None:
        if self.worker is None:
            self.status.setText("模型仍在后台加载，请稍候")
            return
        self.worker.set_runtime_parameters({})
        self.parameter_status.setText("正在读取当前生效值…")

    def restore_default_parameters(self) -> None:
        if self.pipeline is None or self._runtime_parameter_state is None:
            self.status.setText("模型仍在后台加载，请稍候")
            return
        defaults = self.pipeline.runtime_parameter_defaults()
        available = set(self._runtime_parameter_state.available_keys)
        specs = {item.key: item for item in self.pipeline.runtime_parameter_specs}
        self._parameter_syncing = True
        try:
            for key in available:
                self.parameter_vars[key].setText(specs[key].format(defaults[key]))
        finally:
            self._parameter_syncing = False
        self.parameter_status.setText("已填入默认值；点击“应用到运行中”后才会生效")

    def close(self) -> None:  # noqa: D401 - Qt close slot
        if self._closed:
            return
        self._closed = True
        self._preload_cancel.set()
        if hasattr(self, "_preload_timer"):
            self._preload_timer.stop()
        self._message_timer.stop()
        if self.worker is not None:
            self.worker.stop()
        self.verifier.close()
        super().close()


def launch_gui(
    database_path: str = "data/verifier-production-v1.sqlite3",
    vision_backend: str = "production",
) -> int:
    """Start the Qt GUI; all heavy work remains in the existing worker layer."""

    app = QApplication.instance() or QApplication(sys.argv)
    window = VerifierWindow(database_path, vision_backend)
    window.show()
    return app.exec()


__all__ = ["VerifierWindow", "launch_gui"]
