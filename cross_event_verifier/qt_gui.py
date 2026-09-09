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
import os
import queue
import sys
import threading
from typing import Any

import cv2

from .automation import AutomationPolicy
from .engine import CrossEventVerifier
from .ground_control import (
    GroundControlPoint,
    HomographyCheck,
    compute_ground_homography,
    load_ground_control_file,
    parse_ground_control_point,
    save_ground_control_file,
)
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
from .transports import (
    action_publisher_from_environment,
    normalize_transport_mode,
    transport_label,
)
from .vision_factory import build_vision_adapter


try:  # Keep the import failure local to the optional Qt adapter.
    from PyQt6.QtCore import QLineF, QTimer, Qt, QRectF, pyqtSignal
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
        QComboBox,
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


from .actions import ActionType, MANUAL_ACTION_TYPES


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


class GcpCanvas(QLabel):
    """用现有任务画布承载固定画面和可点击的地面控制点。"""

    point_clicked = pyqtSignal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._source_pixmap: QPixmap | None = None
        self._source_size = (0, 0)
        self._points: tuple[GroundControlPoint, ...] = ()
        self._display_rect = QRectF()
        self.setMinimumSize(320, 240)

    def set_frame(self, frame: object) -> None:
        if not hasattr(frame, "shape"):
            raise ValueError("标定画面必须是 OpenCV 图像")
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(
            rgb.data,
            width,
            height,
            int(rgb.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        self._source_pixmap = QPixmap.fromImage(image)
        self._source_size = (int(width), int(height))
        self.update()

    def set_points(self, points: tuple[GroundControlPoint, ...]) -> None:
        self._points = points
        self.update()

    def display_size(self) -> tuple[int, int] | None:
        if self._source_size == (0, 0):
            return None
        return self._source_size

    def _fit_rect(self) -> QRectF:
        if self._source_pixmap is None:
            return QRectF()
        source_width, source_height = self._source_size
        scale = min(
            self.width() / max(source_width, 1),
            self.height() / max(source_height, 1),
            1.0,
        )
        width = source_width * scale
        height = source_height * scale
        return QRectF((self.width() - width) / 2.0, (self.height() - height) / 2.0, width, height)

    def paintEvent(self, event: Any) -> None:  # pragma: no cover - visual drawing
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(PANEL_CREAM))
        self._display_rect = self._fit_rect()
        if self._source_pixmap is None:
            painter.setPen(QPen(QColor(INDUSTRIAL_NAVY), 1))
            painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "加载固定机位画面后，在此标记地面控制点")
            painter.end()
            return
        painter.drawPixmap(self._display_rect, self._source_pixmap, QRectF(self._source_pixmap.rect()))
        source_width, source_height = self._source_size
        scale_x = self._display_rect.width() / max(source_width, 1)
        scale_y = self._display_rect.height() / max(source_height, 1)
        for point in self._points:
            x = self._display_rect.left() + point.image_u * scale_x
            y = self._display_rect.top() + point.image_v * scale_y
            painter.setPen(QPen(QColor(RETRO_ORANGE), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QRectF(x - 5, y - 5, 10, 10))
            painter.setPen(QPen(QColor(INDUSTRIAL_NAVY), 1))
            painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
            painter.drawText(QRectF(x + 8, y - 16, 80, 20), point.point_id)
        painter.end()

    def mousePressEvent(self, event: Any) -> None:  # pragma: no cover - native UI event
        if (
            event.button() != Qt.MouseButton.LeftButton
            or self._source_size == (0, 0)
            or not self._display_rect.contains(event.position())
        ):
            return
        source_width, source_height = self._source_size
        u = (event.position().x() - self._display_rect.left()) * source_width / max(self._display_rect.width(), 1.0)
        v = (event.position().y() - self._display_rect.top()) * source_height / max(self._display_rect.height(), 1.0)
        self.point_clicked.emit(float(u), float(v))


class VerifierWindow(QMainWindow):
    """Qt presentation adapter over the existing worker and verification pipeline."""

    def __init__(
        self,
        database_path: str = "data/verifier-production-v1.sqlite3",
        vision_backend: str = "production",
        action_classifier_path: Path | None = None,
        posec3d_checkpoint_path: Path | None = None,
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
        self.action_classifier_path = (
            Path(action_classifier_path) if action_classifier_path is not None else None
        )
        self.posec3d_checkpoint_path = (
            Path(posec3d_checkpoint_path)
            if posec3d_checkpoint_path is not None
            else None
        )
        self._parameter_syncing = False
        self._runtime_parameter_state: RuntimeParameterState | None = None
        self.parameter_vars: dict[str, QLineEdit] = {}
        self.parameter_entries: dict[str, QLineEdit] = {}
        self.parameter_scales: dict[str, QSlider] = {}
        self.parameter_specs: dict[str, RuntimeParameterSpec] = {}
        self._scale_steps: dict[str, int] = {}
        self._table_rows: dict[str, int] = {}
        self.gcp_camera_id = QLineEdit("camera-1")
        self.gcp_coordinate_frame = QLineEdit("ground-meters")
        self.gcp_unit = QLineEdit("m")
        self.gcp_point_id = QLineEdit("P1")
        self.gcp_image_u = QLineEdit()
        self.gcp_image_v = QLineEdit()
        self.gcp_world_x = QLineEdit()
        self.gcp_world_y = QLineEdit()
        self.gcp_status: QLabel | None = None
        self.gcp_matrix: QLabel | None = None
        self.gcp_rows: dict[str, GroundControlPoint] = {}
        self.gcp_check: HomographyCheck | None = None
        self.gcp_image_size_value: tuple[int, int] | None = None
        raw_transport = os.getenv("CROSS_EVENT_TRANSPORT", "mqtt").strip()
        try:
            transport = normalize_transport_mode(raw_transport).value
        except ValueError:
            transport = "mqtt"
        self.transport_mode = transport_label(transport)
        self._active_transport_mode: str | None = None
        self.transport_status: QLabel | None = None
        self.manual_action_target = ""
        self.manual_action_type = MANUAL_ACTION_TYPES[0].value
        self._manual_action_targets: dict[str, int] = {}
        self._manual_action_syncing = False

        database = Path(database_path)
        if str(database) != ":memory:":
            database.parent.mkdir(parents=True, exist_ok=True)
        self.verifier = CrossEventVerifier(store=SqliteStore(str(database)))
        self._build_layout()
        self._message_timer = QTimer(self)
        self._message_timer.timeout.connect(self._poll_messages)
        selected = vision_backend.strip().lower()
        if selected == "demo":
            self.vision = build_vision_adapter(
                vision_backend,
                action_classifier_path=self.action_classifier_path,
                posec3d_checkpoint_path=self.posec3d_checkpoint_path,
            )
            self._finish_runtime_initialization()
        else:
            self._preload_timer = QTimer(self)
            self._preload_timer.timeout.connect(self._poll_preload)
            self._preload_timer.start(50)
            self._preload_thread = threading.Thread(
                target=self._preload_backend,
                args=(
                    vision_backend,
                    self.action_classifier_path,
                    self.posec3d_checkpoint_path,
                ),
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
        controls_layout.addWidget(self.backend_status, 1, 3, 1, 5)
        controls_layout.addWidget(QLabel("事件传输"), 1, 8)
        self.transport_combo = QComboBox()
        self.transport_combo.addItems(("MQTT", "BLE Light"))
        self.transport_combo.setCurrentText(self.transport_mode)
        self.transport_combo.setMaximumWidth(110)
        self.transport_combo.currentTextChanged.connect(self._transport_mode_changed)
        controls_layout.addWidget(self.transport_combo, 1, 9)
        self.transport_status = _named(
            QLabel(f"{self.transport_mode}：未配置"), "mutedLabel"
        )
        controls_layout.addWidget(self.transport_status, 1, 10, 1, 2)
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
        self.gcp_page = QWidget()
        self.notebook.addTab(self.monitor_page, "01 / 实时识别")
        self.notebook.addTab(self.parameter_page, "02 / 实时参数")
        self.notebook.addTab(self.gcp_page, "03 / 地面控制点")
        outer.addWidget(self.notebook, 1)
        self._build_monitor_page()
        self._build_parameter_page()
        self._build_gcp_page()

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
        self.track_tree = QTableWidget(0, 6)
        self.track_tree.setHorizontalHeaderLabels(("Track", "动作", "身份/候选", "结果", "自动流程", "分数"))
        self.track_tree.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.track_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.track_tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.track_tree.setAlternatingRowColors(True)
        self.track_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.track_tree.verticalHeader().setDefaultSectionSize(24)
        header = self.track_tree.horizontalHeader()
        for column, width in ((0, 58), (1, 86), (2, 105), (3, 125), (5, 62)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            self.track_tree.setColumnWidth(column, width)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        side_layout.addWidget(self.track_tree)
        self.track_tree.itemSelectionChanged.connect(self._sync_manual_action_target)

        action_box = MissionGroupBox("手动动作触发", accent=RETRO_ORANGE)
        action_layout = QGridLayout(action_box)
        self.manual_action_target_combo = QComboBox()
        self.manual_action_target_combo.setEditable(False)
        self.manual_action_target_combo.setEnabled(False)
        self.manual_action_combo = QComboBox()
        self.manual_action_combo.addItems(
            tuple(item.value for item in MANUAL_ACTION_TYPES)
        )
        self.manual_action_combo.setEnabled(False)
        self.manual_action_button = QPushButton("按所选动作触发")
        self.manual_action_button.setEnabled(False)
        self.manual_action_button.clicked.connect(self.trigger_manual_action)
        action_layout.addWidget(QLabel("目标（Track / 身份）"), 0, 0)
        action_layout.addWidget(self.manual_action_target_combo, 0, 1)
        action_layout.addWidget(QLabel("动作"), 1, 0)
        action_layout.addWidget(self.manual_action_combo, 1, 1)
        action_layout.addWidget(self.manual_action_button, 2, 0, 1, 2)
        action_note = _named(
            QLabel("动作只在下一帧再次确认身份为 KNOWN 后发布；不会写入步态库。"),
            "mutedLabel",
        )
        action_note.setWordWrap(True)
        action_layout.addWidget(action_note, 3, 0, 1, 2)
        action_layout.setColumnStretch(1, 1)
        side_layout.addWidget(action_box)

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

    def _build_gcp_page(self) -> None:
        layout = QHBoxLayout(self.gcp_page)
        layout.setContentsMargins(22, 14, 22, 18)
        layout.setSpacing(10)

        preview_panel = _named(MissionPanelFrame(accent=NASA_BLUE), "missionPanelFrame")
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(1, 1, 1, 1)
        self.gcp_canvas = GcpCanvas()
        self.gcp_canvas.point_clicked.connect(self._gcp_canvas_click)
        preview_layout.addWidget(self.gcp_canvas, 1)
        preview_layout.addWidget(
            _named(
                QLabel("点击地面控制点获取像素坐标；然后在右侧录入对应世界坐标。"),
                "mutedLabel",
            )
        )
        layout.addWidget(preview_panel, 3)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        metadata = MissionGroupBox("标定元数据", accent=NASA_BLUE)
        metadata_layout = QGridLayout(metadata)
        metadata_layout.setContentsMargins(7, 8, 7, 6)
        metadata_layout.addWidget(QLabel("摄像头 ID"), 0, 0)
        metadata_layout.addWidget(self.gcp_camera_id, 0, 1)
        metadata_layout.addWidget(QLabel("世界坐标系"), 1, 0)
        metadata_layout.addWidget(self.gcp_coordinate_frame, 1, 1)
        metadata_layout.addWidget(QLabel("单位"), 2, 0)
        metadata_layout.addWidget(self.gcp_unit, 2, 1)
        metadata_layout.addWidget(QLabel("画面尺寸"), 3, 0)
        self.gcp_image_size = _named(QLabel("未加载"), "mutedLabel")
        metadata_layout.addWidget(self.gcp_image_size, 3, 1)
        metadata_layout.setColumnStretch(1, 1)
        right_layout.addWidget(metadata)

        point_form = MissionGroupBox("控制点标记 / 录入", accent=RETRO_ORANGE)
        form_layout = QGridLayout(point_form)
        form_layout.setContentsMargins(7, 8, 7, 6)
        for row, column, label, field in (
            (0, 0, "点 ID", self.gcp_point_id),
            (0, 2, "像素 U", self.gcp_image_u),
            (1, 0, "像素 V", self.gcp_image_v),
            (1, 2, "世界 X", self.gcp_world_x),
            (2, 0, "世界 Y", self.gcp_world_y),
        ):
            form_layout.addWidget(QLabel(label), row, column)
            form_layout.addWidget(field, row, column + 1)
        add_button = QPushButton("添加 / 更新")
        delete_button = QPushButton("删除选中")
        add_button.clicked.connect(self._gcp_upsert)
        delete_button.clicked.connect(self._gcp_delete)
        form_layout.addWidget(add_button, 2, 2)
        form_layout.addWidget(delete_button, 2, 3)
        form_layout.setColumnStretch(1, 1)
        form_layout.setColumnStretch(3, 1)
        right_layout.addWidget(point_form)

        table_box = MissionGroupBox("控制点清单", accent=NASA_BLUE)
        table_layout = QVBoxLayout(table_box)
        table_layout.setContentsMargins(7, 8, 7, 6)
        self.gcp_table = QTableWidget(0, 6)
        self.gcp_table.setHorizontalHeaderLabels(("ID", "U", "V", "X", "Y", "校核"))
        self.gcp_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.gcp_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.gcp_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.gcp_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.gcp_table.itemSelectionChanged.connect(self._gcp_select)
        table_layout.addWidget(self.gcp_table)
        right_layout.addWidget(table_box, 1)

        action_row = QHBoxLayout()
        load_button = QPushButton("加载画面…")
        import_button = QPushButton("导入标定…")
        export_button = _named(QPushButton("导出标定…"), "primaryButton")
        load_button.clicked.connect(self._gcp_load_image)
        import_button.clicked.connect(self._gcp_import)
        export_button.clicked.connect(self._gcp_export)
        action_row.addWidget(load_button)
        action_row.addWidget(import_button)
        action_row.addWidget(export_button)
        right_layout.addLayout(action_row)
        check_button = _named(QPushButton("计算并校核 Homography"), "primaryButton")
        clear_button = _named(QPushButton("清空控制点"), "dangerButton")
        check_button.clicked.connect(self._gcp_check_homography)
        clear_button.clicked.connect(self._gcp_clear)
        right_layout.addWidget(check_button)
        right_layout.addWidget(clear_button)
        self.gcp_status = _named(
            QLabel("请加载一张固定机位画面，然后在画布上点击地面点"),
            "mutedLabel",
        )
        self.gcp_status.setWordWrap(True)
        self.gcp_matrix = _named(QLabel("H 尚未校核"), "rangeLabel")
        self.gcp_matrix.setWordWrap(True)
        right_layout.addWidget(self.gcp_status)
        right_layout.addWidget(self.gcp_matrix)
        layout.addWidget(right, 2)

    def _gcp_next_id(self) -> str:
        index = 1
        while f"P{index}" in self.gcp_rows:
            index += 1
        return f"P{index}"

    def _gcp_current_point(self) -> GroundControlPoint:
        return parse_ground_control_point(
            self.gcp_point_id.text(),
            self.gcp_image_u.text(),
            self.gcp_image_v.text(),
            self.gcp_world_x.text(),
            self.gcp_world_y.text(),
        )

    def _gcp_upsert(self) -> None:
        try:
            point = self._gcp_current_point()
        except ValueError as error:
            QMessageBox.critical(self, "控制点录入失败", str(error))
            return
        self.gcp_rows[point.point_id] = point
        self.gcp_point_id.setText(self._gcp_next_id())
        self.gcp_check = None
        self.gcp_matrix.setText("H 尚未校核")
        self._gcp_refresh_table()
        self.gcp_canvas.set_points(tuple(self.gcp_rows.values()))
        self.gcp_status.setText(f"已记录 {point.point_id}；还可继续点击画面标记下一点")

    def _gcp_select(self) -> None:
        rows = self.gcp_table.selectionModel().selectedRows()
        if not rows:
            return
        point_ids = tuple(self.gcp_rows)
        row = rows[0].row()
        if row >= len(point_ids):
            return
        point = self.gcp_rows[point_ids[row]]
        self.gcp_point_id.setText(point.point_id)
        self.gcp_image_u.setText(f"{point.image_u:.3f}")
        self.gcp_image_v.setText(f"{point.image_v:.3f}")
        self.gcp_world_x.setText(f"{point.world_x:.3f}")
        self.gcp_world_y.setText(f"{point.world_y:.3f}")

    def _gcp_delete(self) -> None:
        rows = self.gcp_table.selectionModel().selectedRows()
        if not rows:
            self.gcp_status.setText("请先在清单中选择控制点")
            return
        point_ids = tuple(self.gcp_rows)
        row = rows[0].row()
        if row < len(point_ids):
            self.gcp_rows.pop(point_ids[row], None)
        self.gcp_check = None
        self.gcp_matrix.setText("H 尚未校核")
        self._gcp_refresh_table()
        self.gcp_canvas.set_points(tuple(self.gcp_rows.values()))
        self.gcp_status.setText("已删除选中控制点")

    def _gcp_refresh_table(self) -> None:
        self.gcp_table.setRowCount(0)
        for index, point in enumerate(self.gcp_rows.values()):
            self.gcp_table.insertRow(index)
            values = [point.point_id, f"{point.image_u:.2f}", f"{point.image_v:.2f}", f"{point.world_x:.3f}", f"{point.world_y:.3f}", "未校核"]
            if self.gcp_check is not None:
                inlier = self.gcp_check.inliers[index]
                values[5] = f"{'内点' if inlier else '外点'} {self.gcp_check.reprojection_errors[index]:.3f}"
            for column, value in enumerate(values):
                self.gcp_table.setItem(index, column, QTableWidgetItem(value))

    def _gcp_canvas_click(self, image_u: float, image_v: float) -> None:
        self.gcp_image_u.setText(f"{image_u:.3f}")
        self.gcp_image_v.setText(f"{image_v:.3f}")
        if not self.gcp_point_id.text().strip() or self.gcp_point_id.text() in self.gcp_rows:
            self.gcp_point_id.setText(self._gcp_next_id())
        self.gcp_status.setText(f"已标记像素坐标 ({image_u:.2f}, {image_v:.2f})；请输入世界坐标后添加")

    def _gcp_load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择固定机位画面", "", "图像 (*.png *.jpg *.jpeg *.bmp *.webp);;全部文件 (*)")
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            QMessageBox.critical(self, "画面加载失败", "无法读取所选图像")
            return
        self.gcp_canvas.set_frame(image)
        height, width = image.shape[:2]
        self.gcp_image_size_value = (int(width), int(height))
        self.gcp_image_size.setText(f"{width} × {height}")
        self.gcp_status.setText(f"已加载画面 {Path(path).name}；点击地面点开始标记")

    def _gcp_check_homography(self) -> HomographyCheck | None:
        try:
            check = compute_ground_homography(tuple(self.gcp_rows.values()))
        except ValueError as error:
            self.gcp_check = None
            self.gcp_matrix.setText("H 尚未校核")
            self.gcp_status.setText(str(error))
            QMessageBox.critical(self, "Homography 校核失败", str(error))
            return None
        self.gcp_check = check
        self._gcp_refresh_table()
        matrix = check.matrix
        rows = (matrix[0:3], matrix[3:6], matrix[6:9])
        self.gcp_matrix.setText("H = " + "; ".join(", ".join(f"{value:.5g}" for value in row) for row in rows))
        self.gcp_status.setText(
            f"校核通过：{len(self.gcp_rows)} 点，内点 {sum(check.inliers)}，"
            f"RMSE {check.rmse:.4f} {self.gcp_unit.text() or 'm'}，最大误差 {check.maximum_error:.4f}"
        )
        return check

    def _gcp_clear(self) -> None:
        self.gcp_rows.clear()
        self.gcp_check = None
        self.gcp_matrix.setText("H 尚未校核")
        self.gcp_point_id.setText("P1")
        self._gcp_refresh_table()
        self.gcp_canvas.set_points(())
        self.gcp_status.setText("已清空控制点")

    def _gcp_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "导入地面控制点标定", "", "GCP JSON (*.json);;全部文件 (*)")
        if not path:
            return
        try:
            payload = load_ground_control_file(path)
        except ValueError as error:
            QMessageBox.critical(self, "标定导入失败", str(error))
            return
        self.gcp_camera_id.setText(str(payload["camera_id"]))
        self.gcp_coordinate_frame.setText(str(payload["coordinate_frame"]))
        self.gcp_unit.setText(str(payload["unit"]))
        image_size = payload.get("image_size")
        if isinstance(image_size, list) and len(image_size) == 2:
            self.gcp_image_size_value = (int(image_size[0]), int(image_size[1]))
            self.gcp_image_size.setText(
                f"{self.gcp_image_size_value[0]} × {self.gcp_image_size_value[1]}"
            )
        self.gcp_rows = {point.point_id: point for point in payload["points"]}
        self.gcp_point_id.setText(self._gcp_next_id())
        self.gcp_check = None
        self._gcp_refresh_table()
        self.gcp_canvas.set_points(tuple(self.gcp_rows.values()))
        self._gcp_check_homography()
        self.gcp_status.setText(f"已导入 {Path(path).name}；请确认画面与标定文件属于同一机位")

    def _gcp_export(self) -> None:
        check = self.gcp_check or self._gcp_check_homography()
        if check is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出地面控制点标定", "ground_control_points.json", "GCP JSON (*.json)")
        if not path:
            return
        try:
            save_ground_control_file(
                path,
                camera_id=self.gcp_camera_id.text(),
                coordinate_frame=self.gcp_coordinate_frame.text(),
                unit=self.gcp_unit.text(),
                image_size=self.gcp_canvas.display_size() or self.gcp_image_size_value,
                points=tuple(self.gcp_rows.values()),
                check=check,
            )
        except ValueError as error:
            QMessageBox.critical(self, "标定导出失败", str(error))
            return
        self.gcp_status.setText(f"已导出标定文件：{Path(path).name}")

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

    def _preload_backend(
        self,
        backend: str,
        action_classifier_path: Path | None = None,
        posec3d_checkpoint_path: Path | None = None,
    ) -> None:
        try:
            vision = build_vision_adapter(
                backend,
                preload=True,
                on_stage=self._publish_preload,
                action_classifier_path=action_classifier_path,
                posec3d_checkpoint_path=posec3d_checkpoint_path,
            )
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
            action_publisher=self._new_action_publisher(self.transport_mode),
            auto_action_publisher=False,
        )
        self._active_transport_mode = normalize_transport_mode(
            self.transport_mode
        ).value
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
            self.transport_combo.setEnabled(False)
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
        self.worker.stop(timeout=None)
        self.transport_combo.setEnabled(True)
        self.status.setText("已停止")

    def _new_action_publisher(self, mode: str):
        """Create the selected event transport without breaking model startup."""

        selected = normalize_transport_mode(mode)
        label = transport_label(selected)
        try:
            publisher = action_publisher_from_environment(selected)
        except Exception as error:
            if self.transport_status is not None:
                self.transport_status.setText(f"{label}：不可用")
            self.status.setText(f"{label} 初始化失败：{error}")
            return None
        if self.transport_status is not None:
            if publisher is None:
                self.transport_status.setText(f"{label}：未配置")
            else:
                self.transport_status.setText(f"{label}：已启用")
        return publisher

    def _transport_mode_changed(self, mode: str) -> None:
        """Switch event transport while the capture worker is stopped."""

        try:
            selected = normalize_transport_mode(mode)
        except ValueError:
            self.status.setText("不支持的动作传输方式；仅可选择 MQTT 或 BLE Light")
            return
        label = transport_label(selected)
        if self.pipeline is None:
            return
        if self.worker is not None and self.worker.running:
            if self._active_transport_mode is not None:
                self.transport_combo.blockSignals(True)
                self.transport_combo.setCurrentText(
                    transport_label(self._active_transport_mode)
                )
                self.transport_combo.blockSignals(False)
            QMessageBox.warning(self, "传输方式未切换", "请先停止采集，再切换 MQTT 或 BLE Light。")
            return
        try:
            publisher = action_publisher_from_environment(selected)
        except Exception as error:
            if self._active_transport_mode is not None:
                self.transport_combo.blockSignals(True)
                self.transport_combo.setCurrentText(
                    transport_label(self._active_transport_mode)
                )
                self.transport_combo.blockSignals(False)
            if self.transport_status is not None:
                self.transport_status.setText(f"{label}：不可用")
            QMessageBox.critical(self, "传输方式不可用", str(error))
            return
        self.pipeline.set_action_publisher(publisher)
        self._active_transport_mode = selected.value
        if self.transport_status is not None:
            if publisher is None:
                self.transport_status.setText(f"{label}：未配置")
            else:
                self.transport_status.setText(f"{label}：已启用")
        self.status.setText(f"动作事件传输已切换为 {label}")

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

    def _sync_manual_action_target(self) -> None:
        """Mirror the selected Track row into the action target selector."""

        if self._manual_action_syncing:
            return
        rows = self.track_tree.selectionModel().selectedRows()
        if not rows:
            return
        item = self.track_tree.item(rows[0].row(), 0)
        if item is None:
            return
        try:
            track_id = int(item.text())
        except (TypeError, ValueError):
            return
        for label, candidate_track_id in self._manual_action_targets.items():
            if candidate_track_id == track_id:
                self._manual_action_syncing = True
                self.manual_action_target_combo.setCurrentText(label)
                self.manual_action_target = label
                self._manual_action_syncing = False
                return

    def _refresh_manual_action_targets(self, tracks: tuple[object, ...]) -> None:
        """Expose current Tracks and their known identity labels to the GUI."""

        previous = self.manual_action_target
        targets: dict[str, int] = {}
        for track in tracks:
            track_id = int(track.track_id)
            decision = track.decision
            identity = (
                decision.identity_id
                if decision.open_set_label == "KNOWN" and decision.identity_id
                else "未确认"
            )
            targets[f"Track {track_id} · {identity}"] = track_id
        self._manual_action_targets = targets
        labels = tuple(targets)
        self._manual_action_syncing = True
        self.manual_action_target_combo.clear()
        self.manual_action_target_combo.addItems(labels)
        if previous in targets:
            self.manual_action_target_combo.setCurrentText(previous)
            self.manual_action_target = previous
        elif labels:
            self.manual_action_target_combo.setCurrentIndex(0)
            self.manual_action_target = labels[0]
        else:
            self.manual_action_target = ""
        self._manual_action_syncing = False
        enabled = bool(labels)
        self.manual_action_target_combo.setEnabled(enabled)
        self.manual_action_combo.setEnabled(enabled)
        self.manual_action_button.setEnabled(enabled)

    def trigger_manual_action(self) -> None:
        """Queue the selected action without bypassing identity authorization."""

        if self.worker is None:
            self.status.setText("模型仍在后台加载，请稍候")
            return
        label = self.manual_action_target_combo.currentText().strip()
        target = self._manual_action_targets.get(label)
        if target is None:
            rows = self.track_tree.selectionModel().selectedRows()
            if rows:
                item = self.track_tree.item(rows[0].row(), 0)
                try:
                    target = int(item.text()) if item is not None else None
                except (TypeError, ValueError):
                    target = None
        if target is None:
            QMessageBox.warning(self, "未选择目标", "请先在当前目标表中选择一个 Track 或身份。")
            return
        action = ActionType.parse(self.manual_action_combo.currentText())
        try:
            self.worker.trigger_manual_action(target, action)
        except Exception as error:
            QMessageBox.critical(self, "手动动作失败", str(error))
            return
        self.status.setText(
            f"已为 Track {target} 排队 {action.value}；下一帧将验证身份并发布动作事件"
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
        self.worker.stop(timeout=None)
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
        self._manual_action_targets.clear()
        self.manual_action_target = ""
        self.manual_action_target_combo.clear()
        self.manual_action_target_combo.setEnabled(False)
        self.manual_action_combo.setEnabled(False)
        self.manual_action_button.setEnabled(False)
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
            values = (
                str(track.track_id),
                track.action_type,
                subject,
                decision.kind.value,
                track.automation.message,
                score,
            )
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
        self._refresh_manual_action_targets(result.tracks)
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
        if result.transport_errors:
            self.status.setText("动作事件传输失败：" + "；".join(result.transport_errors[:2]))
        elif request_message:
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
                if message.level in {"ended", "error"}:
                    self.transport_combo.setEnabled(True)
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
            self.worker.stop(timeout=None)
        if self.pipeline is not None:
            self.pipeline.close()
        self.verifier.close()
        super().close()


def launch_gui(
    database_path: str = "data/verifier-production-v1.sqlite3",
    vision_backend: str = "production",
    action_classifier_path: Path | None = None,
    posec3d_checkpoint_path: Path | None = None,
) -> int:
    """Start the Qt GUI; all heavy work remains in the existing worker layer."""

    app = QApplication.instance() or QApplication(sys.argv)
    window = VerifierWindow(
        database_path,
        vision_backend,
        action_classifier_path,
        posec3d_checkpoint_path,
    )
    window.show()
    return app.exec()


__all__ = ["VerifierWindow", "launch_gui"]
