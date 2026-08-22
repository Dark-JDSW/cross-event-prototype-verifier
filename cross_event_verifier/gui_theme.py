"""Tk/ttk 主题令牌与集中样式。

这个模块只描述 GUI 的外观，不持有业务状态，也不参与模型、媒体或数据库
流程。``configure_theme`` 基于 Tk 自带的 ttk 主题引擎设置统一的复古航天
仪器风格，避免颜色和控件状态散落在业务页面代码中。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


# Retro White Nasapunk palette. 这些颜色只用于表现层，业务参数不依赖它们。
RETRO_WHITE = "#F3EBD7"
WARM_IVORY = "#F7F0DC"
PANEL_CREAM = "#EDE3C7"
VINTAGE_PAPER = "#E8DCC0"

NASA_BLUE = "#1677B8"
SIGNAL_RED = "#D94A38"
SOLAR_YELLOW = "#F2C84B"
RETRO_ORANGE = "#E98A3A"
INDUSTRIAL_NAVY = "#17324D"
METALLIC_GRAY = "#8D918F"
MUTED_NAVY = "#315A7D"

# 低对比度工程图纸装饰色。它们与 Retro White 只保留轻微明度差，避免
# 抢过控件、参数和视频内容的注意力。
ENGINEERING_GRID_TAG = "engineering-grid"
ENGINEERING_GRID_MINOR = "#ECE5D8"
ENGINEERING_GRID_MAJOR = "#E5DCCB"
ENGINEERING_GRID_MARKER = "#DED2BF"

MODULE_SUBTITLES = {
    "开放集与步态锚点": "OPEN-SET / GAIT ANCHOR",
    "质量门控": "QUALITY GATE",
    "自动注册稳定性": "AUTO REGISTRATION",
    "融合与外观响应": "FUSION / APPEARANCE",
    "视觉身份与步态就绪": "IDENTITY / GAIT READINESS",
    "分支概率校准": "BRANCH CALIBRATION",
    "生产视觉前端": "VISION FRONT-END",
}

STATUS_GREEN = "#2E6E58"
STATUS_PENDING = "#8A681A"
PALE_BLUE = "#DCEBF3"
PALE_RED = "#F3D6CF"
PALE_YELLOW = "#F7E9B7"


def draw_engineering_grid(
    canvas: tk.Canvas,
    *,
    spacing: int = 28,
    major_every: int = 5,
) -> None:
    """绘制静态的 NASA 工程制图网格，不承载任何业务状态。"""

    canvas.delete(ENGINEERING_GRID_TAG)
    width = max(int(canvas.winfo_width()), int(canvas.cget("width")), 1)
    height = max(int(canvas.winfo_height()), int(canvas.cget("height")), 1)
    spacing = max(int(spacing), 8)
    major_every = max(int(major_every), 2)

    for column, x in enumerate(range(0, width + spacing, spacing)):
        color = (
            ENGINEERING_GRID_MAJOR
            if column % major_every == 0
            else ENGINEERING_GRID_MINOR
        )
        canvas.create_line(
            x,
            0,
            x,
            height,
            fill=color,
            width=1,
            tags=ENGINEERING_GRID_TAG,
        )
    for row, y in enumerate(range(0, height + spacing, spacing)):
        color = (
            ENGINEERING_GRID_MAJOR
            if row % major_every == 0
            else ENGINEERING_GRID_MINOR
        )
        canvas.create_line(
            0,
            y,
            width,
            y,
            fill=color,
            width=1,
            tags=ENGINEERING_GRID_TAG,
        )

    # 少量坐标十字只落在大网格交点，保持“能感觉到、但不喧宾夺主”。
    major_spacing = spacing * major_every
    marker_color = ENGINEERING_GRID_MARKER
    for x in range(major_spacing, width, major_spacing * 2):
        for y in range(major_spacing, height, major_spacing * 2):
            canvas.create_line(
                x - 3,
                y,
                x + 3,
                y,
                fill=marker_color,
                width=1,
                tags=ENGINEERING_GRID_TAG,
            )
            canvas.create_line(
                x,
                y - 3,
                x,
                y + 3,
                fill=marker_color,
                width=1,
                tags=ENGINEERING_GRID_TAG,
            )


def draw_header_orbit(canvas: tk.Canvas) -> None:
    """绘制不承载业务状态的 Mission Control 轨道标记。"""

    canvas.delete("nasapunk-decoration")
    width = max(int(canvas.winfo_width()), int(canvas.cget("width")))
    height = max(int(canvas.winfo_height()), int(canvas.cget("height")))
    center_x = width // 2
    center_y = height // 2
    canvas.create_oval(
        center_x - 24,
        center_y - 13,
        center_x + 24,
        center_y + 13,
        outline=NASA_BLUE,
        width=2,
        tags="nasapunk-decoration",
    )
    canvas.create_oval(
        center_x - 34,
        center_y - 7,
        center_x + 34,
        center_y + 7,
        outline=SOLAR_YELLOW,
        width=1,
        tags="nasapunk-decoration",
    )
    canvas.create_arc(
        center_x - 30,
        center_y - 18,
        center_x + 30,
        center_y + 18,
        start=205,
        extent=130,
        outline=RETRO_ORANGE,
        width=1,
        tags="nasapunk-decoration",
    )
    canvas.create_oval(
        center_x - 5,
        center_y - 5,
        center_x + 5,
        center_y + 5,
        fill=RETRO_ORANGE,
        outline=RETRO_WHITE,
        tags="nasapunk-decoration",
    )
    for x, y in ((center_x - 29, center_y - 15), (center_x + 29, center_y + 15)):
        canvas.create_oval(
            x - 2,
            y - 2,
            x + 2,
            y + 2,
            fill=SOLAR_YELLOW,
            outline=SOLAR_YELLOW,
            tags="nasapunk-decoration",
        )
    canvas.create_text(
        4,
        height - 5,
        text="SYS / 01",
        anchor="sw",
        fill=SOLAR_YELLOW,
        font=("Consolas", 7, "bold"),
        tags="nasapunk-decoration",
    )


def draw_video_standby(canvas: tk.Canvas, width: int | None = None, height: int | None = None) -> None:
    """绘制低干扰的无视频雷达待机画面。"""

    canvas.delete("nasapunk-decoration")
    actual_width = max(int(canvas.winfo_width()), 1)
    actual_height = max(int(canvas.winfo_height()), 1)
    width = max(int(width or actual_width), 1)
    height = max(int(height or actual_height), 1)
    center_x = width // 2
    center_y = max(height // 2 - 25, 90)
    radius = max(min(width // 6, height // 5), 42)
    muted = METALLIC_GRAY
    canvas.create_oval(
        center_x - radius,
        center_y - radius // 2,
        center_x + radius,
        center_y + radius // 2,
        outline=muted,
        width=1,
        tags="nasapunk-decoration",
    )
    canvas.create_oval(
        center_x - radius // 2,
        center_y - radius // 2,
        center_x + radius // 2,
        center_y + radius // 2,
        outline=muted,
        width=1,
        tags="nasapunk-decoration",
    )
    canvas.create_line(
        center_x - radius - 18,
        center_y,
        center_x + radius + 18,
        center_y,
        fill=muted,
        width=1,
        tags="nasapunk-decoration",
    )
    canvas.create_line(
        center_x,
        center_y - radius // 2 - 18,
        center_x,
        center_y + radius // 2 + 18,
        fill=muted,
        width=1,
        tags="nasapunk-decoration",
    )
    for x, y in (
        (center_x - radius - 18, center_y),
        (center_x + radius + 18, center_y),
        (center_x, center_y - radius // 2 - 18),
    ):
        canvas.create_oval(
            x - 3,
            y - 3,
            x + 3,
            y + 3,
            fill=SOLAR_YELLOW,
            outline=SOLAR_YELLOW,
            tags="nasapunk-decoration",
        )
    canvas.create_text(
        center_x,
        center_y + radius // 2 + 48,
        text="没有画面",
        fill=INDUSTRIAL_NAVY,
        font=("Segoe UI", 12, "bold"),
        tags="nasapunk-decoration",
    )
    canvas.create_text(
        center_x,
        center_y + radius // 2 + 70,
        text="NO VISUAL FEED",
        fill=MUTED_NAVY,
        font=("Consolas", 10, "bold"),
        tags="nasapunk-decoration",
    )
    canvas.create_text(
        center_x,
        center_y + radius // 2 + 88,
        text="CAMERA CHANNEL / STANDBY",
        fill=RETRO_ORANGE,
        font=("Consolas", 8),
        tags="nasapunk-decoration",
    )


def configure_theme(root: tk.Misc) -> ttk.Style:
    """为当前 Tk 根窗口安装集中式复古仪器主题。"""

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        # clam 暴露了更稳定的背景、边框和状态选项，同时仍是 Tk 自带主题。
        style.theme_use("clam")

    style.configure("TFrame", background=RETRO_WHITE)
    # Tk/ttk 不提供可靠的逐控件 alpha；用接近主背景的暖米白表面，
    # 配合外层网格留白，形成低不透明度的视觉效果。
    style.configure("Retro.Paper.TFrame", background=WARM_IVORY)
    style.configure(
        "TLabel",
        background=RETRO_WHITE,
        foreground=INDUSTRIAL_NAVY,
        font=("Segoe UI", 10),
    )
    style.configure(
        "TLabelframe",
        background=RETRO_WHITE,
        foreground=INDUSTRIAL_NAVY,
        bordercolor=INDUSTRIAL_NAVY,
        relief="groove",
        borderwidth=2,
        padding=(4, 3),
    )
    style.configure(
        "TLabelframe.Label",
        background=RETRO_WHITE,
        foreground=INDUSTRIAL_NAVY,
        font=("Segoe UI", 10, "bold"),
    )
    style.configure(
        "TButton",
        background=WARM_IVORY,
        foreground=INDUSTRIAL_NAVY,
        bordercolor=INDUSTRIAL_NAVY,
        relief="raised",
        borderwidth=1,
        padding=(8, 4),
        font=("Segoe UI", 10, "bold"),
    )
    style.map(
        "TButton",
        background=[
            ("disabled", PANEL_CREAM),
            ("pressed", RETRO_ORANGE),
            ("active", SOLAR_YELLOW),
        ],
        foreground=[("disabled", METALLIC_GRAY), ("pressed", INDUSTRIAL_NAVY)],
        bordercolor=[("focus", NASA_BLUE), ("active", INDUSTRIAL_NAVY)],
        relief=[("pressed", "sunken"), ("!pressed", "raised")],
    )
    style.configure(
        "Retro.Primary.TButton",
        background=NASA_BLUE,
        foreground=RETRO_WHITE,
        bordercolor=INDUSTRIAL_NAVY,
        padding=(10, 4),
    )
    style.map(
        "Retro.Primary.TButton",
        background=[
            ("disabled", METALLIC_GRAY),
            ("pressed", RETRO_ORANGE),
            ("active", "#2A87B8"),
        ],
        foreground=[("disabled", PANEL_CREAM), ("!disabled", RETRO_WHITE)],
        relief=[("pressed", "sunken"), ("!pressed", "raised")],
    )
    style.configure(
        "Retro.Danger.TButton",
        background=SIGNAL_RED,
        foreground=RETRO_WHITE,
        bordercolor=INDUSTRIAL_NAVY,
        padding=(9, 4),
    )
    style.map(
        "Retro.Danger.TButton",
        background=[("disabled", METALLIC_GRAY), ("pressed", RETRO_ORANGE), ("active", "#E4513E")],
        foreground=[("disabled", PANEL_CREAM), ("!disabled", RETRO_WHITE)],
        relief=[("pressed", "sunken"), ("!pressed", "raised")],
    )
    style.configure(
        "TEntry",
        fieldbackground=WARM_IVORY,
        foreground=INDUSTRIAL_NAVY,
        bordercolor=INDUSTRIAL_NAVY,
        insertcolor=INDUSTRIAL_NAVY,
        padding=(4, 3),
    )
    style.map(
        "TEntry",
        fieldbackground=[("disabled", PANEL_CREAM), ("focus", RETRO_WHITE)],
        bordercolor=[("focus", NASA_BLUE), ("disabled", METALLIC_GRAY)],
        foreground=[("disabled", METALLIC_GRAY), ("!disabled", INDUSTRIAL_NAVY)],
    )
    style.configure(
        "Retro.Numeric.TEntry",
        fieldbackground=WARM_IVORY,
        foreground=INDUSTRIAL_NAVY,
        bordercolor=INDUSTRIAL_NAVY,
        insertcolor=NASA_BLUE,
        padding=(5, 3),
        font=("Consolas", 10, "bold"),
    )
    style.map(
        "Retro.Numeric.TEntry",
        fieldbackground=[("disabled", PANEL_CREAM), ("focus", RETRO_WHITE)],
        bordercolor=[("focus", RETRO_ORANGE), ("disabled", METALLIC_GRAY)],
        foreground=[("disabled", METALLIC_GRAY), ("!disabled", INDUSTRIAL_NAVY)],
    )
    style.configure(
        "TCheckbutton",
        background=RETRO_WHITE,
        foreground=INDUSTRIAL_NAVY,
        padding=(2, 2),
    )
    style.map(
        "TCheckbutton",
        background=[("active", PANEL_CREAM)],
        foreground=[("disabled", METALLIC_GRAY), ("!disabled", INDUSTRIAL_NAVY)],
    )
    style.configure(
        "TRadiobutton",
        background=RETRO_WHITE,
        foreground=INDUSTRIAL_NAVY,
        padding=(2, 2),
    )
    style.map(
        "TRadiobutton",
        background=[("active", PANEL_CREAM)],
        foreground=[("disabled", METALLIC_GRAY), ("!disabled", INDUSTRIAL_NAVY)],
    )
    style.configure(
        "TNotebook",
        background=RETRO_WHITE,
        bordercolor=INDUSTRIAL_NAVY,
        tabmargins=(2, 2, 2, 0),
    )
    style.configure(
        "TNotebook.Tab",
        background=PANEL_CREAM,
        foreground=INDUSTRIAL_NAVY,
        padding=(11, 5),
        font=("Segoe UI", 10, "bold"),
        borderwidth=1,
        lightcolor=SOLAR_YELLOW,
        darkcolor=INDUSTRIAL_NAVY,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", NASA_BLUE), ("active", SOLAR_YELLOW)],
        foreground=[("selected", RETRO_WHITE), ("active", INDUSTRIAL_NAVY)],
        bordercolor=[("selected", SOLAR_YELLOW), ("active", RETRO_ORANGE)],
    )
    style.configure(
        "Retro.Module.TLabelframe",
        background=WARM_IVORY,
        foreground=INDUSTRIAL_NAVY,
        bordercolor=INDUSTRIAL_NAVY,
        relief="solid",
        borderwidth=1,
        padding=(5, 4),
    )
    style.configure(
        "Retro.Section.TLabelframe",
        background=RETRO_WHITE,
        foreground=INDUSTRIAL_NAVY,
        bordercolor=INDUSTRIAL_NAVY,
        relief="solid",
        borderwidth=2,
        padding=(5, 4),
    )
    style.configure(
        "Retro.Section.TLabelframe.Label",
        background=RETRO_WHITE,
        foreground=INDUSTRIAL_NAVY,
        font=("Segoe UI", 10, "bold"),
        padding=(3, 1),
    )
    style.configure(
        "Retro.Module.TLabelframe.Label",
        background=WARM_IVORY,
        foreground=INDUSTRIAL_NAVY,
        font=("Consolas", 9, "bold"),
        padding=(3, 1),
    )
    style.configure(
        "Retro.ModuleSubtitle.TLabel",
        background=WARM_IVORY,
        foreground=RETRO_ORANGE,
        font=("Consolas", 8, "bold"),
        padding=(3, 1),
    )
    style.configure(
        "Retro.Module.TLabel",
        background=WARM_IVORY,
        foreground=INDUSTRIAL_NAVY,
    )
    style.configure(
        "Retro.Module.Muted.TLabel",
        background=WARM_IVORY,
        foreground=MUTED_NAVY,
        font=("Segoe UI", 9),
    )
    style.configure(
        "Retro.Range.TLabel",
        background=WARM_IVORY,
        foreground=RETRO_ORANGE,
        font=("Consolas", 8, "bold"),
        padding=(0, 0),
    )
    style.configure(
        "Treeview",
        background=WARM_IVORY,
        fieldbackground=WARM_IVORY,
        foreground=INDUSTRIAL_NAVY,
        rowheight=24,
        bordercolor=INDUSTRIAL_NAVY,
        relief="solid",
        font=("Consolas", 9),
    )
    style.configure(
        "Treeview.Heading",
        background=INDUSTRIAL_NAVY,
        foreground=RETRO_WHITE,
        relief="flat",
        padding=(5, 5),
        font=("Segoe UI", 9, "bold"),
    )
    style.map(
        "Treeview",
        background=[("selected", NASA_BLUE)],
        foreground=[("selected", RETRO_WHITE)],
    )
    style.configure(
        "Retro.Horizontal.TProgressbar",
        background=NASA_BLUE,
        troughcolor=PANEL_CREAM,
        bordercolor=INDUSTRIAL_NAVY,
        lightcolor=NASA_BLUE,
        darkcolor=NASA_BLUE,
        thickness=12,
    )
    style.configure(
        "Horizontal.TScale",
        background=RETRO_WHITE,
        troughcolor=PANEL_CREAM,
        bordercolor=INDUSTRIAL_NAVY,
        lightcolor=NASA_BLUE,
        darkcolor=NASA_BLUE,
    )
    style.configure(
        "Retro.Horizontal.TScale",
        background=RETRO_WHITE,
        troughcolor=PANEL_CREAM,
        bordercolor=INDUSTRIAL_NAVY,
        lightcolor=NASA_BLUE,
        darkcolor=NASA_BLUE,
        sliderlength=17,
        borderwidth=1,
    )
    style.map(
        "Retro.Horizontal.TScale",
        background=[("pressed", RETRO_ORANGE), ("active", RETRO_ORANGE)],
        troughcolor=[("disabled", PANEL_CREAM), ("!disabled", VINTAGE_PAPER)],
    )
    style.configure(
        "Retro.Vertical.TScrollbar",
        background=INDUSTRIAL_NAVY,
        troughcolor=PANEL_CREAM,
        bordercolor=INDUSTRIAL_NAVY,
        arrowcolor=INDUSTRIAL_NAVY,
        relief="flat",
        borderwidth=0,
        width=10,
    )
    style.map(
        "Retro.Vertical.TScrollbar",
        background=[("active", NASA_BLUE), ("pressed", RETRO_ORANGE)],
    )
    style.configure(
        "Retro.Horizontal.TScrollbar",
        background=INDUSTRIAL_NAVY,
        troughcolor=PANEL_CREAM,
        bordercolor=INDUSTRIAL_NAVY,
        arrowcolor=INDUSTRIAL_NAVY,
        relief="flat",
        borderwidth=0,
        width=10,
    )
    style.map(
        "Retro.Horizontal.TScrollbar",
        background=[("active", NASA_BLUE), ("pressed", RETRO_ORANGE)],
    )
    # 复古仪器滑轨不需要系统上下/左右箭头，保留 thumb 的鼠标操作和键盘
    # 滚动语义不变。
    style.layout(
        "Retro.Vertical.TScrollbar",
        [
            (
                "Vertical.Scrollbar.trough",
                {
                    "sticky": "ns",
                    "children": [
                        ("Vertical.Scrollbar.thumb", {"sticky": "nswe"}),
                    ],
                },
            ),
        ],
    )
    style.layout(
        "Retro.Horizontal.TScrollbar",
        [
            (
                "Horizontal.Scrollbar.trough",
                {
                    "sticky": "we",
                    "children": [
                        ("Horizontal.Scrollbar.thumb", {"sticky": "nswe"}),
                    ],
                },
            ),
        ],
    )
    style.configure("Retro.Header.TFrame", background=INDUSTRIAL_NAVY)
    style.configure(
        "Retro.Header.TLabel",
        background=INDUSTRIAL_NAVY,
        foreground=RETRO_WHITE,
        font=("Segoe UI", 15, "bold"),
        padding=(2, 2),
    )
    style.configure(
        "Retro.Header.Subtitle.TLabel",
        background=INDUSTRIAL_NAVY,
        foreground=SOLAR_YELLOW,
        font=("Consolas", 9),
        padding=(2, 1),
    )
    style.configure(
        "Retro.Badge.TLabel",
        background=NASA_BLUE,
        foreground=RETRO_WHITE,
        font=("Consolas", 9, "bold"),
        padding=(9, 5),
    )
    style.configure(
        "Retro.Muted.TLabel",
        background=RETRO_WHITE,
        foreground=MUTED_NAVY,
    )
    style.configure(
        "Retro.Instrument.TLabel",
        background=PANEL_CREAM,
        foreground=INDUSTRIAL_NAVY,
        font=("Consolas", 9),
        padding=(5, 3),
    )
    style.configure(
        "Retro.Status.TLabel",
        background=RETRO_WHITE,
        foreground=MUTED_NAVY,
        font=("Consolas", 9),
        padding=(3, 2),
    )
    style.configure(
        "Retro.PanelMarker.TLabel",
        background=RETRO_WHITE,
        foreground=RETRO_ORANGE,
        font=("Consolas", 8, "bold"),
        padding=(2, 1),
    )
    root.configure(background=RETRO_WHITE)
    return style
