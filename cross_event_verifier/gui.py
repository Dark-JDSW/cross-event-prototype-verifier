"""用于摄像头/视频验证的独立 Tk 桌面应用。

Tk 只负责展示。``FrameWorker`` 执行采集和推理，``VideoVerifierPipeline`` 负责
编排，实时参数页提交经过校验的事务，而不是直接编辑验证器内部状态。
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

from .engine import CrossEventVerifier
from .gui_theme import (
    INDUSTRIAL_NAVY,
    MODULE_SUBTITLES,
    NASA_BLUE,
    PALE_BLUE,
    PALE_RED,
    PALE_YELLOW,
    PANEL_CREAM,
    RETRO_ORANGE,
    RETRO_WHITE,
    SIGNAL_RED,
    SOLAR_YELLOW,
    STATUS_GREEN,
    STATUS_PENDING,
    VINTAGE_PAPER,
    configure_theme,
    draw_engineering_grid,
    draw_header_orbit,
    draw_video_standby,
)
from .media import (
    FrameMessage,
    FrameWorker,
    ParameterUpdateMessage,
    RegistrationMessage,
    SourceSpec,
    StatusMessage,
)
from .automation import AutomationPolicy
from .pipeline import FrameResult, VideoVerifierPipeline
from .runtime_parameters import RuntimeParameterSpec, RuntimeParameterState
from .storage import SqliteStore
from .vision_factory import build_vision_adapter


GUI_POLL_INTERVAL_MS = 16
VIDEO_STANDBY_TEXT = "◎\n\n没有画面\nNO VISUAL FEED\nCAMERA CHANNEL STANDBY"


def _frame_to_photo(
    frame_bgr: object,
    maximum_width: int = 960,
    maximum_height: int = 640,
    *,
    master: tk.Misc | None = None,
) -> tk.PhotoImage:
    """不依赖 Pillow，将 OpenCV BGR 帧转换为 Tk 图像。"""

    frame = frame_bgr
    if not hasattr(frame, "shape"):
        raise ValueError("frame must be a NumPy image")
    height, width = frame.shape[:2]
    scale = min(1.0, maximum_width / max(width, 1), maximum_height / max(height, 1))
    if scale < 0.999:
        frame = cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    ppm = f"P6\n{w} {h}\n255\n".encode("ascii") + rgb.tobytes()
    # 显式指定 ``format='PPM'`` 时，Tk 需要原始 PPM 字节。这里传入 base64 文本
    # 会导致某些 Tk 构建对每一帧都报 ``couldn't recognize image data``。
    return tk.PhotoImage(master=master, data=ppm, format="PPM")


class VerifierWindow:
    """GUI 外壳；媒体和验证复杂性隐藏在工作线程接口之后。

    第一个 notebook 页面是监控页，第二个页面公开非 GUI 控制器使用的同一套
    参数说明。这样标签、默认值和校验规则只有一个真实来源。
    """

    def __init__(
        self,
        database_path: str = "data/verifier-production-v1.sqlite3",
        vision_backend: str = "production",
    ) -> None:
        """创建验证器，并在生产后端下后台预加载模型后再启用采集。"""
        self.root = tk.Tk()
        self.root.title("Cross-event Prototype Verifier")
        self.root.geometry("1440x860")
        self.root.minsize(980, 640)
        configure_theme(self.root)

        self._closed = False
        self._preload_cancel = threading.Event()
        self._preload_messages: queue.Queue[tuple[str, str, float, object | None]] = (
            queue.Queue()
        )
        self._preload_thread: threading.Thread | None = None
        self._preload_after_id: str | None = None
        self._message_after_id: str | None = None
        self._idle_after_ids: list[str] = []
        self._scroll_canvases: tuple[tk.Canvas, ...] = ()
        self._mousewheel_bound = False
        self.vision = None
        self.pipeline = None
        self.worker = None
        self._photo: tk.PhotoImage | None = None

        database = Path(database_path)
        if str(database) != ":memory:":
            database.parent.mkdir(parents=True, exist_ok=True)
        self.verifier = CrossEventVerifier(store=SqliteStore(str(database)))
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        selected = vision_backend.strip().lower()
        self._initialize_variables(loading=selected != "demo")
        self._build_layout()
        if selected == "demo":
            self.vision = build_vision_adapter(vision_backend)
            self._finish_runtime_initialization()
        else:
            self._preload_after_id = self.root.after(50, self._poll_preload)
            self._preload_thread = threading.Thread(
                target=self._preload_backend,
                args=(vision_backend,),
                name="cross-event-model-preload",
                daemon=True,
            )
            self._preload_thread.start()

    def _initialize_variables(self, *, loading: bool) -> None:
        """创建正式页面和后台加载状态共用的 Tk 变量。"""

        self.source_kind = tk.StringVar(value="camera")
        self.camera_index = tk.StringVar(value="0")
        self.video_path = tk.StringVar()
        self.video_repeat_count = tk.StringVar(value="1")
        self.camera_id = tk.StringVar(value="camera-1")
        self.identity_id = tk.StringVar(value="P1")
        self.candidate_id = tk.StringVar()
        self.automatic_registration = tk.BooleanVar(value=False)
        self.automation_status = tk.StringVar(
            value="正在后台加载生产视觉模型…" if loading else "诊断后端：自动注册已安全关闭"
        )
        self.backend_status = tk.StringVar(
            value="视觉后端：正在后台准备…" if loading else "视觉后端：准备中"
        )
        self.appearance_request_id = tk.StringVar()
        self.pending_requests = tk.StringVar(value="无")
        self.status = tk.StringVar(
            value="正在后台分阶段加载模型，请稍候…"
            if loading
            else "请选择摄像头或视频文件，然后点击“开始”"
        )
        self.parameter_status = tk.StringVar(value="参数尚未修改")
        self.parameter_vars: dict[str, tk.StringVar] = {}
        self.parameter_entries: dict[str, ttk.Entry] = {}
        self.parameter_scales: dict[str, ttk.Scale] = {}
        self.parameter_specs: dict[str, RuntimeParameterSpec] = {}
        self._parameter_syncing = False
        self._runtime_parameter_state = None

    def _publish_preload(self, text: str, progress: float) -> None:
        """在线程安全队列中记录预加载阶段，Tk 只在主线程消费它。"""

        if not self._preload_cancel.is_set():
            self._preload_messages.put(("stage", text, progress, None))

    def _preload_backend(self, backend: str) -> None:
        """在后台构造视觉适配器、加载模型并执行 Dummy warmup。"""

        try:
            vision = build_vision_adapter(
                backend,
                preload=True,
                on_stage=self._publish_preload,
            )
        except Exception as error:
            self._preload_messages.put(
                ("error", f"视觉后端加载失败：{error}", 1.0, error)
            )
            return
        self._preload_messages.put(("done", "模型预加载完成，正在打开界面…", 1.0, vision))

    def _poll_preload(self) -> None:
        """在 Tk 主线程更新进度，并在后台完成后构造正式页面。"""

        after_id = self._preload_after_id
        self._preload_after_id = None
        if after_id is not None:
            try:
                self.root.after_cancel(after_id)
            except tk.TclError:
                pass
        if self._closed:
            return
        while True:
            try:
                kind, text, progress, payload = self._preload_messages.get_nowait()
            except queue.Empty:
                break
            self.preload_progress.set(progress * 100.0)
            self.preload_phase.set(text)
            self.status.set(text)
            if kind == "done":
                self.vision = payload
                self._finish_runtime_initialization()
                return
            if kind == "error":
                self._preload_failed(text)
                return
        if not self._closed:
            self._preload_after_id = self.root.after(50, self._poll_preload)

    def _preload_failed(self, text: str) -> None:
        """保留加载页并显示失败原因，避免错误后误启用采集。"""

        self.preload_phase.set("模型加载失败")
        self.status.set(text)
        messagebox.showerror("视觉后端加载失败", text)

    def _finish_runtime_initialization(self) -> None:
        """在 Tk 主线程接管已经预热的适配器并创建正式 GUI。"""

        if self._closed or self.vision is None:
            return
        automatic_capable = bool(
            getattr(self.vision, "supports_automatic_registration", False)
        )
        self.pipeline = VideoVerifierPipeline(
            self.verifier,
            self.vision,
            automation_policy=AutomationPolicy(enabled=automatic_capable),
            appearance_first=True,
        )
        self.worker = FrameWorker(self.pipeline)
        self._runtime_parameter_state = self.pipeline.runtime_parameter_state()
        self.automatic_registration.set(automatic_capable)
        self.automation_status.set(
            "自动注册：开启，等待人物进入画面"
            if automatic_capable
            else "诊断后端：自动注册已安全关闭"
        )
        self.backend_status.set(
            f"视觉后端：{getattr(self.vision, 'backend_status', type(self.vision).__name__)}"
        )
        self.status.set("请选择摄像头或视频文件，然后点击“开始”")
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="normal")
        preload_frame = getattr(self, "preload_frame", None)
        if preload_frame is not None:
            preload_frame.destroy()
        for child in self.parameter_page.winfo_children():
            if child is not getattr(self, "parameter_background_canvas", None):
                child.destroy()
        self.parameter_vars.clear()
        self.parameter_entries.clear()
        self.parameter_scales.clear()
        self.parameter_specs.clear()
        self._build_parameter_page(self.parameter_page)
        self._refresh_scroll_canvases()
        warning = getattr(self.vision, "startup_warning", None)
        if warning:
            self.root.after(100, lambda: self._show_startup_warning(warning))
        self._message_after_id = self.root.after(GUI_POLL_INTERVAL_MS, self._poll_messages)

    def _show_startup_warning(self, warning: str) -> None:
        """在窗口仍存在时展示自动回退后端提示。"""

        if not self._closed:
            messagebox.showwarning("视觉后端降级", warning)

    def _redraw_engineering_grid(self, event: tk.Event | None = None) -> None:
        """随窗口尺寸重绘背景工程网格，不覆盖上层控件。"""

        if self._closed or not hasattr(self, "background_canvas"):
            return
        draw_engineering_grid(self.background_canvas)

    def _after_idle(self, callback: object) -> None:
        """登记纯视觉的空闲重绘，关闭窗口时可安全取消。"""

        if self._closed:
            return
        self._idle_after_ids.append(self.root.after_idle(callback))

    def _redraw_page_engineering_grid(self, canvas: tk.Canvas) -> None:
        """重绘隐藏或可见 Notebook 页面自己的装饰网格。"""

        if not self._closed:
            draw_engineering_grid(canvas)

    def _redraw_video_standby(self, event: tk.Event | None = None) -> None:
        """只重绘无视频时的装饰层，不参与视频帧处理。"""

        if self._closed or not hasattr(self, "video_empty_canvas"):
            return
        width = getattr(event, "width", None)
        height = getattr(event, "height", None)
        draw_video_standby(self.video_empty_canvas, width, height)

    def _show_video_standby(self) -> None:
        """在开始新的输入源前显示待机装饰，首帧到达后立即隐藏。"""

        if self._closed:
            return
        self._photo = None
        self.video_label.configure(image="", text=VIDEO_STANDBY_TEXT)
        self.video_empty_canvas.grid()
        self._redraw_video_standby()

    def _hide_video_standby(self) -> None:
        """让真实视频帧完全接管视频区域。"""

        self.video_empty_canvas.grid_remove()
        self.video_label.configure(text="")

    def _refresh_scroll_canvases(self) -> None:
        """注册当前窗口内可滚动画布的跨平台滚轮处理。"""

        canvases = (
            getattr(self, "side_canvas", None),
            getattr(self, "parameter_canvas", None),
        )
        self._scroll_canvases = tuple(
            canvas for canvas in canvases if isinstance(canvas, tk.Canvas)
        )
        if self._mousewheel_bound:
            return

        # Tk 在 Windows/macOS 使用 MouseWheel，在 Linux 常见的是 Button-4/5。
        # 统一使用 bind_all，是因为滚轮通常落在 Canvas 内嵌的 ttk 子控件上，
        # 仅绑定 Canvas 自身无法接收到这些子控件产生的事件。
        self.root.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._on_mousewheel, add="+")
        self._mousewheel_bound = True

    def _on_mousewheel(self, event: tk.Event) -> str | None:
        """只滚动鼠标所在的 Canvas 视口，不吞掉其他控件的滚轮事件。"""

        widget = getattr(event, "widget", None)
        canvas: tk.Canvas | None = None
        for candidate in self._scroll_canvases:
            current = widget
            while current is not None:
                if current is candidate:
                    canvas = candidate
                    break
                current = getattr(current, "master", None)
            if canvas is not None:
                break
        if canvas is None:
            return None

        number = getattr(event, "num", None)
        if number == 4:
            steps = -1
        elif number == 5:
            steps = 1
        else:
            delta = int(getattr(event, "delta", 0) or 0)
            if delta == 0:
                return None
            # Windows 通常以 120 为一个滚轮刻度；保留小 delta，兼容
            # 高精度触控板和 macOS 的滚轮事件。
            steps = -int(delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)

        canvas.yview_scroll(steps, "units")
        return "break"

    def _build_layout(self) -> None:
        """构造输入源控件、监控页、登记面板和状态栏。"""
        self.background_canvas = tk.Canvas(
            self.root,
            background=RETRO_WHITE,
            highlightthickness=0,
            borderwidth=0,
        )
        self.background_canvas.place(
            relx=0,
            rely=0,
            relwidth=1,
            relheight=1,
        )
        self.background_canvas.bind("<Configure>", self._redraw_engineering_grid)
        self._after_idle(self._redraw_engineering_grid)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        header = ttk.Frame(self.root, style="Retro.Header.TFrame")
        header.grid(row=0, column=0, padx=24, pady=(12, 8), sticky="ew")
        header.columnconfigure(1, weight=1)
        orbit = tk.Canvas(
            header,
            width=112,
            height=56,
            background=INDUSTRIAL_NAVY,
            highlightthickness=0,
            borderwidth=0,
        )
        orbit.grid(row=0, column=0, rowspan=2, padx=(10, 6), pady=4)
        draw_header_orbit(orbit)
        orbit.bind(
            "<Configure>",
            lambda _event: draw_header_orbit(orbit),
        )
        ttk.Label(
            header,
            text="CROSS-EVENT PROTOTYPE VERIFIER",
            style="Retro.Header.TLabel",
        ).grid(row=0, column=1, sticky="sw")
        ttk.Label(
            header,
            text=(
                "CEPV // GAIT IDENTITY SYSTEM  ·  "
                "视觉身份 / 步态验证  ·  MISSION CONTROL"
            ),
            style="Retro.Header.Subtitle.TLabel",
        ).grid(row=1, column=1, sticky="nw")
        ttk.Label(
            header,
            text="● SYSTEM ONLINE",
            style="Retro.Badge.TLabel",
        ).grid(row=0, column=2, rowspan=2, padx=(8, 10), sticky="e")

        controls = ttk.LabelFrame(
            self.root,
            text="输入源",
            style="Retro.Section.TLabelframe",
        )
        controls.grid(row=1, column=0, padx=24, pady=(5, 9), sticky="ew")
        controls.columnconfigure(4, weight=1)

        ttk.Radiobutton(
            controls,
            text="摄像头",
            variable=self.source_kind,
            value="camera",
            command=self._source_mode_changed,
        ).grid(row=0, column=0, padx=(8, 4), pady=4)
        ttk.Entry(controls, textvariable=self.camera_index, width=6).grid(row=0, column=1, padx=4)
        ttk.Label(controls, text="设备号").grid(row=0, column=2, padx=(0, 12))
        ttk.Radiobutton(
            controls,
            text="视频文件",
            variable=self.source_kind,
            value="file",
            command=self._source_mode_changed,
        ).grid(row=0, column=3, padx=4, pady=4)
        self.file_entry = ttk.Entry(controls, textvariable=self.video_path)
        self.file_entry.grid(row=0, column=4, padx=4, sticky="ew")
        self.browse_button = ttk.Button(controls, text="浏览…", command=self._browse_video)
        self.browse_button.grid(row=0, column=5, padx=4)
        ttk.Label(controls, text="来源 ID").grid(row=0, column=6, padx=(12, 4))
        ttk.Entry(controls, textvariable=self.camera_id, width=16).grid(row=0, column=7, padx=4)
        ttk.Label(controls, text="视频重复学习").grid(row=0, column=8, padx=(12, 4))
        self.repeat_count_entry = ttk.Entry(
            controls,
            textvariable=self.video_repeat_count,
            width=6,
        )
        self.repeat_count_entry.grid(row=0, column=9, padx=4)
        self.start_button = ttk.Button(
            controls,
            text="开始",
            command=self.start,
            state="disabled",
            style="Retro.Primary.TButton",
        )
        self.start_button.grid(row=0, column=10, padx=(12, 4))
        self.stop_button = ttk.Button(
            controls,
            text="停止",
            command=self.stop,
            state="disabled",
            style="Retro.Danger.TButton",
        )
        self.stop_button.grid(row=0, column=11, padx=(4, 8))
        ttk.Label(
            controls,
            textvariable=self.backend_status,
            style="Retro.Muted.TLabel",
        ).grid(row=1, column=3, columnspan=9, padx=8, pady=(0, 2), sticky="w")
        ttk.Label(
            controls,
            text="MODULE / 01  ·  INPUT ARRAY",
            style="Retro.PanelMarker.TLabel",
        ).grid(row=1, column=0, columnspan=3, padx=8, pady=(0, 2), sticky="w")
        self.preload_frame = ttk.Frame(controls)
        self.preload_frame.columnconfigure(1, weight=1)
        self.preload_phase = tk.StringVar(value="正在后台准备生产视觉后端…")
        ttk.Label(
            self.preload_frame,
            textvariable=self.preload_phase,
            style="Retro.Muted.TLabel",
        ).grid(row=0, column=0, padx=(8, 10), pady=(0, 3), sticky="w")
        self.preload_progress = tk.DoubleVar(value=0.0)
        ttk.Progressbar(
            self.preload_frame,
            variable=self.preload_progress,
            maximum=100.0,
            mode="determinate",
            length=220,
            style="Retro.Horizontal.TProgressbar",
        ).grid(row=0, column=1, padx=(0, 8), pady=(0, 3), sticky="ew")
        self.preload_frame.grid(
            row=2,
            column=0,
            columnspan=12,
            padx=0,
            pady=(0, 1),
            sticky="ew",
        )

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=2, column=0, padx=24, pady=(6, 12), sticky="nsew")
        monitor_page = ttk.Frame(self.notebook)
        parameter_page = ttk.Frame(self.notebook)
        self.parameter_page = parameter_page
        self.notebook.add(monitor_page, text="01 / 实时识别")
        self.notebook.add(parameter_page, text="02 / 实时参数")

        monitor_grid = tk.Canvas(
            monitor_page,
            background=RETRO_WHITE,
            highlightthickness=0,
            borderwidth=0,
        )
        self.monitor_background_canvas = monitor_grid
        monitor_grid.place(relx=0, rely=0, relwidth=1, relheight=1)
        monitor_grid.bind(
            "<Configure>",
            lambda _event, canvas=monitor_grid: self._redraw_page_engineering_grid(
                canvas
            ),
        )
        self._after_idle(
            lambda canvas=monitor_grid: self._redraw_page_engineering_grid(canvas)
        )

        parameter_grid = tk.Canvas(
            parameter_page,
            background=RETRO_WHITE,
            highlightthickness=0,
            borderwidth=0,
        )
        self.parameter_background_canvas = parameter_grid
        parameter_grid.place(relx=0, rely=0, relwidth=1, relheight=1)
        parameter_grid.bind(
            "<Configure>",
            lambda _event, canvas=parameter_grid: self._redraw_page_engineering_grid(
                canvas
            ),
        )
        self._after_idle(
            lambda canvas=parameter_grid: self._redraw_page_engineering_grid(canvas)
        )

        body = ttk.Frame(monitor_page)
        body.pack(fill="both", expand=True, padx=22, pady=18)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        video_panel = tk.Frame(
            body,
            background=PANEL_CREAM,
            highlightbackground=INDUSTRIAL_NAVY,
            highlightthickness=1,
        )
        video_panel.grid(row=0, column=0, sticky="nsew")
        video_panel.columnconfigure(0, weight=1)
        video_panel.rowconfigure(0, weight=1)
        self.video_label = tk.Label(
            video_panel,
            text=VIDEO_STANDBY_TEXT,
            background=PANEL_CREAM,
            foreground=INDUSTRIAL_NAVY,
            anchor="center",
            justify="center",
            font=("Consolas", 12, "bold"),
            padx=12,
            pady=12,
            highlightbackground=INDUSTRIAL_NAVY,
            highlightthickness=1,
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")
        self.video_empty_canvas = tk.Canvas(
            video_panel,
            background=PANEL_CREAM,
            highlightthickness=0,
            borderwidth=0,
        )
        self.video_empty_canvas.grid(row=0, column=0, sticky="nsew")
        self.video_empty_canvas.bind(
            "<Configure>",
            self._redraw_video_standby,
        )
        self._after_idle(self._redraw_video_standby)

        # 右侧控件总高度会随 Track 列表和状态文案变化，不能直接把它放进
        # 固定高度的监控页。使用一个独立的画布视口承载右栏，窗口较矮时仍
        # 可以通过滚动条访问“外观吸收”和图库区域，而不会被父 Frame 裁掉。
        side_content_width = 520
        side_viewport = ttk.Frame(body, width=side_content_width + 18)
        side_viewport.grid(row=0, column=1, padx=(10, 0), sticky="nsew")
        side_viewport.grid_propagate(False)
        side_viewport.columnconfigure(0, weight=1)
        side_viewport.rowconfigure(0, weight=1)
        side_canvas = tk.Canvas(
            side_viewport,
            background=RETRO_WHITE,
            highlightthickness=0,
            borderwidth=0,
        )
        side_canvas.grid(row=0, column=0, sticky="nsew")
        side_scrollbar = ttk.Scrollbar(
            side_viewport,
            orient="vertical",
            command=side_canvas.yview,
            style="Retro.Vertical.TScrollbar",
        )
        side_scrollbar.grid(row=0, column=1, sticky="ns")
        side_canvas.configure(yscrollcommand=side_scrollbar.set)
        side = ttk.Frame(side_canvas, width=side_content_width)
        side_window = side_canvas.create_window(
            (0, 0),
            window=side,
            anchor="nw",
            width=side_content_width,
        )

        def update_side_scrollregion(_event: tk.Event[tk.Misc] | None = None) -> None:
            """根据右栏实际内容更新画布可滚动范围。"""

            side_canvas.configure(scrollregion=side_canvas.bbox("all"))

        def resize_side_window(event: tk.Event[tk.Misc]) -> None:
            """让右栏内容保持固定设计宽度，不因滚动条出现而水平溢出。"""

            side_canvas.itemconfigure(
                side_window,
                width=max(side_content_width, int(event.width)),
            )

        side.bind("<Configure>", update_side_scrollregion)
        side_canvas.bind("<Configure>", resize_side_window)
        # 将画布和滚动条保留为属性，便于后续主题或无障碍交互扩展。
        self.side_canvas = side_canvas
        self.side_scrollbar = side_scrollbar
        ttk.Label(
            side,
            text="MODULE / 03  ·  TARGET ACQUISITION",
            style="Retro.PanelMarker.TLabel",
        ).pack(anchor="w", pady=(0, 2))
        ttk.Label(
            side,
            text="当前目标",
            font=("Segoe UI", 11, "bold"),
            style="Retro.Instrument.TLabel",
        ).pack(anchor="w", pady=(0, 6))
        self.track_tree = ttk.Treeview(
            side,
            columns=("track", "identity", "kind", "automation", "score"),
            show="headings",
            height=18,
        )
        for column, heading, width in (
            ("track", "Track", 58),
            ("identity", "身份/候选", 105),
            ("kind", "结果", 125),
            ("automation", "自动流程", 170),
            ("score", "分数", 62),
        ):
            self.track_tree.heading(column, text=heading)
            self.track_tree.column(column, width=width, anchor="center")
        self.track_tree.tag_configure("verified", foreground=STATUS_GREEN)
        self.track_tree.tag_configure(
            "learning",
            foreground=NASA_BLUE,
            background=PALE_BLUE,
        )
        self.track_tree.tag_configure(
            "waiting",
            foreground=RETRO_ORANGE,
            background=PALE_YELLOW,
        )
        self.track_tree.tag_configure(
            "unknown",
            foreground=STATUS_PENDING,
            background=PALE_YELLOW,
        )
        self.track_tree.tag_configure(
            "conflict",
            foreground=SIGNAL_RED,
            background=PALE_RED,
        )
        self.track_tree.pack(fill="x", expand=False)

        enrollment = ttk.LabelFrame(
            side,
            text="人物注册",
            style="Retro.Module.TLabelframe",
        )
        enrollment.pack(fill="x", pady=(14, 0))
        ttk.Checkbutton(
            enrollment,
            text="自动注册新人物（默认开启）",
            variable=self.automatic_registration,
            command=self._toggle_automatic_registration,
        ).grid(row=0, column=0, columnspan=2, padx=6, pady=(8, 2), sticky="w")
        ttk.Label(
            enrollment,
            textvariable=self.automation_status,
            wraplength=470,
        ).grid(row=1, column=0, columnspan=2, padx=6, pady=(2, 8), sticky="w")
        ttk.Label(enrollment, text="人工身份 ID").grid(row=2, column=0, padx=6, pady=8)
        ttk.Entry(enrollment, textvariable=self.identity_id, width=18).grid(row=2, column=1, padx=6, pady=8)
        ttk.Label(enrollment, text="跨事件候选键").grid(row=3, column=0, padx=6, pady=8)
        ttk.Entry(enrollment, textvariable=self.candidate_id, width=18).grid(
            row=3,
            column=1,
            padx=6,
            pady=8,
        )
        ttk.Label(
            enrollment,
            text="视觉身份尚未完成时，临时 Track 换视频/摄像头仍需保持不变；已有 P 会由 OSNet 自动重新绑定。",
            wraplength=470,
        ).grid(row=4, column=0, columnspan=2, padx=6, pady=(0, 8), sticky="w")
        ttk.Button(
            enrollment,
            text="人工登记选中目标（兜底）",
            command=self.register_selected,
        ).grid(row=5, column=0, columnspan=2, padx=6, pady=(0, 8), sticky="ew")
        ttk.Label(
            enrollment,
            text="当前流程：OSNet 先确认视觉身份并编号；随后按独立步态事件采集 GaitGraph2 原型。",
            wraplength=470,
        ).grid(row=6, column=0, columnspan=2, padx=6, pady=(0, 8), sticky="w")

        request_box = ttk.LabelFrame(
            side,
            text="外观吸收（自动；此处为人工兜底）",
            style="Retro.Module.TLabelframe",
        )
        request_box.pack(fill="x", pady=(14, 0))
        self.appearance_request_box = request_box
        ttk.Label(request_box, text="请求令牌").grid(row=0, column=0, padx=6, pady=8)
        ttk.Entry(request_box, textvariable=self.appearance_request_id, width=22).grid(
            row=0,
            column=1,
            padx=6,
            pady=8,
        )
        ttk.Button(
            request_box,
            text="应用到后续帧",
            command=self.apply_appearance_request,
        ).grid(row=1, column=0, columnspan=2, padx=6, pady=(0, 8), sticky="ew")
        ttk.Label(request_box, text="当前待响应请求：").grid(
            row=2,
            column=0,
            columnspan=2,
            padx=6,
            sticky="w",
        )
        ttk.Label(
            request_box,
            textvariable=self.pending_requests,
            wraplength=330,
        ).grid(row=3, column=0, columnspan=2, padx=6, pady=(2, 8), sticky="w")

        gallery = ttk.LabelFrame(
            side,
            text="当前正式身份",
            style="Retro.Module.TLabelframe",
        )
        gallery.pack(fill="x", pady=(14, 0))
        self.gallery_label = ttk.Label(gallery, text="无")
        self.gallery_label.pack(anchor="w", padx=8, pady=8)
        ttk.Button(
            gallery,
            text="清除现有 ID（重新建库）",
            command=self.clear_existing_ids,
            style="Retro.Danger.TButton",
        ).pack(fill="x", padx=8, pady=(0, 8))

        self._build_parameter_page(parameter_page)
        self._refresh_scroll_canvases()

        status = ttk.Label(
            self.root,
            textvariable=self.status,
            anchor="w",
            style="Retro.Status.TLabel",
        )
        status.grid(row=3, column=0, padx=24, pady=(0, 14), sticky="ew")
        self._source_mode_changed()

    def _build_parameter_page(self, page: ttk.Frame) -> None:
        """根据运行时模块的参数说明构造可滚动页面。"""

        if self.pipeline is None:
            ttk.Label(
                page,
                text="生产视觉模型加载完成后，这里会显示实时参数。",
                style="Retro.Muted.TLabel",
            ).pack(anchor="w", padx=24, pady=20)
            return

        page.columnconfigure(0, weight=1)
        page.rowconfigure(1, weight=1)
        toolbar = ttk.Frame(page)
        toolbar.grid(row=0, column=0, sticky="ew", padx=24, pady=(14, 10))
        ttk.Button(
            toolbar,
            text="应用到运行中",
            command=self.apply_runtime_parameters,
            style="Retro.Primary.TButton",
        ).pack(side="left")
        ttk.Button(
            toolbar,
            text="重新读取当前值",
            command=self.reload_runtime_parameters,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            toolbar,
            text="填入默认值（未应用）",
            command=self.restore_default_parameters,
        ).pack(side="left", padx=(8, 0))
        ttk.Label(
            toolbar,
            textvariable=self.parameter_status,
            style="Retro.Muted.TLabel",
        ).pack(side="left", padx=(18, 0))
        ttk.Label(
            page,
            text=(
                "所有输入会先整组校验，再在采集线程的帧边界一次性生效。"
                "可通过滑条调节，也可直接在输入框中精确编辑。"
                "参数仅影响本次程序运行；生产阈值应以目标摄像头验证集为依据。"
            ),
            wraplength=1250,
        ).grid(row=2, column=0, sticky="w", padx=24, pady=(4, 14))

        canvas = tk.Canvas(
            page,
            background=PANEL_CREAM,
            highlightbackground=INDUSTRIAL_NAVY,
            highlightthickness=1,
        )
        self.parameter_canvas = canvas
        scrollbar = ttk.Scrollbar(
            page,
            orient="vertical",
            command=canvas.yview,
            style="Retro.Vertical.TScrollbar",
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=1, column=0, sticky="nsew", padx=(24, 0))
        scrollbar.grid(row=1, column=1, sticky="ns", padx=(0, 24))
        inner = ttk.Frame(canvas, style="Retro.Paper.TFrame")
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.columnconfigure(0, weight=1)
        inner.columnconfigure(1, weight=1)

        def update_scroll_region(_event: object = None) -> None:
            """让滚动条范围与动态构造的表单保持一致。"""
            canvas.configure(scrollregion=canvas.bbox("all"))

        def fill_canvas(event: tk.Event) -> None:
            """将嵌入式表单拉伸到当前画布宽度。"""
            canvas.itemconfigure(window_id, width=event.width)

        inner.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", fill_canvas)

        sections: list[str] = []
        for spec in self.pipeline.runtime_parameter_specs:
            if spec.section not in sections:
                sections.append(spec.section)
        for section_index, section in enumerate(sections):
            group = ttk.LabelFrame(
                inner,
                text=section,
                style="Retro.Module.TLabelframe",
            )
            group.grid(
                row=section_index // 2,
                column=section_index % 2,
                padx=12,
                pady=10,
                sticky="nsew",
            )
            group.columnconfigure(2, weight=1)
            group.columnconfigure(3, weight=2)
            ttk.Label(
                group,
                text=(
                    f"MODULE / {section_index + 4:02d}  ·  "
                    f"{MODULE_SUBTITLES.get(section, 'RUNTIME CONTROL')}"
                ),
                style="Retro.ModuleSubtitle.TLabel",
            ).grid(
                row=0,
                column=0,
                columnspan=4,
                padx=(5, 5),
                pady=(0, 4),
                sticky="w",
            )
            rows = [
                item
                for item in self.pipeline.runtime_parameter_specs
                if item.section == section
            ]
            for row_index, spec in enumerate(rows):
                variable = tk.StringVar()
                entry = ttk.Entry(
                    group,
                    textvariable=variable,
                    width=10,
                    justify="center",
                    style="Retro.Numeric.TEntry",
                )
                scale = ttk.Scale(
                    group,
                    from_=float(spec.minimum),
                    to=float(spec.maximum),
                    orient="horizontal",
                    length=150,
                    style="Retro.Horizontal.TScale",
                    command=lambda value, key=spec.key: self._parameter_scale_changed(
                        key, value
                    ),
                )
                self.parameter_vars[spec.key] = variable
                self.parameter_entries[spec.key] = entry
                self.parameter_scales[spec.key] = scale
                self.parameter_specs[spec.key] = spec
                variable.trace_add(
                    "write",
                    lambda *_args, key=spec.key: self._parameter_var_changed(key),
                )
                ttk.Label(
                    group,
                    text=spec.label,
                    width=19,
                    style="Retro.Module.TLabel",
                ).grid(
                    row=row_index + 1,
                    column=0,
                    padx=(8, 4),
                    pady=4,
                    sticky="w",
                )
                entry.grid(row=row_index + 1, column=1, padx=4, pady=4, sticky="w")
                scale.grid(
                    row=row_index + 1,
                    column=2,
                    padx=(3, 7),
                    pady=4,
                    sticky="ew",
                )
                detail = ttk.Frame(group, style="Retro.Paper.TFrame")
                ttk.Label(
                    detail,
                    text=f"RANGE  {spec.minimum:g}—{spec.maximum:g}",
                    style="Retro.Range.TLabel",
                ).pack(anchor="w")
                ttk.Label(
                    detail,
                    text=spec.description,
                    wraplength=280,
                    style="Retro.Module.Muted.TLabel",
                ).pack(anchor="w")
                detail.grid(
                    row=row_index + 1,
                    column=3,
                    padx=(5, 8),
                    pady=4,
                    sticky="w",
                )
        self._load_runtime_parameter_state(self._runtime_parameter_state)

    def _parameter_var_changed(self, key: str) -> None:
        """将手工输入框的合法数值同步到同一参数的滑条。"""

        if self._parameter_syncing:
            return
        spec = self.parameter_specs.get(key)
        scale = self.parameter_scales.get(key)
        variable = self.parameter_vars.get(key)
        if spec is None or scale is None or variable is None:
            return
        try:
            value = spec.coerce(variable.get())
        except ValueError:
            # 输入框允许用户暂时处于编辑中的非法状态，等应用时再由原有
            # RuntimeParameterController 做整组校验；此时不要移动滑条。
            return
        self._parameter_syncing = True
        try:
            scale.set(float(value))
        finally:
            self._parameter_syncing = False
        self.parameter_status.set("参数已修改；点击“应用到运行中”后生效")

    def _parameter_scale_changed(self, key: str, raw_value: str) -> None:
        """将滑条值格式化回原有 StringVar，保留既有提交路径。"""

        if self._parameter_syncing:
            return
        spec = self.parameter_specs.get(key)
        variable = self.parameter_vars.get(key)
        scale = self.parameter_scales.get(key)
        if spec is None or variable is None or scale is None:
            return
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return
        if spec.kind == "int":
            value = float(round(value))
        value = min(max(value, float(spec.minimum)), float(spec.maximum))
        self._parameter_syncing = True
        try:
            scale.set(value)
            variable.set(spec.format(value))
        finally:
            self._parameter_syncing = False
        self.parameter_status.set("参数已修改；点击“应用到运行中”后生效")

    def _load_runtime_parameter_state(self, state: RuntimeParameterState) -> None:
        """渲染已校验快照，并禁用当前后端不适用的字段。"""
        self._runtime_parameter_state = state
        available = set(state.available_keys)
        for spec in self.pipeline.runtime_parameter_specs:
            variable = self.parameter_vars.get(spec.key)
            entry = self.parameter_entries.get(spec.key)
            scale = self.parameter_scales.get(spec.key)
            if variable is None or entry is None or scale is None:
                continue
            entry.configure(state="normal")
            scale.configure(state="normal")
            if spec.key in available:
                value = state.values[spec.key]
                self._parameter_syncing = True
                try:
                    variable.set(spec.format(value))
                    scale.set(float(value))
                finally:
                    self._parameter_syncing = False
            else:
                self._parameter_syncing = True
                try:
                    variable.set("当前后端不可用")
                finally:
                    self._parameter_syncing = False
                entry.configure(state="disabled")
                scale.configure(state="disabled")
        self.parameter_status.set(f"当前运行时参数版本：{state.revision}")

    def apply_runtime_parameters(self) -> None:
        """收集可见表单值，并排队一组原子工作线程事务。"""
        if self.worker is None or self._runtime_parameter_state is None:
            self.status.set("模型仍在后台加载，请稍候")
            return
        available = set(self._runtime_parameter_state.available_keys)
        values = {
            key: variable.get()
            for key, variable in self.parameter_vars.items()
            if key in available
        }
        self.worker.set_runtime_parameters(values)
        self.parameter_status.set("参数更新已排队，将在下一帧前整组生效")

    def reload_runtime_parameters(self) -> None:
        """请求管线生成最新快照，但不修改参数值。"""
        if self.worker is None:
            self.status.set("模型仍在后台加载，请稍候")
            return
        self.worker.set_runtime_parameters({})
        self.parameter_status.set("正在读取当前生效值…")

    def restore_default_parameters(self) -> None:
        """将默认值填入表单；用户仍需点击应用。"""
        if self.pipeline is None or self._runtime_parameter_state is None:
            self.status.set("模型仍在后台加载，请稍候")
            return
        defaults = self.pipeline.runtime_parameter_defaults()
        available = set(self._runtime_parameter_state.available_keys)
        specs = {item.key: item for item in self.pipeline.runtime_parameter_specs}
        for key in available:
            self.parameter_vars[key].set(specs[key].format(defaults[key]))
        self.parameter_status.set("已填入默认值；点击“应用到运行中”后才会生效")

    def _source_mode_changed(self) -> None:
        """仅在选中视频文件单选按钮时启用文件控件。"""
        enabled = self.source_kind.get() == "file"
        state = "normal" if enabled else "disabled"
        self.file_entry.configure(state=state)
        self.browse_button.configure(state=state)
        self.repeat_count_entry.configure(state=state)

    def _browse_video(self) -> None:
        """打开系统文件选择器，并将选中的路径复制到表单。"""
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=(
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("所有文件", "*.*"),
            ),
        )
        if path:
            self.video_path.set(path)

    def _source_spec(self) -> SourceSpec:
        """校验输入源控件，并将其转换为工作线程使用的 ``SourceSpec``。"""
        candidate_id = self.candidate_id.get().strip() or None
        if self.source_kind.get() == "file":
            path = self.video_path.get().strip()
            if not path:
                raise ValueError("请先选择视频文件")
            try:
                repeat_count = int(self.video_repeat_count.get().strip())
            except ValueError as error:
                raise ValueError("视频重复学习次数必须是正整数") from error
            if repeat_count < 1:
                raise ValueError("视频重复学习次数必须是正整数")
            return SourceSpec(
                "file",
                path,
                Path(path).name,
                candidate_id,
                repeat_count,
            )
        try:
            index = int(self.camera_index.get().strip())
        except ValueError as error:
            raise ValueError("摄像头设备号必须是整数") from error
        label = self.camera_id.get().strip() or f"camera-{index}"
        return SourceSpec("camera", index, label, candidate_id)

    def start(self) -> None:
        """校验所选摄像头或文件输入源后开始采集。"""
        if self.worker is None:
            self.status.set("模型仍在后台加载，请稍候")
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
            self.status.set(f"正在打开：{spec.label}{repeat_text}")
        except Exception as error:
            messagebox.showerror("无法开始", str(error))

    def stop(self) -> None:
        """停止采集并更新状态栏，但不销毁窗口。"""
        if self.worker is None:
            self.status.set("模型仍在后台加载，请稍候")
            return
        self.worker.stop()
        self.status.set("已停止")

    def _toggle_automatic_registration(self) -> None:
        """排队切换自动化；拒绝在不安全诊断后端上启用自动注册。"""
        if self.worker is None:
            self.automatic_registration.set(False)
            self.status.set("模型仍在后台加载，请稍候")
            return
        enabled = self.automatic_registration.get()
        if enabled and not bool(
            getattr(self.vision, "supports_automatic_registration", False)
        ):
            self.automatic_registration.set(False)
            messagebox.showwarning(
                "自动注册已阻止",
                "HOG 诊断后端不能提供强步态证据，请使用 production 后端。",
            )
            return
        self.worker.set_automatic_registration(enabled)
        self.automation_status.set(
            "自动注册：开启，OSNet 将先确认视觉身份，随后学习步态原型"
            if enabled
            else "自动注册：关闭；识别和外观令牌响应仍继续"
        )

    def register_selected(self) -> None:
        """为选中的 Track 或当前最佳 Track 排队手工登记。"""
        if self.worker is None:
            self.status.set("模型仍在后台加载，请稍候")
            return
        identity_id = self.identity_id.get().strip()
        if not identity_id:
            messagebox.showwarning("缺少身份 ID", "请输入身份 ID，例如 P001")
            return
        selected = self.track_tree.selection()
        track_id = int(selected[0]) if selected else None
        self.worker.register_identity(identity_id, track_id)
        self.status.set("登记请求已排队，将在采集线程安全执行")

    def clear_existing_ids(self) -> None:
        """备份后清空全部身份数据，供重新识别/建库。"""

        if self.worker is None or self.pipeline is None:
            self.status.set("模型仍在后台加载，请稍候")
            return

        if not messagebox.askyesno(
            "确认清除身份",
            "将清除全部视觉身份、步态原型、事件和审计记录。\n"
            "清除前会自动创建数据库备份，是否继续？",
            icon="warning",
        ):
            return
        self.worker.stop()
        store = self.verifier.store
        backup_path: Path | None = None
        if store.path != ":memory:":
            database = Path(store.path)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_path = database.with_name(
                f"{database.stem}-before-clear-{stamp}{database.suffix}"
            )
            store.backup_to(str(backup_path))
        try:
            self.pipeline.clear_gallery()
        except Exception as error:
            messagebox.showerror("清除失败", f"身份数据未完成清除：{error}")
            return
        for item in self.track_tree.get_children():
            self.track_tree.delete(item)
        self.gallery_label.configure(text="无")
        self.pending_requests.set("无")
        self.automation_status.set("自动注册：开启，等待人物进入画面")
        self.status.set(
            "已清除全部身份数据"
            + (f"；备份：{backup_path.name}" if backup_path is not None else "")
        )

    def apply_appearance_request(self) -> None:
        """排队一次性外观令牌，可选绑定到选中的 Track。"""
        if self.worker is None:
            self.status.set("模型仍在后台加载，请稍候")
            return
        request_id = self.appearance_request_id.get().strip()
        selected = self.track_tree.selection()
        track_id = int(selected[0]) if selected else None
        self.worker.set_appearance_request(request_id or None, track_id)
        self.status.set(
            "已清除外观响应令牌"
            if not request_id
            else (
                f"外观响应令牌已绑定到 Track {track_id}"
                if track_id is not None
                else "外观响应令牌将应用到后续目标"
            )
        )

    @staticmethod
    def _track_visual_tag(kind: str) -> str:
        """把既有决策类别映射为仅用于显示的状态色标签。"""

        if kind in {
            "formal_match",
            "visual_identity_created",
            "appearance_response_accepted",
        }:
            return "verified"
        if kind == "conflict":
            return "conflict"
        if kind in {"unknown", "ambiguous"}:
            return "unknown"
        if kind == "appearance_requested":
            return "waiting"
        if kind in {
            "deferred",
            "need_more_data",
            "candidate_created",
            "candidate_updated",
        }:
            return "learning"
        return ""

    def _update_frame(self, result: FrameResult) -> None:
        """渲染一帧处理结果、决策、自动化阶段和图库 ID。"""
        self.backend_status.set(
            f"视觉后端：{getattr(self.vision, 'backend_status', type(self.vision).__name__)}"
        )
        try:
            self._photo = _frame_to_photo(result.frame_bgr, master=self.root)
            self.video_label.configure(image=self._photo, text="")
            self._hide_video_standby()
        except Exception as error:
            self.status.set(f"画面显示失败：{error}")
        existing_items = set(self.track_tree.get_children())
        seen_items: set[str] = set()
        request_message = ""
        automation_messages: list[str] = []
        for track in result.tracks:
            decision = track.decision
            subject = decision.identity_id or decision.candidate_id or "-"
            score = "-" if decision.score is None else f"{decision.score:.3f}"
            item_id = str(track.track_id)
            visual_tag = self._track_visual_tag(decision.kind.value)
            tags = (visual_tag,) if visual_tag else ()
            values = (
                track.track_id,
                subject,
                decision.kind.value,
                track.automation.message,
                score,
            )
            if item_id in existing_items:
                self.track_tree.item(item_id, values=values, tags=tags)
            else:
                self.track_tree.insert(
                    "",
                    "end",
                    iid=item_id,
                    values=values,
                    tags=tags,
                )
            seen_items.add(item_id)
            automation_messages.append(
                f"T{track.track_id}: {track.automation.message}"
            )
            if decision.appearance_request_id:
                self.appearance_request_id.set(decision.appearance_request_id)
                request_message = track.automation.message
            if decision.kind.value == "appearance_response_accepted":
                self.appearance_request_id.set("")
                request_message = track.automation.message
        for item_id in existing_items - seen_items:
            self.track_tree.delete(item_id)
        self.automation_status.set(
            " | ".join(automation_messages[:2])
            if automation_messages
            else (
                "自动注册：开启，等待人物进入画面"
                if self.pipeline.automatic_registration_enabled
                else "自动注册：关闭"
            )
        )
        self.gallery_label.configure(text=", ".join(result.formal_identities) or "无")
        self.pending_requests.set(
            "\n".join(result.pending_request_ids[:3])
            if result.pending_request_ids
            else "无"
        )
        self.status.set(
            f"帧 {result.frame_index} | 目标 {len(result.tracks)} | "
            f"处理 {result.processing_seconds * 1000:.0f} ms"
        )
        if request_message:
            self.status.set(request_message)

    def _poll_messages(self) -> None:
        """取出工作线程消息，并安排下一次 Tk 轮询。"""
        after_id = self._message_after_id
        self._message_after_id = None
        if after_id is not None:
            try:
                self.root.after_cancel(after_id)
            except tk.TclError:
                pass
        if self._closed or self.worker is None:
            return
        latest_frame: FrameMessage | None = None
        while True:
            try:
                message = self.worker.messages.get_nowait()
            except queue.Empty:
                break
            if isinstance(message, FrameMessage):
                # 工作线程在高负载时会丢弃旧帧，但轮询开始前仍可能积压多条
                # FrameMessage。只渲染最新帧，避免 Tk 连续转换/绘制已经过时的
                # 画面，造成额外背压和可见延迟。
                latest_frame = message
            elif isinstance(message, RegistrationMessage):
                self.status.set(message.text)
                if message.success:
                    messagebox.showinfo("登记成功", message.text)
                else:
                    messagebox.showerror("登记失败", message.text)
            elif isinstance(message, ParameterUpdateMessage):
                self.parameter_status.set(message.text)
                self.status.set(message.text)
                if message.success and message.state is not None:
                    self._load_runtime_parameter_state(message.state)
                elif not message.success:
                    messagebox.showerror("参数应用失败", message.text)
            elif isinstance(message, StatusMessage):
                self.status.set(message.text)
        if latest_frame is not None:
            self._update_frame(latest_frame.result)
        if not self._closed:
            self._message_after_id = self.root.after(
                GUI_POLL_INTERVAL_MS,
                self._poll_messages,
            )

    def close(self) -> None:
        """停止后台工作、关闭 SQLite，并销毁 Tk 根窗口。"""
        if self._closed:
            return
        self._closed = True
        self._preload_cancel.set()
        for attribute in ("_preload_after_id", "_message_after_id"):
            after_id = getattr(self, attribute, None)
            if after_id is None:
                continue
            try:
                self.root.after_cancel(after_id)
            except tk.TclError:
                pass
            setattr(self, attribute, None)
        for idle_id in self._idle_after_ids:
            try:
                self.root.after_cancel(idle_id)
            except tk.TclError:
                pass
        self._idle_after_ids.clear()
        if self.worker is not None:
            self.worker.stop()
        self.verifier.close()
        self.root.destroy()


def launch_tk_gui(
    database_path: str = "data/verifier-production-v1.sqlite3",
    vision_backend: str = "production",
) -> int:
    """启动独立桌面应用，并阻塞在 Tk 事件循环中。"""

    window = VerifierWindow(database_path, vision_backend)
    window.root.mainloop()
    return 0


def launch_gui(
    database_path: str = "data/verifier-production-v1.sqlite3",
    vision_backend: str = "production",
) -> int:
    """启动 Qt 表现层；保留 ``launch_tk_gui`` 作为兼容回退入口。"""

    try:
        from .qt_gui import launch_gui as launch_qt_gui
    except ImportError as error:
        raise RuntimeError(
            "Qt GUI requires PyQt6; install it with `pip install -e .[gui-qt]`."
        ) from error

    return launch_qt_gui(database_path, vision_backend)


__all__ = ["VerifierWindow", "launch_gui", "launch_tk_gui"]
