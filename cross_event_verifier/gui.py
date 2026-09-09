"""用于摄像头/视频验证的独立 Tk 桌面应用。

Tk 只负责展示。``FrameWorker`` 执行采集和推理，``VideoVerifierPipeline`` 负责
编排，实时参数页提交经过校验的事务，而不是直接编辑验证器内部状态。
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from .actions import ActionType, MANUAL_ACTION_TYPES
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
    STATUS_GREEN,
    STATUS_PENDING,
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
from .ground_control import (
    GroundControlPoint,
    HomographyCheck,
    compute_ground_homography,
    load_ground_control_file,
    parse_ground_control_point,
    save_ground_control_file,
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
    参数说明，第三个页面独立管理地面控制点标定。
    """

    def __init__(
        self,
        database_path: str = "data/verifier-production-v1.sqlite3",
        vision_backend: str = "production",
        action_classifier_path: Path | None = None,
        posec3d_checkpoint_path: Path | None = None,
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
        self.action_classifier_path = (
            Path(action_classifier_path) if action_classifier_path is not None else None
        )
        self.posec3d_checkpoint_path = (
            Path(posec3d_checkpoint_path)
            if posec3d_checkpoint_path is not None
            else None
        )

        database = Path(database_path)
        if str(database) != ":memory:":
            database.parent.mkdir(parents=True, exist_ok=True)
        self.verifier = CrossEventVerifier(store=SqliteStore(str(database)))
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        selected = vision_backend.strip().lower()
        self._initialize_variables(loading=selected != "demo")
        self._build_layout()
        if selected == "demo":
            self.vision = build_vision_adapter(
                vision_backend,
                action_classifier_path=self.action_classifier_path,
                posec3d_checkpoint_path=self.posec3d_checkpoint_path,
            )
            self._finish_runtime_initialization()
        else:
            self._preload_after_id = self.root.after(50, self._poll_preload)
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

    def _initialize_variables(self, *, loading: bool) -> None:
        """创建正式页面和后台加载状态共用的 Tk 变量。"""

        self.source_kind = tk.StringVar(value="camera")
        self.camera_index = tk.StringVar(value="0")
        self.video_path = tk.StringVar()
        self.video_repeat_count = tk.StringVar(value="1")
        self.camera_id = tk.StringVar(value="camera-1")
        raw_transport = os.getenv("CROSS_EVENT_TRANSPORT", "mqtt").strip()
        try:
            transport = normalize_transport_mode(raw_transport).value
        except ValueError:
            transport = "mqtt"
        self.transport_mode = tk.StringVar(value=transport_label(transport))
        self.transport_status = tk.StringVar(
            value=f"{transport_label(transport)}：未配置"
        )
        self._active_transport_mode: str | None = None
        self.manual_action_target = tk.StringVar()
        self.manual_action_type = tk.StringVar(value=MANUAL_ACTION_TYPES[0].value)
        self._manual_action_targets: dict[str, int] = {}
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
        self.gcp_camera_id = tk.StringVar(value=self.camera_id.get())
        self.gcp_coordinate_frame = tk.StringVar(value="ground-meters")
        self.gcp_unit = tk.StringVar(value="m")
        self.gcp_point_id = tk.StringVar(value="P1")
        self.gcp_image_u = tk.StringVar()
        self.gcp_image_v = tk.StringVar()
        self.gcp_world_x = tk.StringVar()
        self.gcp_world_y = tk.StringVar()
        self.gcp_image_size = tk.StringVar(value="未加载")
        self.gcp_status = tk.StringVar(value="请加载一张固定机位画面，然后在画布上点击地面点")
        self.gcp_matrix = tk.StringVar(value="H 尚未校核")
        self.gcp_rows: dict[str, GroundControlPoint] = {}
        self.gcp_check: HomographyCheck | None = None
        self._gcp_image: object | None = None
        self._gcp_photo: tk.PhotoImage | None = None
        self._gcp_display_scale = 1.0
        self._gcp_display_offset = (0.0, 0.0)
        self._gcp_image_size_value: tuple[int, int] | None = None

    def _publish_preload(self, text: str, progress: float) -> None:
        """在线程安全队列中记录预加载阶段，Tk 只在主线程消费它。"""

        if not self._preload_cancel.is_set():
            self._preload_messages.put(("stage", text, progress, None))

    def _preload_backend(
        self,
        backend: str,
        action_classifier_path: Path | None = None,
        posec3d_checkpoint_path: Path | None = None,
    ) -> None:
        """在后台构造视觉适配器、加载模型并执行 Dummy warmup。"""

        try:
            vision = build_vision_adapter(
                backend,
                preload=True,
                on_stage=self._publish_preload,
                action_classifier_path=action_classifier_path,
                posec3d_checkpoint_path=posec3d_checkpoint_path,
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
            action_publisher=self._new_action_publisher(self.transport_mode.get()),
            auto_action_publisher=False,
        )
        self._active_transport_mode = normalize_transport_mode(
            self.transport_mode.get()
        ).value
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
        ).grid(row=1, column=3, columnspan=5, padx=8, pady=(0, 2), sticky="w")
        ttk.Label(
            controls,
            text="MODULE / 01  ·  INPUT ARRAY",
            style="Retro.PanelMarker.TLabel",
        ).grid(row=1, column=0, columnspan=3, padx=8, pady=(0, 2), sticky="w")
        ttk.Label(controls, text="事件传输").grid(
            row=1, column=8, padx=(8, 3), pady=(0, 2), sticky="e"
        )
        self.transport_combo = ttk.Combobox(
            controls,
            textvariable=self.transport_mode,
            values=("MQTT", "BLE Light"),
            state="readonly",
            width=10,
        )
        self.transport_combo.grid(row=1, column=9, padx=3, pady=(0, 2), sticky="w")
        self.transport_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._transport_mode_changed(),
        )
        ttk.Label(
            controls,
            textvariable=self.transport_status,
            style="Retro.Muted.TLabel",
        ).grid(row=1, column=10, columnspan=2, padx=(5, 8), pady=(0, 2), sticky="w")
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
        gcp_page = ttk.Frame(self.notebook)
        self.parameter_page = parameter_page
        self.gcp_page = gcp_page
        self.notebook.add(monitor_page, text="01 / 实时识别")
        self.notebook.add(parameter_page, text="02 / 实时参数")
        self.notebook.add(gcp_page, text="03 / 地面控制点")

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

        gcp_grid = tk.Canvas(
            gcp_page,
            background=RETRO_WHITE,
            highlightthickness=0,
            borderwidth=0,
        )
        self.gcp_background_canvas = gcp_grid
        gcp_grid.place(relx=0, rely=0, relwidth=1, relheight=1)
        gcp_grid.bind(
            "<Configure>",
            lambda _event, canvas=gcp_grid: self._redraw_page_engineering_grid(canvas),
        )
        self._after_idle(
            lambda canvas=gcp_grid: self._redraw_page_engineering_grid(canvas)
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
            columns=("track", "action", "identity", "kind", "automation", "score"),
            show="headings",
            height=18,
        )
        for column, heading, width in (
            ("track", "Track", 58),
            ("action", "动作", 86),
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
        self.track_tree.bind(
            "<<TreeviewSelect>>",
            lambda _event: self._sync_manual_action_target(),
        )

        action_box = ttk.LabelFrame(
            side,
            text="手动动作触发",
            style="Retro.Module.TLabelframe",
        )
        action_box.pack(fill="x", pady=(14, 0))
        ttk.Label(action_box, text="目标（Track / 身份）").grid(
            row=0, column=0, padx=6, pady=(8, 4), sticky="w"
        )
        self.manual_action_target_combo = ttk.Combobox(
            action_box,
            textvariable=self.manual_action_target,
            values=(),
            state="disabled",
            width=34,
        )
        self.manual_action_target_combo.grid(
            row=0, column=1, padx=6, pady=(8, 4), sticky="ew"
        )
        ttk.Label(action_box, text="动作").grid(
            row=1, column=0, padx=6, pady=4, sticky="w"
        )
        self.manual_action_combo = ttk.Combobox(
            action_box,
            textvariable=self.manual_action_type,
            values=tuple(item.value for item in MANUAL_ACTION_TYPES),
            state="disabled",
            width=18,
        )
        self.manual_action_combo.grid(
            row=1, column=1, padx=6, pady=4, sticky="ew"
        )
        self.manual_action_button = ttk.Button(
            action_box,
            text="按所选动作触发",
            command=self.trigger_manual_action,
            state="disabled",
        )
        self.manual_action_button.grid(
            row=2, column=0, columnspan=2, padx=6, pady=(4, 6), sticky="ew"
        )
        ttk.Label(
            action_box,
            text="动作只在下一帧再次确认身份为 KNOWN 后发布；不会写入步态库。",
            wraplength=470,
        ).grid(row=3, column=0, columnspan=2, padx=6, pady=(0, 8), sticky="w")
        action_box.columnconfigure(1, weight=1)

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
        self._build_gcp_page(gcp_page)
        self._refresh_scroll_canvases()

        status = ttk.Label(
            self.root,
            textvariable=self.status,
            anchor="w",
            style="Retro.Status.TLabel",
        )
        status.grid(row=3, column=0, padx=24, pady=(0, 14), sticky="ew")
        self._source_mode_changed()

    def _build_gcp_page(self, page: ttk.Frame) -> None:
        """构造独立的地面控制点标定页，不改变实时识别数据流。"""

        page.columnconfigure(0, weight=1)
        page.rowconfigure(1, weight=1)
        toolbar = ttk.Frame(page)
        toolbar.grid(row=0, column=0, sticky="ew", padx=24, pady=(14, 8))
        ttk.Label(
            toolbar,
            text="MODULE / 05  ·  GROUND CONTROL POINTS",
            style="Retro.PanelMarker.TLabel",
        ).pack(side="left")
        ttk.Button(toolbar, text="加载画面…", command=self._gcp_load_image).pack(
            side="right"
        )
        ttk.Button(toolbar, text="导入标定…", command=self._gcp_import).pack(
            side="right", padx=(0, 8)
        )
        ttk.Button(
            toolbar,
            text="导出标定…",
            command=self._gcp_export,
            style="Retro.Primary.TButton",
        ).pack(side="right", padx=(0, 8))

        body = ttk.Frame(page)
        body.grid(row=1, column=0, sticky="nsew", padx=22, pady=(0, 18))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        preview_panel = tk.Frame(
            body,
            background=PANEL_CREAM,
            highlightbackground=INDUSTRIAL_NAVY,
            highlightthickness=1,
        )
        preview_panel.grid(row=0, column=0, sticky="nsew")
        preview_panel.columnconfigure(0, weight=1)
        preview_panel.rowconfigure(0, weight=1)
        self.gcp_canvas = tk.Canvas(
            preview_panel,
            background=PANEL_CREAM,
            highlightthickness=0,
            borderwidth=0,
        )
        self.gcp_canvas.grid(row=0, column=0, sticky="nsew")
        self.gcp_canvas.bind("<Configure>", lambda _event: self._gcp_render())
        self.gcp_canvas.bind("<Button-1>", self._gcp_canvas_click)
        ttk.Label(
            preview_panel,
            text="点击地面控制点获取像素坐标；然后在右侧录入对应世界坐标。",
            style="Retro.Muted.TLabel",
        ).grid(row=1, column=0, padx=8, pady=6, sticky="w")

        right = ttk.Frame(body)
        right.grid(row=0, column=1, padx=(10, 0), sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        metadata = ttk.LabelFrame(
            right,
            text="标定元数据",
            style="Retro.Module.TLabelframe",
        )
        metadata.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        metadata.columnconfigure(1, weight=1)
        for row, label, variable in (
            (0, "摄像头 ID", self.gcp_camera_id),
            (1, "世界坐标系", self.gcp_coordinate_frame),
            (2, "单位", self.gcp_unit),
        ):
            ttk.Label(metadata, text=label).grid(row=row, column=0, padx=6, pady=5, sticky="w")
            ttk.Entry(metadata, textvariable=variable).grid(
                row=row, column=1, padx=6, pady=5, sticky="ew"
            )
        ttk.Label(metadata, text="画面尺寸").grid(row=3, column=0, padx=6, pady=5, sticky="w")
        ttk.Label(metadata, textvariable=self.gcp_image_size, style="Retro.Muted.TLabel").grid(
            row=3, column=1, padx=6, pady=5, sticky="w"
        )

        point_form = ttk.LabelFrame(
            right,
            text="控制点标记 / 录入",
            style="Retro.Module.TLabelframe",
        )
        point_form.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        point_form.columnconfigure(1, weight=1)
        point_form.columnconfigure(3, weight=1)
        fields = (
            (0, 0, "点 ID", self.gcp_point_id),
            (0, 2, "像素 U", self.gcp_image_u),
            (1, 0, "像素 V", self.gcp_image_v),
            (1, 2, "世界 X", self.gcp_world_x),
            (2, 0, "世界 Y", self.gcp_world_y),
        )
        for row, column, label, variable in fields:
            ttk.Label(point_form, text=label).grid(
                row=row, column=column, padx=6, pady=5, sticky="w"
            )
            ttk.Entry(point_form, textvariable=variable, width=12).grid(
                row=row, column=column + 1, padx=6, pady=5, sticky="ew"
            )
        ttk.Button(point_form, text="添加 / 更新", command=self._gcp_upsert).grid(
            row=2, column=2, padx=6, pady=5, sticky="ew"
        )
        ttk.Button(point_form, text="删除选中", command=self._gcp_delete).grid(
            row=2, column=3, padx=6, pady=5, sticky="ew"
        )

        table_box = ttk.LabelFrame(
            right,
            text="控制点清单",
            style="Retro.Module.TLabelframe",
        )
        table_box.grid(row=2, column=0, sticky="nsew")
        table_box.columnconfigure(0, weight=1)
        table_box.rowconfigure(0, weight=1)
        self.gcp_tree = ttk.Treeview(
            table_box,
            columns=("id", "u", "v", "x", "y", "check"),
            show="headings",
            selectmode="browse",
        )
        for column, heading, width in (
            ("id", "ID", 52),
            ("u", "U", 72),
            ("v", "V", 72),
            ("x", "X", 72),
            ("y", "Y", 72),
            ("check", "校核", 100),
        ):
            self.gcp_tree.heading(column, text=heading)
            self.gcp_tree.column(column, width=width, anchor="center")
        self.gcp_tree.tag_configure("inlier", foreground=STATUS_GREEN)
        self.gcp_tree.tag_configure("outlier", foreground=SIGNAL_RED)
        self.gcp_tree.grid(row=0, column=0, sticky="nsew")
        gcp_scrollbar = ttk.Scrollbar(
            table_box,
            orient="vertical",
            command=self.gcp_tree.yview,
            style="Retro.Vertical.TScrollbar",
        )
        gcp_scrollbar.grid(row=0, column=1, sticky="ns")
        self.gcp_tree.configure(yscrollcommand=gcp_scrollbar.set)
        self.gcp_tree.bind("<<TreeviewSelect>>", self._gcp_select)

        footer = ttk.Frame(right)
        footer.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Button(
            footer,
            text="计算并校核 Homography",
            command=self._gcp_check_homography,
            style="Retro.Primary.TButton",
        ).grid(row=0, column=0, sticky="ew")
        ttk.Button(
            footer,
            text="清空控制点",
            command=self._gcp_clear,
            style="Retro.Danger.TButton",
        ).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(
            footer,
            textvariable=self.gcp_status,
            wraplength=470,
            style="Retro.Muted.TLabel",
        ).grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Label(
            footer,
            textvariable=self.gcp_matrix,
            wraplength=470,
            style="Retro.Range.TLabel",
        ).grid(row=3, column=0, sticky="w", pady=(4, 0))

    def _gcp_next_id(self) -> str:
        index = 1
        while f"P{index}" in self.gcp_rows:
            index += 1
        return f"P{index}"

    def _gcp_current_point(self) -> GroundControlPoint:
        return parse_ground_control_point(
            self.gcp_point_id.get(),
            self.gcp_image_u.get(),
            self.gcp_image_v.get(),
            self.gcp_world_x.get(),
            self.gcp_world_y.get(),
        )

    def _gcp_upsert(self) -> None:
        try:
            point = self._gcp_current_point()
        except ValueError as error:
            messagebox.showerror("控制点录入失败", str(error))
            return
        self.gcp_rows[point.point_id] = point
        self.gcp_point_id.set(self._gcp_next_id())
        self.gcp_check = None
        self.gcp_matrix.set("H 尚未校核")
        self._gcp_refresh_tree()
        self._gcp_render()
        self.gcp_status.set(f"已记录 {point.point_id}；还可继续点击画面标记下一点")

    def _gcp_select(self, _event: tk.Event | None = None) -> None:
        selection = self.gcp_tree.selection()
        if not selection:
            return
        point = self.gcp_rows.get(selection[0])
        if point is None:
            return
        self.gcp_point_id.set(point.point_id)
        self.gcp_image_u.set(f"{point.image_u:.3f}")
        self.gcp_image_v.set(f"{point.image_v:.3f}")
        self.gcp_world_x.set(f"{point.world_x:.3f}")
        self.gcp_world_y.set(f"{point.world_y:.3f}")

    def _gcp_delete(self) -> None:
        selection = self.gcp_tree.selection()
        if not selection:
            self.gcp_status.set("请先在清单中选择控制点")
            return
        self.gcp_rows.pop(selection[0], None)
        self.gcp_check = None
        self.gcp_matrix.set("H 尚未校核")
        self._gcp_refresh_tree()
        self._gcp_render()
        self.gcp_status.set("已删除选中控制点")

    def _gcp_refresh_tree(self) -> None:
        for item in self.gcp_tree.get_children():
            self.gcp_tree.delete(item)
        for point_id, point in self.gcp_rows.items():
            check_text = "未校核"
            tag = ()
            if self.gcp_check is not None:
                index = list(self.gcp_rows).index(point_id)
                inlier = self.gcp_check.inliers[index]
                error = self.gcp_check.reprojection_errors[index]
                check_text = f"{'内点' if inlier else '外点'} {error:.3f}"
                tag = ("inlier" if inlier else "outlier",)
            self.gcp_tree.insert(
                "",
                "end",
                iid=point_id,
                values=(
                    point_id,
                    f"{point.image_u:.2f}",
                    f"{point.image_v:.2f}",
                    f"{point.world_x:.3f}",
                    f"{point.world_y:.3f}",
                    check_text,
                ),
                tags=tag,
            )

    def _gcp_load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择固定机位画面",
            filetypes=(
                ("图像", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("全部文件", "*.*"),
            ),
        )
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("画面加载失败", "无法读取所选图像")
            return
        self._gcp_image = image
        height, width = image.shape[:2]
        self._gcp_image_size_value = (int(width), int(height))
        self.gcp_image_size.set(f"{width} × {height}")
        self._gcp_render()
        self.gcp_status.set(f"已加载画面 {Path(path).name}；点击地面点开始标记")

    def _gcp_render(self) -> None:
        if not hasattr(self, "gcp_canvas"):
            return
        canvas = self.gcp_canvas
        canvas.delete("gcp-image")
        canvas.delete("gcp-marker")
        image = self._gcp_image
        if image is None or not hasattr(image, "shape"):
            canvas.create_text(
                max(canvas.winfo_width() // 2, 1),
                max(canvas.winfo_height() // 2, 1),
                text="加载固定机位画面后，在此标记地面控制点",
                fill=INDUSTRIAL_NAVY,
                font=("Segoe UI", 11, "bold"),
                tags="gcp-marker",
            )
            return
        height, width = image.shape[:2]
        available_width = max(canvas.winfo_width(), 2)
        available_height = max(canvas.winfo_height(), 2)
        scale = min(available_width / width, available_height / height)
        scale = min(scale, 1.0)
        display_width = max(1, int(round(width * scale)))
        display_height = max(1, int(round(height * scale)))
        display = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)
        self._gcp_photo = _frame_to_photo(
            display,
            maximum_width=display_width,
            maximum_height=display_height,
            master=canvas,
        )
        offset_x = (available_width - display_width) / 2.0
        offset_y = (available_height - display_height) / 2.0
        self._gcp_display_scale = scale
        self._gcp_display_offset = (offset_x, offset_y)
        canvas.create_image(
            offset_x,
            offset_y,
            anchor="nw",
            image=self._gcp_photo,
            tags="gcp-image",
        )
        for point in self.gcp_rows.values():
            x = offset_x + point.image_u * scale
            y = offset_y + point.image_v * scale
            canvas.create_oval(x - 5, y - 5, x + 5, y + 5, outline=RETRO_ORANGE, width=2, tags="gcp-marker")
            canvas.create_text(x + 8, y - 8, text=point.point_id, anchor="sw", fill=INDUSTRIAL_NAVY, font=("Consolas", 9, "bold"), tags="gcp-marker")
        try:
            pending_u = float(self.gcp_image_u.get())
            pending_v = float(self.gcp_image_v.get())
        except (TypeError, ValueError):
            return
        if 0 <= pending_u <= width and 0 <= pending_v <= height:
            x = offset_x + pending_u * scale
            y = offset_y + pending_v * scale
            canvas.create_line(x - 8, y, x + 8, y, fill=NASA_BLUE, width=1, tags="gcp-marker")
            canvas.create_line(x, y - 8, x, y + 8, fill=NASA_BLUE, width=1, tags="gcp-marker")

    def _gcp_canvas_click(self, event: tk.Event) -> None:
        image = self._gcp_image
        if image is None or not hasattr(image, "shape"):
            return
        height, width = image.shape[:2]
        offset_x, offset_y = self._gcp_display_offset
        scale = self._gcp_display_scale
        if scale <= 0:
            return
        image_u = (float(event.x) - offset_x) / scale
        image_v = (float(event.y) - offset_y) / scale
        if not (0 <= image_u <= width and 0 <= image_v <= height):
            return
        self.gcp_image_u.set(f"{image_u:.3f}")
        self.gcp_image_v.set(f"{image_v:.3f}")
        if not self.gcp_point_id.get().strip() or self.gcp_point_id.get() in self.gcp_rows:
            self.gcp_point_id.set(self._gcp_next_id())
        self.gcp_status.set(f"已标记像素坐标 ({image_u:.2f}, {image_v:.2f})；请输入世界坐标后添加")
        self._gcp_render()

    def _gcp_check_homography(self, *, show_error: bool = True) -> HomographyCheck | None:
        try:
            check = compute_ground_homography(list(self.gcp_rows.values()))
        except ValueError as error:
            self.gcp_check = None
            self.gcp_matrix.set("H 尚未校核")
            self.gcp_status.set(str(error))
            if show_error:
                messagebox.showerror("Homography 校核失败", str(error))
            return None
        self.gcp_check = check
        self._gcp_refresh_tree()
        matrix = np.asarray(check.matrix, dtype=float).reshape(3, 3)
        matrix_text = "H = " + "; ".join(", ".join(f"{value:.5g}" for value in row) for row in matrix)
        self.gcp_matrix.set(matrix_text)
        self.gcp_status.set(
            f"校核通过：{len(self.gcp_rows)} 点，内点 {sum(check.inliers)}，"
            f"RMSE {check.rmse:.4f} {self.gcp_unit.get() or 'm'}，最大误差 {check.maximum_error:.4f}"
        )
        return check

    def _gcp_clear(self) -> None:
        self.gcp_rows.clear()
        self.gcp_check = None
        self.gcp_matrix.set("H 尚未校核")
        self.gcp_point_id.set("P1")
        self._gcp_refresh_tree()
        self._gcp_render()
        self.gcp_status.set("已清空控制点")

    def _gcp_import(self) -> None:
        path = filedialog.askopenfilename(
            title="导入地面控制点标定",
            filetypes=(("GCP JSON", "*.json"), ("全部文件", "*.*")),
        )
        if not path:
            return
        try:
            payload = load_ground_control_file(path)
        except ValueError as error:
            messagebox.showerror("标定导入失败", str(error))
            return
        self.gcp_camera_id.set(str(payload["camera_id"]))
        self.gcp_coordinate_frame.set(str(payload["coordinate_frame"]))
        self.gcp_unit.set(str(payload["unit"]))
        image_size = payload.get("image_size")
        if isinstance(image_size, list) and len(image_size) == 2:
            self._gcp_image_size_value = (int(image_size[0]), int(image_size[1]))
            self.gcp_image_size.set(
                f"{self._gcp_image_size_value[0]} × {self._gcp_image_size_value[1]}"
            )
        self.gcp_rows = {point.point_id: point for point in payload["points"]}
        self.gcp_point_id.set(self._gcp_next_id())
        self._gcp_check_homography(show_error=False)
        self._gcp_render()
        self.gcp_status.set(f"已导入 {Path(path).name}；请确认画面与标定文件属于同一机位")

    def _gcp_export(self) -> None:
        check = self.gcp_check or self._gcp_check_homography()
        if check is None:
            return
        path = filedialog.asksaveasfilename(
            title="导出地面控制点标定",
            defaultextension=".json",
            filetypes=(("GCP JSON", "*.json"),),
        )
        if not path:
            return
        image_size: tuple[int, int] | None = None
        if self._gcp_image is not None and hasattr(self._gcp_image, "shape"):
            image_size = (int(self._gcp_image.shape[1]), int(self._gcp_image.shape[0]))
        if image_size is None:
            image_size = self._gcp_image_size_value
        try:
            save_ground_control_file(
                path,
                camera_id=self.gcp_camera_id.get(),
                coordinate_frame=self.gcp_coordinate_frame.get(),
                unit=self.gcp_unit.get(),
                image_size=image_size,
                points=list(self.gcp_rows.values()),
                check=check,
            )
        except ValueError as error:
            messagebox.showerror("标定导出失败", str(error))
            return
        self.gcp_status.set(f"已导出标定文件：{Path(path).name}")

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
            self.transport_combo.configure(state="disabled")
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
        self.worker.stop(timeout=None)
        self.transport_combo.configure(state="readonly")
        self.status.set("已停止")

    def _new_action_publisher(self, mode: str):
        """Create the selected event transport without breaking model startup."""

        selected = normalize_transport_mode(mode)
        label = transport_label(selected)
        try:
            publisher = action_publisher_from_environment(selected)
        except Exception as error:
            self.transport_status.set(f"{label}：不可用")
            self.status.set(f"{label} 初始化失败：{error}")
            return None
        if publisher is None:
            self.transport_status.set(f"{label}：未配置")
        else:
            self.transport_status.set(f"{label}：已启用")
        return publisher

    def _transport_mode_changed(self) -> None:
        """Switch event transport while the capture worker is stopped."""

        try:
            selected = normalize_transport_mode(self.transport_mode.get())
        except ValueError:
            self.status.set("不支持的动作传输方式；仅可选择 MQTT 或 BLE Light")
            return
        label = transport_label(selected)
        if self.pipeline is None:
            return
        if self.worker is not None and self.worker.running:
            if self._active_transport_mode is not None:
                self.transport_mode.set(transport_label(self._active_transport_mode))
            messagebox.showwarning("传输方式未切换", "请先停止采集，再切换 MQTT 或 BLE Light。")
            return
        try:
            publisher = action_publisher_from_environment(selected)
        except Exception as error:
            if self._active_transport_mode is not None:
                self.transport_mode.set(transport_label(self._active_transport_mode))
            self.transport_status.set(f"{label}：不可用")
            messagebox.showerror("传输方式不可用", str(error))
            return
        self.pipeline.set_action_publisher(publisher)
        self._active_transport_mode = selected.value
        if publisher is None:
            self.transport_status.set(f"{label}：未配置")
        else:
            self.transport_status.set(f"{label}：已启用")
        self.status.set(f"动作事件传输已切换为 {label}")

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

    def _sync_manual_action_target(self) -> None:
        """Mirror the selected Track row into the action target selector."""

        selected = self.track_tree.selection()
        if not selected:
            return
        try:
            track_id = int(selected[0])
        except (TypeError, ValueError):
            return
        for label, candidate_track_id in self._manual_action_targets.items():
            if candidate_track_id == track_id:
                self.manual_action_target.set(label)
                return

    def _refresh_manual_action_targets(self, tracks: tuple[object, ...]) -> None:
        """Expose current Tracks and their known identity labels to the GUI."""

        previous = self.manual_action_target.get()
        targets: dict[str, int] = {}
        for track in tracks:
            track_id = int(track.track_id)
            decision = track.decision
            identity = (
                decision.identity_id
                if decision.open_set_label == "KNOWN" and decision.identity_id
                else "未确认"
            )
            label = f"Track {track_id} · {identity}"
            targets[label] = track_id
        self._manual_action_targets = targets
        labels = tuple(targets)
        self.manual_action_target_combo.configure(values=labels)
        if previous in targets:
            self.manual_action_target.set(previous)
        elif labels:
            self.manual_action_target.set(labels[0])
        else:
            self.manual_action_target.set("")
        state = "readonly" if labels else "disabled"
        self.manual_action_target_combo.configure(state=state)
        self.manual_action_combo.configure(state=state)
        self.manual_action_button.configure(state="normal" if labels else "disabled")

    def trigger_manual_action(self) -> None:
        """Queue the selected action without bypassing identity authorization."""

        if self.worker is None:
            self.status.set("模型仍在后台加载，请稍候")
            return
        target = self._manual_action_targets.get(self.manual_action_target.get())
        if target is None:
            selected = self.track_tree.selection()
            if selected:
                try:
                    target = int(selected[0])
                except (TypeError, ValueError):
                    target = None
        if target is None:
            messagebox.showwarning("未选择目标", "请先在当前目标表中选择一个 Track 或身份。")
            return
        action = ActionType.parse(self.manual_action_type.get())
        try:
            self.worker.trigger_manual_action(target, action)
        except Exception as error:
            messagebox.showerror("手动动作失败", str(error))
            return
        self.status.set(
            f"已为 Track {target} 排队 {action.value}；下一帧将验证身份并发布动作事件"
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
        self.worker.stop(timeout=None)
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
        self._manual_action_targets.clear()
        self.manual_action_target.set("")
        self.manual_action_target_combo.configure(values=(), state="disabled")
        self.manual_action_combo.configure(state="disabled")
        self.manual_action_button.configure(state="disabled")
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
                track.action_type,
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
        self._refresh_manual_action_targets(result.tracks)
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
        if result.transport_errors:
            self.status.set("动作事件传输失败：" + "；".join(result.transport_errors[:2]))
        elif request_message:
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
                if message.level in {"ended", "error"}:
                    self.transport_combo.configure(state="readonly")
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
            self.worker.stop(timeout=None)
        if self.pipeline is not None:
            self.pipeline.close()
        self.verifier.close()
        self.root.destroy()


def launch_tk_gui(
    database_path: str = "data/verifier-production-v1.sqlite3",
    vision_backend: str = "production",
    action_classifier_path: Path | None = None,
    posec3d_checkpoint_path: Path | None = None,
) -> int:
    """启动独立桌面应用，并阻塞在 Tk 事件循环中。"""

    window = VerifierWindow(
        database_path,
        vision_backend,
        action_classifier_path,
        posec3d_checkpoint_path,
    )
    window.root.mainloop()
    return 0


def launch_gui(
    database_path: str = "data/verifier-production-v1.sqlite3",
    vision_backend: str = "production",
    action_classifier_path: Path | None = None,
    posec3d_checkpoint_path: Path | None = None,
) -> int:
    """启动 Qt 表现层；保留 ``launch_tk_gui`` 作为兼容回退入口。"""

    try:
        from .qt_gui import launch_gui as launch_qt_gui
    except ImportError as error:
        raise RuntimeError(
            "Qt GUI requires PyQt6; install it with `pip install -e .[gui-qt]`."
        ) from error

    return launch_qt_gui(
        database_path,
        vision_backend,
        action_classifier_path,
        posec3d_checkpoint_path,
    )


__all__ = ["VerifierWindow", "launch_gui", "launch_tk_gui"]
