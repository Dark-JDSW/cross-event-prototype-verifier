"""用于摄像头/视频验证的独立 Tk 桌面应用。

Tk 只负责展示。``FrameWorker`` 执行采集和推理，``VideoVerifierPipeline`` 负责
编排，实时参数页提交经过校验的事务，而不是直接编辑验证器内部状态。
"""

from __future__ import annotations

from pathlib import Path
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

from ..participant_a.engine import CrossEventVerifier
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
from .runtime_parameters import RuntimeParameterState
from .storage import SqliteStore
from ..participant_b.vision_factory import build_vision_adapter


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
        """创建验证器、所选视觉后端、工作线程和 Tk 布局。"""
        self.root = tk.Tk()
        self.root.title("Cross-event Prototype Verifier")
        self.root.geometry("1440x860")
        self.root.minsize(980, 640)

        database = Path(database_path)
        if str(database) != ":memory:":
            database.parent.mkdir(parents=True, exist_ok=True)
        self.verifier = CrossEventVerifier(store=SqliteStore(str(database)))
        self.vision = build_vision_adapter(vision_backend)
        automatic_capable = bool(
            getattr(self.vision, "supports_automatic_registration", False)
        )
        self.pipeline = VideoVerifierPipeline(
            self.verifier,
            self.vision,
            automation_policy=AutomationPolicy(enabled=automatic_capable),
        )
        self.worker = FrameWorker(self.pipeline)
        self._photo: tk.PhotoImage | None = None

        self.source_kind = tk.StringVar(value="camera")
        self.camera_index = tk.StringVar(value="0")
        self.video_path = tk.StringVar()
        self.camera_id = tk.StringVar(value="camera-1")
        self.identity_id = tk.StringVar(value="P1")
        self.automatic_registration = tk.BooleanVar(value=automatic_capable)
        self.automation_status = tk.StringVar(
            value=(
                "自动注册：开启，等待人物进入画面"
                if automatic_capable
                else "诊断后端：自动注册已安全关闭"
            )
        )
        self.backend_status = tk.StringVar(
            value=f"视觉后端：{getattr(self.vision, 'backend_status', type(self.vision).__name__)}"
        )
        self.appearance_request_id = tk.StringVar()
        self.pending_requests = tk.StringVar(value="无")
        self.status = tk.StringVar(value="请选择摄像头或视频文件，然后点击“开始”")
        self.parameter_status = tk.StringVar(value="参数尚未修改")
        self.parameter_vars: dict[str, tk.StringVar] = {}
        self.parameter_entries: dict[str, ttk.Entry] = {}
        self._runtime_parameter_state = self.pipeline.runtime_parameter_state()

        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(50, self._poll_messages)

    def _build_layout(self) -> None:
        """构造输入源控件、监控页、登记面板和状态栏。"""
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(self.root, text="输入源")
        controls.grid(row=0, column=0, padx=10, pady=(10, 6), sticky="ew")
        controls.columnconfigure(4, weight=1)

        ttk.Radiobutton(
            controls,
            text="摄像头",
            variable=self.source_kind,
            value="camera",
            command=self._source_mode_changed,
        ).grid(row=0, column=0, padx=(8, 4), pady=8)
        ttk.Entry(controls, textvariable=self.camera_index, width=6).grid(row=0, column=1, padx=4)
        ttk.Label(controls, text="设备号").grid(row=0, column=2, padx=(0, 12))
        ttk.Radiobutton(
            controls,
            text="视频文件",
            variable=self.source_kind,
            value="file",
            command=self._source_mode_changed,
        ).grid(row=0, column=3, padx=4)
        self.file_entry = ttk.Entry(controls, textvariable=self.video_path)
        self.file_entry.grid(row=0, column=4, padx=4, sticky="ew")
        self.browse_button = ttk.Button(controls, text="浏览…", command=self._browse_video)
        self.browse_button.grid(row=0, column=5, padx=4)
        ttk.Label(controls, text="来源 ID").grid(row=0, column=6, padx=(12, 4))
        ttk.Entry(controls, textvariable=self.camera_id, width=16).grid(row=0, column=7, padx=4)
        ttk.Button(controls, text="开始", command=self.start).grid(row=0, column=8, padx=(12, 4))
        ttk.Button(controls, text="停止", command=self.stop).grid(row=0, column=9, padx=(4, 8))
        ttk.Label(
            controls,
            textvariable=self.backend_status,
            foreground="#315a7d",
        ).grid(row=1, column=0, columnspan=10, padx=8, pady=(0, 7), sticky="w")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, padx=10, pady=6, sticky="nsew")
        monitor_page = ttk.Frame(self.notebook)
        parameter_page = ttk.Frame(self.notebook)
        self.notebook.add(monitor_page, text="实时识别")
        self.notebook.add(parameter_page, text="实时参数")

        body = ttk.Frame(monitor_page)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        self.video_label = tk.Label(
            body,
            text="没有画面",
            background="#20242b",
            foreground="#d8dee9",
            anchor="center",
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")

        side = ttk.Frame(body, width=520)
        side.grid(row=0, column=1, padx=(10, 0), sticky="ns")
        side.grid_propagate(False)
        ttk.Label(side, text="当前目标", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))
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
        self.track_tree.pack(fill="x", expand=False)

        enrollment = ttk.LabelFrame(side, text="人物注册")
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
        ttk.Button(
            enrollment,
            text="人工登记选中目标（兜底）",
            command=self.register_selected,
        ).grid(row=3, column=0, columnspan=2, padx=6, pady=(0, 8), sticky="ew")
        ttk.Label(
            enrollment,
            text="自动流程只用强步态建号；外观必须经一次性请求后才会吸收。",
            wraplength=470,
        ).grid(row=4, column=0, columnspan=2, padx=6, pady=(0, 8), sticky="w")

        request_box = ttk.LabelFrame(side, text="外观吸收（自动；此处为人工兜底）")
        request_box.pack(fill="x", pady=(14, 0))
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

        gallery = ttk.LabelFrame(side, text="当前正式身份")
        gallery.pack(fill="x", pady=(14, 0))
        self.gallery_label = ttk.Label(gallery, text="无")
        self.gallery_label.pack(anchor="w", padx=8, pady=8)

        self._build_parameter_page(parameter_page)

        status = ttk.Label(self.root, textvariable=self.status, anchor="w")
        status.grid(row=2, column=0, padx=10, pady=(2, 10), sticky="ew")
        self._source_mode_changed()

    def _build_parameter_page(self, page: ttk.Frame) -> None:
        """根据运行时模块的参数说明构造可滚动页面。"""

        page.columnconfigure(0, weight=1)
        page.rowconfigure(1, weight=1)
        toolbar = ttk.Frame(page)
        toolbar.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        ttk.Button(
            toolbar,
            text="应用到运行中",
            command=self.apply_runtime_parameters,
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
            foreground="#315a7d",
        ).pack(side="left", padx=(18, 0))
        ttk.Label(
            page,
            text=(
                "所有输入会先整组校验，再在采集线程的帧边界一次性生效。"
                "参数仅影响本次程序运行；生产阈值应以目标摄像头验证集为依据。"
            ),
            wraplength=1250,
        ).grid(row=2, column=0, sticky="w", padx=12, pady=(4, 10))

        canvas = tk.Canvas(page, highlightthickness=0)
        scrollbar = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=1, column=0, sticky="nsew", padx=(10, 0))
        scrollbar.grid(row=1, column=1, sticky="ns", padx=(0, 10))
        inner = ttk.Frame(canvas)
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
            group = ttk.LabelFrame(inner, text=section)
            group.grid(
                row=section_index // 2,
                column=section_index % 2,
                padx=6,
                pady=6,
                sticky="nsew",
            )
            group.columnconfigure(2, weight=1)
            rows = [
                item
                for item in self.pipeline.runtime_parameter_specs
                if item.section == section
            ]
            for row_index, spec in enumerate(rows):
                variable = tk.StringVar()
                entry = ttk.Entry(group, textvariable=variable, width=10)
                self.parameter_vars[spec.key] = variable
                self.parameter_entries[spec.key] = entry
                ttk.Label(group, text=spec.label, width=19).grid(
                    row=row_index,
                    column=0,
                    padx=(8, 4),
                    pady=4,
                    sticky="w",
                )
                entry.grid(row=row_index, column=1, padx=4, pady=4, sticky="w")
                ttk.Label(
                    group,
                    text=(
                        f"[{spec.minimum:g}～{spec.maximum:g}] "
                        f"{spec.description}"
                    ),
                    wraplength=390,
                    foreground="#555555",
                ).grid(
                    row=row_index,
                    column=2,
                    padx=(5, 8),
                    pady=4,
                    sticky="w",
                )
        self._load_runtime_parameter_state(self._runtime_parameter_state)

    def _load_runtime_parameter_state(self, state: RuntimeParameterState) -> None:
        """渲染已校验快照，并禁用当前后端不适用的字段。"""
        self._runtime_parameter_state = state
        available = set(state.available_keys)
        for spec in self.pipeline.runtime_parameter_specs:
            variable = self.parameter_vars.get(spec.key)
            entry = self.parameter_entries.get(spec.key)
            if variable is None or entry is None:
                continue
            entry.configure(state="normal")
            if spec.key in available:
                variable.set(spec.format(state.values[spec.key]))
            else:
                variable.set("当前后端不可用")
                entry.configure(state="disabled")
        self.parameter_status.set(f"当前运行时参数版本：{state.revision}")

    def apply_runtime_parameters(self) -> None:
        """收集可见表单值，并排队一组原子工作线程事务。"""
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
        self.worker.set_runtime_parameters({})
        self.parameter_status.set("正在读取当前生效值…")

    def restore_default_parameters(self) -> None:
        """将默认值填入表单；用户仍需点击应用。"""
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
        if self.source_kind.get() == "file":
            path = self.video_path.get().strip()
            if not path:
                raise ValueError("请先选择视频文件")
            return SourceSpec("file", path, Path(path).name)
        try:
            index = int(self.camera_index.get().strip())
        except ValueError as error:
            raise ValueError("摄像头设备号必须是整数") from error
        label = self.camera_id.get().strip() or f"camera-{index}"
        return SourceSpec("camera", index, label)

    def start(self) -> None:
        """校验所选摄像头或文件输入源后开始采集。"""
        try:
            spec = self._source_spec()
            self.worker.start(spec)
            self.status.set(f"正在打开：{spec.label}")
        except Exception as error:
            messagebox.showerror("无法开始", str(error))

    def stop(self) -> None:
        """停止采集并更新状态栏，但不销毁窗口。"""
        self.worker.stop()
        self.status.set("已停止")

    def _toggle_automatic_registration(self) -> None:
        """排队切换自动化；拒绝在不安全诊断后端上启用自动注册。"""
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
            "自动注册：开启，稳定强步态将自动生成 P 身份"
            if enabled
            else "自动注册：关闭；识别和外观令牌响应仍继续"
        )

    def register_selected(self) -> None:
        """为选中的 Track 或当前最佳 Track 排队手工登记。"""
        identity_id = self.identity_id.get().strip()
        if not identity_id:
            messagebox.showwarning("缺少身份 ID", "请输入身份 ID，例如 P001")
            return
        selected = self.track_tree.selection()
        track_id = int(selected[0]) if selected else None
        self.worker.register_identity(identity_id, track_id)
        self.status.set("登记请求已排队，将在采集线程安全执行")

    def apply_appearance_request(self) -> None:
        """排队一次性外观令牌，可选绑定到选中的 Track。"""
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

    def _update_frame(self, result: FrameResult) -> None:
        """渲染一帧处理结果、决策、自动化阶段和图库 ID。"""
        self.backend_status.set(
            f"视觉后端：{getattr(self.vision, 'backend_status', type(self.vision).__name__)}"
        )
        try:
            self._photo = _frame_to_photo(result.frame_bgr, master=self.root)
            self.video_label.configure(image=self._photo, text="")
        except Exception as error:
            self.status.set(f"画面显示失败：{error}")
        for item in self.track_tree.get_children():
            self.track_tree.delete(item)
        request_message = ""
        automation_messages: list[str] = []
        for track in result.tracks:
            decision = track.decision
            subject = decision.identity_id or decision.candidate_id or "-"
            score = "-" if decision.score is None else f"{decision.score:.3f}"
            self.track_tree.insert(
                "",
                "end",
                iid=str(track.track_id),
                values=(
                    track.track_id,
                    subject,
                    decision.kind.value,
                    track.automation.message,
                    score,
                ),
            )
            automation_messages.append(
                f"T{track.track_id}: {track.automation.message}"
            )
            if decision.appearance_request_id:
                self.appearance_request_id.set(decision.appearance_request_id)
                request_message = track.automation.message
            if decision.kind.value == "appearance_response_accepted":
                self.appearance_request_id.set("")
                request_message = track.automation.message
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
        while True:
            try:
                message = self.worker.messages.get_nowait()
            except queue.Empty:
                break
            if isinstance(message, FrameMessage):
                self._update_frame(message.result)
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
        self.root.after(50, self._poll_messages)

    def close(self) -> None:
        """停止后台工作、关闭 SQLite，并销毁 Tk 根窗口。"""
        self.worker.stop()
        self.verifier.close()
        self.root.destroy()


def launch_gui(
    database_path: str = "data/verifier-production-v1.sqlite3",
    vision_backend: str = "production",
) -> int:
    """启动独立桌面应用，并阻塞在 Tk 事件循环中。"""

    window = VerifierWindow(database_path, vision_backend)
    warning = getattr(window.vision, "startup_warning", None)
    if warning:
        window.root.after(100, lambda: messagebox.showwarning("视觉后端降级", warning))
    window.root.mainloop()
    return 0


__all__ = ["VerifierWindow", "launch_gui"]
