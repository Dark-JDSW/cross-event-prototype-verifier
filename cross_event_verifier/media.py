"""独立桌面 GUI 使用的 OpenCV 采集和工作线程适配器。

GUI 线程只负责渲染 Tk 控件。``FrameWorker`` 负责采集、管线调用和排队命令，
从而把摄像头 I/O 与 GPU 推理移出 Tk 事件循环，并为运行时参数更新提供帧边界
提交点。
"""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Literal, Mapping

import cv2

from .pipeline import FrameResult, VideoVerifierPipeline
from .runtime_parameters import RuntimeParameterState


class CaptureError(RuntimeError):
    """摄像头或视频文件无法打开时抛出。"""


@dataclass(frozen=True)
class SourceSpec:
    """描述待采集的摄像头索引或本地视频路径。"""

    kind: Literal["camera", "file"]
    value: int | str
    label: str
    candidate_id: str | None = None


class OpenCvCapture:
    """为文件和摄像头提供一致元数据的小型输入源适配器。

    Windows 摄像头优先尝试 DirectShow，然后使用 OpenCV 默认后端；文件则直接
    使用路径打开。
    """

    def __init__(self, spec: SourceSpec) -> None:
        """创建一个尚未打开的输入源采集包装器。"""
        self.spec = spec
        self.capture: cv2.VideoCapture | None = None
        self.fps = 0.0
        self.frame_count = 0
        self.width = 0
        self.height = 0

    def open(self) -> None:
        """打开输入源并缓存 FPS、尺寸和帧数。"""
        if self.spec.kind == "camera":
            index = int(self.spec.value)
            capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(index)
            # Do not let an overloaded inference thread accumulate seconds of
            # stale camera frames.  Backends may ignore this hint; processing
            # remains synchronous and the worker still reports its latency.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            capture = cv2.VideoCapture(str(self.spec.value))
        if not capture.isOpened():
            capture.release()
            raise CaptureError(f"无法打开输入源: {self.spec.label}")
        self.capture = capture
        self.fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    def read(self) -> tuple[bool, object]:
        """读取一帧；如果跳过 ``open``，则抛出明确错误。"""
        if self.capture is None:
            raise CaptureError("capture has not been opened")
        return self.capture.read()

    def release(self) -> None:
        """如果底层采集句柄已打开，则释放它。"""
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> "OpenCvCapture":
        """为 ``with`` 语句使用打开采集器。"""
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        """离开 ``with`` 块时释放采集器。"""
        self.release()


@dataclass(frozen=True)
class FrameMessage:
    """从工作线程发送到 GUI、携带一帧处理结果的消息。"""

    result: FrameResult


@dataclass(frozen=True)
class StatusMessage:
    """从工作线程发送到 GUI 的生命周期或错误状态。"""

    level: Literal["info", "error", "ended"]
    text: str


@dataclass(frozen=True)
class RegistrationMessage:
    """用户请求的手工登记操作结果。"""

    success: bool
    text: str


@dataclass(frozen=True)
class ParameterUpdateMessage:
    """一组运行时参数事务的结果及可选快照。"""

    success: bool
    text: str
    state: RuntimeParameterState | None = None


class FrameWorker:
    """在 GUI 事件循环之外执行采集和帧处理。

    命令通过队列传递，而不是在 Tk 回调中直接修改管线；这样可以保证时序状态
    始终由工作线程持有。
    """

    def __init__(self, pipeline: VideoVerifierPipeline) -> None:
        """为一个管线创建有界输出队列和无界命令队列。"""
        self.pipeline = pipeline
        self.messages: queue.Queue[
            FrameMessage | StatusMessage | RegistrationMessage | ParameterUpdateMessage
        ] = queue.Queue(
            maxsize=4
        )
        self._commands: queue.Queue[
            tuple[str, str | dict[str, object], int | None]
        ] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        """采集线程是否存活并接受排队命令。"""
        return self._thread is not None and self._thread.is_alive()

    def _put(
        self,
        message: FrameMessage | StatusMessage | RegistrationMessage | ParameterUpdateMessage,
    ) -> None:
        """发布消息；只丢弃旧帧，不丢弃控制/状态/登记结果。"""
        try:
            self.messages.put_nowait(message)
        except queue.Full:
            retained: list[object] = []
            removed_frame = False
            while True:
                try:
                    older = self.messages.get_nowait()
                except queue.Empty:
                    break
                if isinstance(older, FrameMessage) and not removed_frame:
                    removed_frame = True
                    continue
                retained.append(older)
            for older in retained:
                try:
                    self.messages.put_nowait(older)
                except queue.Full:
                    break
            if isinstance(message, FrameMessage) and not removed_frame:
                # A full queue containing only control messages must preserve
                # those messages; the newest frame is intentionally dropped.
                return
            if not removed_frame and not isinstance(message, FrameMessage):
                # Control/status messages are never silently discarded.  Block
                # until the GUI drains one when the bounded queue contains only
                # control messages; dropping a frame is preferable to losing a
                # registration result or an error notification.
                self.messages.put(message)
                return
            try:
                self.messages.put_nowait(message)
            except queue.Full:
                pass

    def start(self, spec: SourceSpec, *, candidate_id: str | None = None) -> None:
        """停止旧工作线程，并为 ``spec`` 启动守护线程。

        ``candidate_id`` 由采集任务/操作员显式提供，允许同一未知人物在
        不同输入源之间继续积累独立事件；缺省时保持会话隔离。
        """
        self.stop()
        self._stop.clear()
        effective_candidate = (
            candidate_id if candidate_id is not None else spec.candidate_id
        )
        if effective_candidate is not None and not effective_candidate.strip():
            raise ValueError("candidate_id cannot be empty")
        self._thread = threading.Thread(
            target=self._run,
            args=(spec, effective_candidate.strip() if effective_candidate else None),
            name="cross-event-capture",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """请求关闭并短暂等待，以便及时释放资源。"""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

    def register_identity(self, identity_id: str, track_id: int | None = None) -> None:
        """在采集线程排队登记操作，以保持管线状态局部性。"""

        if not self.running:
            try:
                enrolled_track = self.pipeline.register_current_track(identity_id, track_id)
                self._put(
                    RegistrationMessage(
                        True,
                        f"已登记身份 {identity_id}（track {enrolled_track}）",
                    )
                )
            except Exception as error:
                self._put(RegistrationMessage(False, f"登记失败：{error}"))
            return
        self._commands.put(("register", identity_id, track_id))

    def set_appearance_request(
        self,
        request_id: str | None,
        track_id: int | None = None,
    ) -> None:
        """附加或清除一次性外观响应令牌。"""

        value = request_id.strip() if request_id else ""
        if not self.running:
            self.pipeline.set_appearance_request(value or None, track_id)
            return
        self._commands.put(("request", value, track_id))

    def set_automatic_registration(self, enabled: bool) -> None:
        """排队一次自动注册策略变更。"""

        value = "1" if enabled else "0"
        if not self.running:
            self.pipeline.set_automatic_registration(enabled)
            return
        self._commands.put(("automation", value, None))

    def set_runtime_parameters(self, values: Mapping[str, object]) -> None:
        """在采集线程应用一组参数事务。"""

        payload = dict(values)
        if not self.running:
            self._apply_runtime_parameters(payload)
            return
        self._commands.put(("parameters", payload, None))

    def _apply_runtime_parameters(self, values: dict[str, object]) -> None:
        """应用参数映射，并发布成功消息或面向用户的错误。"""
        try:
            state = self.pipeline.update_runtime_parameters(values)
            action = "已读取当前参数" if not values else "参数已实时应用"
            self._put(
                ParameterUpdateMessage(
                    True,
                    f"{action}（运行时版本 {state.revision}）",
                    state,
                )
            )
        except Exception as error:
            self._put(ParameterUpdateMessage(False, f"参数应用失败：{error}"))

    def _drain_commands(self) -> None:
        """处理读取下一帧之前排队的所有命令。"""
        while True:
            try:
                command, payload, track_id = self._commands.get_nowait()
            except queue.Empty:
                return
            if command == "request":
                request_id = str(payload).strip()
                self.pipeline.set_appearance_request(request_id or None, track_id)
                continue
            if command == "automation":
                self.pipeline.set_automatic_registration(payload == "1")
                continue
            if command == "parameters":
                if isinstance(payload, dict):
                    self._apply_runtime_parameters(payload)
                continue
            if command != "register":
                continue
            identity_id = str(payload)
            try:
                enrolled_track = self.pipeline.register_current_track(identity_id, track_id)
                self._put(
                    RegistrationMessage(
                        True,
                        f"已登记身份 {identity_id}（track {enrolled_track}）",
                    )
                )
            except Exception as error:  # 将用户操作错误显示在 GUI 中
                self._put(RegistrationMessage(False, f"登记失败：{error}"))

    def _run(self, spec: SourceSpec, candidate_id: str | None) -> None:
        """持续采集、处理并发布帧，直到停止或输入结束。"""
        capture: OpenCvCapture | None = None
        try:
            capture = OpenCvCapture(spec)
            capture.open()
            session_id = f"{spec.kind}-{int(time.time() * 1000)}"
            self.pipeline.set_source(
                spec.label,
                capture_session_id=session_id,
                candidate_id=candidate_id,
            )
            session_start = time.time()
            last_maintenance = time.monotonic()
            fps_text = f"{capture.fps:.1f} FPS" if capture.fps > 1.0 else "实时输入"
            backend_status = str(
                getattr(
                    self.pipeline.vision,
                    "backend_status",
                    type(self.pipeline.vision).__name__,
                )
            )
            self._put(
                StatusMessage(
                    "info",
                    (
                        f"已打开：{spec.label} | {capture.width}×{capture.height} | "
                        f"{fps_text} | {backend_status}"
                    ),
                )
            )
            while not self._stop.is_set():
                self._drain_commands()
                ok, frame = capture.read()
                if not ok:
                    self._put(StatusMessage("ended", "视频已结束"))
                    break
                started = time.perf_counter()
                if spec.kind == "file" and capture.capture is not None:
                    position_ms = float(
                        capture.capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0
                    )
                    frame_timestamp = float(session_start + position_ms / 1000.0)
                else:
                    frame_timestamp = time.time()
                result = self.pipeline.process_frame(frame, timestamp=frame_timestamp)
                if time.monotonic() - last_maintenance >= 30.0:
                    removed_candidates = self.pipeline.verifier.maintenance(now=time.time())
                    self.pipeline.automation.discard_candidates(removed_candidates)
                    last_maintenance = time.monotonic()
                self._put(FrameMessage(result))
                if spec.kind == "file" and capture.fps > 1.0:
                    remaining = (1.0 / capture.fps) - (time.perf_counter() - started)
                    if remaining > 0:
                        self._stop.wait(min(remaining, 0.20))
        except CaptureError as error:
            self._put(StatusMessage("error", str(error)))
        except Exception as error:  # 保持 GUI 存活并报告工作线程错误
            self._put(StatusMessage("error", f"处理输入源失败：{error}"))
        finally:
            if capture is not None:
                capture.release()
            self._thread = None


__all__ = [
    "CaptureError",
    "FrameMessage",
    "FrameWorker",
    "OpenCvCapture",
    "ParameterUpdateMessage",
    "RegistrationMessage",
    "SourceSpec",
    "StatusMessage",
]
