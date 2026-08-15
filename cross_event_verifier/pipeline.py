"""独立桌面 GUI 使用的帧到验证管线。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Sequence
from uuid import uuid4

import cv2
import numpy as np

from .automation import (
    AutomationPolicy,
    AutomationStatus,
    AutomaticVerificationController,
)
from .engine import CrossEventVerifier
from .runtime_parameters import (
    RuntimeParameterController,
    RuntimeParameterSpec,
    RuntimeParameterState,
)
from .types import Decision, FeatureBundle, Observation, TrackQuality
from .vision import Box, VisionAdapter, VisionTrack


@dataclass(frozen=True)
class PipelineTrack:
    """一条待渲染轨迹及其决策和自动化状态。"""

    track_id: int
    box: Box
    features: FeatureBundle
    quality: TrackQuality
    decision: Decision
    automation: AutomationStatus


@dataclass(frozen=True)
class FrameResult:
    """从媒体工作线程传递给 GUI 的不可变帧消息。"""

    frame_bgr: np.ndarray
    tracks: tuple[PipelineTrack, ...]
    frame_index: int
    timestamp: float
    processing_seconds: float
    formal_identities: tuple[str, ...]
    pending_request_ids: tuple[str, ...]


class VideoVerifierPipeline:
    """对 GUI 隐藏帧记录细节的深模块。

    接口：

    * :meth:`set_source` 开始新的来源会话，同时保留正式身份图库；
    * :meth:`process_frame` 接收一帧 BGR 图像并返回可渲染结果；
    * :meth:`register_current_track` 将所选轨迹的最新特征快照登记到正式记忆。

    GUI 无需知道轨迹 ID、事件 ID、质量元数据或候选 ID 是如何构造的。
    """

    def __init__(
        self,
        verifier: CrossEventVerifier,
        vision: VisionAdapter,
        *,
        camera_id: str = "camera-1",
        automation_policy: AutomationPolicy | None = None,
    ) -> None:
        """连接验证器、可替换视觉适配器和自动化策略。"""
        self.verifier = verifier
        self.vision = vision
        self.automation = AutomaticVerificationController(
            verifier,
            automation_policy,
        )
        self._runtime_parameters = RuntimeParameterController(
            verifier,
            self.automation,
            vision,
        )
        self.camera_id = camera_id
        self.capture_session_id = f"capture-{uuid4().hex}"
        self.frame_index = 0
        self._latest: dict[int, VisionTrack] = {}
        # 保留一段输入源本地的短期历史供操作员登记使用。生产环境中的单帧可能
        # 短暂丢失腿部关键点或触碰边界，即使同一 Track 之前已经提供了强步态证据。
        # 手工登记应使用那份有效证据，而不是让最后一个瞬时帧决定结果。
        self._track_history: dict[int, deque[VisionTrack]] = {}

    @property
    def appearance_request_id(self) -> str | None:
        """保留操作员提供的全局令牌，以兼容旧调用方式。"""

        return self.automation.manual_request_id

    @property
    def automatic_registration_enabled(self) -> bool:
        """返回当前的自动创建新 ID 设置。"""
        return self.automation.registration_enabled

    @property
    def runtime_parameter_specs(self) -> tuple[RuntimeParameterSpec, ...]:
        """公开用于构造 GUI 参数页的说明。"""
        return self._runtime_parameters.specs

    def runtime_parameter_defaults(self) -> dict[str, int | float]:
        """返回重置值，但不实际应用。"""
        return self._runtime_parameters.defaults()

    def runtime_parameter_state(self) -> RuntimeParameterState:
        """返回当前已原子应用的运行时快照。"""
        return self._runtime_parameters.state()

    def update_runtime_parameters(
        self,
        updates: dict[str, object],
    ) -> RuntimeParameterState:
        """在两帧处理之间应用一组已经校验的参数事务。"""

        state = self._runtime_parameters.apply(updates)
        if updates:
            # 手工登记历史同样属于证据。不要把旧阈值下捕获的快照与新的运行时
            # 策略混合使用。
            self._track_history.clear()
        return state

    def set_source(self, camera_id: str, *, capture_session_id: str | None = None) -> None:
        """开始新的摄像头/文件会话，但不删除正式身份。"""

        self.camera_id = camera_id.strip() or "camera-1"
        self.capture_session_id = capture_session_id or f"capture-{uuid4().hex}"
        self.frame_index = 0
        self._latest.clear()
        self._track_history.clear()
        self.automation.reset_tracks()
        self.vision.reset()

    def set_automatic_registration(self, enabled: bool) -> None:
        """启用或禁用根据稳定步态创建新 ID。"""

        self.automation.set_registration_enabled(enabled)

    def set_appearance_request(
        self,
        request_id: str | None,
        track_id: int | None = None,
    ) -> None:
        """为一条轨迹设置一次性令牌，或将其作为全局回退令牌。"""

        if track_id is None and self._latest:
            # 令牌只能使用一次，绝不能在多人画面中广播给每个人。没有显式选择时，
            # 将它绑定到当前最大目标（单摄像头场景下的便利行为）。
            track_id = max(
                self._latest,
                key=lambda item: (
                    self._latest[item].box[2] - self._latest[item].box[0]
                ) * (self._latest[item].box[3] - self._latest[item].box[1]),
            )
        candidate_id = (
            self._candidate_id(
                self.camera_id,
                self.capture_session_id,
                int(track_id),
            )
            if track_id is not None
            else None
        )
        self.automation.set_manual_request(
            request_id,
            candidate_id=candidate_id,
        )

    @staticmethod
    def _candidate_id(camera_id: str, session_id: str, track_id: int) -> str:
        """构造在连续帧之间保持稳定的输入源本地候选键。"""
        return f"{camera_id}:{session_id}:track-{track_id}"

    @staticmethod
    def _color(decision: Decision) -> tuple[int, int, int]:
        """选择反映决策状态的 BGR 叠加颜色。"""
        if decision.kind.value == "conflict":
            return 0, 0, 255
        if decision.kind.value in {
            "formal_match",
            "appearance_requested",
            "appearance_response_accepted",
        }:
            return 0, 190, 0
        if decision.kind.value == "deferred":
            # deferred 不是确认身份；使用黄色避免把“疑似 P1”误显示成已识别。
            return 0, 190, 255
        if decision.kind.value == "ambiguous":
            # Top-2 步态过近时明确显示歧义，不把它伪装成普通未知或旧身份。
            return 0, 120, 255
        if decision.kind.value in {"candidate_created", "candidate_updated"}:
            return 0, 190, 255
        return 150, 150, 150

    @staticmethod
    def _label(track: PipelineTrack) -> str:
        """为叠加层格式化身份、决策、分数和请求状态。"""
        decision = track.decision
        subject = decision.identity_id or decision.candidate_id or f"T{track.track_id}"
        score = "-" if decision.score is None else f"{decision.score:.2f}"
        request = "  gait->appearance" if decision.appearance_request_id else ""
        return f"{subject} | {decision.kind.value} | {score}{request}"

    def _annotate(self, frame: np.ndarray, tracks: Sequence[PipelineTrack]) -> np.ndarray:
        """在帧副本上绘制检测框和紧凑的决策标签。"""
        rendered = frame.copy()
        for track in tracks:
            x1, y1, x2, y2 = track.box
            color = self._color(track.decision)
            cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
            label = self._label(track)
            baseline_y = max(18, y1 - 8)
            cv2.rectangle(
                rendered,
                (x1, max(0, baseline_y - 18)),
                (min(rendered.shape[1] - 1, x1 + max(160, len(label) * 8)), baseline_y + 3),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                rendered,
                label,
                (x1 + 3, baseline_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
        return rendered

    def process_frame(self, frame_bgr: np.ndarray, *, timestamp: float | None = None) -> FrameResult:
        """处理一帧并返回所有可供 UI 使用的验证结果。

        这是管线唯一的帧接口：视觉模块产出轨迹，观察值获得来源信息，自动化
        模块调用批量验证，最后为显示结果添加标注。GUI 不会自行执行这些步骤。
        """

        started = time.perf_counter()
        now = time.time() if timestamp is None else float(timestamp)
        self.frame_index += 1
        vision_tracks = self.vision.process(frame_bgr)
        prepared: list[tuple[VisionTrack, str, Observation]] = []
        for item in vision_tracks:
            event_id = (
                f"{self.capture_session_id}:frame-{self.frame_index}:track-{item.track_id}"
            )
            candidate_id = self._candidate_id(
                self.camera_id,
                self.capture_session_id,
                item.track_id,
            )
            prepared.append(
                (
                    item,
                    candidate_id,
                    Observation(
                        event_id=event_id,
                        camera_id=self.camera_id,
                        capture_session_id=self.capture_session_id,
                        track_id=str(item.track_id),
                        timestamp=now,
                        features=item.features,
                        quality=item.quality,
                        model_version=str(
                            getattr(
                                self.vision,
                                "model_version",
                                self.verifier.config.model_version,
                            )
                        ),
                        threshold_version=self.verifier.config.threshold_version,
                        metadata={
                            "source": "standalone-gui",
                            "vision_adapter": type(self.vision).__name__,
                            "vision_backend": str(
                                getattr(self.vision, "backend_status", "unknown")
                            ),
                        },
                    ),
                )
            )
        verified = self.automation.verify_batch(
            [item[2] for item in prepared],
            candidate_ids=[item[1] for item in prepared],
        )
        tracks: list[PipelineTrack] = []
        for (item, _, _), (decision, automation) in zip(prepared, verified):
            tracks.append(
                PipelineTrack(
                    track_id=item.track_id,
                    box=item.box,
                    features=item.features,
                    quality=item.quality,
                    decision=decision,
                    automation=automation,
                )
            )
            self._latest[item.track_id] = item
            self._remember_track(item)
        active_ids = {item.track_id for item in vision_tracks}
        for track_id in tuple(self._latest):
            if track_id not in active_ids:
                # 跟踪器仍存活时保留最新快照；一旦它从适配器输出中消失，就丢弃。
                if not hasattr(self.vision, "latest") or track_id not in getattr(self.vision, "latest", {}):
                    del self._latest[track_id]
                    self._track_history.pop(track_id, None)
        rendered = self._annotate(frame_bgr, tracks)
        return FrameResult(
            frame_bgr=rendered,
            tracks=tuple(tracks),
            frame_index=self.frame_index,
            timestamp=now,
            processing_seconds=time.perf_counter() - started,
            formal_identities=self.verifier.formal_identities,
            pending_request_ids=tuple(
                item.request_id
                for item in self.verifier.pending_appearance_requests(now=now)
            ),
        )

    def latest_track_ids(self) -> tuple[int, ...]:
        """返回登记控件当前可用的轨迹 ID。"""

        return tuple(sorted(self._latest))

    def _remember_track(self, item: VisionTrack) -> None:
        """在跟踪器身份保持稳定期间保留近期快照。"""

        history = self._track_history.setdefault(
            item.track_id,
            deque(maxlen=max(self.automation.policy.gait_sample_window, 16)),
        )
        if history:
            previous = history[-1].quality
            current = item.quality
            # 生产适配器会在 ID 切换后清除时序步态状态。手工登记历史也要这样做，
            # 防止新人物继承旧轨迹的强快照。
            if (
                current.id_switches > previous.id_switches
                or current.frame_count < previous.frame_count
            ):
                history.clear()
        history.append(item)

    def _best_manual_gait_snapshot(self, track_id: int) -> VisionTrack:
        """为手工登记选择近期最强的步态快照。

        选择近期最强的有效帧，可以避免因为最后一帧短暂触碰图像边界或丢失一个
        脚踝关键点而拒绝操作员登记。
        """

        history = tuple(self._track_history.get(track_id, ()))
        if not history:
            latest = self._latest.get(track_id)
            if latest is None:
                raise ValueError(f"track {track_id} is not currently available")
            history = (latest,)
        gait_snapshots = [item for item in history if item.features.has_gait]
        if not gait_snapshots:
            raise ValueError("人工登记需要可用的步态特征，请让目标完整行走后重试")
        return max(
            gait_snapshots,
            key=lambda item: item.quality.gait_availability(
                self.verifier.config.minimum_frames,
                self.verifier.config.minimum_gait_cycles,
            ),
        )

    def register_current_track(self, identity_id: str, track_id: int | None = None) -> int:
        """将当前轨迹的特征快照登记到正式记忆。

        返回已登记的轨迹 ID。调用方可以省略 ``track_id``，此时使用当前可用的
        最大轨迹，适合单人摄像头场景。
        """

        identity_id = identity_id.strip()
        if not identity_id:
            raise ValueError("identity_id cannot be empty")
        if not self._latest:
            raise ValueError("no tracked person is available")
        if track_id is None:
            track_id = max(
                self._latest,
                key=lambda item: (
                    self._latest[item].box[2] - self._latest[item].box[0]
                ) * (self._latest[item].box[3] - self._latest[item].box[1]),
            )
        snapshot = self._best_manual_gait_snapshot(int(track_id))
        gait_quality = snapshot.quality.gait_availability(
            self.verifier.config.minimum_frames,
            self.verifier.config.minimum_gait_cycles,
        )
        if gait_quality < self.verifier.config.strong_gait_quality:
            raise ValueError(
                "人工登记需要强步态质量："
                f"当前最佳 Pg={gait_quality:.2f}/"
                f"{self.verifier.config.strong_gait_quality:.2f}；"
                "请让目标完整入镜并连续行走后重试"
            )
        candidate_id = self._candidate_id(
            self.camera_id,
            self.capture_session_id,
            snapshot.track_id,
        )
        observation = Observation(
            event_id=f"{self.capture_session_id}:enrollment:track-{snapshot.track_id}",
            camera_id=self.camera_id,
            capture_session_id=self.capture_session_id,
            track_id=str(snapshot.track_id),
            features=FeatureBundle(gait=snapshot.features.normalized().gait),
            quality=snapshot.quality,
            metadata={
                "source": "standalone-gui-manual-gait-enrollment",
                "vision_adapter": type(self.vision).__name__,
                "gait_quality": gait_quality,
                "source_frame_count": snapshot.quality.frame_count,
            },
        )
        decision = self.verifier.enroll_gait_identity(
            observation,
            identity_id=identity_id,
            candidate_id=candidate_id,
            # 操作员提供身份意图，但验证器仍然强制执行强步态质量门控。
            gait_confidence=1.0,
        )
        if decision.appearance_request_id:
            self.automation.bind_request(
                candidate_id,
                decision.appearance_request_id,
                identity_id=decision.identity_id,
            )
        return snapshot.track_id


__all__ = ["FrameResult", "PipelineTrack", "VideoVerifierPipeline"]
