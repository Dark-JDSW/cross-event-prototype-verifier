"""Toolkit neutral state models for the desktop GUIs.

This module intentionally has no Tk, Qt, OpenCV, model, or verifier imports.
Both desktop adapters use it to validate controls and present pipeline facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping


@dataclass(frozen=True)
class SourceSelection:
    kind: Literal["camera", "file"]
    value: int | str
    label: str
    candidate_id: str | None = None
    repeat_count: int = 1

    def as_source_args(self) -> tuple[str, int | str, str, str | None, int]:
        return (
            self.kind,
            self.value,
            self.label,
            self.candidate_id,
            self.repeat_count,
        )


@dataclass(frozen=True)
class ManualActionTargetsViewModel:
    targets: Mapping[str, int]
    selected_label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", MappingProxyType(dict(self.targets)))

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.targets)


@dataclass(frozen=True)
class TrackRowViewModel:
    track_id: str
    values: tuple[str, str, str, str, str, str]
    visual_tag: str


@dataclass(frozen=True)
class FrameViewModel:
    rows: tuple[TrackRowViewModel, ...]
    action_targets: ManualActionTargetsViewModel
    automation_text: str
    gallery_text: str
    pending_requests_text: str
    status_text: str
    appearance_request_id: str | None = None
    clear_appearance_request: bool = False


@dataclass(frozen=True)
class RegistrationToggleViewModel:
    enabled: bool
    enqueue: bool
    status_text: str | None = None
    warning_message: str | None = None


class GuiPresenter:
    """Convert widget values and pipeline results into immutable view models."""

    @staticmethod
    def source_selection(
        *,
        source_kind: str,
        camera_index: str,
        camera_id: str,
        video_path: str,
        video_repeat_count: str,
        candidate_id: str,
    ) -> SourceSelection:
        normalized_candidate = candidate_id.strip() or None
        if source_kind == "file":
            path = video_path.strip()
            if not path:
                raise ValueError("请先选择视频文件")
            try:
                repeat_count = int(video_repeat_count.strip())
            except ValueError as error:
                raise ValueError("视频重复学习次数必须是正整数") from error
            if repeat_count < 1:
                raise ValueError("视频重复学习次数必须是正整数")
            return SourceSelection(
                "file",
                path,
                Path(path).name,
                normalized_candidate,
                repeat_count,
            )
        try:
            index = int(camera_index.strip())
        except ValueError as error:
            raise ValueError("摄像头设备号必须是整数") from error
        label = camera_id.strip() or f"camera-{index}"
        return SourceSelection("camera", index, label, normalized_candidate)

    @staticmethod
    def track_visual_tag(kind: str) -> str:
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

    @staticmethod
    def manual_action_targets(
        tracks: tuple[object, ...],
        *,
        previous_label: str,
    ) -> ManualActionTargetsViewModel:
        targets: dict[str, int] = {}
        for track in tracks:
            track_id = int(getattr(track, "track_id"))
            decision = getattr(track, "decision")
            identity = (
                decision.identity_id
                if decision.open_set_label == "KNOWN" and decision.identity_id
                else "未确认"
            )
            targets[f"Track {track_id} · {identity}"] = track_id
        labels = tuple(targets)
        selected = previous_label if previous_label in targets else (labels[0] if labels else "")
        return ManualActionTargetsViewModel(targets, selected)

    @classmethod
    def frame(
        cls,
        result: Any,
        *,
        automatic_registration_enabled: bool,
        previous_action_target: str,
    ) -> FrameViewModel:
        rows: list[TrackRowViewModel] = []
        automation_messages: list[str] = []
        request_message = ""
        appearance_request_id: str | None = None
        clear_request = False
        for track in result.tracks:
            decision = track.decision
            subject = decision.identity_id or decision.candidate_id or "-"
            score = "-" if decision.score is None else f"{decision.score:.3f}"
            track_id = str(track.track_id)
            rows.append(
                TrackRowViewModel(
                    track_id=track_id,
                    values=(
                        track_id,
                        str(track.action_type),
                        subject,
                        decision.kind.value,
                        track.automation.message,
                        score,
                    ),
                    visual_tag=cls.track_visual_tag(decision.kind.value),
                )
            )
            automation_messages.append(f"T{track.track_id}: {track.automation.message}")
            if decision.appearance_request_id:
                appearance_request_id = decision.appearance_request_id
                clear_request = False
                request_message = track.automation.message
            if decision.kind.value == "appearance_response_accepted":
                appearance_request_id = None
                clear_request = True
                request_message = track.automation.message

        automation_text = (
            " | ".join(automation_messages[:2])
            if automation_messages
            else (
                "自动注册：开启，等待人物进入画面"
                if automatic_registration_enabled
                else "自动注册：关闭"
            )
        )
        status_text = (
            f"帧 {result.frame_index} | 目标 {len(result.tracks)} | "
            f"处理 {result.processing_seconds * 1000:.0f} ms"
        )
        if result.transport_errors:
            status_text = "动作事件传输失败：" + "；".join(result.transport_errors[:2])
        elif request_message:
            status_text = request_message
        return FrameViewModel(
            rows=tuple(rows),
            action_targets=cls.manual_action_targets(
                tuple(result.tracks),
                previous_label=previous_action_target,
            ),
            automation_text=automation_text,
            gallery_text=", ".join(result.formal_identities) or "无",
            pending_requests_text=(
                "\n".join(result.pending_request_ids[:3])
                if result.pending_request_ids
                else "无"
            ),
            status_text=status_text,
            appearance_request_id=appearance_request_id,
            clear_appearance_request=clear_request,
        )

    @staticmethod
    def appearance_request(
        request_id: str,
        track_id: int | None,
    ) -> tuple[str | None, str]:
        normalized = request_id.strip() or None
        if normalized is None:
            return None, "已清除外观响应令牌"
        if track_id is not None:
            return normalized, f"外观响应令牌已绑定到 Track {track_id}"
        return normalized, "外观响应令牌将应用到后续目标"

    @staticmethod
    def automatic_registration(
        requested: bool,
        *,
        runtime_ready: bool,
        backend_supported: bool,
    ) -> RegistrationToggleViewModel:
        if not runtime_ready:
            return RegistrationToggleViewModel(
                enabled=False,
                enqueue=False,
                status_text="模型仍在后台加载，请稍候",
            )
        if requested and not backend_supported:
            return RegistrationToggleViewModel(
                enabled=False,
                enqueue=False,
                warning_message=(
                    "HOG 诊断后端不能提供强步态证据，请使用 production 后端。"
                ),
            )
        return RegistrationToggleViewModel(
            enabled=requested,
            enqueue=True,
            status_text=(
                "自动注册：开启，OSNet 将先确认视觉身份，随后学习步态原型"
                if requested
                else "自动注册：关闭；识别和外观令牌响应仍继续"
            ),
        )

    @staticmethod
    def selected_action_target(
        targets: Mapping[str, int],
        selected_label: str,
        fallback_track_id: int | None,
    ) -> int | None:
        return targets.get(selected_label, fallback_track_id)

    @staticmethod
    def runtime_parameter_updates(
        available_keys: tuple[str, ...],
        values: Mapping[str, object],
    ) -> dict[str, object]:
        available = set(available_keys)
        return {key: value for key, value in values.items() if key in available}
