"""动作类型、质量和路由契约。

V1 只负责把动作证据和现有 WALK/GaitGraph2 分支隔离开。动作分类器本身可以
在适配器或离线模型中实现，但下游不应再依赖未版本化的字符串或把低置信度
动作当成 WALK。没有动作预测时保留旧 API 的兼容行为；一旦调用方显式提供
动作结果，路由器就采用保守的显式门控。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np


class ActionType(str, Enum):
    """当前规划支持的动作类型。"""

    WALK = "WALK"
    SQUAT = "SQUAT"
    SIT_STAND = "SIT_STAND"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def parse(cls, value: object) -> "ActionType":
        """将外部标签规范化；无法确认的标签进入 UNKNOWN。"""

        if isinstance(value, cls):
            return value
        normalized = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "WALKING": cls.WALK,
            "SITSTAND": cls.SIT_STAND,
            "SIT_TO_STAND": cls.SIT_STAND,
            "STAND_TO_SIT": cls.SIT_STAND,
            "NONE": cls.UNKNOWN,
            "": cls.UNKNOWN,
        }
        return aliases.get(normalized, cls._value2member_map_.get(normalized, cls.UNKNOWN))


class ActionQuality(str, Enum):
    """动作窗口质量，而不是身份相似度。"""

    INVALID = "INVALID"
    PARTIAL = "PARTIAL"
    STRONG = "STRONG"

    @classmethod
    def parse(cls, value: object) -> "ActionQuality":
        """将质量标签规范化；缺失或未知值按 INVALID 处理。"""

        if isinstance(value, cls):
            return value
        normalized = str(value or "").strip().upper()
        return cls._value2member_map_.get(normalized, cls.INVALID)


@dataclass(frozen=True)
class ActionPrediction:
    """一个可审计的动作分类结果。"""

    action_type: ActionType | str
    confidence: float
    quality: ActionQuality | str = ActionQuality.INVALID
    completion: float = 0.0
    source: str = "unknown"
    model_version: str = "unconfigured"

    def __post_init__(self) -> None:
        """规范化枚举和有限数值，防止路由层接收 NaN。"""

        object.__setattr__(self, "action_type", ActionType.parse(self.action_type))
        object.__setattr__(self, "quality", ActionQuality.parse(self.quality))
        try:
            confidence = float(self.confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        try:
            completion = float(self.completion)
        except (TypeError, ValueError):
            completion = 0.0
        object.__setattr__(
            self,
            "confidence",
            float(np.clip(confidence if np.isfinite(confidence) else 0.0, 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "completion",
            float(np.clip(completion if np.isfinite(completion) else 0.0, 0.0, 1.0)),
        )
        if not str(self.source).strip():
            object.__setattr__(self, "source", "unknown")
        if not str(self.model_version).strip():
            object.__setattr__(self, "model_version", "unconfigured")


class ActionRouter:
    """把显式动作预测转换为 WALK/GaitGraph2 是否可写的决定。"""

    def __init__(
        self,
        *,
        minimum_confidence: float = 0.70,
        minimum_completion: float = 0.50,
    ) -> None:
        """创建保守路由器。"""

        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")
        if not 0.0 <= minimum_completion <= 1.0:
            raise ValueError("minimum_completion must be in [0, 1]")
        self.minimum_confidence = float(minimum_confidence)
        self.minimum_completion = float(minimum_completion)

    @staticmethod
    def from_metadata(metadata: Mapping[str, Any] | None) -> ActionPrediction | None:
        """从观察元数据读取动作结果。

        ``None`` 表示调用方没有接入动作分类器，此时旧的 WALK/GaitGraph2 API
        保持兼容。显式提供 ``action_type`` 后，即使字段不完整也不会默认放行。
        """

        values = dict(metadata or {})
        nested = values.get("action_prediction")
        if isinstance(nested, Mapping):
            values = {**values, **dict(nested)}
        if "action_type" not in values:
            return None
        return ActionPrediction(
            action_type=values.get("action_type"),
            confidence=values.get("action_confidence", values.get("confidence", 0.0)),
            quality=values.get("action_quality", ActionQuality.INVALID.value),
            completion=values.get("action_completion", 0.0),
            source=str(values.get("action_source", values.get("source", "metadata"))),
            model_version=str(values.get("action_model_version", "unconfigured")),
        )

    def allows_walk(self, prediction: ActionPrediction | None) -> bool:
        """判断显式动作证据是否足够进入当前 WALK/GaitGraph2 分支。"""

        return bool(
            prediction is not None
            and prediction.action_type == ActionType.WALK
            and prediction.confidence >= self.minimum_confidence
            and prediction.quality == ActionQuality.STRONG
            and prediction.completion >= self.minimum_completion
        )

    def quarantine_reason(self, prediction: ActionPrediction | None) -> str:
        """返回稳定的审计原因，不把动作失败伪装成身份负证据。"""

        if prediction is None:
            return "action_prediction_missing_after_explicit_routing"
        if prediction.action_type != ActionType.WALK:
            return f"action_{prediction.action_type.value.lower()}_routed_to_quarantine"
        if prediction.quality != ActionQuality.STRONG:
            return "action_quality_not_strong"
        if prediction.confidence < self.minimum_confidence:
            return "action_confidence_below_walk_gate"
        if prediction.completion < self.minimum_completion:
            return "action_completion_below_walk_gate"
        return "action_walk_gate_rejected"


def conservative_walk_prediction(
    *,
    walking_ratio: float,
    gait_cycles: float,
    valid_pose_frames: int,
    valid_leg_frames: int,
    minimum_frames: int,
) -> ActionPrediction:
    """生成一个只负责 WALK/非 WALK 隔离的保守启发式结果。

    它不是 SQUAT/SIT_STAND 分类器：无法确认具体非 WALK 动作时返回 UNKNOWN，
    这样 V1 可以阻止污染而不会把蹲起或坐立错误命名为某个动作。后续训练好的
    动作模型可通过 ``VisionTrack.metadata`` 提供更细粒度的显式预测。
    """

    walking = float(np.clip(walking_ratio, 0.0, 1.0))
    cycles = max(float(gait_cycles), 0.0)
    pose_ratio = float(np.clip(valid_pose_frames / max(minimum_frames, 1), 0.0, 1.0))
    leg_ratio = float(
        np.clip(valid_leg_frames / max(valid_pose_frames, 1), 0.0, 1.0)
    )
    completion = float(np.clip(min(walking, cycles / 1.0), 0.0, 1.0))
    strong_window = (
        valid_pose_frames >= minimum_frames
        and leg_ratio >= 0.75
        and walking >= 0.55
        and cycles >= 0.75
    )
    if not strong_window:
        quality = (
            ActionQuality.PARTIAL
            if valid_pose_frames > 0 and leg_ratio >= 0.45
            else ActionQuality.INVALID
        )
        return ActionPrediction(
            ActionType.UNKNOWN,
            confidence=float(np.clip(1.0 - max(walking, min(cycles, 1.0)), 0.0, 1.0)),
            quality=quality,
            completion=completion,
            source="pose_heuristic_v1",
            model_version="pose-heuristic-v1",
        )
    confidence = float(
        np.clip(0.55 + 0.20 * walking + 0.15 * min(cycles, 1.0) + 0.10 * pose_ratio, 0.0, 1.0)
    )
    return ActionPrediction(
        ActionType.WALK,
        confidence=confidence,
        quality=ActionQuality.STRONG,
        completion=completion,
        source="pose_heuristic_v1",
        model_version="pose-heuristic-v1",
    )


__all__ = [
    "ActionPrediction",
    "ActionQuality",
    "ActionRouter",
    "ActionType",
    "conservative_walk_prediction",
]
