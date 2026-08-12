"""各个包接口之间共享的值对象和契约。

项目有意在参与者之间传递普通且可序列化的领域值。视觉适配器产出
:class:`FeatureBundle` 和 :class:`TrackQuality`；:class:`Observation` 添加事件
来源；验证器返回 :class:`Decision`。将这些对象集中在这里，可以避免 GUI、
存储适配器和模型实现彼此直接导入。

大多数对象都是冻结数据类，因此决策可以安全地跨工作队列或写入审计日志。
原型和候选记录保持可变，因为内存图库会逐步更新它们。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np


def normalize_vector(value: np.ndarray | Sequence[float] | None) -> np.ndarray | None:
    """返回有限的 float32 单位向量；信号为空时返回 ``None``。"""

    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return None
    return (vector / norm).astype(np.float32)


class VerificationState(str, Enum):
    """未知候选或正式身份的生命周期状态。

    ``ISOLATED_CANDIDATE`` 和 ``PROVISIONAL_IDENTITY`` 有意区别于
    ``CONFIRMED_IDENTITY``：前两者可能只能存在于隔离图库中，直到独立证据将其提升。
    """
    UNKNOWN = "unknown"
    ISOLATED_CANDIDATE = "isolated_candidate"
    PROVISIONAL_IDENTITY = "provisional_identity"
    CONFIRMED_IDENTITY = "confirmed_identity"
    SUSPENDED = "suspended"
    MERGED = "merged"
    REVOKED = "revoked"


class DecisionKind(str, Enum):
    """验证器返回的外部可见结果类别。"""
    FORMAL_MATCH = "formal_match"
    UNKNOWN = "unknown"
    DEFERRED = "deferred"
    NEED_MORE_DATA = "need_more_data"
    CONFLICT = "conflict"
    CANDIDATE_CREATED = "candidate_created"
    CANDIDATE_UPDATED = "candidate_updated"
    APPEARANCE_REQUESTED = "appearance_requested"
    APPEARANCE_RESPONSE_ACCEPTED = "appearance_response_accepted"


@dataclass(frozen=True)
class FeatureBundle:
    """视觉适配器提供的归一化外观和步态嵌入。

    ``appearance`` 通常是 OSNet 向量，``gait`` 是时序 GaitGraph2 向量。接口字段
    接受通用序列，因此测试和替代模型不必依赖特定张量库。验证器会在计算余弦
    相似度前调用 :meth:`normalized`。
    """

    appearance: np.ndarray | Sequence[float] | None = None
    gait: np.ndarray | Sequence[float] | None = None

    def normalized(self) -> "FeatureBundle":
        """返回副本，其中存在的向量具有单位 L2 范数。"""
        return FeatureBundle(normalize_vector(self.appearance), normalize_vector(self.gait))

    @property
    def has_appearance(self) -> bool:
        """是否存在有限且非零的外观向量。"""
        return normalize_vector(self.appearance) is not None

    @property
    def has_gait(self) -> bool:
        """是否存在有限且非零的步态向量。"""
        return normalize_vector(self.gait) is not None


@dataclass(frozen=True)
class TrackQuality:
    """从检测、跟踪和姿态上游收集的质量元数据。

    这些字段是证据门控，而不是身份分数。它们描述当前裁剪图和时序窗口的
    可信程度，使决策层能在比较身份前拒绝较弱观察值。``gait_branch_quality``
    是可选的，因为轻量调用方可能只有关键点可见度，而生产适配器可以提供
    更强的模型专用步态质量估计。
    """

    detection_confidence: float = 0.0
    box_height: float = 0.0
    box_valid: bool = True
    sharpness: float = 1.0
    occlusion: float = 0.0
    keypoint_visibility: float = 0.0
    # OpenGait 轮廓/深度分支质量可以独立于关键点可见度提供；只有骨架的调用方
    # 可以将其保留为 None。
    gait_branch_quality: float | None = None
    contour_area: float = 0.0
    contour_jitter: float = 0.0
    id_switches: int = 0
    frame_count: int = 1
    gait_cycles: float = 0.0
    walking_ratio: float = 0.0
    view_angle: str | None = None
    carrying_object: bool = False
    reasons: tuple[str, ...] = ()

    def appearance_availability(self, detection_floor: float = 0.35) -> float:
        """计算 ``[0, 1]`` 范围内的外观可用度 ``P_a``。

        检测置信度、裁剪清晰度、遮挡、轮廓大小和轨迹抖动会以保守方式组合。
        该值表示外观向量是否可用，不表示它属于哪个身份。
        """

        if not self.box_valid or self.box_height <= 0:
            return 0.0
        confidence = np.clip(
            (self.detection_confidence - detection_floor)
            / (1.0 - detection_floor + 1e-8),
            0.0,
            1.0,
        )
        sharpness = float(np.clip(self.sharpness, 0.0, 1.0))
        contour = 1.0 if self.contour_area <= 0 else float(
            np.clip(np.sqrt(self.contour_area / (self.contour_area + 400.0)), 0.0, 1.0)
        )
        jitter_penalty = float(np.clip(1.0 - self.contour_jitter, 0.0, 1.0))
        return float(
            np.clip(
                np.sqrt(confidence) * (1.0 - np.clip(self.occlusion, 0.0, 1.0))
                * sharpness * max(contour, 0.5) * jitter_penalty,
                0.0,
                1.0,
            )
        )

    def gait_availability(
        self,
        minimum_frames: int = 8,
        minimum_gait_cycles: float = 1.0,
    ) -> float:
        """返回 ``[0, 1]`` 范围内的步态可用度 ``P_g``。

        序列成熟度、检测到的行走周期、腿部可见度、运动比例、遮挡和轨迹切换
        会相乘，使任一缺失前提都能抑制看似相似的步态嵌入。
        """

        if self.frame_count <= 0:
            return 0.0
        sequence = min(1.0, self.frame_count / max(minimum_frames, 1))
        cycles = (
            min(1.0, self.gait_cycles / minimum_gait_cycles)
            if minimum_gait_cycles > 0
            else 1.0
        )
        walking = float(np.clip(self.walking_ratio, 0.0, 1.0))
        keypoints = float(np.clip(self.keypoint_visibility, 0.0, 1.0))
        gait_signal = (
            float(np.clip(self.gait_branch_quality, 0.0, 1.0))
            if self.gait_branch_quality is not None
            else np.sqrt(keypoints)
        )
        switch_penalty = float(np.clip(1.0 - 0.20 * self.id_switches, 0.0, 1.0))
        return float(
            np.clip(
                gait_signal
                * np.sqrt(sequence)
                * np.sqrt(max(cycles, 0.0))
                * max(walking, 0.25 if cycles > 0 else 0.0)
                * (1.0 - np.clip(self.occlusion, 0.0, 1.0))
                * switch_penalty,
                0.0,
                1.0,
            )
        )

    def overall(self, detection_floor: float = 0.35) -> float:
        """返回可用分支中最强的质量，用于早期门控。

        该汇总值有意不作为融合身份分数；验证器会保持外观和步态概率分离。
        """

        values = [self.appearance_availability(detection_floor)]
        gait = self.gait_availability()
        if gait > 0:
            values.append(gait)
        return float(max(values))

    def quality_reasons(
        self,
        detection_floor: float = 0.35,
        minimum_frames: int = 8,
        minimum_gait_cycles: float = 1.0,
    ) -> tuple[str, ...]:
        """说明当前失败的每一个明显质量门。"""
        reasons = list(self.reasons)
        if not self.box_valid or self.box_height <= 0:
            reasons.append("invalid_box")
        if self.detection_confidence < detection_floor:
            reasons.append("low_detection_confidence")
        if self.occlusion > 0.40:
            reasons.append("occluded")
        if self.frame_count < minimum_frames:
            reasons.append("too_short")
        if self.gait_cycles < minimum_gait_cycles:
            reasons.append("too_few_gait_cycles")
        if self.walking_ratio < 0.50:
            reasons.append("not_walking")
        if self.id_switches > 0:
            reasons.append("track_id_switch")
        return tuple(dict.fromkeys(reasons))


@dataclass(frozen=True)
class Observation:
    """一个可独立归因的验证事件。

    事件/摄像头/会话/轨迹标识让证据来源明确。之后会使用 ``source_event_ids``
    和 ``capture_session_id``，防止同一次实际采集中的多帧被计为独立建号证据。
    """

    event_id: str = field(default_factory=lambda: f"evt-{uuid4().hex}")
    camera_id: str = "unknown-camera"
    capture_session_id: str = "unknown-session"
    track_id: str = "unknown-track"
    timestamp: float = field(default_factory=time.time)
    end_timestamp: float | None = None
    features: FeatureBundle = field(default_factory=FeatureBundle)
    quality: TrackQuality = field(default_factory=TrackQuality)
    model_version: str = "unconfigured"
    threshold_version: str = "default-v1"
    source_event_ids: tuple[str, ...] = ()
    challenge_id: str | None = None
    challenge_response: Mapping[str, Any] = field(default_factory=dict)
    appearance_request_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def normalized(self) -> "Observation":
        """返回特征向量已归一化的等价观察值。"""
        return Observation(
            event_id=self.event_id,
            camera_id=self.camera_id,
            capture_session_id=self.capture_session_id,
            track_id=self.track_id,
            timestamp=self.timestamp,
            end_timestamp=self.end_timestamp,
            features=self.features.normalized(),
            quality=self.quality,
            model_version=self.model_version,
            threshold_version=self.threshold_version,
            source_event_ids=tuple(self.source_event_ids),
            challenge_id=self.challenge_id,
            challenge_response=dict(self.challenge_response),
            appearance_request_id=self.appearance_request_id,
            metadata=dict(self.metadata),
        )


@dataclass
class Prototype:
    """存储在图库区域中的一个归一化外观或步态向量。

    ``formal`` 原型可以参与身份决策；``quarantine`` 原型会被隔离，直到候选
    提升策略明确接受它。构造函数会归一化并校验向量，使下游相似度代码可以依赖该不变量。
    """
    identity_id: str
    modality: str
    vector: np.ndarray
    zone: str = "formal"
    quality: float = 1.0
    camera_id: str | None = None
    view_angle: str | None = None
    clothing_tag: str | None = None
    source_event_id: str | None = None
    prototype_id: str = field(default_factory=lambda: f"proto-{uuid4().hex}")
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """归一化向量，并强制执行支持的模态/区域不变量。"""
        normalized = normalize_vector(self.vector)
        if normalized is None:
            raise ValueError("prototype vector must be a non-zero finite vector")
        self.vector = normalized
        if self.modality not in {"appearance", "gait"}:
            raise ValueError(f"unsupported modality: {self.modality}")
        if self.zone not in {"formal", "quarantine"}:
            raise ValueError(f"unsupported prototype zone: {self.zone}")


@dataclass
class CandidateRecord:
    """未知人物提升前的持久化记录。"""
    candidate_id: str
    state: VerificationState = VerificationState.ISOLATED_CANDIDATE
    proposed_identity: str | None = None
    confirmed_identity: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    event_ids: list[str] = field(default_factory=list)
    independent_event_count: int = 0
    high_quality_evidence_count: int = 0
    conflict_count: int = 0
    challenge_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreBreakdown:
    """用于排序、可供审计的每个身份分数组成。"""
    identity_id: str
    appearance_similarity: float | None
    gait_similarity: float | None
    appearance_probability: float | None
    gait_probability: float | None
    spatial_probability: float
    appearance_weight: float
    gait_weight: float
    fused_probability: float
    appearance_quality: float
    gait_quality: float
    appearance_prototype_id: str | None = None
    gait_prototype_id: str | None = None
    conflict: bool = False


@dataclass(frozen=True)
class Decision:
    """一次验证或建号尝试的不可变公开结果。"""
    kind: DecisionKind
    state: VerificationState
    identity_id: str | None = None
    candidate_id: str | None = None
    score: float | None = None
    margin: float | None = None
    reasons: tuple[str, ...] = ()
    ranking: tuple[ScoreBreakdown, ...] = ()
    challenge_prompt: str | None = None
    evidence_id: str | None = None
    appearance_request_id: str | None = None


@dataclass(frozen=True)
class PromotionResult:
    """隔离候选进入正式记忆后返回的摘要。"""
    candidate_id: str
    identity_id: str
    state: VerificationState
    independent_event_count: int
    prototype_count: int
    reasons: tuple[str, ...] = ()


@dataclass
class AppearanceAbsorptionRequest:
    """授权吸收外观样本的一次性请求。

    请求由强步态事件签发，响应观察值必须显式引用它。请求本身永远不是身份
    凭据；它只授权指定的外观分支。
    """

    request_id: str
    identity_id: str
    issued_by_event_id: str
    candidate_id: str | None
    gait_probability: float
    gait_quality: float
    issued_at: float
    expires_at: float
    status: str = "pending"
    response_event_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
