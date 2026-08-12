"""验证器决策和记忆策略的配置。

默认值有意遵循设计中较保守的策略：低质量轨迹不会被强行归入某个身份，
不确定匹配会留在隔离区，正式记忆写入的门槛也高于输出匹配门槛。
"""

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class VerifierConfig:
    """控制证据、决策和图库写入的全部阈值。

    字段按协议职责分组：质量门、开放集接受、步态授权外观吸收、质量感知
    融合、正式记忆、隔离区晋升和来源信息。数据类被冻结，使运行时更新可以
    原子地替换整份配置。
    """
    # 轨迹质量门，改编自 videotracker 的 P_a/P_p 表述。
    detection_confidence_floor: float = 0.35
    keypoint_confidence_floor: float = 0.35
    minimum_frames: int = 8
    minimum_gait_cycles: float = 1.0
    minimum_matching_quality: float = 0.38
    maximum_write_occlusion: float = 0.40

    # 开放集决策策略。这些是校准概率而不是原始余弦相似度，应使用部署验证
    # 数据进行调节。
    accept_threshold: float = 0.82
    defer_threshold: float = 0.62
    margin_threshold: float = 0.08
    appearance_floor: float = 0.45
    gait_floor: float = 0.58
    conflict_probability: float = 0.72
    allow_single_modality_match: bool = True
    # 证据协议：步态是确认锚点；外观只有在强步态证据签发一次性请求后才能
    # 被吸收。
    require_gait_for_formal_match: bool = True
    # 高质量步态结果若低于该校准概率，就是明确的开放集证据：不允许外观将
    # 轨迹继续粘连到最接近的旧身份。此时自动控制器可以采集稳定序列并创建
    # 真正的新身份。
    gait_novelty_threshold: float = 0.35
    strong_gait_probability: float = 0.90
    strong_gait_quality: float = 0.70
    strong_gait_margin: float = 0.08
    strong_appearance_probability: float = 0.90
    strong_appearance_quality: float = 0.70
    appearance_request_ttl_seconds: float = 90.0

    # 质量感知融合。两个分支都有数据时外观仍是有界锚点；只有外观缺失时步态
    # 才可以接管。
    maximum_gait_weight: float = 0.35
    spatial_prior_weight: float = 0.10

    # 正式图库记忆。
    maximum_prototypes: int = 5
    gallery_diversity_threshold: float = 0.88
    appearance_max_learning_rate: float = 0.15
    gait_max_learning_rate: float = 0.10
    minimum_append_quality: float = 0.70
    minimum_formal_write_quality: float = 0.70
    minimum_formal_write_score: float = 0.80

    # 隔离区/晋升策略。
    minimum_independent_events: int = 2
    minimum_promotion_score: float = 0.76
    minimum_promotion_margin: float = 0.06
    candidate_ttl_seconds: float = 24 * 3600
    auto_provisional_transition: bool = True
    auto_confirmation: bool = False

    # 运行来源信息。
    model_version: str = "unconfigured"
    threshold_version: str = "default-v1"
    camera_transitions: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    maximum_transition_seconds: float = 90.0

    def __post_init__(self) -> None:
        """在构造时校验取值范围和跨字段不变量。"""
        bounded = {
            "detection_confidence_floor": self.detection_confidence_floor,
            "keypoint_confidence_floor": self.keypoint_confidence_floor,
            "minimum_matching_quality": self.minimum_matching_quality,
            "maximum_write_occlusion": self.maximum_write_occlusion,
            "accept_threshold": self.accept_threshold,
            "defer_threshold": self.defer_threshold,
            "margin_threshold": self.margin_threshold,
            "appearance_floor": self.appearance_floor,
            "gait_floor": self.gait_floor,
            "conflict_probability": self.conflict_probability,
            "gait_novelty_threshold": self.gait_novelty_threshold,
            "strong_gait_probability": self.strong_gait_probability,
            "strong_gait_quality": self.strong_gait_quality,
            "strong_gait_margin": self.strong_gait_margin,
            "strong_appearance_probability": self.strong_appearance_probability,
            "strong_appearance_quality": self.strong_appearance_quality,
            "maximum_gait_weight": self.maximum_gait_weight,
            "spatial_prior_weight": self.spatial_prior_weight,
            "gallery_diversity_threshold": self.gallery_diversity_threshold,
            "minimum_append_quality": self.minimum_append_quality,
            "minimum_formal_write_quality": self.minimum_formal_write_quality,
            "minimum_formal_write_score": self.minimum_formal_write_score,
            "minimum_promotion_score": self.minimum_promotion_score,
            "minimum_promotion_margin": self.minimum_promotion_margin,
        }
        for name, value in bounded.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
        if self.defer_threshold >= self.accept_threshold:
            raise ValueError("defer_threshold must be lower than accept_threshold")
        if self.gait_novelty_threshold >= self.strong_gait_probability:
            raise ValueError(
                "gait_novelty_threshold must be lower than strong_gait_probability"
            )
        if self.minimum_frames < 1:
            raise ValueError("minimum_frames must be positive")
        if self.minimum_gait_cycles < 0:
            raise ValueError("minimum_gait_cycles cannot be negative")
        if self.maximum_prototypes < 1:
            raise ValueError("maximum_prototypes must be positive")
        if self.minimum_independent_events < 1:
            raise ValueError("minimum_independent_events must be positive")
        if self.appearance_request_ttl_seconds <= 0:
            raise ValueError("appearance_request_ttl_seconds must be positive")
