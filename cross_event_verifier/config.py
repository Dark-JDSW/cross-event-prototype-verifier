"""验证器决策和记忆策略的配置。

默认值有意遵循设计中较保守的策略：低质量轨迹不会被强行归入某个身份，
不确定匹配会留在隔离区，正式记忆写入的门槛也高于输出匹配门槛。
"""

from dataclasses import dataclass, field
from types import MappingProxyType
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
    partial_gait_quality: float = 0.35
    maximum_write_occlusion: float = 0.40
    # A quality-weighted gait window may include partial observations, but the
    # accumulated contribution must still reach this fraction of the nominal
    # sample mass before it can produce a stable template.
    minimum_weighted_gait_mass: float = 0.70
    allow_partial_gait_samples: bool = True

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
    # 单一 formal gait 身份时，校准概率本身没有真正的负类。允许第二个
    # 身份进入的条件因此更严格：原始步态相似度不能接近重复，且外观必须
    # 以强质量明确拒绝当前唯一身份。
    single_gallery_gait_similarity_limit: float = 0.985
    single_gallery_appearance_novelty_threshold: float = 0.30
    # 多身份开放集的绝对上限。margin 只表达“相对更像”，不能替代
    # max-impostor 门；该值必须在目标域校准后覆盖默认研究值。
    open_set_max_impostor_similarity: float = 0.90
    strong_gait_probability: float = 0.90
    strong_gait_quality: float = 0.70
    strong_gait_margin: float = 0.08
    maximum_formal_gait_dispersion: float = 0.60
    minimum_gait_event_support_for_strong_match: int = 1
    minimum_view_evidence_for_strong_match: float = 0.30
    strong_appearance_probability: float = 0.90
    strong_appearance_quality: float = 0.70
    # Only a high-quality appearance below this raw cosine is a hard visual
    # contradiction.  The lower-than-match value leaves ordinary cross-view
    # variance recoverable through the bound Track and multi-prototype memory.
    appearance_conflict_similarity: float = 0.35
    appearance_request_ttl_seconds: float = 90.0

    # OSNet-first visual identity and GaitGraph2 enrollment.  These thresholds
    # describe readiness of a visual identity's gait gallery; they do not
    # decide whether a Track may receive its initial visual label.
    appearance_identity_min_samples: int = 8
    appearance_identity_min_stability: float = 0.90
    appearance_identity_novelty_threshold: float = 0.90
    gait_provisional_min_events: int = 2
    gait_ready_min_events: int = 3
    gait_ready_min_coverage: int = 2
    gait_event_min_similarity: float = 0.70
    gait_holdout_min_similarity: float = 0.70
    gait_duplicate_event_similarity: float = 0.985
    # GaitGraph2 can resample a short window to its fixed input length.  The
    # shorter threshold is for learning an event; formal gait assignment needs
    # a longer real observation window and sufficient pose coverage.
    gait_learning_min_frames: int = 25
    gait_identity_min_frames: int = 45
    gait_min_pose_coverage: float = 0.75

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
    feature_schema: str = "unconfigured-v1"
    calibration_version: str = "heuristic-default-v1"
    # Research/demo defaults remain available for local tests, but production
    # deployments can require a fitted target-domain calibration profile.
    require_calibrated_scores: bool = False
    threshold_version: str = "default-v1"
    artifact_sha256: str = "unverified"
    preprocess_version: str = "unversioned-v1"
    joint_format: str = "unknown"
    sequence_length: int | None = None
    tta_mode: str = "unknown"
    coordinate_contract: str = "unknown"
    embedding_dimensions: Mapping[str, int] = field(default_factory=dict)
    vector_recall_k: int = 64
    quarantine_max_prototypes: int = 5
    quarantine_max_candidates: int = 10000
    camera_transitions: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    maximum_transition_seconds: float = 90.0

    def __post_init__(self) -> None:
        """在构造时校验取值范围和跨字段不变量。"""
        if not str(self.model_version).strip():
            raise ValueError("model_version cannot be empty")
        if not str(self.feature_schema).strip():
            raise ValueError("feature_schema cannot be empty")
        if not str(self.calibration_version).strip():
            raise ValueError("calibration_version cannot be empty")
        if not str(self.threshold_version).strip():
            raise ValueError("threshold_version cannot be empty")
        for name in (
            "artifact_sha256",
            "preprocess_version",
            "joint_format",
            "tta_mode",
            "coordinate_contract",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be empty")
        bounded = {
            "detection_confidence_floor": self.detection_confidence_floor,
            "keypoint_confidence_floor": self.keypoint_confidence_floor,
            "minimum_matching_quality": self.minimum_matching_quality,
            "partial_gait_quality": self.partial_gait_quality,
            "maximum_write_occlusion": self.maximum_write_occlusion,
            "minimum_weighted_gait_mass": self.minimum_weighted_gait_mass,
            "accept_threshold": self.accept_threshold,
            "defer_threshold": self.defer_threshold,
            "margin_threshold": self.margin_threshold,
            "appearance_floor": self.appearance_floor,
            "gait_floor": self.gait_floor,
            "conflict_probability": self.conflict_probability,
            "gait_novelty_threshold": self.gait_novelty_threshold,
            "single_gallery_gait_similarity_limit": self.single_gallery_gait_similarity_limit,
            "single_gallery_appearance_novelty_threshold": self.single_gallery_appearance_novelty_threshold,
            "open_set_max_impostor_similarity": self.open_set_max_impostor_similarity,
            "strong_gait_probability": self.strong_gait_probability,
            "strong_gait_quality": self.strong_gait_quality,
            "strong_gait_margin": self.strong_gait_margin,
            "maximum_formal_gait_dispersion": self.maximum_formal_gait_dispersion,
            "minimum_view_evidence_for_strong_match": self.minimum_view_evidence_for_strong_match,
            "strong_appearance_probability": self.strong_appearance_probability,
            "strong_appearance_quality": self.strong_appearance_quality,
            "appearance_conflict_similarity": self.appearance_conflict_similarity,
            "appearance_identity_min_stability": self.appearance_identity_min_stability,
            "appearance_identity_novelty_threshold": self.appearance_identity_novelty_threshold,
            "gait_event_min_similarity": self.gait_event_min_similarity,
            "gait_holdout_min_similarity": self.gait_holdout_min_similarity,
            "gait_duplicate_event_similarity": self.gait_duplicate_event_similarity,
            "gait_min_pose_coverage": self.gait_min_pose_coverage,
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
        if self.partial_gait_quality >= self.strong_gait_quality:
            raise ValueError(
                "partial_gait_quality must be lower than strong_gait_quality"
            )
        if self.minimum_frames < 1:
            raise ValueError("minimum_frames must be positive")
        if self.minimum_gait_event_support_for_strong_match < 1:
            raise ValueError(
                "minimum_gait_event_support_for_strong_match must be positive"
            )
        if self.gait_learning_min_frames < 1:
            raise ValueError("gait_learning_min_frames must be positive")
        if self.gait_identity_min_frames < self.gait_learning_min_frames:
            raise ValueError(
                "gait_identity_min_frames cannot be shorter than gait_learning_min_frames"
            )
        for name in (
            "appearance_identity_min_samples",
            "gait_provisional_min_events",
            "gait_ready_min_events",
            "gait_ready_min_coverage",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.gait_provisional_min_events > self.gait_ready_min_events:
            raise ValueError(
                "gait_provisional_min_events cannot exceed gait_ready_min_events"
            )
        if self.gait_ready_min_coverage > self.gait_ready_min_events:
            raise ValueError(
                "gait_ready_min_coverage cannot exceed gait_ready_min_events"
            )
        if self.gait_duplicate_event_similarity <= self.gait_event_min_similarity:
            raise ValueError(
                "gait_duplicate_event_similarity must exceed gait_event_min_similarity"
            )
        if self.minimum_gait_cycles < 0:
            raise ValueError("minimum_gait_cycles cannot be negative")
        if self.maximum_prototypes < 1:
            raise ValueError("maximum_prototypes must be positive")
        if self.minimum_independent_events < 1:
            raise ValueError("minimum_independent_events must be positive")
        if self.appearance_request_ttl_seconds <= 0:
            raise ValueError("appearance_request_ttl_seconds must be positive")
        if self.sequence_length is not None and self.sequence_length < 1:
            raise ValueError("sequence_length must be positive when configured")
        if self.vector_recall_k < 1:
            raise ValueError("vector_recall_k must be positive")
        if self.quarantine_max_prototypes < 1:
            raise ValueError("quarantine_max_prototypes must be positive")
        if self.quarantine_max_candidates < 1:
            raise ValueError("quarantine_max_candidates must be positive")
        dimensions = {
            str(key): int(value)
            for key, value in dict(self.embedding_dimensions).items()
        }
        if any(value <= 0 for value in dimensions.values()):
            raise ValueError("embedding_dimensions must contain positive values")
        object.__setattr__(self, "embedding_dimensions", MappingProxyType(dimensions))
