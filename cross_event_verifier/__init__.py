"""跨事件步态确认与外观吸收验证器。

本包将模型推理隐藏在适配器接口之后。公开入口是
:class:`CrossEventVerifier`；调用方提供归一化外观和步态特征，以及轨迹质量
和事件来源信息。
"""

from .automation import AutomationPolicy, AutomationStage, AutomationStatus
from .appearance_first import AppearanceFirstGaitEnrollmentController
from .actions import (
    ActionPrediction,
    ActionQuality,
    ActionRouter,
    ActionType,
    conservative_walk_prediction,
)
from .config import VerifierConfig
from .engine import CrossEventVerifier
from .gait_readiness import GaitReadinessEvaluator
from .evaluation import (
    EncoderEvaluation,
    OpenSetProtocolReport,
    VerificationMetrics,
    compare_encoder_embeddings,
    d_prime,
    evaluate_encoder_embeddings,
    evaluate_open_set_protocol,
    equal_error_rate,
    fnir_at_fpir,
    max_formal_similarity,
    threshold_at_fpir,
    threshold_metrics,
)
from .types import (
    AppearanceAbsorptionRequest,
    CandidateRecord,
    CleanSubTracklet,
    Decision,
    DecisionKind,
    EmbeddingContract,
    EvidenceSummary,
    FeatureBundle,
    AppearanceIdentityBinding,
    GaitEnrollmentEvent,
    GaitQualityBand,
    GaitReadinessReport,
    GaitReadinessState,
    Observation,
    Prototype,
    TrackQuality,
    VerificationState,
)

__all__ = [
    "CandidateRecord",
    "CleanSubTracklet",
    "AppearanceAbsorptionRequest",
    "AutomationPolicy",
    "AutomationStage",
    "AutomationStatus",
    "ActionPrediction",
    "ActionQuality",
    "ActionRouter",
    "ActionType",
    "conservative_walk_prediction",
    "AppearanceFirstGaitEnrollmentController",
    "AppearanceIdentityBinding",
    "CrossEventVerifier",
    "GaitReadinessEvaluator",
    "Decision",
    "DecisionKind",
    "EmbeddingContract",
    "EvidenceSummary",
    "FeatureBundle",
    "GaitEnrollmentEvent",
    "GaitQualityBand",
    "GaitReadinessReport",
    "GaitReadinessState",
    "Observation",
    "Prototype",
    "TrackQuality",
    "VerificationState",
    "VerifierConfig",
    "VerificationMetrics",
    "EncoderEvaluation",
    "OpenSetProtocolReport",
    "compare_encoder_embeddings",
    "d_prime",
    "evaluate_encoder_embeddings",
    "evaluate_open_set_protocol",
    "equal_error_rate",
    "fnir_at_fpir",
    "max_formal_similarity",
    "threshold_at_fpir",
    "threshold_metrics",
]
