"""跨事件步态确认与外观吸收验证器。

本包将模型推理隐藏在适配器接口之后。公开入口是
:class:`CrossEventVerifier`；调用方提供归一化外观和步态特征，以及轨迹质量
和事件来源信息。
"""

from .automation import AutomationPolicy, AutomationStage, AutomationStatus
from .config import VerifierConfig
from .engine import CrossEventVerifier
from .evaluation import (
    EncoderEvaluation,
    VerificationMetrics,
    compare_encoder_embeddings,
    d_prime,
    evaluate_encoder_embeddings,
    equal_error_rate,
    fnir_at_fpir,
    max_formal_similarity,
    threshold_at_fpir,
    threshold_metrics,
)
from .types import (
    AppearanceAbsorptionRequest,
    CandidateRecord,
    Decision,
    DecisionKind,
    FeatureBundle,
    GaitQualityBand,
    Observation,
    Prototype,
    TrackQuality,
    VerificationState,
)

__all__ = [
    "CandidateRecord",
    "AppearanceAbsorptionRequest",
    "AutomationPolicy",
    "AutomationStage",
    "AutomationStatus",
    "CrossEventVerifier",
    "Decision",
    "DecisionKind",
    "FeatureBundle",
    "GaitQualityBand",
    "Observation",
    "Prototype",
    "TrackQuality",
    "VerificationState",
    "VerifierConfig",
    "VerificationMetrics",
    "EncoderEvaluation",
    "compare_encoder_embeddings",
    "d_prime",
    "evaluate_encoder_embeddings",
    "equal_error_rate",
    "fnir_at_fpir",
    "max_formal_similarity",
    "threshold_at_fpir",
    "threshold_metrics",
]
