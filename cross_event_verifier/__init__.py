"""跨事件步态确认与外观吸收验证器。

本包将模型推理隐藏在适配器接口之后。公开入口是
:class:`CrossEventVerifier`；调用方提供归一化外观和步态特征，以及轨迹质量
和事件来源信息。
"""

from .participant_c.automation import AutomationPolicy, AutomationStage, AutomationStatus
from .participant_a.config import VerifierConfig
from .participant_a.engine import CrossEventVerifier
from .participant_a.evaluation import VerificationMetrics, equal_error_rate, threshold_metrics
from .types import (
    AppearanceAbsorptionRequest,
    CandidateRecord,
    Decision,
    DecisionKind,
    FeatureBundle,
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
    "Observation",
    "Prototype",
    "TrackQuality",
    "VerificationState",
    "VerifierConfig",
    "VerificationMetrics",
    "equal_error_rate",
    "threshold_metrics",
]
