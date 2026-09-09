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
    MANUAL_ACTION_TYPES,
    conservative_walk_prediction,
)
from .action_classifier import (
    ActionClassifierModel,
    PoseActionClassifier,
    extract_pose_action_features,
)
from .posec3d_action import (
    PoseC3DActionClassifier,
    PoseC3DPrediction,
    PoseC3DError,
)
from .config import VerifierConfig
from .engine import CrossEventVerifier
from .ble_bridge import (
    BleActionPublisher,
    BleBridgeError,
    BleCommandMode,
    BleConfig,
    DEFAULT_COMMAND_UUID,
    DEFAULT_SERVICE_UUID,
    DEFAULT_STATUS_UUID,
    LIGHT_OFF_OPCODE,
    LIGHT_ON_OPCODE,
)
from .smart_home import (
    ACTION_EVENT_PROTOCOL,
    ActionPublisher,
    MqttActionPublisher,
    MqttConfig,
    event_id,
)
from .transports import (
    ActionTransportMode,
    action_publisher_from_environment,
    normalize_transport_mode,
    transport_label,
)
from .transient_actions import ActionEvent, TransientActionConfig, TransientActionManager
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
    "MANUAL_ACTION_TYPES",
    "conservative_walk_prediction",
    "ActionClassifierModel",
    "PoseActionClassifier",
    "extract_pose_action_features",
    "PoseC3DActionClassifier",
    "PoseC3DPrediction",
    "PoseC3DError",
    "AppearanceFirstGaitEnrollmentController",
    "AppearanceIdentityBinding",
    "CrossEventVerifier",
    "ActionEvent",
    "TransientActionConfig",
    "TransientActionManager",
    "ActionPublisher",
    "ACTION_EVENT_PROTOCOL",
    "MqttActionPublisher",
    "MqttConfig",
    "event_id",
    "BleActionPublisher",
    "BleBridgeError",
    "BleCommandMode",
    "BleConfig",
    "DEFAULT_COMMAND_UUID",
    "DEFAULT_SERVICE_UUID",
    "DEFAULT_STATUS_UUID",
    "LIGHT_OFF_OPCODE",
    "LIGHT_ON_OPCODE",
    "ActionTransportMode",
    "action_publisher_from_environment",
    "normalize_transport_mode",
    "transport_label",
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
