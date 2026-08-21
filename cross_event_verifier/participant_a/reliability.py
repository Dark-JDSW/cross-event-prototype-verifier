"""独立的数值可靠性辅助函数。

这些确定性函数为不需要完整在线管线的调用方归一化检测置信度并估计分支可
用性。它们不识别人物，只描述某个分支可以被信任的程度。
"""

from __future__ import annotations

import numpy as np

from .fusion import similarity_confidence


def det_conf_norm(confidence: float, threshold: float) -> float:
    """归一化高于配置底线的检测置信度。"""
    return float(np.clip((confidence - threshold) / (1.0 - threshold + 1e-8), 0.0, 1.0))


def appearance_availability(box_legal: bool, detection_norm: float, occlusion: float) -> float:
    """估计检测裁剪区域是否可用于外观证据。"""
    if not box_legal:
        return 0.0
    return float(
        np.clip(
            np.sqrt(max(detection_norm, 0.0)) * (1.0 - np.clip(occlusion, 0.0, 1.0)),
            0.0,
            1.0,
        )
    )


def pose_availability(keypoint_confidences: np.ndarray | None, threshold: float) -> float:
    """将关键点置信度概括为步态/姿态可用性分数。"""
    if keypoint_confidences is None:
        return 0.0
    values = np.asarray(keypoint_confidences, dtype=np.float32)
    if values.size == 0:
        return 0.0
    normalized = np.clip((values.reshape(-1) - threshold) / (1.0 - threshold + 1e-8), 0.0, 1.0)
    return float(np.sqrt(normalized.mean()))


def fuse_similarity(
    appearance_similarity: float,
    gait_similarity: float | None,
    appearance_quality: float,
    gait_quality: float,
    appearance_stability: float = 1.0,
    gait_stability: float = 1.0,
    *,
    maximum_gait_weight: float = 0.35,
    appearance_floor: float = 0.45,
    gait_floor: float = 0.58,
) -> tuple[float, float]:
    """返回轻量原始分数融合结果和有界步态权重。"""
    """为轻量测试返回 ``(fused_raw_similarity, gait_weight)``。

    为了与旧项目保持一致，该辅助函数有意直接处理原始相似度。生产决策应
    使用 ``fusion.fuse_calibrated_scores``，先对每个模型分支进行校准。
    """

    if gait_similarity is None:
        return float(np.clip(appearance_similarity, 0.0, 1.0)), 0.0
    qa = appearance_quality * similarity_confidence(appearance_similarity, appearance_floor) * appearance_stability
    qg = gait_quality * similarity_confidence(gait_similarity, gait_floor) * gait_stability
    gait_weight = min(
        float(np.clip(maximum_gait_weight, 0.0, 1.0)),
        float(maximum_gait_weight) * qg / (qa + qg + 1e-8),
    )
    fused = (1.0 - gait_weight) * appearance_similarity + gait_weight * gait_similarity
    return float(np.clip(fused, 0.0, 1.0)), float(gait_weight)
