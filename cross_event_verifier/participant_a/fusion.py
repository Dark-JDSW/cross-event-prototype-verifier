"""质量感知的校准外观、步态和空间证据融合。

融合接收的是概率而不是原始嵌入相似度。外观通常是锚点，步态权重受到限制，
只有在外观缺失时才能占主导。返回的分解结果记录所有组成部分，便于操作员
解释某身份为何被接受或拒绝。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .calibration import ScoreCalibrator


def _sigmoid(value: float) -> float:
    """计算用于融合对数几率的有界 Logistic 函数。"""
    if value >= 0:
        z = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -60.0))
    return z / (1.0 + z)


def _logit(value: float) -> float:
    """将概率转换为经过截断的对数几率。"""
    p = float(np.clip(value, 1e-6, 1.0 - 1e-6))
    return math.log(p / (1.0 - p))


def similarity_confidence(similarity: float | None, floor: float) -> float:
    """将一个分支的相似度和质量底线转换为置信度。"""
    if similarity is None:
        return 0.0
    return float(np.clip((similarity - floor) / (1.0 - floor + 1e-8), 0.0, 1.0))


@dataclass(frozen=True)
class FusionResult:
    """组合分支概率和质量后的可审计结果。"""
    appearance_probability: float | None
    gait_probability: float | None
    fused_probability: float
    appearance_weight: float
    gait_weight: float
    spatial_bonus: float
    appearance_evidence: float
    gait_evidence: float


def fuse_calibrated_scores(
    *,
    appearance_similarity: float | None,
    gait_similarity: float | None,
    appearance_quality: float,
    gait_quality: float,
    appearance_stability: float,
    gait_stability: float,
    spatial_probability: float,
    appearance_calibrator: ScoreCalibrator,
    gait_calibrator: ScoreCalibrator,
    appearance_floor: float = 0.45,
    gait_floor: float = 0.58,
    maximum_gait_weight: float = 0.35,
    spatial_prior_weight: float = 0.10,
) -> FusionResult:
    """在质量门之后融合分支对数几率。

    两个分支都存在时，步态权重会像 videotracker 中由外观锚定的
    ``W_p <= pi_p`` 一样受 ``maximum_gait_weight`` 动态限制。若外观缺失，
    步态是唯一可用的生物特征信号；调用方仍需应用更严格的开放集策略。
    """

    # 缺失分支会被省略，而不是当作零证据。这一区分对仅步态建号和外观请求
    # 非常重要。
    app_probability = (
        appearance_calibrator.probability(appearance_similarity)
        if appearance_similarity is not None
        else None
    )
    gait_probability = (
        gait_calibrator.probability(gait_similarity)
        if gait_similarity is not None
        else None
    )

    app_evidence = (
        float(np.clip(appearance_quality, 0.0, 1.0))
        * similarity_confidence(appearance_similarity, appearance_floor)
        * float(np.clip(appearance_stability, 0.0, 1.0))
        if app_probability is not None
        else 0.0
    )
    gait_evidence = (
        float(np.clip(gait_quality, 0.0, 1.0))
        * similarity_confidence(gait_similarity, gait_floor)
        * float(np.clip(gait_stability, 0.0, 1.0))
        if gait_probability is not None
        else 0.0
    )

    if app_probability is not None and gait_probability is not None:
        gait_weight = min(
            float(np.clip(maximum_gait_weight, 0.0, 1.0)),
            float(maximum_gait_weight)
            * gait_evidence
            / (app_evidence + gait_evidence + 1e-8),
        )
        appearance_weight = 1.0 - gait_weight
        log_odds = appearance_weight * _logit(app_probability) + gait_weight * _logit(
            gait_probability
        )
    elif app_probability is not None:
        appearance_weight, gait_weight = 1.0, 0.0
        log_odds = _logit(app_probability)
    elif gait_probability is not None:
        appearance_weight, gait_weight = 0.0, 1.0
        log_odds = _logit(gait_probability)
    else:
        appearance_weight, gait_weight = 0.0, 0.0
        log_odds = _logit(0.5)

    spatial = float(np.clip(spatial_probability, 0.0, 1.0))
    spatial_bonus = float(np.clip(spatial_prior_weight, 0.0, 1.0)) * (2.0 * spatial - 1.0)
    fused = float(np.clip(_sigmoid(log_odds + spatial_bonus), 0.0, 1.0))
    return FusionResult(
        appearance_probability=app_probability,
        gait_probability=gait_probability,
        fused_probability=fused,
        appearance_weight=appearance_weight,
        gait_weight=gait_weight,
        spatial_bonus=spatial_bonus,
        appearance_evidence=app_evidence,
        gait_evidence=gait_evidence,
    )
