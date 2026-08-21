"""用于部署阈值选择的轻量开放集验证指标。

项目报告 TAR/FAR/FRR，而不是单一的闭集准确率，因为未知人物是正式结果
之一。这些纯函数适合离线扫描阈值，不依赖在线验证器或模型框架。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class VerificationMetrics:
    """某一阈值下真实样本/冒充样本的工作点测量结果。"""
    threshold: float
    tar: float
    far: float
    frr: float
    genuine_count: int
    impostor_count: int


def threshold_metrics(
    genuine_scores: Iterable[float],
    impostor_scores: Iterable[float],
    threshold: float,
) -> VerificationMetrics:
    """计算一个校准分数阈值下的 TAR、FAR 和 FRR。"""
    genuine = np.asarray(list(genuine_scores), dtype=np.float64)
    impostor = np.asarray(list(impostor_scores), dtype=np.float64)
    tar = float(np.mean(genuine >= threshold)) if genuine.size else float("nan")
    far = float(np.mean(impostor >= threshold)) if impostor.size else float("nan")
    return VerificationMetrics(
        threshold=float(threshold),
        tar=tar,
        far=far,
        frr=1.0 - tar if np.isfinite(tar) else float("nan"),
        genuine_count=int(genuine.size),
        impostor_count=int(impostor.size),
    )


def equal_error_rate(
    genuine_scores: Iterable[float],
    impostor_scores: Iterable[float],
) -> tuple[float, float]:
    """使用所有观测到的分数断点，返回 ``(eer, threshold)``。"""

    genuine = np.asarray(list(genuine_scores), dtype=np.float64)
    impostor = np.asarray(list(impostor_scores), dtype=np.float64)
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    if thresholds.size == 0:
        return float("nan"), float("nan")
    reports = [threshold_metrics(genuine, impostor, float(threshold)) for threshold in thresholds]
    report = min(reports, key=lambda item: abs(item.far - item.frr))
    return (report.far + report.frr) / 2.0, report.threshold
