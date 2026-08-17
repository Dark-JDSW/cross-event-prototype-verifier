"""分支专用的分数校准。

FastReID/OpenGait 输出的原始余弦分数不能直接互换。本模块先把每个分支转换
为校准概率（以及对数几率），再交给融合模块。这里包含一个轻量逻辑回归拟
合，使部署验证数据可以替换保守默认值，而不必增加 scikit-learn 依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


def _sigmoid(value: float) -> float:
    """计算经过数值范围限制的 Logistic 函数。"""
    if value >= 0:
        z = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -60.0))
    return z / (1.0 + z)


def _logit(probability: float) -> float:
    """将截断后的概率转换为可用于加法融合的对数几率。"""
    p = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class ScoreCalibrator:
    """针对一个模态的一维 Platt 风格校准器。"""

    scale: float = 8.0
    midpoint: float = 0.55
    name: str = "default"
    source: str = "heuristic"
    sample_count: int = 0
    version: str = "unversioned"

    def __post_init__(self) -> None:
        """校验正斜率以及余弦相似度定义域内的中点。"""
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("calibrator scale must be a positive finite value")
        if not math.isfinite(self.midpoint) or not -1.0 <= self.midpoint <= 1.0:
            raise ValueError("calibrator midpoint must be in [-1, 1]")
        if not str(self.source).strip():
            raise ValueError("calibrator source cannot be empty")
        if not str(self.version).strip():
            raise ValueError("calibrator version cannot be empty")
        if int(self.sample_count) < 0:
            raise ValueError("calibrator sample_count cannot be negative")

    @property
    def is_target_calibrated(self) -> bool:
        """是否来自足够大的目标域数据集，而不是启发式默认值。"""

        return self.source == "target-data" and int(self.sample_count) >= 32

    def probability(self, raw_similarity: float) -> float:
        """将一个原始余弦相似度映射为校准概率。"""
        score = float(np.clip(raw_similarity, -1.0, 1.0))
        return _sigmoid(self.scale * (score - self.midpoint))

    def log_likelihood(self, raw_similarity: float) -> float:
        """返回用于分支融合和诊断的校准对数几率。"""
        return _logit(self.probability(raw_similarity))

    def to_dict(self) -> dict[str, float | str | int]:
        """将校准参数序列化，用于清单和审计元数据。"""
        return {
            "scale": self.scale,
            "midpoint": self.midpoint,
            "name": self.name,
            "source": self.source,
            "sample_count": int(self.sample_count),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "ScoreCalibrator":
        """从宽松的 JSON 风格映射重建校准器。"""
        return cls(
            scale=float(values.get("scale", 8.0)),
            midpoint=float(values.get("midpoint", 0.55)),
            name=str(values.get("name", "loaded")),
            source=str(values.get("source", "heuristic")),
            sample_count=int(values.get("sample_count", 0)),
            version=str(values.get("version", values.get("name", "loaded"))),
        )

    @classmethod
    def fit(
        cls,
        similarities: Sequence[float] | Iterable[float],
        labels: Sequence[int] | Iterable[int],
        *,
        name: str = "fitted",
        iterations: int = 800,
        learning_rate: float = 0.05,
        l2: float = 1e-3,
        minimum_pairs: int = 32,
    ) -> "ScoreCalibrator":
        """在验证样本对上拟合 ``sigmoid(scale * (score - midpoint))``。

        函数要求使用二值标签，其中 ``1`` 表示真实配对。这里有意将斜率限制
        为正数，从而保证相似度升高时校准概率也会升高。
        """

        x = np.asarray(list(similarities), dtype=np.float64).reshape(-1)
        y = np.asarray(list(labels), dtype=np.float64).reshape(-1)
        minimum_pairs = max(4, int(minimum_pairs))
        if x.size != y.size or x.size < minimum_pairs:
            raise ValueError(
                f"at least {minimum_pairs} score/label pairs of equal length are required"
            )
        if not np.all(np.isin(y, [0.0, 1.0])) or np.unique(y).size < 2:
            raise ValueError("labels must contain both 0 and 1")
        positive = int(np.count_nonzero(y == 1.0))
        negative = int(np.count_nonzero(y == 0.0))
        if min(positive, negative) < 8:
            raise ValueError(
                "target calibration requires at least eight genuine and eight impostor pairs"
            )

        # 先在 [score, 1] 上进行逻辑回归，再转换为正斜率/中点参数形式。
        design = np.column_stack([x, np.ones_like(x)])
        weights = np.array([6.0, -3.0], dtype=np.float64)
        for _ in range(max(1, iterations)):
            logits = design @ weights
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
            gradient = (design.T @ (probs - y)) / x.size
            gradient += l2 * np.array([weights[0], 0.0])
            weights -= learning_rate * gradient

        scale = max(float(abs(weights[0])), 1e-3)
        midpoint = float(-weights[1] / weights[0]) if abs(weights[0]) > 1e-6 else 0.5
        return cls(
            scale=scale,
            midpoint=float(np.clip(midpoint, -1.0, 1.0)),
            name=name,
            source="target-data",
            sample_count=int(x.size),
            version=name,
        )


DEFAULT_APPEARANCE_CALIBRATOR = ScoreCalibrator(
    scale=8.0,
    midpoint=0.56,
    name="appearance-default",
    source="heuristic",
    version="heuristic-default-v1",
)
DEFAULT_GAIT_CALIBRATOR = ScoreCalibrator(
    scale=8.0,
    midpoint=0.68,
    name="gait-default",
    source="heuristic",
    version="heuristic-default-v1",
)


__all__ = [
    "DEFAULT_APPEARANCE_CALIBRATOR",
    "DEFAULT_GAIT_CALIBRATOR",
    "ScoreCalibrator",
]
