"""用于部署阈值选择的轻量开放集验证指标。

项目报告 TAR/FAR/FRR，而不是单一的闭集准确率，因为未知人物是正式结果
之一。这些纯函数适合离线扫描阈值，不依赖在线验证器或模型框架。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Sequence

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


@dataclass(frozen=True)
class EncoderEvaluation:
    """一个 gait 编码器在同一批标注序列上的可比较报告。"""

    encoder: str
    genuine_similarity: tuple[float, ...]
    impostor_similarity: tuple[float, ...]
    max_impostor_similarity: float | None
    d_prime: float
    fnir_at_fpir: tuple[tuple[float, float, float], ...]
    novelty_threshold_at_fpir: tuple[tuple[float, float], ...]

    def to_dict(self) -> dict[str, object]:
        """返回可写入 JSON/审计日志的报告。"""

        return {
            "encoder": self.encoder,
            "genuine_similarity": list(self.genuine_similarity),
            "impostor_similarity": list(self.impostor_similarity),
            "max_impostor_similarity": self.max_impostor_similarity,
            "d_prime": self.d_prime,
            "fnir_at_fpir": [
                {
                    "target_fpir": target,
                    "fnir": fnir,
                    "threshold": threshold,
                }
                for target, fnir, threshold in self.fnir_at_fpir
            ],
            "novelty_threshold_at_fpir": [
                {"target_fpir": target, "threshold": threshold}
                for target, threshold in self.novelty_threshold_at_fpir
            ],
        }


def _unit(value: Sequence[float] | np.ndarray) -> np.ndarray | None:
    """将离线评估向量转为单位向量；无效向量返回 ``None``。"""

    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.size == 0 or not np.all(np.isfinite(vector)) or norm <= 1e-12:
        return None
    return vector / norm


def d_prime(
    genuine_scores: Iterable[float],
    impostor_scores: Iterable[float],
) -> float:
    """计算 genuine/impostor 分布的标准化分离度 ``d'``。"""

    genuine = np.asarray(list(genuine_scores), dtype=np.float64)
    impostor = np.asarray(list(impostor_scores), dtype=np.float64)
    if not genuine.size or not impostor.size:
        return float("nan")
    pooled = 0.5 * (float(np.var(genuine)) + float(np.var(impostor)))
    if pooled <= 1e-12:
        if float(np.mean(genuine)) == float(np.mean(impostor)):
            return 0.0
        return float("inf") if float(np.mean(genuine)) > float(np.mean(impostor)) else float("-inf")
    return float((np.mean(genuine) - np.mean(impostor)) / np.sqrt(pooled))


def fnir_at_fpir(
    genuine_scores: Iterable[float],
    impostor_scores: Iterable[float],
    target_fpir: float,
) -> tuple[float, float]:
    """返回固定 FPIR 上的 ``(FNIR, threshold)``。

    评估分数按“越高越像同一人”解释。阈值从观测断点中选择，并保证实际
    FPIR 不高于目标；没有可行断点时使用高于最大冒充分数的安全阈值。
    """

    if not 0.0 <= target_fpir <= 1.0:
        raise ValueError("target_fpir must be in [0, 1]")
    genuine = np.asarray(list(genuine_scores), dtype=np.float64)
    impostor = np.asarray(list(impostor_scores), dtype=np.float64)
    if not genuine.size or not impostor.size:
        return float("nan"), float("nan")
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    reports = [threshold_metrics(genuine, impostor, float(value)) for value in thresholds]
    eligible = [report for report in reports if report.far <= target_fpir + 1e-12]
    if eligible:
        report = min(eligible, key=lambda item: item.threshold)
        return report.frr, report.threshold
    threshold = float(np.nextafter(np.max(impostor), np.inf))
    report = threshold_metrics(genuine, impostor, threshold)
    return report.frr, threshold


def threshold_at_fpir(
    impostor_scores: Iterable[float],
    target_fpir: float,
) -> float:
    """返回使负样本误接收率不超过目标 FPIR 的最低阈值。"""

    if not 0.0 <= target_fpir <= 1.0:
        raise ValueError("target_fpir must be in [0, 1]")
    impostor = np.asarray(list(impostor_scores), dtype=np.float64)
    if not impostor.size:
        return float("nan")
    thresholds = np.unique(impostor)
    eligible = [
        float(value)
        for value in thresholds
        if float(np.mean(impostor >= value)) <= target_fpir + 1e-12
    ]
    if eligible:
        return min(eligible)
    return float(np.nextafter(np.max(impostor), np.inf))


def max_formal_similarity(
    unknown_embeddings: Iterable[Sequence[float] | np.ndarray],
    formal_embeddings: Iterable[Sequence[float] | np.ndarray],
) -> tuple[float, ...]:
    """返回每个 unknown 对 formal gallery 的最大原始余弦相似度。

    这是开放集阈值标定所需的 ``max impostor similarity``，而不是所有负配对
    的平均值。维度不兼容或向量无效的配对会被跳过。
    """

    formal = [item for value in formal_embeddings if (item := _unit(value)) is not None]
    if not formal:
        return ()
    result: list[float] = []
    for value in unknown_embeddings:
        query = _unit(value)
        if query is None:
            continue
        scores = [float(np.dot(query, candidate)) for candidate in formal if candidate.size == query.size]
        if scores:
            result.append(float(max(scores)))
    return tuple(result)


def _pairwise_scores(
    embeddings_by_identity: Mapping[str, Iterable[Sequence[float] | np.ndarray]],
) -> tuple[list[float], list[float]]:
    """从带身份标签的序列 embedding 生成 genuine/impostor 配对。"""

    groups: dict[str, list[np.ndarray]] = {}
    for identity, values in embeddings_by_identity.items():
        groups[str(identity)] = [
            item for value in values if (item := _unit(value)) is not None
        ]
    genuine: list[float] = []
    impostor: list[float] = []
    for values in groups.values():
        for left, right in combinations(values, 2):
            if left.size == right.size:
                genuine.append(float(np.dot(left, right)))
    identities = tuple(groups)
    for index, left_identity in enumerate(identities):
        for right_identity in identities[index + 1:]:
            for left in groups[left_identity]:
                for right in groups[right_identity]:
                    if left.size == right.size:
                        impostor.append(float(np.dot(left, right)))
    return genuine, impostor


def evaluate_encoder_embeddings(
    encoder: str,
    embeddings_by_identity: Mapping[str, Iterable[Sequence[float] | np.ndarray]],
    *,
    unknown_embeddings: Iterable[Sequence[float] | np.ndarray] = (),
    formal_embeddings: Iterable[Sequence[float] | np.ndarray] = (),
    target_fpirs: Iterable[float] = (0.01, 0.05),
) -> EncoderEvaluation:
    """生成一个 HRNet/RTMPose 等编码器的离线开放集评估报告。

    ``unknown_embeddings`` 与 ``formal_embeddings`` 可选；提供后，会把每个
    unknown 对 formal gallery 的最大相似度追加到 impostor 分布，避免只看
    平均负配对而漏掉真正危险的最高错误匹配。
    """

    genuine, impostor = _pairwise_scores(embeddings_by_identity)
    unknown_max = max_formal_similarity(unknown_embeddings, formal_embeddings)
    impostor.extend(unknown_max)
    workpoints: list[tuple[float, float, float]] = []
    novelty_workpoints: list[tuple[float, float]] = []
    for target in target_fpirs:
        fnir, threshold = fnir_at_fpir(genuine, impostor, float(target))
        workpoints.append((float(target), fnir, threshold))
        novelty_workpoints.append(
            (float(target), threshold_at_fpir(unknown_max or impostor, float(target)))
        )
    return EncoderEvaluation(
        encoder=str(encoder),
        genuine_similarity=tuple(genuine),
        impostor_similarity=tuple(impostor),
        max_impostor_similarity=max(impostor) if impostor else None,
        d_prime=d_prime(genuine, impostor),
        fnir_at_fpir=tuple(workpoints),
        novelty_threshold_at_fpir=tuple(novelty_workpoints),
    )


def compare_encoder_embeddings(
    embeddings_by_encoder: Mapping[
        str,
        Mapping[str, Iterable[Sequence[float] | np.ndarray]],
    ],
    *,
    unknown_by_encoder: Mapping[str, Iterable[Sequence[float] | np.ndarray]] | None = None,
    formal_by_encoder: Mapping[str, Iterable[Sequence[float] | np.ndarray]] | None = None,
    target_fpirs: Iterable[float] = (0.01, 0.05),
) -> tuple[EncoderEvaluation, ...]:
    """对齐输出 HRNet/RTMPose 等多个编码器的 A/B 报告。"""

    unknown_by_encoder = unknown_by_encoder or {}
    formal_by_encoder = formal_by_encoder or {}
    return tuple(
        evaluate_encoder_embeddings(
            encoder,
            embeddings,
            unknown_embeddings=unknown_by_encoder.get(encoder, ()),
            formal_embeddings=formal_by_encoder.get(encoder, ()),
            target_fpirs=target_fpirs,
        )
        for encoder, embeddings in embeddings_by_encoder.items()
    )


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
