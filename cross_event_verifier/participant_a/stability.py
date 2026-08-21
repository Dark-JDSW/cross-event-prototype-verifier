"""由确认样本驱动的验证器时序稳定性。

跟踪器为每个身份/模态组合保存指数加权均值和方差。只有严格的正式记忆写
入成功后才会更新，因此隔离候选人无法提高自己的稳定性估计。
"""

from __future__ import annotations

import math


class StabilityTracker:
    """按身份和模态跟踪指数移动平均方差。

    调用方只能在严格的正式记忆写入成功后更新本模块。因此候选人的观测不
    能提高自己的稳定性分数，也不能让被污染的正式原型看起来稳定。
    """

    minimum_samples = 3

    def __init__(self, eta: float = 1.0 / 24.0, variance_clip: float = 1.0) -> None:
        """创建一个限制方差贡献范围的指数移动平均跟踪器。"""
        if not 0 < eta <= 1:
            raise ValueError("eta must be in (0, 1]")
        self.eta = eta
        self.variance_clip = max(float(variance_clip), 1e-8)
        self._stats: dict[tuple[str, str], list[float | int]] = {}

    def update(self, identity_id: str, modality: str, similarity: float, gap: int = 1) -> None:
        """纳入一个相似度样本，并考虑跳过的帧数。"""
        gap = max(int(gap), 1)
        eta_eff = 1.0 - (1.0 - self.eta) ** gap
        key = (identity_id, modality)
        state = self._stats.get(key)
        if state is None:
            self._stats[key] = [float(similarity), 0.0, 1]
            return
        mean, variance, count = state
        mean_new = (1.0 - eta_eff) * float(mean) + eta_eff * float(similarity)
        variance_new = (1.0 - eta_eff) * float(variance) + eta_eff * (
            float(similarity) - float(mean)
        ) * (float(similarity) - mean_new)
        state[0] = mean_new
        state[1] = min(max(variance_new, 0.0), self.variance_clip)
        state[2] = int(count) + gap

    def get(self, identity_id: str, modality: str) -> float:
        """返回由观测方差推导出的置信度乘数。"""
        state = self._stats.get((identity_id, modality))
        if state is None or int(state[2]) < self.minimum_samples:
            return 1.0
        return 1.0 / (1.0 + 4.0 * math.sqrt(max(float(state[1]), 0.0)) + 1e-8)

    def remove(self, identity_id: str) -> None:
        """删除某个身份的全部模态统计信息。"""
        for key in [key for key in self._stats if key[0] == identity_id]:
            del self._stats[key]

    def snapshot(self) -> dict[str, list[float | int]]:
        """返回适合 JSON 序列化的指数移动平均状态，用于持久化或回滚。"""
        return {f"{identity}\x1f{modality}": list(state) for (identity, modality), state in self._stats.items()}

    def restore(self, values: dict[str, list[float | int]] | None) -> None:
        """使用持久化快照替换当前指数移动平均状态。"""
        self._stats.clear()
        for key, state in (values or {}).items():
            identity, modality = key.split("\x1f", 1)
            self._stats[(identity, modality)] = list(state)
