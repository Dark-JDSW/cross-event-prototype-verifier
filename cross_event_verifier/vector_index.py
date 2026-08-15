"""独立的外观/步态向量索引及可选 FAISS 适配器。

身份引擎通过本模块查询最近原型，而不需要知道底层实现是 NumPy 还是 FAISS。
将不同模态放在独立索引中，也能让步态/外观冲突显式暴露，避免检索时一个分支
隐藏另一个分支。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .types import Prototype, normalize_vector


@dataclass(frozen=True)
class VectorHit:
    """模态专用最近邻查询返回的一个原型。"""

    identity_id: str
    prototype_id: str
    similarity: float


class NumpyVectorIndex:
    """供开发和测试使用的小型无依赖精确余弦索引。"""

    def __init__(self) -> None:
        """创建按向量维度分组的空精确余弦索引。"""
        self._records: list[Prototype] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        self._groups: dict[int, tuple[list[Prototype], np.ndarray]] = {}

    def rebuild(self, prototypes: Iterable[Prototype]) -> None:
        """替换索引内容，并安全分离混合维度。"""
        self._records = list(prototypes)
        if not self._records:
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            self._groups = {}
            return
        grouped: dict[int, list[Prototype]] = {}
        for prototype in self._records:
            grouped.setdefault(int(prototype.vector.size), []).append(prototype)
        self._groups = {
            dimension: (
                records,
                np.stack([prototype.vector for prototype in records]).astype(np.float32),
            )
            for dimension, records in grouped.items()
        }
        self._matrix = (
            next(iter(self._groups.values()))[1]
            if len(self._groups) == 1
            else np.zeros((0, 0), dtype=np.float32)
        )

    def add(self, prototype: Prototype) -> None:
        """插入或替换一个原型，然后重建紧凑矩阵。"""
        self._records = [item for item in self._records if item.prototype_id != prototype.prototype_id]
        self._records.append(prototype)
        self.rebuild(self._records)

    def remove_identity(self, identity_id: str) -> None:
        """删除属于某个身份的全部原型。"""
        self.rebuild(item for item in self._records if item.identity_id != identity_id)

    def search(self, query: np.ndarray, k: int = 10) -> tuple[VectorHit, ...]:
        """返回最多 ``k`` 个余弦相似度最高的原型。"""
        normalized = normalize_vector(query)
        if normalized is None or not self._records:
            return ()
        group = self._groups.get(int(normalized.size))
        if group is None:
            return ()
        records, matrix = group
        scores = matrix @ normalized
        order = np.argsort(-scores)[: max(int(k), 0)]
        return tuple(
            VectorHit(
                identity_id=records[int(index)].identity_id,
                prototype_id=records[int(index)].prototype_id,
                similarity=float(scores[int(index)]),
            )
            for index in order
        )


class FaissVectorIndex(NumpyVectorIndex):
    """安装 ``faiss`` 后使用 FAISS 支持的精确内积索引。"""

    def __init__(self, dimension: int | None = None) -> None:
        """在可选依赖存在时创建 FAISS 精确内积索引。"""
        try:
            import faiss  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("FaissVectorIndex requires faiss-cpu or faiss-gpu") from exc
        super().__init__()
        self._faiss = faiss
        self._fixed_dimension = dimension is not None
        self._dimension = dimension
        self._index = None
        self._faiss_records: list[Prototype] = []

    def rebuild(self, prototypes: Iterable[Prototype]) -> None:
        """重建 NumPy 回退数据和单一维度的 FAISS 索引。"""
        super().rebuild(prototypes)
        if not self._records:
            self._index = None
            self._faiss_records = []
            return
        if len(self._groups) != 1:
            # 模型迁移期间，SQLite 中可能暂时同时存在新旧嵌入维度。NumPy
            # 可以安全分组；FAISS 无法在一个索引中表示混合维度。
            self._index = None
            self._faiss_records = []
            if not self._fixed_dimension:
                self._dimension = None
            return
        records, matrix = next(iter(self._groups.values()))
        dimension = matrix.shape[1]
        if self._fixed_dimension and self._dimension != dimension:
            raise ValueError(f"expected dimension {self._dimension}, got {dimension}")
        self._dimension = dimension
        self._index = self._faiss.IndexFlatIP(dimension)
        self._index.add(matrix)
        self._faiss_records = records

    def search(self, query: np.ndarray, k: int = 10) -> tuple[VectorHit, ...]:
        """可用时搜索 FAISS，否则使用继承的精确搜索路径。"""
        normalized = normalize_vector(query)
        if normalized is None:
            return ()
        if self._index is None:
            return super().search(normalized, k)
        if self._dimension != int(normalized.size):
            return ()
        scores, indexes = self._index.search(normalized.reshape(1, -1).astype(np.float32), max(int(k), 1))
        hits: list[VectorHit] = []
        for score, index in zip(scores[0], indexes[0]):
            if index < 0 or index >= len(self._faiss_records):
                continue
            prototype = self._faiss_records[int(index)]
            hits.append(VectorHit(prototype.identity_id, prototype.prototype_id, float(score)))
        return tuple(hits)


class DualModalityIndex:
    """维护独立索引，避免分支候选召回掩盖冲突。

    FAISS 只在条件允许时使用：依赖不可用或不兼容时，会回退到确定性的 NumPy
    实现，而不改变调用方。
    """

    def __init__(self, *, prefer_faiss: bool = False) -> None:
        """使用请求的后端构造外观和步态索引。"""
        factory = FaissVectorIndex if prefer_faiss else NumpyVectorIndex
        try:
            self.appearance = factory()
            self.gait = factory()
        except RuntimeError:
            self.appearance = NumpyVectorIndex()
            self.gait = NumpyVectorIndex()

    def rebuild(self, prototypes: Iterable[Prototype]) -> None:
        """按模态划分原型，并重建两个分支。"""
        values = list(prototypes)
        self.appearance.rebuild(item for item in values if item.modality == "appearance")
        self.gait.rebuild(item for item in values if item.modality == "gait")

    def search(self, modality: str, query: np.ndarray, k: int = 10) -> tuple[VectorHit, ...]:
        """搜索指定模态，并拒绝未知分支名称。"""
        if modality == "appearance":
            return self.appearance.search(query, k)
        if modality == "gait":
            return self.gait.search(query, k)
        raise ValueError(f"unknown modality: {modality}")
