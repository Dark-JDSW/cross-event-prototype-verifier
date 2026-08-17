"""隔离区多原型记忆。

本模块有意独立于 SQLite 和模型推理，是记忆接口背后的深模块：调用方添加
观测或晋升候选人，具体实现负责归一化、指数移动平均更新、多样性、容量上限
以及正式区/隔离区的分离。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .types import FeatureBundle, Prototype, normalize_vector


@dataclass(frozen=True)
class MemoryUpdate:
    """描述一次追加、指数移动平均更新或被阻止的图库写入。"""
    identity_id: str
    modality: str
    action: str
    similarity: float | None
    prototype_id: str | None
    blocked: bool = False


class PrototypeMemory:
    """分别维护正式原型图库和隔离原型图库。

    每个身份在每个模态下拥有一个有上限的列表。相似观测通过指数移动平均
    合并；足够不同的视角会成为额外原型，直到多样性/容量策略淘汰冗余原
    型。两个区域不会隐式共享存储，只有 ``promote_candidate`` 可以把隔离
    向量复制到正式记忆。
    """

    def __init__(
        self,
        *,
        maximum_prototypes: int = 5,
        diversity_threshold: float = 0.88,
        appearance_max_learning_rate: float = 0.15,
        gait_max_learning_rate: float = 0.10,
        minimum_append_quality: float = 0.70,
        minimum_candidate_quality: float = 0.20,
        maximum_quarantine_prototypes: int | None = None,
    ) -> None:
        """初始化有容量上限的图库策略以及空的正式区/隔离区。"""
        self.maximum_prototypes = max(int(maximum_prototypes), 1)
        self.maximum_quarantine_prototypes = max(
            int(
                maximum_prototypes
                if maximum_quarantine_prototypes is None
                else maximum_quarantine_prototypes
            ),
            1,
        )
        self.diversity_threshold = float(np.clip(diversity_threshold, -1.0, 1.0))
        self.appearance_max_learning_rate = float(np.clip(appearance_max_learning_rate, 0.0, 1.0))
        self.gait_max_learning_rate = float(np.clip(gait_max_learning_rate, 0.0, 1.0))
        self.minimum_append_quality = float(np.clip(minimum_append_quality, 0.0, 1.0))
        self.minimum_candidate_quality = float(np.clip(minimum_candidate_quality, 0.0, 1.0))
        self.formal: dict[str, dict[str, list[Prototype]]] = {}
        self.quarantine: dict[str, dict[str, list[Prototype]]] = {}

    @staticmethod
    def _modalities(features: FeatureBundle) -> Iterable[tuple[str, np.ndarray | None]]:
        """生成归一化分支向量，同时保留缺失模态的信息。"""
        yield "appearance", normalize_vector(features.appearance)
        yield "gait", normalize_vector(features.gait)

    @staticmethod
    def _similarity(left: np.ndarray, right: np.ndarray) -> float:
        """计算余弦相似度；维度冲突时返回 ``-1``。"""
        if left.shape != right.shape:
            return -1.0
        return float(np.dot(left.astype(np.float32), right.astype(np.float32)))

    def identities(self) -> tuple[str, ...]:
        """返回排序后的正式身份 ID。"""
        return tuple(sorted(self.formal))

    def candidate_ids(self) -> tuple[str, ...]:
        """返回排序后的隔离候选人 ID。"""
        return tuple(sorted(self.quarantine))

    def formal_prototypes(self, identity_id: str, modality: str | None = None) -> tuple[Prototype, ...]:
        """读取正式原型，也可以限定为某个模态。"""
        values = self.formal.get(identity_id, {})
        if modality is not None:
            return tuple(values.get(modality, ()))
        return tuple(proto for group in values.values() for proto in group)

    def quarantine_prototypes(self, candidate_id: str, modality: str | None = None) -> tuple[Prototype, ...]:
        """读取某个候选人的隔离原型。"""
        values = self.quarantine.get(candidate_id, {})
        if modality is not None:
            return tuple(values.get(modality, ()))
        return tuple(proto for group in values.values() for proto in group)

    def best_formal(
        self,
        identity_id: str,
        modality: str,
        query: np.ndarray,
        *,
        model_version: str | None = None,
        feature_schema: str | None = None,
        artifact_sha256: str | None = None,
        preprocess_version: str | None = None,
        joint_format: str | None = None,
        sequence_length: int | None = None,
        tta_mode: str | None = None,
        coordinate_contract: str | None = None,
        embedding_dimensions: dict[str, int] | None = None,
    ) -> tuple[float | None, Prototype | None]:
        """返回与查询协议兼容的最近正式原型及其余弦相似度。"""
        normalized = normalize_vector(query)
        if normalized is None:
            return None, None
        prototypes = self.formal.get(identity_id, {}).get(modality, [])
        best: tuple[float | None, Prototype | None] = (None, None)
        for prototype in prototypes:
            if model_version is not None and prototype.model_version != model_version:
                continue
            if feature_schema is not None and prototype.feature_schema != feature_schema:
                continue
            if artifact_sha256 is not None and prototype.artifact_sha256 != artifact_sha256:
                continue
            if preprocess_version is not None and prototype.preprocess_version != preprocess_version:
                continue
            if joint_format is not None and prototype.joint_format != joint_format:
                continue
            if sequence_length is not None and prototype.sequence_length != sequence_length:
                continue
            if tta_mode is not None and prototype.tta_mode != tta_mode:
                continue
            if coordinate_contract is not None and prototype.coordinate_contract != coordinate_contract:
                continue
            if embedding_dimensions is not None and dict(prototype.embedding_dimensions) != dict(embedding_dimensions):
                continue
            if prototype.vector.shape != normalized.shape:
                # A matching label/schema is not enough to make a malformed or
                # partially migrated vector comparable.  Treat it as absent
                # instead of exposing the sentinel similarity -1 downstream.
                continue
            score = self._similarity(normalized, prototype.vector)
            if best[0] is None or score > best[0]:
                best = (score, prototype)
        return best

    def _evict_redundant(
        self,
        group: list[Prototype],
        *,
        maximum: int | None = None,
    ) -> None:
        """删除最冗余的原型，使容量保持在上限以内。"""
        limit = self.maximum_prototypes if maximum is None else max(int(maximum), 1)
        if len(group) <= limit:
            return
        # 删除最近邻相似度最高的原型。若出现并列，则丢弃质量较低的那个，
        # 从而保留多样化视角。
        worst_index = 0
        worst_key: tuple[float, float] | None = None
        for index, prototype in enumerate(group):
            neighbours = [
                self._similarity(prototype.vector, other.vector)
                for other_index, other in enumerate(group)
                if other_index != index
            ]
            redundancy = max(neighbours) if neighbours else -1.0
            key = (redundancy, -prototype.quality)
            if worst_key is None or key > worst_key:
                worst_key = key
                worst_index = index
        group.pop(worst_index)

    def _write_one(
        self,
        group: list[Prototype],
        prototype: Prototype,
        *,
        max_learning_rate: float,
        quality: float,
        append_gate: float,
        enforce_gate: bool,
        maximum_prototypes: int,
    ) -> tuple[str, float | None, str | None]:
        """合并邻近原型，或在通过门控后追加一个多样样本。"""
        if not group:
            group.append(prototype)
            return "append", None, prototype.prototype_id

        scores = [self._similarity(prototype.vector, current.vector) for current in group]
        nearest_index = int(np.argmax(scores))
        nearest_score = float(scores[nearest_index])
        nearest = group[nearest_index]
        if nearest_score >= self.diversity_threshold:
            alpha = float(np.clip(max_learning_rate * quality, 0.0, 1.0))
            merged = (1.0 - alpha) * nearest.vector + alpha * prototype.vector
            nearest.vector = normalize_vector(merged)  # type: ignore[assignment]
            nearest.quality = max(nearest.quality, quality)
            return "ema", nearest_score, nearest.prototype_id

        if enforce_gate and quality < append_gate:
            return "blocked", nearest_score, None

        group.append(prototype)
        self._evict_redundant(group, maximum=maximum_prototypes)
        return "append", nearest_score, prototype.prototype_id

    def _add_to_zone(
        self,
        zone: dict[str, dict[str, list[Prototype]]],
        identity_id: str,
        features: FeatureBundle,
        *,
        appearance_quality: float,
        gait_quality: float,
        camera_id: str | None,
        view_angle: str | None,
        clothing_tag: str | None,
        source_event_id: str | None,
        model_version: str,
        feature_schema: str,
        artifact_sha256: str,
        preprocess_version: str,
        joint_format: str,
        sequence_length: int | None,
        tta_mode: str,
        coordinate_contract: str,
        embedding_dimensions: dict[str, int] | None,
        formal: bool,
        enforce_append_gate: bool = True,
    ) -> tuple[MemoryUpdate, ...]:
        """将外观/步态向量写入一个图库区域。"""
        updates: list[MemoryUpdate] = []
        for modality, vector in self._modalities(features):
            if vector is None:
                continue
            quality = appearance_quality if modality == "appearance" else gait_quality
            max_lr = (
                self.appearance_max_learning_rate
                if modality == "appearance"
                else self.gait_max_learning_rate
            )
            group = zone.setdefault(identity_id, {}).setdefault(modality, [])
            if group:
                incompatible = [
                    current
                    for current in group
                    if (
                        current.vector.size != vector.size
                        or current.model_version != model_version
                        or current.feature_schema != feature_schema
                        or current.artifact_sha256 != artifact_sha256
                        or current.preprocess_version != preprocess_version
                        or current.joint_format != joint_format
                        or current.sequence_length != sequence_length
                        or current.tta_mode != tta_mode
                        or current.coordinate_contract != coordinate_contract
                        or dict(current.embedding_dimensions)
                        != dict(embedding_dimensions or {})
                    )
                ]
                if incompatible:
                    raise ValueError(
                        "incompatible embedding contract for "
                        f"{identity_id}/{modality}: "
                        f"existing={incompatible[0].model_version}/"
                        f"{incompatible[0].feature_schema}/"
                        f"{incompatible[0].artifact_sha256}/"
                        f"{incompatible[0].vector.size}, "
                        f"new={model_version}/{feature_schema}/{artifact_sha256}/{vector.size}"
                    )
            prototype = Prototype(
                identity_id=identity_id,
                modality=modality,
                vector=vector,
                zone="formal" if formal else "quarantine",
                quality=quality,
                camera_id=camera_id,
                view_angle=view_angle,
                clothing_tag=clothing_tag,
                source_event_id=source_event_id,
                model_version=model_version,
                feature_schema=feature_schema,
                artifact_sha256=artifact_sha256,
                preprocess_version=preprocess_version,
                joint_format=joint_format,
                sequence_length=sequence_length,
                tta_mode=tta_mode,
                coordinate_contract=coordinate_contract,
                embedding_dimensions=dict(embedding_dimensions or {}),
            )
            action, similarity, prototype_id = self._write_one(
                group,
                prototype,
                max_learning_rate=max_lr,
                quality=quality,
                append_gate=self.minimum_append_quality,
                enforce_gate=formal and enforce_append_gate,
                maximum_prototypes=(
                    self.maximum_prototypes
                    if formal
                    else self.maximum_quarantine_prototypes
                ),
            )
            updates.append(
                MemoryUpdate(
                    identity_id=identity_id,
                    modality=modality,
                    action=action,
                    similarity=similarity,
                    prototype_id=prototype_id,
                    blocked=action == "blocked",
                )
            )
        return tuple(updates)

    def add_formal(
        self,
        identity_id: str,
        features: FeatureBundle,
        *,
        appearance_quality: float = 1.0,
        gait_quality: float = 1.0,
        camera_id: str | None = None,
        view_angle: str | None = None,
        clothing_tag: str | None = None,
        source_event_id: str | None = None,
        model_version: str = "unconfigured",
        feature_schema: str = "unconfigured-v1",
        artifact_sha256: str = "unverified",
        preprocess_version: str = "unversioned-v1",
        joint_format: str = "unknown",
        sequence_length: int | None = None,
        tta_mode: str = "unknown",
        coordinate_contract: str = "unknown",
        embedding_dimensions: dict[str, int] | None = None,
        enforce_append_gate: bool = True,
    ) -> tuple[MemoryUpdate, ...]:
        """将严格样本写入正式记忆。

        与已有原型接近的视角使用指数移动平均细化。只有分支质量通过追加门
        时，才会添加多样视角；这是 videotracker 图库门控中的防污染规则。
        """

        # 正式写入是唯一会影响未来身份决策的写入，因此默认启用追加门。
        if identity_id not in self.formal:
            self.formal[identity_id] = {}
        updates = self._add_to_zone(
            self.formal,
            identity_id,
            features,
            appearance_quality=appearance_quality,
            gait_quality=gait_quality,
            camera_id=camera_id,
            view_angle=view_angle,
            clothing_tag=clothing_tag,
            source_event_id=source_event_id,
            model_version=model_version,
            feature_schema=feature_schema,
            artifact_sha256=artifact_sha256,
            preprocess_version=preprocess_version,
            joint_format=joint_format,
            sequence_length=sequence_length,
            tta_mode=tta_mode,
            coordinate_contract=coordinate_contract,
            embedding_dimensions=embedding_dimensions,
            formal=True,
            enforce_append_gate=enforce_append_gate,
        )
        if not enforce_append_gate:
            # 正式基线只用于受控消融测试；生产调用方应保留默认的严格策略。
            for update in updates:
                if update.blocked:
                    group = self.formal[identity_id].setdefault(update.modality, [])
                    vector = normalize_vector(
                        features.appearance if update.modality == "appearance" else features.gait
                    )
                    if vector is not None:
                        group.append(
                            Prototype(
                                identity_id=identity_id,
                                modality=update.modality,
                                vector=vector,
                                zone="formal",
                                quality=(
                                    appearance_quality
                                    if update.modality == "appearance"
                                    else gait_quality
                                ),
                                camera_id=camera_id,
                                view_angle=view_angle,
                                clothing_tag=clothing_tag,
                                source_event_id=source_event_id,
                                model_version=model_version,
                                feature_schema=feature_schema,
                                artifact_sha256=artifact_sha256,
                                preprocess_version=preprocess_version,
                                joint_format=joint_format,
                                sequence_length=sequence_length,
                                tta_mode=tta_mode,
                                coordinate_contract=coordinate_contract,
                                embedding_dimensions=dict(embedding_dimensions or {}),
                            )
                        )
                        self._evict_redundant(group)
        return updates

    def add_quarantine(
        self,
        candidate_id: str,
        features: FeatureBundle,
        *,
        appearance_quality: float = 1.0,
        gait_quality: float = 1.0,
        camera_id: str | None = None,
        view_angle: str | None = None,
        clothing_tag: str | None = None,
        source_event_id: str | None = None,
        model_version: str = "unconfigured",
        feature_schema: str = "unconfigured-v1",
        artifact_sha256: str = "unverified",
        preprocess_version: str = "unversioned-v1",
        joint_format: str = "unknown",
        sequence_length: int | None = None,
        tta_mode: str = "unknown",
        coordinate_contract: str = "unknown",
        embedding_dimensions: dict[str, int] | None = None,
    ) -> tuple[MemoryUpdate, ...]:
        """将观测添加到隔离记忆，不触碰正式身份。"""

        # 弱证据会保留，用于审计和后续晋升复核，但在这里期间永不参与正式匹配。
        if candidate_id not in self.quarantine:
            self.quarantine[candidate_id] = {}
        # 候选人记忆可以保留低质量样本供后续审计，但不能保留损坏向量或低于
        # 硬性最低值的分支。
        appearance_quality = max(float(appearance_quality), self.minimum_candidate_quality)
        gait_quality = max(float(gait_quality), self.minimum_candidate_quality)
        return self._add_to_zone(
            self.quarantine,
            candidate_id,
            features,
            appearance_quality=appearance_quality,
            gait_quality=gait_quality,
            camera_id=camera_id,
            view_angle=view_angle,
            clothing_tag=clothing_tag,
            source_event_id=source_event_id,
            model_version=model_version,
            feature_schema=feature_schema,
            artifact_sha256=artifact_sha256,
            preprocess_version=preprocess_version,
            joint_format=joint_format,
            sequence_length=sequence_length,
            tta_mode=tta_mode,
            coordinate_contract=coordinate_contract,
            embedding_dimensions=embedding_dimensions,
            formal=False,
        )

    def snapshot(self, identity_id: str) -> dict[str, list[Prototype]]:
        """深复制一个正式身份，用于事务回滚。"""
        return deepcopy(self.formal.get(identity_id, {}))

    def restore(self, identity_id: str, snapshot: dict[str, list[Prototype]]) -> None:
        """持久化事务失败后恢复正式记忆快照。"""
        self.formal[identity_id] = deepcopy(snapshot)

    def promote_candidate(
        self,
        candidate_id: str,
        identity_id: str,
        *,
        minimum_quality: float = 0.70,
    ) -> int:
        """将获准候选人移动到正式记忆。

        只有在向量复制完成后才会清理候选区。调用方可以在执行本方法前备份
        正式记忆，并在后续数据库事务失败时恢复它。
        """

        candidate = self.quarantine.get(candidate_id)
        if not candidate:
            raise KeyError(f"candidate has no quarantined prototypes: {candidate_id}")
        target = self.formal.setdefault(identity_id, {})
        copied = 0
        for modality, prototypes in candidate.items():
            group = target.setdefault(modality, [])
            for source in sorted(prototypes, key=lambda item: item.quality, reverse=True):
                if source.quality < minimum_quality:
                    continue
                clone = Prototype(
                    identity_id=identity_id,
                    modality=modality,
                    vector=source.vector.copy(),
                    zone="formal",
                    quality=source.quality,
                    camera_id=source.camera_id,
                    view_angle=source.view_angle,
                    clothing_tag=source.clothing_tag,
                    source_event_id=source.source_event_id,
                    model_version=source.model_version,
                    feature_schema=source.feature_schema,
                    artifact_sha256=source.artifact_sha256,
                    preprocess_version=source.preprocess_version,
                    joint_format=source.joint_format,
                    sequence_length=source.sequence_length,
                    tta_mode=source.tta_mode,
                    coordinate_contract=source.coordinate_contract,
                    embedding_dimensions=dict(source.embedding_dimensions),
                )
                group.append(clone)
                self._evict_redundant(group)
                copied += 1
        if copied == 0:
            raise ValueError("candidate contains no prototype above promotion quality")
        del self.quarantine[candidate_id]
        return copied

    def remove_formal(self, identity_id: str) -> None:
        """删除某身份的正式图库，但不影响隔离区。"""
        self.formal.pop(identity_id, None)

    def remove_candidate(self, candidate_id: str) -> None:
        """删除一个隔离候选人及其全部隔离向量。"""
        self.quarantine.pop(candidate_id, None)
