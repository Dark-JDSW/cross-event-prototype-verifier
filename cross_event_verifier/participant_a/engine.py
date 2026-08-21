"""公开的跨事件验证深模块。

``CrossEventVerifier`` 是原始视觉观测与应用身份决策之间的策略接口。其实现
协调校准、质量门、分支融合、一对一指派、候选人状态、隔离区/正式区记忆、
持久化和审计记录。调用方应提供 :class:`Observation` 并消费 :class:`Decision`，
不应复制这些检查的内部顺序。

重要安全规则是：高外观分数不能单独创建或确认身份。强步态可以确认已有
身份，或创建仅含步态的身份；之后，一次性外观请求才可以授权更新外观。
"""

from __future__ import annotations

from dataclasses import replace
import time
from typing import Any, Iterable, Sequence
from uuid import uuid4

import numpy as np

from .assignment import gated_global_assignment
from .absorption import AppearanceAbsorptionManager
from .calibration import (
    DEFAULT_APPEARANCE_CALIBRATOR,
    DEFAULT_GAIT_CALIBRATOR,
    ScoreCalibrator,
)
from .challenge import ChallengeManager
from .config import VerifierConfig
from .fusion import fuse_calibrated_scores
from .memory import MemoryUpdate, PrototypeMemory
from .state import check_independence, mark_evidence, transition
from ..participant_c.storage import SqliteStore
from .stability import StabilityTracker
from ..types import (
    AppearanceAbsorptionRequest,
    CandidateRecord,
    Decision,
    DecisionKind,
    FeatureBundle,
    Observation,
    PromotionResult,
    ScoreBreakdown,
    VerificationState,
)
from ..participant_c.vector_index import DualModalityIndex, VectorHit


class CrossEventVerifier:
    """针对隔离且带版本的原型图库验证观测。

    接口不变量：

    * 模型适配器传入一个 :class:`Observation`；验证器负责归一化向量并拥有
      全部决策阈值；
    * 正式匹配和正式记忆写入是两条独立路径；
    * 只有来源分组全新时，观测才算独立；
    * 只有 ``promote_candidate`` 这个公开操作可以把隔离原型移入正式记忆。

    实现既可以在测试中使用内存 SQLite 适配器，也可以在应用部署中使用文件
    形式的适配器。
    """

    def __init__(
        self,
        config: VerifierConfig | None = None,
        *,
        store: SqliteStore | None = None,
        memory: PrototypeMemory | None = None,
        appearance_calibrator: ScoreCalibrator | None = None,
        gait_calibrator: ScoreCalibrator | None = None,
        challenge_manager: ChallengeManager | None = None,
    ) -> None:
        """构建验证器并恢复内存索引。

        依赖可以注入，因此测试能够使用内存存储和确定性记忆，生产环境则可
        传入文件形式的 ``SqliteStore``。恢复发生在重建索引之前，保证进程重
        启动后看到的正式图库与之前一致。
        """
        self.config = config or VerifierConfig()
        self.store = store or SqliteStore(":memory:")
        self.memory = memory or PrototypeMemory(
            maximum_prototypes=self.config.maximum_prototypes,
            diversity_threshold=self.config.gallery_diversity_threshold,
            appearance_max_learning_rate=self.config.appearance_max_learning_rate,
            gait_max_learning_rate=self.config.gait_max_learning_rate,
            minimum_append_quality=self.config.minimum_append_quality,
        )
        self.appearance_calibrator = appearance_calibrator or DEFAULT_APPEARANCE_CALIBRATOR
        self.gait_calibrator = gait_calibrator or DEFAULT_GAIT_CALIBRATOR
        self.challenge_manager = challenge_manager or ChallengeManager()
        self.appearance_absorption = AppearanceAbsorptionManager(
            self.config.appearance_request_ttl_seconds
        )
        self.stability = StabilityTracker()
        self._identity_states = self.store.identity_states()
        self._last_seen: dict[str, tuple[str, float]] = {}
        self._candidates: dict[str, CandidateRecord] = {}
        self._candidate_observations: dict[str, list[Observation]] = {}
        self.vector_index = DualModalityIndex()
        self._hydrate()
        self.rebuild_index()

    def _hydrate(self) -> None:
        """加载持久化原型、候选人、请求和状态。

        SQLite 是跨重启的事实来源；字典和向量索引只是可丢弃的运行时加速
        结构，在这里重新构建。
        """

        for prototype in self.store.load_prototypes():
            target = (
                self.memory.formal
                if prototype.zone == "formal"
                else self.memory.quarantine
            )
            target.setdefault(prototype.identity_id, {}).setdefault(
                prototype.modality, []
            ).append(prototype)
        for candidate in self.store.list_candidates():
            self._candidates[candidate.candidate_id] = candidate
            self._candidate_observations[candidate.candidate_id] = self.store.observations_for_candidate(
                candidate.candidate_id
            )
        for request in self.store.load_appearance_requests():
            self.appearance_absorption.restore(request)
        for identity_id in self.memory.identities():
            self._identity_states.setdefault(
                identity_id, VerificationState.CONFIRMED_IDENTITY
            )

    @property
    def formal_identities(self) -> tuple[str, ...]:
        """返回活跃正式 ID，排除已暂停、已合并或已撤销的身份。"""
        blocked = {
            VerificationState.SUSPENDED,
            VerificationState.MERGED,
            VerificationState.REVOKED,
        }
        return tuple(
            identity_id
            for identity_id in self.memory.identities()
            if self._identity_states.get(identity_id, VerificationState.CONFIRMED_IDENTITY)
            not in blocked
        )

    def register_identity(
        self,
        identity_id: str,
        features: FeatureBundle,
        *,
        metadata: dict[str, Any] | None = None,
        camera_id: str | None = None,
        view_angle: str | None = None,
        clothing_tag: str | None = None,
        quality: float = 1.0,
        source_event_id: str | None = None,
    ) -> tuple[MemoryUpdate, ...]:
        """使用可信建号证据创建正式身份。

        该显式建号路径用于已知/可信的初始化。它会直接把归一化原型写入正
        式区；未知人物的自动建号则使用 :meth:`enroll_gait_identity`。
        """

        if identity_id in self.memory.formal:
            raise ValueError(f"identity already exists: {identity_id}")
        normalized = features.normalized()
        if not normalized.has_appearance and not normalized.has_gait:
            raise ValueError("an identity needs at least one non-empty feature branch")
        updates = self.memory.add_formal(
            identity_id,
            normalized,
            appearance_quality=quality,
            gait_quality=quality,
            camera_id=camera_id,
            view_angle=view_angle,
            clothing_tag=clothing_tag,
            source_event_id=source_event_id,
            enforce_append_gate=False,
        )
        if not self.memory.formal_prototypes(identity_id):
            raise ValueError("enrollment produced no prototype")
        self.store.upsert_identity(
            identity_id,
            VerificationState.CONFIRMED_IDENTITY,
            metadata=metadata,
        )
        self.store.save_prototypes(
            list(self.memory.formal_prototypes(identity_id)),
            replace_identity=identity_id,
            zone="formal",
        )
        self.store.audit(
            "identity_registered",
            identity_id,
            {"modalities": [item.modality for item in self.memory.formal_prototypes(identity_id)]},
        )
        self._identity_states[identity_id] = VerificationState.CONFIRMED_IDENTITY
        self.rebuild_index()
        return updates

    def rebuild_index(self, *, prefer_faiss: bool = False) -> None:
        """在记忆变化后重建外观和步态检索索引。"""

        if prefer_faiss:
            self.vector_index = DualModalityIndex(prefer_faiss=True)
        self.vector_index.rebuild(
            prototype
            for identity_id in self.formal_identities
            for prototype in self.memory.formal_prototypes(identity_id)
        )

    def search_vectors(
        self,
        modality: str,
        query: Sequence[float] | np.ndarray,
        *,
        k: int = 10,
    ) -> tuple[VectorHit, ...]:
        """为大图库集成暴露按分支检索的能力。"""

        return self.vector_index.search(modality, np.asarray(query, dtype=np.float32), k)

    def get_appearance_request(
        self,
        request_id: str | None,
    ) -> AppearanceAbsorptionRequest | None:
        """读取由步态签发的外观吸收请求。"""

        return self.appearance_absorption.get(request_id)

    def pending_appearance_requests(
        self,
        *,
        now: float | None = None,
    ) -> tuple[AppearanceAbsorptionRequest, ...]:
        """为外部采集界面返回未消费且未过期的请求。"""

        now = time.time() if now is None else float(now)
        pending: list[AppearanceAbsorptionRequest] = []
        for request in self.appearance_absorption.pending():
            if now <= request.expires_at:
                pending.append(request)
            else:
                request.status = "expired"
                self.store.save_appearance_request(request)
        return tuple(pending)

    def _issue_appearance_request(
        self,
        *,
        identity_id: str,
        observation: Observation,
        gait_probability: float,
        gait_quality: float,
        candidate_id: str | None,
        reason: str,
    ) -> tuple[AppearanceAbsorptionRequest | None, bool]:
        """为某身份签发或复用唯一的有效外观请求。"""

        request = self.appearance_absorption.pending_for_identity(
            identity_id,
            now=observation.timestamp,
        )
        if request is not None:
            return request, False
        completed_in_session = any(
            historical.identity_id == identity_id
            and historical.status == "consumed"
            and historical.metadata.get("capture_session_id")
            == observation.capture_session_id
            for historical in self.appearance_absorption.all()
        )
        if completed_in_session:
            return None, False
        request = self.appearance_absorption.issue(
            identity_id=identity_id,
            issued_by_event_id=observation.event_id,
            gait_probability=float(np.clip(gait_probability, 0.0, 1.0)),
            gait_quality=float(np.clip(gait_quality, 0.0, 1.0)),
            # 令牌是跨事件授权边界。它必须跨越跟踪器/摄像头交接继续有效，
            # 因此有意不绑定到短生命周期的候选人 ID。
            candidate_id=None,
            now=observation.timestamp,
            metadata={
                "reason": reason,
                "source_candidate_id": candidate_id,
                "capture_session_id": observation.capture_session_id,
            },
        )
        self.store.save_appearance_request(request)
        self.store.audit(
            "appearance_absorption_requested",
            identity_id,
            {
                "request_id": request.request_id,
                "event_id": observation.event_id,
                "gait_probability": request.gait_probability,
                "reason": reason,
            },
        )
        return request, True

    def enroll_gait_identity(
        self,
        observation: Observation,
        *,
        identity_id: str | None = None,
        candidate_id: str | None = None,
        gait_confidence: float | None = None,
    ) -> Decision:
        """使用强且经时序确认的步态创建正式身份。

        这是狭窄的自动建号接口。它只写入步态分支，并立即签发一次性外观请
        求；外观只有通过响应路径才能进入正式记忆。调用方负责从稳定序列而
        非单帧推导 ``gait_confidence``。
        """

        normalized = observation.normalized()
        features = normalized.features
        if not features.has_gait:
            raise ValueError("automatic enrollment requires a gait feature")
        gait_quality = normalized.quality.gait_availability(
            self.config.minimum_frames,
            self.config.minimum_gait_cycles,
        )
        if gait_quality < self.config.strong_gait_quality:
            raise ValueError(
                "automatic enrollment gait quality is below the strong gate"
            )
        confidence = float(
            np.clip(
                gait_quality if gait_confidence is None else gait_confidence,
                0.0,
                1.0,
            )
        )
        if confidence < self.config.strong_gait_probability:
            raise ValueError(
                "automatic enrollment gait confidence is below the strong gate"
            )

        candidate = self._candidates.get(candidate_id) if candidate_id else None
        if candidate is not None and candidate.state in {
            VerificationState.SUSPENDED,
            VerificationState.REVOKED,
            VerificationState.MERGED,
        }:
            raise ValueError(
                f"candidate is not eligible for gait enrollment: {candidate.state.value}"
            )

        target = identity_id or self._next_identity_id()
        gait_only = FeatureBundle(gait=features.gait)
        self.register_identity(
            target,
            gait_only,
            metadata={
                **dict(normalized.metadata),
                "enrollment": "strong_gait_automatic",
            },
            camera_id=normalized.camera_id,
            view_angle=normalized.quality.view_angle,
            quality=gait_quality,
            source_event_id=normalized.event_id,
        )
        self.store.save_observation(normalized, candidate_id=candidate_id)
        self._last_seen[target] = (normalized.camera_id, normalized.timestamp)

        if candidate is not None:
            if candidate.state == VerificationState.UNKNOWN:
                transition(candidate, VerificationState.ISOLATED_CANDIDATE)
            if candidate.state == VerificationState.ISOLATED_CANDIDATE:
                transition(candidate, VerificationState.PROVISIONAL_IDENTITY)
            if candidate.state == VerificationState.PROVISIONAL_IDENTITY:
                transition(candidate, VerificationState.CONFIRMED_IDENTITY)
            if candidate.state == VerificationState.CONFIRMED_IDENTITY:
                transition(candidate, VerificationState.MERGED)
            candidate.proposed_identity = target
            candidate.confirmed_identity = target
            candidate.updated_at = normalized.timestamp
            candidate.metadata["enrollment"] = "strong_gait_automatic"
            self.memory.remove_candidate(candidate.candidate_id)
            self.store.save_prototypes(
                [],
                replace_identity=candidate.candidate_id,
                zone="quarantine",
            )
            self.store.save_candidate(candidate)

        ranking = self.rank(normalized)
        target_score = next(
            (item for item in ranking if item.identity_id == target),
            None,
        )
        gait_probability = max(
            confidence,
            target_score.gait_probability or 0.0 if target_score is not None else 0.0,
        )
        request, _ = self._issue_appearance_request(
            identity_id=target,
            observation=normalized,
            gait_probability=gait_probability,
            gait_quality=gait_quality,
            candidate_id=candidate_id,
            reason="automatic_gait_enrollment",
        )
        assert request is not None
        self.store.audit(
            "gait_identity_enrolled",
            target,
            {
                "candidate_id": candidate_id,
                "event_id": normalized.event_id,
                "gait_confidence": confidence,
                "gait_quality": gait_quality,
                "appearance_request_id": request.request_id,
            },
        )
        return Decision(
            kind=DecisionKind.APPEARANCE_REQUESTED,
            state=VerificationState.CONFIRMED_IDENTITY,
            identity_id=target,
            candidate_id=candidate_id,
            score=gait_probability,
            margin=(
                self._branch_margin(ranking, target_score, "gait")
                if target_score is not None
                else None
            ),
            reasons=(
                "automatic_gait_enrollment",
                "strong_gait_confirmation",
                "appearance_absorption_requested",
            ),
            ranking=tuple(ranking),
            appearance_request_id=request.request_id,
        )

    def _spatial_probability(self, observation: Observation, identity_id: str) -> float:
        """返回某个身份的有界摄像头转场先验概率。"""
        previous = self._last_seen.get(identity_id)
        if previous is None:
            return 0.5
        previous_camera, previous_time = previous
        elapsed = observation.timestamp - previous_time
        if elapsed < 0 or elapsed > self.config.maximum_transition_seconds:
            return 0.5
        if previous_camera == observation.camera_id:
            return 0.76
        allowed = self.config.camera_transitions.get(previous_camera, ())
        if observation.camera_id in allowed:
            return 0.90
        if not allowed and previous_camera != "unknown-camera":
            # 未声明摄像头拓扑时保持中性，不虚构不可能的转场。
            return 0.5
        return 0.20

    def _score_identity(self, observation: Observation, identity_id: str) -> ScoreBreakdown:
        """在两个生物特征分支上独立评估一个正式身份。"""
        features = observation.features.normalized()
        appearance_similarity, appearance_proto = self.memory.best_formal(
            identity_id, "appearance", features.appearance
        ) if features.has_appearance else (None, None)
        gait_similarity, gait_proto = self.memory.best_formal(
            identity_id, "gait", features.gait
        ) if features.has_gait else (None, None)
        appearance_quality = observation.quality.appearance_availability(
            self.config.detection_confidence_floor
        ) if appearance_similarity is not None else 0.0
        gait_quality = observation.quality.gait_availability(
            self.config.minimum_frames,
            self.config.minimum_gait_cycles,
        ) if gait_similarity is not None else 0.0
        fusion = fuse_calibrated_scores(
            appearance_similarity=appearance_similarity,
            gait_similarity=gait_similarity,
            appearance_quality=appearance_quality,
            gait_quality=gait_quality,
            appearance_stability=self.stability.get(identity_id, "appearance"),
            gait_stability=self.stability.get(identity_id, "gait"),
            spatial_probability=self._spatial_probability(observation, identity_id),
            appearance_calibrator=self.appearance_calibrator,
            gait_calibrator=self.gait_calibrator,
            appearance_floor=self.config.appearance_floor,
            gait_floor=self.config.gait_floor,
            maximum_gait_weight=self.config.maximum_gait_weight,
            spatial_prior_weight=self.config.spatial_prior_weight,
        )
        return ScoreBreakdown(
            identity_id=identity_id,
            appearance_similarity=appearance_similarity,
            gait_similarity=gait_similarity,
            appearance_probability=fusion.appearance_probability,
            gait_probability=fusion.gait_probability,
            spatial_probability=self._spatial_probability(observation, identity_id),
            appearance_weight=fusion.appearance_weight,
            gait_weight=fusion.gait_weight,
            fused_probability=fusion.fused_probability,
            appearance_quality=appearance_quality,
            gait_quality=gait_quality,
            appearance_prototype_id=appearance_proto.prototype_id if appearance_proto else None,
            gait_prototype_id=gait_proto.prototype_id if gait_proto else None,
        )

    def rank(self, observation: Observation) -> tuple[ScoreBreakdown, ...]:
        """计算针对活跃身份的校准、质量感知分数。

        排名先独立计算每个分支，只有完成质量加权后才融合概率；当强外观和
        强步态分别选中不同身份时，结果会标记为冲突。
        """

        observation = observation.normalized()
        values = [self._score_identity(observation, identity_id) for identity_id in self.formal_identities]
        if not values:
            return ()
        values.sort(key=lambda item: item.fused_probability, reverse=True)
        appearance_winner = max(
            (item for item in values if item.appearance_probability is not None),
            key=lambda item: item.appearance_probability or 0.0,
            default=None,
        )
        gait_winner = max(
            (item for item in values if item.gait_probability is not None),
            key=lambda item: item.gait_probability or 0.0,
            default=None,
        )
        conflict = bool(
            appearance_winner
            and gait_winner
            and appearance_winner.identity_id != gait_winner.identity_id
            and (appearance_winner.appearance_probability or 0.0) >= self.config.conflict_probability
            and (gait_winner.gait_probability or 0.0) >= self.config.conflict_probability
        )
        if conflict:
            values = [replace(item, conflict=True) for item in values]
        return tuple(values)

    @staticmethod
    def _branch_winner(
        ranking: Sequence[ScoreBreakdown],
        branch: str,
    ) -> ScoreBreakdown | None:
        """选择某个模态中可用分数最高的身份。"""
        if branch == "gait":
            values = [item for item in ranking if item.gait_probability is not None]
            return max(values, key=lambda item: item.gait_probability or 0.0, default=None)
        values = [item for item in ranking if item.appearance_probability is not None]
        return max(values, key=lambda item: item.appearance_probability or 0.0, default=None)

    @staticmethod
    def _branch_margin(
        ranking: Sequence[ScoreBreakdown],
        winner: ScoreBreakdown | None,
        branch: str,
    ) -> float:
        """计算某个分支中胜者与次优者之间的概率间隔。"""
        if winner is None:
            return 0.0
        if branch == "gait":
            values = sorted(
                (item.gait_probability or 0.0 for item in ranking if item.gait_probability is not None),
                reverse=True,
            )
        else:
            values = sorted(
                (item.appearance_probability or 0.0 for item in ranking if item.appearance_probability is not None),
                reverse=True,
            )
        return float(values[0] - values[1]) if len(values) > 1 else float(values[0])

    def _strong_gait_match(
        self,
        ranking: Sequence[ScoreBreakdown],
    ) -> tuple[ScoreBreakdown | None, float]:
        """对步态排序应用概率、质量和 Top-2 间隔门控。"""
        winner = self._branch_winner(ranking, "gait")
        margin = self._branch_margin(ranking, winner, "gait")
        if winner is None:
            return None, margin
        if (
            (winner.gait_probability or 0.0) >= self.config.strong_gait_probability
            and winner.gait_quality >= self.config.strong_gait_quality
            and margin >= self.config.strong_gait_margin
        ):
            return winner, margin
        return None, margin

    def _strong_gait_signal(self, observation: Observation) -> bool:
        """在没有图库时，判断新建号候选人的步态质量是否足够强。"""

        features = observation.features.normalized()
        if not features.has_gait:
            return False
        quality = observation.quality.gait_availability(
            self.config.minimum_frames,
            self.config.minimum_gait_cycles,
        )
        return quality >= self.config.strong_gait_quality

    def _appearance_response(
        self,
        observation: Observation,
        ranking: Sequence[ScoreBreakdown],
        candidate_id: str | None,
    ) -> tuple[Decision | None, AppearanceAbsorptionRequest | None, str | None]:
        """检查观测是否是有效的步态授权响应。"""

        request_id = observation.appearance_request_id
        if not request_id:
            return None, None, None
        request = self.appearance_absorption.get(request_id)
        if request is None:
            return None, None, "unknown_appearance_request"
        effective_candidate = candidate_id or observation.metadata.get("candidate_id")
        valid, reason, request = self.appearance_absorption.validate(
            request_id,
            identity_id=request.identity_id,
            event_id=observation.event_id,
            candidate_id=str(effective_candidate) if effective_candidate is not None else None,
            now=observation.timestamp,
        )
        if not valid:
            if request is not None:
                self.store.save_appearance_request(request)
            return None, request, reason
        quality_reasons = self._appearance_response_quality_reasons(observation)
        if quality_reasons:
            return None, request, ":".join(quality_reasons)
        appearance_winner = self._branch_winner(ranking, "appearance")
        target = next(
            (item for item in ranking if item.identity_id == request.identity_id),
            None,
        )
        if target is None:
            return None, request, "appearance_response_missing_target"
        gait_winner, _ = self._strong_gait_match(ranking)
        if gait_winner is not None and gait_winner.identity_id != request.identity_id:
            return None, request, "appearance_response_gait_conflict"
        target_has_appearance = bool(
            self.memory.formal_prototypes(request.identity_id, "appearance")
        )
        if not target_has_appearance:
            # 仅步态自动建号尚无外观原型可供比较。强步态请求正是用于引导第
            # 一个高质量外观样本的授权。若外观强匹配到另一个身份，仍然阻止。
            if (
                appearance_winner is not None
                and appearance_winner.identity_id != request.identity_id
                and (appearance_winner.appearance_probability or 0.0)
                >= self.config.conflict_probability
            ):
                return None, request, "appearance_response_identity_mismatch"
            appearance_quality = observation.quality.appearance_availability(
                self.config.detection_confidence_floor
            )
            return (
                Decision(
                    kind=DecisionKind.APPEARANCE_RESPONSE_ACCEPTED,
                    state=VerificationState.CONFIRMED_IDENTITY,
                    identity_id=request.identity_id,
                    score=appearance_quality,
                    reasons=(
                        "gait_authorized_appearance_bootstrap",
                        "strong_appearance_direct_pass",
                    ),
                    ranking=tuple(ranking),
                    appearance_request_id=request.request_id,
                ),
                request,
                None,
            )
        confirmed_by_current_gait = bool(
            gait_winner is not None
            and gait_winner.identity_id == request.identity_id
        )
        if not confirmed_by_current_gait:
            if appearance_winner is None:
                return None, request, "appearance_response_missing_target"
            if appearance_winner.identity_id != request.identity_id:
                return None, request, "appearance_response_identity_mismatch"
            if (
                (target.appearance_probability or 0.0)
                < self.config.strong_appearance_probability
                or target.appearance_quality < self.config.strong_appearance_quality
            ):
                return None, request, "appearance_response_not_strong"
        response_reason = (
            "strong_gait_authorized_appearance_refresh"
            if confirmed_by_current_gait
            else "gait_authorized_appearance_absorption"
        )
        return (
            Decision(
                kind=DecisionKind.APPEARANCE_RESPONSE_ACCEPTED,
                state=VerificationState.CONFIRMED_IDENTITY,
                identity_id=request.identity_id,
                score=(
                    max(
                        gait_winner.gait_probability or 0.0,
                        observation.quality.appearance_availability(
                            self.config.detection_confidence_floor
                        ),
                    )
                    if confirmed_by_current_gait and gait_winner is not None
                    else target.appearance_probability
                ),
                margin=(
                    self._branch_margin(ranking, appearance_winner, "appearance")
                    if appearance_winner is not None
                    else None
                ),
                reasons=(response_reason, "strong_appearance_direct_pass"),
                ranking=tuple(ranking),
                appearance_request_id=request.request_id,
            ),
            request,
            None,
        )

    def _appearance_response_quality_reasons(
        self,
        observation: Observation,
    ) -> tuple[str, ...]:
        """不额外要求步态长度的外观响应门。"""

        quality = observation.quality
        reasons: list[str] = []
        if not quality.box_valid or quality.box_height <= 0:
            reasons.append("invalid_box")
        if quality.detection_confidence < self.config.detection_confidence_floor:
            reasons.append("low_detection_confidence")
        if quality.occlusion > self.config.maximum_write_occlusion:
            reasons.append("occluded")
        features = observation.features.normalized()
        if not features.has_appearance:
            reasons.append("no_appearance_signal")
        elif quality.appearance_availability(self.config.detection_confidence_floor) < self.config.strong_appearance_quality:
            reasons.append("appearance_response_quality_too_low")
        return tuple(dict.fromkeys(reasons))

    def _quality_block_reasons(self, observation: Observation) -> tuple[str, ...]:
        """说明观测为何还不能产生可靠证据。"""
        quality = observation.quality
        reasons: list[str] = []
        if not quality.box_valid or quality.box_height <= 0:
            reasons.append("invalid_box")
        if quality.detection_confidence < self.config.detection_confidence_floor:
            reasons.append("low_detection_confidence")
        if quality.occlusion > self.config.maximum_write_occlusion:
            reasons.append("occluded")
        if quality.frame_count < self.config.minimum_frames:
            reasons.append("too_short")
        features = observation.features.normalized()
        app_quality = (
            quality.appearance_availability(self.config.detection_confidence_floor)
            if features.has_appearance
            else 0.0
        )
        gait_quality = (
            quality.gait_availability(self.config.minimum_frames, self.config.minimum_gait_cycles)
            if features.has_gait
            else 0.0
        )
        if not features.has_appearance and not features.has_gait:
            reasons.append("no_feature_signal")
        if max(app_quality, gait_quality) < self.config.minimum_matching_quality:
            reasons.append("low_track_quality")
        return tuple(dict.fromkeys(reasons))

    def _decision_from_ranking(
        self,
        observation: Observation,
        ranking: Sequence[ScoreBreakdown],
        *,
        forced_identity: str | None = None,
    ) -> Decision:
        """将图库排名转换为可审计的开放集决策。"""
        quality_reasons = self._quality_block_reasons(observation)
        if quality_reasons:
            return Decision(
                kind=DecisionKind.NEED_MORE_DATA,
                state=VerificationState.UNKNOWN,
                score=None,
                reasons=quality_reasons,
                ranking=tuple(ranking),
            )
        if not ranking:
            return Decision(
                kind=DecisionKind.UNKNOWN,
                state=VerificationState.UNKNOWN,
                reasons=("no_formal_identity",),
                ranking=(),
            )
        if forced_identity is not None:
            selected = next((item for item in ranking if item.identity_id == forced_identity), None)
            if selected is None:
                selected = ranking[0]
            others = [item for item in ranking if item.identity_id != selected.identity_id]
            ranking_for_margin = sorted(others, key=lambda item: item.fused_probability, reverse=True)
        else:
            selected = ranking[0]
            ranking_for_margin = list(ranking[1:])
        margin = selected.fused_probability - (
            ranking_for_margin[0].fused_probability if ranking_for_margin else 0.0
        )
        reasons: list[str] = []
        if selected.conflict:
            reasons.append("appearance_gait_conflict")
            return Decision(
                kind=DecisionKind.CONFLICT,
                state=VerificationState.ISOLATED_CANDIDATE,
                identity_id=None,
                score=selected.fused_probability,
                margin=margin,
                reasons=tuple(reasons),
                ranking=tuple(ranking),
            )
        strong_gait, gait_margin = self._strong_gait_match(ranking)
        if strong_gait is not None:
            if forced_identity is not None and forced_identity != strong_gait.identity_id:
                return Decision(
                    kind=DecisionKind.CONFLICT,
                    state=VerificationState.ISOLATED_CANDIDATE,
                    score=strong_gait.gait_probability,
                    margin=gait_margin,
                    reasons=("gait_anchor_assignment_conflict",),
                    ranking=tuple(ranking),
                )
            # 步态是确认锚点。外观可以参与排名，但不能在这里替代步态胜者。
            selected = strong_gait
            return Decision(
                kind=DecisionKind.FORMAL_MATCH,
                state=VerificationState.CONFIRMED_IDENTITY,
                identity_id=selected.identity_id,
                score=max(
                    selected.fused_probability,
                    selected.gait_probability or 0.0,
                ),
                margin=gait_margin,
                reasons=("strong_gait_confirmation",),
                ranking=tuple(ranking),
            )
        if self.config.require_gait_for_formal_match:
            gait_candidate = self._branch_winner(ranking, "gait")
            gait_candidate_margin = self._branch_margin(
                ranking,
                gait_candidate,
                "gait",
            )
            if (
                gait_candidate is not None
                and gait_candidate.gait_quality >= self.config.strong_gait_quality
                and (gait_candidate.gait_probability or 0.0)
                < self.config.gait_novelty_threshold
            ):
                # 外观可能因为衣着、场景或过宽的原型而高度相似。一旦成熟的
                # 步态分支明确拒绝整个正式图库，继续保留最近旧身份会让自动
                # 控制器永远无法采集新的建号序列。
                return Decision(
                    kind=DecisionKind.UNKNOWN,
                    state=VerificationState.UNKNOWN,
                    identity_id=None,
                    score=gait_candidate.gait_probability,
                    margin=gait_candidate_margin,
                    reasons=(
                        "high_quality_gait_rejects_formal_gallery",
                        "appearance_cannot_override_gait_rejection",
                    ),
                    ranking=tuple(ranking),
                )
            if gait_candidate is not None:
                # 即使尚未达到强确认门，步态仍是身份锚点。不要把等待中的轨
                # 迹标为另一个仅由外观选出的胜者。
                selected = gait_candidate
                margin = gait_candidate_margin
            reasons.append("appearance_requires_gait_authorization")
            if selected.appearance_probability is not None:
                reasons.append("appearance_is_absorbable_only")
            return Decision(
                kind=DecisionKind.DEFERRED,
                state=VerificationState.ISOLATED_CANDIDATE,
                identity_id=selected.identity_id,
                score=selected.fused_probability,
                margin=margin,
                reasons=tuple(reasons),
                ranking=tuple(ranking),
            )
        branches = int(selected.appearance_probability is not None) + int(selected.gait_probability is not None)
        if not self.config.allow_single_modality_match and branches < 2:
            reasons.append("insufficient_modalities")
            return Decision(
                kind=DecisionKind.DEFERRED,
                state=VerificationState.ISOLATED_CANDIDATE,
                identity_id=selected.identity_id,
                score=selected.fused_probability,
                margin=margin,
                reasons=tuple(reasons),
                ranking=tuple(ranking),
            )
        if (
            selected.fused_probability >= self.config.accept_threshold
            and margin >= self.config.margin_threshold
        ):
            return Decision(
                kind=DecisionKind.FORMAL_MATCH,
                state=VerificationState.CONFIRMED_IDENTITY,
                identity_id=selected.identity_id,
                score=selected.fused_probability,
                margin=margin,
                reasons=(),
                ranking=tuple(ranking),
            )
        if selected.fused_probability >= self.config.defer_threshold:
            reasons.append("open_set_uncertain")
            if margin < self.config.margin_threshold:
                reasons.append("small_top2_margin")
            return Decision(
                kind=DecisionKind.DEFERRED,
                state=VerificationState.ISOLATED_CANDIDATE,
                identity_id=selected.identity_id,
                score=selected.fused_probability,
                margin=margin,
                reasons=tuple(reasons),
                ranking=tuple(ranking),
            )
        return Decision(
            kind=DecisionKind.UNKNOWN,
            state=VerificationState.UNKNOWN,
            identity_id=None,
            score=selected.fused_probability,
            margin=margin,
            reasons=("below_defer_threshold",),
            ranking=tuple(ranking),
        )

    def _get_or_create_candidate(self, candidate_id: str) -> CandidateRecord:
        """返回候选记录；不存在时创建一个隔离候选。"""
        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            candidate = CandidateRecord(candidate_id=candidate_id)
            self._candidates[candidate_id] = candidate
        return candidate

    def _candidate_key(self, observation: Observation, candidate_id: str | None) -> str:
        """解析调用方提供的稳定跨事件候选键。"""
        if candidate_id:
            return candidate_id
        metadata_candidate = observation.metadata.get("candidate_id")
        if metadata_candidate:
            return str(metadata_candidate)
        # 有意不静默合并来自不同会话的两个事件。若应用掌握跟踪器身份并希望
        # 进行跨事件累积，应传入稳定的 candidate_id。
        return f"candidate-{observation.event_id}"

    def _issue_challenge_if_needed(
        self,
        candidate: CandidateRecord,
        observation: Observation,
        reasons: tuple[str, ...],
    ) -> str | None:
        """为需要证据的候选签发或复用挑战提示。"""
        if candidate.state in {VerificationState.REVOKED, VerificationState.MERGED}:
            return None
        if self.challenge_manager.active_for_candidate(candidate.candidate_id):
            return self.challenge_manager.active_for_candidate(candidate.candidate_id)[0].prompt
        challenge = self.challenge_manager.issue(
            candidate.candidate_id,
            reasons,
            camera_id=observation.camera_id,
            now=observation.timestamp,
        )
        if challenge.challenge_id not in candidate.challenge_ids:
            candidate.challenge_ids.append(challenge.challenge_id)
        return challenge.prompt

    def _write_formal_sample(
        self,
        observation: Observation,
        breakdown: ScoreBreakdown,
        *,
        absorb_appearance: bool = False,
        absorb_gait: bool = True,
    ) -> tuple[MemoryUpdate, ...]:
        """将通过门控的样本写入正式记忆，并持久化审计轨迹。"""
        source_features = observation.features.normalized()
        appearance_quality = (
            observation.quality.appearance_availability(
                self.config.detection_confidence_floor
            )
            if source_features.has_appearance
            else 0.0
        )
        gait_quality = (
            observation.quality.gait_availability(
                self.config.minimum_frames,
                self.config.minimum_gait_cycles,
            )
            if source_features.has_gait
            else 0.0
        )
        if (
            not absorb_appearance
            and breakdown.fused_probability < self.config.minimum_formal_write_score
        ):
            return ()
        if breakdown.conflict or observation.quality.occlusion > self.config.maximum_write_occlusion:
            return ()
        branch_quality = appearance_quality if absorb_appearance else gait_quality
        if branch_quality < self.config.minimum_formal_write_quality:
            return ()
        identity_id = breakdown.identity_id
        memory_snapshot = self.memory.snapshot(identity_id)
        db_snapshot = self.store.snapshot_identity(identity_id, reason="before-formal-update")
        try:
            # 证据方向是明确的：普通正式匹配只吸收步态；外观需要有效请求令
            # 牌。有效响应只吸收外观，防止未请求的外观向量静默进入正式记忆。
            write_features = FeatureBundle(
                appearance=source_features.appearance if absorb_appearance else None,
                gait=source_features.gait if absorb_gait else None,
            )
            updates = self.memory.add_formal(
                identity_id,
                write_features,
                appearance_quality=appearance_quality,
                gait_quality=gait_quality,
                camera_id=observation.camera_id,
                view_angle=observation.quality.view_angle,
                clothing_tag=str(observation.metadata.get("clothing_tag")) if observation.metadata.get("clothing_tag") else None,
                source_event_id=observation.event_id,
            )
            successful = tuple(item for item in updates if not item.blocked)
            if not successful:
                return ()
            self.store.save_prototypes(
                list(self.memory.formal_prototypes(identity_id)),
                replace_identity=identity_id,
                zone="formal",
            )
            self.rebuild_index()
            for item in successful:
                similarity = (
                    breakdown.appearance_similarity
                    if item.modality == "appearance"
                    else breakdown.gait_similarity
                )
                if similarity is not None:
                    self.stability.update(identity_id, item.modality, float(np.clip(similarity, 0.0, 1.0)))
            self.store.audit(
                "formal_prototype_update",
                identity_id,
                {"event_id": observation.event_id, "updates": [item.action for item in successful]},
            )
            return updates
        except Exception:
            self.memory.restore(identity_id, memory_snapshot)
            self.store.restore_snapshot(db_snapshot)
            raise

    def _process(
        self,
        observation: Observation,
        *,
        candidate_id: str | None = None,
        ranking: Sequence[ScoreBreakdown] | None = None,
        forced_identity: str | None = None,
    ) -> Decision:
        """运行完整的单观测协议。

        顺序是有意设计的：

        1. 归一化观测并进行排名；
        2. 如果提供了外观令牌，则校验它；
        3. 只有身份门通过时才返回/吸收正式证据；
        4. 否则将观测追加到隔离区，并更新候选人状态及审计轨迹。

        将该顺序集中在一个深模块方法中，可以防止 GUI、批处理和自动建号调
        用方意外实现出不同的安全规则。
        """
        observation = observation.normalized()
        ranking = tuple(self.rank(observation) if ranking is None else ranking)
        # 外观响应先于普通排名检查，因为有效的步态签发令牌明确绑定了一个
        # 身份/轨迹。
        response_decision, response_request, response_error = self._appearance_response(
            observation,
            ranking,
            candidate_id,
        )
        if response_decision is not None:
            decision = response_decision
        else:
            decision = self._decision_from_ranking(
                observation,
                ranking,
                forced_identity=forced_identity,
            )
            if response_error:
                decision = replace(
                    decision,
                    reasons=tuple((*decision.reasons, response_error)),
                )
        # 正式结果可以更新正式图库。其他所有结果都进入隔离区，只能通过证据
        # 累积和下面的晋升策略继续推进。
        if decision.kind in {
            DecisionKind.FORMAL_MATCH,
            DecisionKind.APPEARANCE_RESPONSE_ACCEPTED,
        } and decision.identity_id is not None:
            self.store.save_observation(observation)
            selected = next(
                item for item in ranking if item.identity_id == decision.identity_id
            )
            self._last_seen[decision.identity_id] = (
                observation.camera_id,
                observation.timestamp,
            )
            is_appearance_response = decision.kind == DecisionKind.APPEARANCE_RESPONSE_ACCEPTED
            self._write_formal_sample(
                observation,
                selected,
                absorb_appearance=is_appearance_response,
                absorb_gait=not is_appearance_response,
            )
            request_id = decision.appearance_request_id
            if is_appearance_response and response_request is not None:
                self.appearance_absorption.consume(
                    response_request.request_id,
                    observation.event_id,
                )
                self.store.save_appearance_request(response_request)
                request_id = response_request.request_id
                self.store.audit(
                    "appearance_response_accepted",
                    decision.identity_id,
                    {
                        "request_id": response_request.request_id,
                        "event_id": observation.event_id,
                    },
                )
            elif self._strong_gait_match(ranking)[0] is not None:
                gait_anchor = self._strong_gait_match(ranking)[0]
                assert gait_anchor is not None
                request, issued = self._issue_appearance_request(
                    identity_id=decision.identity_id,
                    observation=observation,
                    gait_probability=float(gait_anchor.gait_probability or 0.0),
                    gait_quality=gait_anchor.gait_quality,
                    candidate_id=candidate_id,
                    reason="strong_gait_confirmation",
                )
                if request is not None and not issued:
                    decision = replace(
                        decision,
                        reasons=tuple(
                            (*decision.reasons, "appearance_absorption_request_pending")
                        ),
                    )
                request_id = request.request_id if request is not None else None
                if (
                    request is not None
                    and "appearance_absorption_requested" not in decision.reasons
                ):
                    decision = replace(
                        decision,
                        reasons=tuple((*decision.reasons, "appearance_absorption_requested")),
                    )
            self.store.audit(
                "formal_match",
                decision.identity_id,
                {
                    "event_id": observation.event_id,
                    "score": decision.score,
                    "margin": decision.margin,
                },
            )
            return replace(decision, appearance_request_id=request_id)

        cid = self._candidate_key(observation, candidate_id)
        # 未知证据优先使用调用方提供的稳定候选人 ID；否则依靠事件/会话/轨迹
        # 来源，防止两个无关采集被意外合并。
        candidate = self._get_or_create_candidate(cid)
        previous = tuple(self._candidate_observations.get(cid, ()))
        self._candidate_observations.setdefault(cid, []).append(observation)
        self.store.save_observation(observation, candidate_id=cid)
        # 在修改候选人观测列表之前计算独立性。当前事件绝不能让自己看起来是
        # 独立证据。
        independence = check_independence(previous, observation)
        challenge_validation = None
        if observation.challenge_id:
            challenge_validation = self.challenge_manager.validate(cid, observation, now=observation.timestamp)
        challenge_valid = bool(challenge_validation and challenge_validation.valid)
        if challenge_valid and observation.challenge_id not in candidate.challenge_ids:
            candidate.challenge_ids.append(observation.challenge_id)  # type: ignore[arg-type]

        top = decision.ranking[0] if decision.ranking else None
        observation_features = observation.features.normalized()
        observation_appearance_quality = (
            observation.quality.appearance_availability(self.config.detection_confidence_floor)
            if observation_features.has_appearance
            else 0.0
        )
        observation_gait_quality = (
            observation.quality.gait_availability(
                self.config.minimum_frames,
                self.config.minimum_gait_cycles,
            )
            if observation_features.has_gait
            else 0.0
        )
        high_quality = max(
            top.appearance_quality if top else observation_appearance_quality,
            top.gait_quality if top else observation_gait_quality,
        ) >= self.config.minimum_formal_write_quality
        proposed_identity = decision.identity_id if decision.kind == DecisionKind.DEFERRED else None
        auto_provisional = bool(
            self.config.auto_provisional_transition
            and decision.score is not None
            and decision.score >= self.config.minimum_promotion_score
            and (decision.margin or 0.0) >= self.config.minimum_promotion_margin
        )
        evidence_id = f"evidence-{uuid4().hex}"
        mark_evidence(
            candidate,
            observation=observation,
            evidence_id=evidence_id,
            identity_id=proposed_identity,
            score=decision.score,
            independent=independence,
            high_quality=high_quality,
            conflict=decision.kind == DecisionKind.CONFLICT,
            auto_provisional=auto_provisional,
        )
        if challenge_valid:
            candidate.metadata.setdefault("valid_challenges", []).append(observation.challenge_id)
        # 将弱/未知证据写入隔离区是安全的，因为在显式晋升前它们不能参与正式
        # 匹配。
        self.memory.add_quarantine(
            cid,
            observation.features,
            appearance_quality=top.appearance_quality if top else observation_appearance_quality,
            gait_quality=top.gait_quality if top else observation_gait_quality,
            camera_id=observation.camera_id,
            view_angle=observation.quality.view_angle,
            clothing_tag=str(observation.metadata.get("clothing_tag")) if observation.metadata.get("clothing_tag") else None,
            source_event_id=observation.event_id,
        )
        self.store.save_prototypes(
            list(self.memory.quarantine_prototypes(cid)),
            replace_identity=cid,
            zone="quarantine",
        )
        evidence_payload = {
            "reasons": decision.reasons,
            "independence_reason": independence.reason,
            "challenge_valid": challenge_valid,
            "gait_anchor": self._strong_gait_signal(observation),
            "appearance_absorption_request_id": observation.appearance_request_id,
            "model_version": observation.model_version,
            "threshold_version": observation.threshold_version,
        }
        if evidence_payload["gait_anchor"]:
            candidate.metadata.setdefault("gait_anchor_events", []).append(observation.event_id)
        self.store.save_evidence(
            evidence_id=evidence_id,
            candidate_id=cid,
            event_id=observation.event_id,
            identity_id=proposed_identity,
            kind=decision.kind.value,
            score=decision.score,
            margin=decision.margin,
            independent=independence.independent or challenge_valid,
            payload=evidence_payload,
        )
        candidate.updated_at = observation.timestamp
        self.store.save_candidate(candidate)
        challenge_prompt = None
        if not challenge_valid and decision.kind in {
            DecisionKind.NEED_MORE_DATA,
            DecisionKind.DEFERRED,
            DecisionKind.CONFLICT,
            DecisionKind.UNKNOWN,
        }:
            challenge_prompt = self._issue_challenge_if_needed(
                candidate,
                observation,
                tuple(dict.fromkeys((*decision.reasons, *self._quality_block_reasons(observation)))),
            )
            self.store.save_candidate(candidate)
        if self.config.auto_confirmation and candidate.state == VerificationState.PROVISIONAL_IDENTITY:
            try:
                self.promote_candidate(cid)
                return replace(
                    decision,
                    kind=DecisionKind.CANDIDATE_UPDATED,
                    state=VerificationState.CONFIRMED_IDENTITY,
                    candidate_id=cid,
                    challenge_prompt=challenge_prompt,
                    evidence_id=evidence_id,
                    reasons=tuple((*decision.reasons, "auto_confirmed")),
                )
            except (ValueError, KeyError):
                # 自动确认是可选便利功能；最终事务失败时，候选人必须仍留在
                # 隔离区且可见。
                pass
        final_kind = (
            DecisionKind.CANDIDATE_CREATED
            if not previous
            else DecisionKind.CANDIDATE_UPDATED
        )
        return replace(
            decision,
            kind=final_kind,
            state=candidate.state,
            candidate_id=cid,
            challenge_prompt=challenge_prompt,
            evidence_id=evidence_id,
        )

    def verify(self, observation: Observation, *, candidate_id: str | None = None) -> Decision:
        """处理一条轨迹/事件，并返回安全、可审计的决策。"""

        return self._process(observation, candidate_id=candidate_id)

    def verify_batch(
        self,
        observations: Sequence[Observation],
        *,
        candidate_ids: Sequence[str | None] | None = None,
    ) -> list[Decision]:
        """使用全局一对一身份指派验证一帧。

        先进行指派再做单项决策，防止两条轨迹消费同一个身份。当步态是正式
        锚点时，指派矩阵使用与单观测路径相同的强步态可行性门。
        """

        if candidate_ids is not None and len(candidate_ids) != len(observations):
            raise ValueError("candidate_ids must align with observations")
        if not observations:
            return []
        normalized = [item.normalized() for item in observations]
        identities = list(self.formal_identities)
        if not identities:
            return [
                self._process(
                    item,
                    candidate_id=candidate_ids[index] if candidate_ids else None,
                    ranking=(),
                )
                for index, item in enumerate(normalized)
            ]
        rankings = [self.rank(item) for item in normalized]
        if self.config.require_gait_for_formal_match:
            # 全局指派必须使用与最终决策相同的步态锚点可行性。过去融合后的
            # 外观分数会让陌生人先占用旧身份列，步态随后才拒绝，导致识别和
            # 新身份注册都被挤饿。
            strong_gait_winners = [self._strong_gait_match(ranking)[0] for ranking in rankings]
            score_matrix = np.asarray(
                [
                    [
                        (
                            float(winner.gait_probability or 0.0)
                            if winner is not None and winner.identity_id == identity_id
                            else 0.0
                        )
                        for identity_id in identities
                    ]
                    for winner in strong_gait_winners
                ],
                dtype=np.float32,
            )
            assignment_threshold = self.config.strong_gait_probability
            assignment_appearance: np.ndarray | None = None
            assignment_margin = 0.0
        else:
            score_matrix = np.asarray(
                [
                    [
                        next(
                            (
                                score.fused_probability
                                for score in ranking
                                if score.identity_id == identity_id
                            ),
                            0.0,
                        )
                        for identity_id in identities
                    ]
                    for ranking in rankings
                ],
                dtype=np.float32,
            )
            assignment_threshold = self.config.accept_threshold
            assignment_margin = self.config.margin_threshold
            assignment_appearance = np.asarray(
                [
                    [
                        next(
                            (
                                (
                                    score.appearance_similarity
                                    if score.appearance_similarity is not None
                                    else 1.0
                                )
                                for score in ranking
                                if score.identity_id == identity_id
                            ),
                            1.0,
                        )
                        for identity_id in identities
                    ]
                    for ranking in rankings
                ],
                dtype=np.float32,
            )
        assigned = gated_global_assignment(
            score_matrix,
            assignment_appearance,
            accept_threshold=assignment_threshold,
            appearance_floor=self.config.appearance_floor,
            margin_threshold=assignment_margin,
        )
        assigned_identities = {
            identities[column] for column in assigned.values()
        }
        decisions: list[Decision] = []
        for index, item in enumerate(normalized):
            candidate_id = candidate_ids[index] if candidate_ids else None
            if index in assigned:
                forced = identities[assigned[index]]
                decisions.append(
                    self._process(
                        item,
                        candidate_id=candidate_id,
                        ranking=rankings[index],
                        forced_identity=forced,
                    )
                )
            else:
                filtered = tuple(
                    score
                    for score in rankings[index]
                    if score.identity_id not in assigned_identities
                )
                decisions.append(
                    self._process(
                        item,
                        candidate_id=candidate_id,
                        ranking=filtered,
                    )
                )
        return decisions

    def get_candidate(self, candidate_id: str) -> CandidateRecord | None:
        """返回一个仍在跟踪中的内存候选人记录。"""
        return self._candidates.get(candidate_id)

    def list_candidates(self) -> tuple[CandidateRecord, ...]:
        """按最近更新时间从新到旧返回候选人。"""
        return tuple(sorted(self._candidates.values(), key=lambda item: item.updated_at, reverse=True))

    def issue_challenge(self, candidate_id: str, reasons: Iterable[str] = ()) -> str:
        """为已知候选人签发并持久化挑战。"""
        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            raise KeyError(candidate_id)
        challenge = self.challenge_manager.issue(candidate_id, tuple(reasons))
        candidate.challenge_ids.append(challenge.challenge_id)
        self.store.save_candidate(candidate)
        return challenge.challenge_id

    def _promotion_check(self, candidate: CandidateRecord) -> tuple[bool, tuple[str, ...]]:
        """说明隔离证据是否满足晋升策略。"""
        reasons: list[str] = []
        if candidate.state in {VerificationState.SUSPENDED, VerificationState.REVOKED, VerificationState.MERGED}:
            reasons.append(f"candidate_state:{candidate.state.value}")
        if candidate.independent_event_count < self.config.minimum_independent_events:
            reasons.append("not_enough_independent_events")
        if candidate.high_quality_evidence_count < 1:
            reasons.append("no_high_quality_evidence")
        if candidate.conflict_count:
            reasons.append("evidence_conflict")
        if not candidate.metadata.get("gait_anchor_events"):
            reasons.append("no_gait_anchor")
        evidence = self.store.list_evidence(candidate.candidate_id)
        qualified = [
            row
            for row in evidence
            if (
                row["score"] is not None
                and float(row["score"]) >= self.config.minimum_promotion_score
                and (row["margin"] is None or float(row["margin"]) >= self.config.minimum_promotion_margin)
            )
            or bool(row["payload"].get("challenge_valid"))
        ]
        if candidate.proposed_identity is not None:
            if len(qualified) < self.config.minimum_independent_events:
                reasons.append("evidence_scores_not_confirmed")
        elif candidate.high_quality_evidence_count < self.config.minimum_independent_events:
            # 真正的新人物尚没有正式身份分数。自动合并仍不安全，但显式晋升
            # 可以使用两条相互独立的高质量观测作为建号证据。
            reasons.append("not_enough_high_quality_enrollment_events")
        return not reasons, tuple(reasons)

    def promote_candidate(
        self,
        candidate_id: str,
        *,
        identity_id: str | None = None,
        force: bool = False,
    ) -> PromotionResult:
        """在证据检查或显式复核后晋升隔离原型。"""

        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            raise KeyError(candidate_id)
        eligible, reasons = self._promotion_check(candidate)
        if not eligible and not force:
            raise ValueError("candidate is not promotable: " + ", ".join(reasons))
        target = identity_id or candidate.proposed_identity or self._next_identity_id()
        if target in self.memory.formal and self._identity_states.get(target) in {
            VerificationState.SUSPENDED,
            VerificationState.REVOKED,
            VerificationState.MERGED,
        }:
            raise ValueError(f"target identity is not active: {target}")

        formal_snapshot = self.memory.snapshot(target)
        candidate_snapshot = {
            modality: list(values)
            for modality, values in self.memory.quarantine.get(candidate_id, {}).items()
        }
        db_snapshot = self.store.snapshot_identity(target, reason="before-candidate-promotion")
        try:
            if force and candidate.state == VerificationState.SUSPENDED:
                # 人工复核是冲突时的显式逃生通道；保留审计标记，不要静默地
                # 把它当成普通状态转换。
                candidate.metadata["manual_review_override"] = True
                candidate.state = VerificationState.PROVISIONAL_IDENTITY
            if candidate.state == VerificationState.ISOLATED_CANDIDATE:
                transition(candidate, VerificationState.PROVISIONAL_IDENTITY)
            if candidate.state == VerificationState.PROVISIONAL_IDENTITY:
                transition(candidate, VerificationState.CONFIRMED_IDENTITY)
            copied = self.memory.promote_candidate(
                candidate_id,
                target,
                minimum_quality=self.config.minimum_append_quality,
            )
            self.store.upsert_identity(target, VerificationState.CONFIRMED_IDENTITY)
            self.store.save_prototypes(
                list(self.memory.formal_prototypes(target)),
                replace_identity=target,
                zone="formal",
            )
            self.store.save_prototypes([], replace_identity=candidate_id, zone="quarantine")
            candidate.confirmed_identity = target
            candidate.updated_at = time.time()
            self._identity_states[target] = VerificationState.CONFIRMED_IDENTITY
            self.store.save_candidate(candidate)
            self.store.audit(
                "candidate_promoted",
                candidate_id,
                {"identity_id": target, "prototype_count": copied, "forced": force},
            )
            self.rebuild_index()
            return PromotionResult(
                candidate_id=candidate_id,
                identity_id=target,
                state=candidate.state,
                independent_event_count=candidate.independent_event_count,
                prototype_count=copied,
                reasons=reasons if force and not eligible else (),
            )
        except Exception:
            self.memory.restore(target, formal_snapshot)
            self.memory.quarantine[candidate_id] = candidate_snapshot
            self.store.restore_snapshot(db_snapshot)
            raise

    def _next_identity_id(self) -> str:
        """选择下一个未使用的、便于人读取的 ``P<number>`` 身份 ID。"""
        known_ids = set(self.memory.formal) | set(self._identity_states)
        numbers = [
            int(identity_id[1:])
            for identity_id in known_ids
            if identity_id.startswith("P") and identity_id[1:].isdigit()
        ]
        number = max(numbers, default=0) + 1
        while f"P{number}" in known_ids:
            number += 1
        return f"P{number}"

    def rollback(self, identity_id: str, snapshot_id: int) -> int:
        """恢复正式图库快照，并保留审计轨迹。"""

        restored = self.store.restore_snapshot(snapshot_id)
        grouped: dict[str, list[Any]] = {}
        for prototype in restored:
            grouped.setdefault(prototype.modality, []).append(prototype)
        self.memory.restore(identity_id, grouped)
        self.store.save_prototypes(
            list(self.memory.formal_prototypes(identity_id)),
            replace_identity=identity_id,
            zone="formal",
        )
        self.rebuild_index()
        self.store.audit(
            "formal_gallery_rollback",
            identity_id,
            {"snapshot_id": snapshot_id, "prototype_count": len(restored)},
        )
        return len(restored)

    def purge_expired_candidates(self, *, now: float | None = None) -> tuple[str, ...]:
        """撤销过期的非正式候选人，并删除其隔离向量。"""
        now = time.time() if now is None else float(now)
        removed: list[str] = []
        for candidate_id, candidate in list(self._candidates.items()):
            if now - candidate.updated_at <= self.config.candidate_ttl_seconds:
                continue
            if candidate.state in {VerificationState.CONFIRMED_IDENTITY, VerificationState.REVOKED}:
                continue
            candidate.state = VerificationState.REVOKED
            candidate.updated_at = now
            self.memory.remove_candidate(candidate_id)
            self.store.save_prototypes([], replace_identity=candidate_id, zone="quarantine")
            self.store.save_candidate(candidate)
            self.store.audit("candidate_expired", candidate_id)
            removed.append(candidate_id)
        return tuple(removed)

    def close(self) -> None:
        """关闭由该验证器持有的存储适配器。"""
        self.store.close()
