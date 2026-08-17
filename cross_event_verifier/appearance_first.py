"""OSNet-first 视觉身份绑定与 GaitGraph2 步态学习控制器。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .assignment import gated_global_assignment
from .automation import AutomationPolicy, AutomationStage, AutomationStatus
from .engine import CrossEventVerifier
from .types import (
    Decision,
    DecisionKind,
    FeatureBundle,
    GaitQualityBand,
    GaitReadinessReport,
    GaitReadinessState,
    Observation,
    VerificationState,
)


@dataclass
class _AppearanceFirstTrackState:
    """一条 Track 的短期视觉绑定和步态采集窗口。"""

    appearance_samples: deque[np.ndarray]
    gait_samples: deque[np.ndarray]
    identity_id: str | None = None
    gait_window_start_timestamp: float | None = None
    conflict: bool = False


@dataclass(frozen=True)
class _StableGaitSample:
    vector: np.ndarray
    stability: float
    sample_count: int


class AppearanceFirstGaitEnrollmentController:
    """先确认视觉身份，再把独立步态事件注册到该身份的控制器。

    该控制器与旧的 ``AutomaticVerificationController`` 并存。旧控制器继续
    支持历史 API 的“强步态创建身份”流程；生产 GUI 使用本控制器，避免把
    步态事件计数误当成视觉身份编号进度。
    """

    def __init__(
        self,
        verifier: CrossEventVerifier,
        policy: AutomationPolicy | None = None,
    ) -> None:
        self.verifier = verifier
        self.policy = policy or AutomationPolicy()
        self._registration_enabled = self.policy.enabled
        self._states: dict[str, _AppearanceFirstTrackState] = {}
        self._manual_request_id: str | None = None
        self._readiness_cache: dict[str, GaitReadinessReport] = {}

    @property
    def registration_enabled(self) -> bool:
        """返回是否允许自动创建新的视觉身份编号。"""

        return self._registration_enabled

    @property
    def manual_request_id(self) -> str | None:
        """兼容旧 GUI 的人工令牌字段；OSNet-first 默认不需要该令牌。"""

        return self._manual_request_id

    def set_registration_enabled(self, enabled: bool) -> None:
        """切换自动创建视觉身份，不影响已有身份的步态学习。"""

        self._registration_enabled = bool(enabled)

    def update_policy(self, policy: AutomationPolicy) -> None:
        """原子替换采集窗口策略，并清除未完成的窗口。"""

        if policy == self.policy:
            return
        self.policy = policy
        self.reset_gait_samples()

    def reset_gait_samples(self) -> None:
        """清除未完成的视觉/步态样本，不删除已接受的步态事件。"""

        for state in self._states.values():
            state.appearance_samples.clear()
            state.gait_samples.clear()
            state.gait_window_start_timestamp = None
            state.conflict = False
        self._readiness_cache.clear()

    def reset_tracks(
        self,
        *,
        preserve_candidate_ids: Sequence[str] = (),
        discard_unpreserved: bool = False,
    ) -> None:
        """在换源时清理短期 Track 窗口，正式身份和事件事实继续保留。"""

        keep = {value for value in preserve_candidate_ids if value}
        if discard_unpreserved:
            self._states = {
                key: value for key, value in self._states.items() if key in keep
            }
        for state in self._states.values():
            state.appearance_samples.clear()
            state.gait_samples.clear()
            state.gait_window_start_timestamp = None
            state.conflict = False

    def discard_candidate(self, candidate_id: str) -> None:
        """清除一个短期采集任务的视觉绑定窗口。"""

        self._states.pop(candidate_id, None)

    def discard_candidates(self, candidate_ids: Sequence[str]) -> None:
        """批量清除已过期的短期采集任务。"""

        for candidate_id in candidate_ids:
            self.discard_candidate(candidate_id)

    def bind_request(
        self,
        candidate_id: str,
        request_id: str,
        *,
        identity_id: str | None = None,
    ) -> None:
        """兼容旧控制器调用；新的 OSNet-first 流程不依赖步态签发令牌。"""

        state = self._states.setdefault(
            candidate_id,
            _AppearanceFirstTrackState(deque(), deque()),
        )
        state.identity_id = identity_id
        self._manual_request_id = request_id

    def set_manual_request(
        self,
        request_id: str | None,
        *,
        candidate_id: str | None = None,
    ) -> None:
        """保留旧 GUI 入口，但不让令牌绕过视觉身份绑定。"""

        self._manual_request_id = request_id.strip() if request_id else None

    @staticmethod
    def _event_key(observation: Observation) -> str:
        """同一会话/挑战中的多个窗口只对应一个步态事件。"""

        if observation.challenge_id:
            return f"challenge:{observation.challenge_id}"
        if observation.capture_session_id and observation.capture_session_id != "unknown-session":
            return f"session:{observation.capture_session_id}"
        return f"event:{observation.event_id}"

    @staticmethod
    def _unit_mean(samples: deque[np.ndarray]) -> np.ndarray | None:
        if not samples:
            return None
        vector = np.mean(np.stack(tuple(samples), axis=0), axis=0)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            return None
        return (vector / norm).astype(np.float32)

    def _state(self, key: str) -> _AppearanceFirstTrackState:
        return self._states.setdefault(
            key,
            _AppearanceFirstTrackState(
                appearance_samples=deque(),
                gait_samples=deque(),
            ),
        )

    def _appearance_assignments(
        self,
        observations: Sequence[Observation],
    ) -> tuple[list[tuple], dict[int, int]]:
        """对当前帧的 OSNet 分数做全局一对一身份指派。"""

        identities = list(self.verifier.formal_identities)
        rankings = [
            self.verifier.rank_appearance(item) if item.features.has_appearance else []
            for item in observations
        ]
        if not identities or not observations:
            return rankings, {}
        score_matrix = np.zeros((len(observations), len(identities)), dtype=np.float32)
        quality_matrix = np.zeros_like(score_matrix)
        for row, ranking in enumerate(rankings):
            for column, identity_id in enumerate(identities):
                score = next(
                    (item for item in ranking if item.identity_id == identity_id),
                    None,
                )
                if score is not None:
                    score_matrix[row, column] = float(score.appearance_probability or 0.0)
                    quality_matrix[row, column] = float(score.appearance_quality)
        assigned = gated_global_assignment(
            score_matrix,
            quality_matrix,
            accept_threshold=self.verifier.config.strong_appearance_probability,
            appearance_floor=self.verifier.config.strong_appearance_quality,
            margin_threshold=self.verifier.config.margin_threshold,
        )
        return rankings, assigned

    def _gait_assignments(
        self,
        observations: Sequence[Observation],
    ) -> dict[int, int]:
        """对已经步态就绪的身份执行 GaitGraph2 全局一对一检索。"""

        identities = [
            identity_id
            for identity_id in self.verifier.formal_identities
            if self._readiness(identity_id).state == GaitReadinessState.READY
        ]
        if not identities or not observations:
            return {}
        decisions = [
            self.verifier.match_gait_identity(
                item,
                require_identity_quality=True,
            )
            if item.features.has_gait
            else None
            for item in observations
        ]
        # A deferred/short query may still carry a useful ranking for audit, but
        # it must not enter the global identity assignment matrix.
        rankings = [
            decision.ranking
            if decision is not None and decision.kind == DecisionKind.FORMAL_MATCH
            else ()
            for decision in decisions
        ]
        score_matrix = np.zeros((len(observations), len(identities)), dtype=np.float32)
        quality_matrix = np.zeros_like(score_matrix)
        for row, ranking in enumerate(rankings):
            for column, identity_id in enumerate(identities):
                score = next(
                    (item for item in ranking if item.identity_id == identity_id),
                    None,
                )
                if score is not None:
                    score_matrix[row, column] = float(score.gait_probability or 0.0)
                    quality_matrix[row, column] = float(score.gait_quality)
        return gated_global_assignment(
            score_matrix,
            quality_matrix,
            accept_threshold=self.verifier.config.strong_gait_probability,
            appearance_floor=self.verifier.config.strong_gait_quality,
            margin_threshold=self.verifier.config.strong_gait_margin,
        )

    def _collect_appearance(
        self,
        state: _AppearanceFirstTrackState,
        observation: Observation,
        ranking: Sequence,
    ) -> tuple[str | None, str, float]:
        """累积 OSNet 样本，必要时创建一个新的视觉身份编号。"""

        features = observation.features.normalized()
        quality = observation.quality.appearance_availability(
            self.verifier.config.detection_confidence_floor
        )
        if not features.has_appearance or quality < self.verifier.config.strong_appearance_quality:
            state.appearance_samples.clear()
            return None, "等待高质量外观样本", quality
        vector = np.asarray(features.appearance, dtype=np.float32)
        reference = self._unit_mean(state.appearance_samples)
        if reference is not None:
            if reference.shape != vector.shape:
                state.appearance_samples.clear()
            else:
                similarity = float(np.dot(reference, vector))
                if similarity < self.verifier.config.appearance_identity_min_stability:
                    state.appearance_samples.clear()
                    state.appearance_samples.append(vector.copy())
                    return None, f"外观波动，重新采集（相似度 {similarity:.2f}）", quality
        state.appearance_samples.append(vector.copy())
        window = max(
            self.policy.gait_sample_window,
            self.verifier.config.appearance_identity_min_samples,
        )
        while len(state.appearance_samples) > window:
            state.appearance_samples.popleft()
        required = self.verifier.config.appearance_identity_min_samples
        if len(state.appearance_samples) < required:
            return (
                None,
                f"视觉身份确认中 {len(state.appearance_samples)}/{required}",
                quality,
            )
        centroid = self._unit_mean(state.appearance_samples)
        if centroid is None:
            state.appearance_samples.clear()
            return None, "外观样本聚合失败，重新采集", quality
        stability = float(
            np.mean([float(np.dot(item, centroid)) for item in state.appearance_samples])
        )
        if stability < self.verifier.config.appearance_identity_min_stability:
            return None, f"视觉身份稳定度 {stability:.2f} 不足", quality
        raw_similarities = [
            float(item.appearance_similarity)
            for item in ranking
            if item.appearance_similarity is not None
        ]
        if raw_similarities and max(raw_similarities) >= self.verifier.config.appearance_identity_novelty_threshold:
            return None, "外观接近已有视觉身份，等待更可靠绑定", quality
        if not self._registration_enabled:
            return None, "自动创建视觉身份已关闭", quality

        identity_id = self.verifier.next_identity_id()
        self.verifier.register_identity(
            identity_id,
            FeatureBundle(appearance=centroid),
            metadata={
                **dict(observation.metadata),
                "enrollment": "osnet_visual_identity",
                "appearance_sample_count": len(state.appearance_samples),
                "appearance_stability": stability,
            },
            camera_id=observation.camera_id,
            view_angle=observation.quality.view_angle,
            quality=quality,
            source_event_id=observation.event_id,
            model_version=observation.model_version,
            feature_schema=observation.feature_schema,
            artifact_sha256=observation.artifact_sha256,
            preprocess_version=observation.preprocess_version,
            joint_format=observation.joint_format,
            sequence_length=observation.sequence_length,
            tta_mode=observation.tta_mode,
            coordinate_contract=observation.coordinate_contract,
            embedding_dimensions=dict(observation.embedding_dimensions),
        )
        state.appearance_samples.clear()
        return identity_id, f"视觉身份 {identity_id} 已确认", quality

    def _appearance_conflicts_with_identity(
        self,
        observation: Observation,
        ranking: Sequence,
        identity_id: str,
    ) -> bool:
        """判断当前高质量外观是否构成该视觉身份的明确反证。

        外观暂时变弱不等于冲突：同一可靠 Track 的跨视角变化应由 must-link
        和多原型吸收。只有高质量样本明显低于冲突余弦门，或另一个身份同时
        以强概率和足够 margin 胜出，才阻止 gait 继续裁决。
        """

        if not observation.features.has_appearance:
            return False
        quality = observation.quality.appearance_availability(
            self.verifier.config.detection_confidence_floor
        )
        if quality < self.verifier.config.strong_appearance_quality:
            return False
        target = next(
            (item for item in ranking if item.identity_id == identity_id),
            None,
        )
        if target is not None and target.appearance_similarity is not None:
            if target.appearance_similarity <= self.verifier.config.appearance_conflict_similarity:
                return True
        candidates = [
            item
            for item in ranking
            if item.appearance_probability is not None
        ]
        winner = max(
            candidates,
            key=lambda item: item.appearance_probability or 0.0,
            default=None,
        )
        if winner is None or winner.identity_id == identity_id:
            return False
        probabilities = sorted(
            (float(item.appearance_probability or 0.0) for item in candidates),
            reverse=True,
        )
        margin = probabilities[0] - probabilities[1] if len(probabilities) > 1 else probabilities[0]
        return bool(
            winner.appearance_quality >= self.verifier.config.strong_appearance_quality
            and (winner.appearance_probability or 0.0) >= self.verifier.config.conflict_probability
            and margin >= self.verifier.config.margin_threshold
        )

    def _collect_bound_appearance(
        self,
        state: _AppearanceFirstTrackState,
        observation: Observation,
        identity_id: str,
    ) -> tuple[Decision | None, str, float]:
        """为已有视觉身份积累一个稳定的新视角 appearance prototype。"""

        features = observation.features.normalized()
        quality = observation.quality.appearance_availability(
            self.verifier.config.detection_confidence_floor
        )
        if not features.has_appearance or quality < self.verifier.config.strong_appearance_quality:
            return None, "等待高质量外观样本", quality
        vector = np.asarray(features.appearance, dtype=np.float32)
        reference = self._unit_mean(state.appearance_samples)
        if reference is not None:
            if reference.shape != vector.shape:
                state.appearance_samples.clear()
            else:
                similarity = float(np.dot(reference, vector))
                if similarity < self.verifier.config.appearance_identity_min_stability:
                    state.appearance_samples.clear()
                    state.appearance_samples.append(vector.copy())
                    return (
                        None,
                        f"新视角外观窗口重新采集（相似度 {similarity:.2f}）",
                        quality,
                    )
        state.appearance_samples.append(vector.copy())
        required = self.verifier.config.appearance_identity_min_samples
        window = max(required, self.policy.gait_sample_window)
        while len(state.appearance_samples) > window:
            state.appearance_samples.popleft()
        if len(state.appearance_samples) < required:
            return (
                None,
                f"视觉身份 {identity_id} 新视角学习中 "
                f"{len(state.appearance_samples)}/{required}",
                quality,
            )
        centroid = self._unit_mean(state.appearance_samples)
        if centroid is None:
            state.appearance_samples.clear()
            return None, "新视角外观聚合失败，重新采集", quality
        stability = float(
            np.mean([float(np.dot(item, centroid)) for item in state.appearance_samples])
        )
        if stability < self.verifier.config.appearance_identity_min_stability:
            return None, f"新视角外观稳定度 {stability:.2f} 不足", quality
        decision = self.verifier.enroll_appearance_prototype(
            identity_id,
            replace(
                observation,
                features=FeatureBundle(appearance=centroid),
                metadata={
                    **dict(observation.metadata),
                    "enrollment": "appearance_first_view_prototype",
                    "appearance_sample_count": len(state.appearance_samples),
                    "appearance_stability": stability,
                },
            ),
            stability=stability,
            sample_count=len(state.appearance_samples),
        )
        state.appearance_samples.clear()
        return decision, f"视觉身份 {identity_id} 已吸收新视角外观原型", quality

    def _collect_gait(
        self,
        state: _AppearanceFirstTrackState,
        observation: Observation,
        identity_id: str,
    ) -> tuple[_StableGaitSample | None, str, GaitQualityBand]:
        """从已绑定身份的 Track 收集一个稳定步态样本。"""

        quality = observation.quality
        band = quality.gait_quality_band(
            minimum_frames=self.verifier.config.gait_learning_min_frames,
            minimum_gait_cycles=self.verifier.config.minimum_gait_cycles,
            partial_threshold=self.verifier.config.partial_gait_quality,
            strong_threshold=self.verifier.config.strong_gait_quality,
        )
        if not observation.features.has_gait:
            return None, "等待 GaitGraph2 步态样本", band
        if band == GaitQualityBand.INVALID:
            state.gait_samples.clear()
            state.gait_window_start_timestamp = None
            return None, "步态样本 INVALID，等待完整下肢和行走周期", band
        if band == GaitQualityBand.PARTIAL:
            return None, "步态样本 PARTIAL，仅等待不写入步态原型", band
        # An event is counted once per session, but its later windows still
        # belong to the same person's learning stream.  The update decision is
        # made after a stable window is formed, so short/noisy samples are not
        # written merely because the event key already exists.
        if not state.gait_samples:
            state.gait_window_start_timestamp = observation.timestamp
        vector = np.asarray(observation.features.normalized().gait, dtype=np.float32)
        reference = self._unit_mean(state.gait_samples)
        if reference is not None:
            if reference.shape != vector.shape:
                state.gait_samples.clear()
                state.gait_window_start_timestamp = observation.timestamp
            else:
                similarity = float(np.dot(reference, vector))
                if similarity < self.policy.minimum_sample_similarity:
                    state.gait_samples.clear()
                    state.gait_window_start_timestamp = observation.timestamp
                    state.gait_samples.append(vector.copy())
                    return None, f"步态样本波动，重新采集（相似度 {similarity:.2f}）", band
        state.gait_samples.append(vector.copy())
        while len(state.gait_samples) > self.policy.gait_sample_window:
            state.gait_samples.popleft()
        required = self.policy.minimum_stable_gait_samples
        if len(state.gait_samples) < required:
            return None, f"步态样本采集中 {len(state.gait_samples)}/{required}", band
        centroid = self._unit_mean(state.gait_samples)
        if centroid is None:
            state.gait_samples.clear()
            return None, "步态样本聚合失败，重新采集", band
        stability = float(
            np.mean([float(np.dot(item, centroid)) for item in state.gait_samples])
        )
        if stability < self.policy.minimum_gait_stability:
            return None, f"步态事件稳定度 {stability:.2f} 不足", band
        return (
            _StableGaitSample(centroid, stability, len(state.gait_samples)),
            f"步态事件稳定（{stability:.2f}）",
            band,
        )

    def _readiness(self, identity_id: str) -> GaitReadinessReport:
        report = self._readiness_cache.get(identity_id)
        if report is None:
            report = self.verifier.evaluate_gait_readiness(identity_id)
            self._readiness_cache[identity_id] = report
        return report

    def gait_readiness(self, identity_id: str) -> GaitReadinessReport:
        """返回一个视觉身份当前的步态就绪报告。"""

        return self._readiness(identity_id)

    def _status(
        self,
        identity_id: str | None,
        report: GaitReadinessReport | None,
        *,
        message: str,
        progress: float = 0.0,
        gait_quality_band: GaitQualityBand | None = None,
        auto_registered: bool = False,
    ) -> AutomationStatus:
        if identity_id is None:
            return AutomationStatus(
                AutomationStage.WAIT_MORE_DATA,
                message,
                progress=float(np.clip(progress, 0.0, 1.0)),
                gait_quality_band=gait_quality_band.value if gait_quality_band else None,
            )
        report = report or self._readiness(identity_id)
        state = report.state
        if state == GaitReadinessState.CONFLICT:
            stage = AutomationStage.GAIT_CONFLICT
        elif state == GaitReadinessState.READY:
            stage = AutomationStage.GAIT_READY
        elif state in {GaitReadinessState.PROVISIONAL}:
            stage = AutomationStage.GAIT_PROVISIONAL
        elif report.accepted_event_count:
            stage = AutomationStage.GAIT_LEARNING
        else:
            stage = AutomationStage.VISUAL_CONFIRMED
        if report.accepted_event_count < self.verifier.config.gait_ready_min_events:
            event_progress = report.accepted_event_count / max(
                self.verifier.config.gait_ready_min_events,
                1,
            )
            progress = max(progress, event_progress)
        return AutomationStatus(
            stage,
            message,
            progress=float(np.clip(progress, 0.0, 1.0)),
            identity_id=identity_id,
            auto_registered=auto_registered,
            gait_quality_band=gait_quality_band.value if gait_quality_band else None,
            readiness_state=state.value,
            gait_event_count=report.accepted_event_count,
            gait_prototype_count=report.accepted_prototype_count,
            visual_identity_confirmed=True,
        )

    def register_visual_identity(
        self,
        identity_id: str,
        observation: Observation,
        *,
        state_key: str,
    ) -> None:
        """由人工确认或外部任务明确登记视觉身份，并绑定当前 Track。"""

        normalized = observation.normalized()
        quality = normalized.quality.appearance_availability(
            self.verifier.config.detection_confidence_floor
        )
        if not normalized.features.has_appearance:
            raise ValueError("视觉身份登记需要 OSNet 外观特征")
        if quality < self.verifier.config.strong_appearance_quality:
            raise ValueError(
                "视觉身份登记需要强外观质量："
                f"当前 {quality:.2f}/{self.verifier.config.strong_appearance_quality:.2f}"
            )
        if identity_id not in self.verifier.formal_identities:
            self.verifier.register_identity(
                identity_id,
                FeatureBundle(appearance=normalized.features.appearance),
                metadata={
                    **dict(normalized.metadata),
                    "enrollment": "manual_visual_identity",
                },
                camera_id=normalized.camera_id,
                view_angle=normalized.quality.view_angle,
                quality=quality,
                source_event_id=normalized.event_id,
                model_version=normalized.model_version,
                feature_schema=normalized.feature_schema,
                artifact_sha256=normalized.artifact_sha256,
                preprocess_version=normalized.preprocess_version,
                joint_format=normalized.joint_format,
                sequence_length=normalized.sequence_length,
                tta_mode=normalized.tta_mode,
                coordinate_contract=normalized.coordinate_contract,
                embedding_dimensions=dict(normalized.embedding_dimensions),
            )
        state = self._state(state_key)
        state.identity_id = identity_id
        state.appearance_samples.clear()

    def _bound_decision(
        self,
        observation: Observation,
        identity_id: str,
        ranking: Sequence,
    ) -> Decision:
        """输出当前 Track 的视觉身份确认结果。"""

        # 先让当前 OSNet 观察确认“没有明显反对这个已绑定身份”。
        # GaitGraph2 只能在这个视觉身份范围内增强结果，不能跳过该检查。
        if observation.features.has_appearance:
            appearance_decision = self.verifier.match_appearance_identity(
                observation,
                ranking=ranking,
                forced_identity=identity_id,
            )
        else:
            appearance_decision = None
        if observation.features.has_gait:
            report = self._readiness(identity_id)
            if report.state == GaitReadinessState.READY:
                gait_decision = self.verifier.match_gait_identity(
                    observation,
                    allowed_identity_ids=(identity_id,),
                    require_identity_quality=True,
                )
                if gait_decision.kind == DecisionKind.FORMAL_MATCH:
                    return replace(
                        gait_decision,
                        reasons=gait_decision.reasons + ("gait_ready_primary_match",),
                    )
        if appearance_decision is not None and appearance_decision.kind == DecisionKind.FORMAL_MATCH:
            return appearance_decision
        return Decision(
            kind=DecisionKind.FORMAL_MATCH,
            state=VerificationState.CONFIRMED_IDENTITY,
            identity_id=identity_id,
            score=None,
            reasons=("visual_identity_binding_continues",),
            ranking=tuple(ranking),
        )

    def _process_one(
        self,
        observation: Observation,
        *,
        state_key: str,
        ranking: Sequence,
        assigned_identity: str | None,
        appearance_identity: str | None = None,
        cannot_link_identity_ids: frozenset[str] = frozenset(),
    ) -> tuple[Decision, AutomationStatus]:
        state = self._state(state_key)
        bound_identity = state.identity_id or assigned_identity
        if (
            bound_identity is not None
            and bound_identity in cannot_link_identity_ids
        ):
            state.conflict = True
            state.gait_samples.clear()
            state.appearance_samples.clear()
            return (
                Decision(
                    kind=DecisionKind.CONFLICT,
                    state=VerificationState.ISOLATED_CANDIDATE,
                    identity_id=None,
                    reasons=("simultaneous_track_cannot_link",),
                    ranking=tuple(ranking),
                ),
                self._status(
                    None,
                    None,
                    message="同时出现的 Track 不能共享同一视觉身份，已隔离",
                ),
            )
        if (
            bound_identity is not None
            and self._appearance_conflicts_with_identity(
                observation,
                ranking,
                bound_identity,
            )
        ):
            state.conflict = True
            state.gait_samples.clear()
            state.appearance_samples.clear()
            return (
                Decision(
                    kind=DecisionKind.CONFLICT,
                    state=VerificationState.ISOLATED_CANDIDATE,
                    identity_id=None,
                    reasons=("appearance_conflicts_with_bound_identity",),
                    ranking=tuple(ranking),
                ),
                self._status(
                    None,
                    None,
                    message="高质量外观与当前视觉身份冲突，已禁止步态合并",
                ),
            )
        if (
            assigned_identity is not None
            and appearance_identity is not None
            and assigned_identity != appearance_identity
        ):
            state.conflict = True
            state.gait_samples.clear()
            state.appearance_samples.clear()
            return (
                Decision(
                    kind=DecisionKind.CONFLICT,
                    state=VerificationState.ISOLATED_CANDIDATE,
                    reasons=("gait_appearance_identity_conflict",),
                    ranking=tuple(ranking),
                ),
                self._status(
                    None,
                    None,
                    message="GaitGraph2 与 OSNet 指向不同视觉身份，已暂停该 Track",
                ),
            )
        if assigned_identity is not None:
            if state.identity_id is not None and state.identity_id != assigned_identity:
                state.conflict = True
                state.gait_samples.clear()
                state.appearance_samples.clear()
                return (
                    Decision(
                        kind=DecisionKind.CONFLICT,
                        state=VerificationState.ISOLATED_CANDIDATE,
                        reasons=("visual_identity_track_switch",),
                        ranking=tuple(ranking),
                    ),
                    self._status(
                        None,
                        None,
                        message="Track 的视觉身份发生冲突，已清空步态窗口",
                        progress=0.0,
                    ),
                )
            state.identity_id = assigned_identity
            state.conflict = False

        if state.conflict and state.identity_id is not None:
            report = self._readiness(state.identity_id)
            return (
                Decision(
                    kind=DecisionKind.CONFLICT,
                    state=VerificationState.ISOLATED_CANDIDATE,
                    reasons=("gait_conflict_requires_review",),
                    ranking=tuple(ranking),
                ),
                self._status(
                    state.identity_id,
                    report,
                    message=f"{state.identity_id}：步态冲突，已暂停写入并等待复核",
                ),
            )

        base_decision: Decision | None = None
        auto_registered = False
        if state.identity_id is None:
            visual_decision = self.verifier.match_appearance_identity(
                observation,
                ranking=ranking,
            )
            if visual_decision.kind == DecisionKind.FORMAL_MATCH and visual_decision.identity_id:
                state.identity_id = visual_decision.identity_id
                base_decision = visual_decision
            elif observation.features.has_appearance:
                created_identity, message, appearance_quality = self._collect_appearance(
                    state,
                    observation,
                    ranking,
                )
                if created_identity is not None:
                    state.identity_id = created_identity
                    auto_registered = True
                    base_decision = Decision(
                        kind=DecisionKind.VISUAL_IDENTITY_CREATED,
                        state=VerificationState.CONFIRMED_IDENTITY,
                        identity_id=created_identity,
                        score=appearance_quality,
                        reasons=("osnet_visual_identity_created",),
                    )
                else:
                    return (
                        visual_decision,
                        self._status(
                            None,
                            None,
                            message=message,
                            progress=min(
                                1.0,
                                len(state.appearance_samples)
                                / max(self.verifier.config.appearance_identity_min_samples, 1),
                            ),
                            gait_quality_band=(
                                observation.quality.gait_quality_band(
                                    minimum_frames=self.verifier.config.gait_learning_min_frames,
                                    minimum_gait_cycles=self.verifier.config.minimum_gait_cycles,
                                    partial_threshold=self.verifier.config.partial_gait_quality,
                                    strong_threshold=self.verifier.config.strong_gait_quality,
                                )
                                if observation.features.has_gait
                                else None
                            ),
                        ),
                    )
        if state.identity_id is None:
            return (
                base_decision
                or Decision(
                    kind=DecisionKind.UNKNOWN,
                    state=VerificationState.UNKNOWN,
                    reasons=("visual_identity_unresolved",),
                ),
                self._status(None, None, message="等待视觉身份确认"),
            )

        identity_id = state.identity_id
        if base_decision is None:
            base_decision = self._bound_decision(observation, identity_id, ranking)
        appearance_update, appearance_message, _ = (
            (None, "", 0.0)
            if auto_registered
            else self._collect_bound_appearance(state, observation, identity_id)
        )
        if appearance_update is not None:
            base_decision = replace(
                base_decision,
                reasons=base_decision.reasons + ("appearance_prototype_absorbed",),
            )
        stable, gait_message, gait_band = self._collect_gait(
            state,
            observation,
            identity_id,
        )
        report = self._readiness(identity_id)
        if stable is not None:
            event_key = self._event_key(observation)
            existing_events = self.verifier.load_gait_enrollment_events(identity_id)
            event_already_counted = any(
                event.identity_id == identity_id and event.event_key == event_key
                for event in existing_events
            )
            other_event_similarities = [
                float(np.dot(stable.vector, event.vector))
                for event in existing_events
                if event.event_key != event_key
                if stable.vector.shape == event.vector.shape
            ]
            if event_already_counted:
                self.verifier.enroll_gait_prototype(
                    identity_id,
                    replace(
                        observation,
                        features=FeatureBundle(gait=stable.vector),
                        metadata={
                            **dict(observation.metadata),
                            "enrollment": "appearance_first_gait_sample_update",
                            "gait_event_key": event_key,
                            "gait_sample_count": stable.sample_count,
                            "gait_stability": stable.stability,
                        },
                    ),
                    event_key=event_key,
                    stability=stable.stability,
                    sample_count=stable.sample_count,
                    update_existing_event=True,
                )
                self._readiness_cache.pop(identity_id, None)
                state.gait_samples.clear()
                state.gait_window_start_timestamp = None
                auto_registered = True
                report = self._readiness(identity_id)
                gait_message = (
                    "本步态事件已计数，继续吸收本会话步态样本"
                    f"（累计 {report.stable_sample_count} 个）"
                )
                base_decision = replace(
                    base_decision,
                    reasons=base_decision.reasons + ("gait_prototype_updated",),
                )
            elif other_event_similarities and max(other_event_similarities) >= self.verifier.config.gait_duplicate_event_similarity:
                state.gait_samples.clear()
                state.gait_window_start_timestamp = None
                gait_message = "步态事件与已有事件近重复，不重复计数"
            elif other_event_similarities and max(other_event_similarities) < self.verifier.config.gait_event_min_similarity:
                state.conflict = True
                state.gait_samples.clear()
                state.gait_window_start_timestamp = None
                report = self._readiness(identity_id)
                return (
                    replace(
                        base_decision,
                        kind=DecisionKind.CONFLICT,
                        identity_id=None,
                        reasons=base_decision.reasons + ("independent_gait_event_conflict",),
                    ),
                    self._status(
                        identity_id,
                        report,
                        message="独立步态事件冲突，暂停写入并等待复核",
                        gait_quality_band=gait_band,
                    ),
                )
            else:
                enrollment = replace(
                    observation,
                    features=FeatureBundle(gait=stable.vector),
                    metadata={
                        **dict(observation.metadata),
                        "enrollment": "appearance_first_gait_prototype",
                        "gait_event_key": event_key,
                        "gait_sample_count": stable.sample_count,
                        "gait_stability": stable.stability,
                    },
                )
                self.verifier.enroll_gait_prototype(
                    identity_id,
                    enrollment,
                    event_key=event_key,
                    stability=stable.stability,
                    sample_count=stable.sample_count,
                )
                self._readiness_cache.pop(identity_id, None)
                state.gait_samples.clear()
                state.gait_window_start_timestamp = None
                auto_registered = True
                report = self._readiness(identity_id)
                base_decision = replace(
                    base_decision,
                    reasons=base_decision.reasons + ("gait_prototype_enrolled",),
                )

        message = f"{identity_id}：{gait_message}"
        if appearance_message:
            message = f"{identity_id}：{appearance_message}；{gait_message}"
        if report.state == GaitReadinessState.READY:
            message = f"{identity_id}：步态就绪，可由 GaitGraph2 主检索"
        elif report.accepted_event_count:
            message = (
                f"{identity_id}：步态学习 {report.accepted_event_count}/"
                f"{self.verifier.config.gait_ready_min_events}，{gait_message}"
            )
        else:
            message = f"{identity_id}：视觉身份已确认，等待步态样本"
        return (
            base_decision,
            self._status(
                identity_id,
                report,
                message=message,
                progress=1.0 if report.state == GaitReadinessState.READY else 0.0,
                gait_quality_band=gait_band,
                auto_registered=auto_registered,
            ),
        )

    def verify(
        self,
        observation: Observation,
        *,
        candidate_id: str,
    ) -> tuple[Decision, AutomationStatus]:
        """处理单条 Track；批量路径负责全局一对一外观指派。"""

        return self.verify_batch(
            [observation],
            candidate_ids=[candidate_id],
        )[0]

    def verify_batch(
        self,
        observations: Sequence[Observation],
        *,
        candidate_ids: Sequence[str],
    ) -> tuple[tuple[Decision, AutomationStatus], ...]:
        """先完成 OSNet 全局绑定，再逐 Track 收集 GaitGraph2 事件。"""

        if len(observations) != len(candidate_ids):
            raise ValueError("candidate_ids must align with observations")
        if not observations:
            return ()
        normalized = [item.normalized() for item in observations]
        rankings, assigned = self._appearance_assignments(normalized)
        identities = list(self.verifier.formal_identities)
        appearance_ids = {
            row: identities[column]
            for row, column in assigned.items()
            if column < len(identities)
        }
        # OSNet-first 的全局指派只负责视觉身份。GaitGraph2 可以在已绑定
        # Track 的身份范围内做主检索，但不能把未绑定 Track 直接改写成正式 P。
        # 这条 seam 还保留了 Track 状态作为短期 must-link。
        state_identities = {
            index: self._state(candidate_id).identity_id
            for index, candidate_id in enumerate(candidate_ids)
        }
        proposed = {
            index: appearance_ids.get(index) or state_identities.get(index)
            for index in range(len(normalized))
        }
        identity_counts: dict[str, int] = {}
        for identity_id in proposed.values():
            if identity_id is not None:
                identity_counts[identity_id] = identity_counts.get(identity_id, 0) + 1
        cannot_link_identity_ids = frozenset(
            identity_id
            for identity_id, count in identity_counts.items()
            if count > 1
        )
        return tuple(
            self._process_one(
                item,
                state_key=candidate_id,
                ranking=rankings[index],
                assigned_identity=appearance_ids.get(index),
                appearance_identity=None,
                cannot_link_identity_ids=cannot_link_identity_ids,
            )
            for index, (item, candidate_id) in enumerate(zip(normalized, candidate_ids))
        )


__all__ = ["AppearanceFirstGaitEnrollmentController"]
