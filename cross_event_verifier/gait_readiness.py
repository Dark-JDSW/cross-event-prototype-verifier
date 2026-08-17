"""步态事件、步态原型与步态就绪判定。

本模块只回答一个领域问题：一个已经拥有视觉身份的人员，是否已经积累了
足以承担后续 GaitGraph2 检索的步态原型集合。它不负责产生姿态或修改模型
权重，也不把同一采集会话中的多个窗口误算成多个独立步态事件。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .config import VerifierConfig
from .types import (
    GaitEnrollmentEvent,
    GaitReadinessReport,
    GaitReadinessState,
    Prototype,
)


class GaitReadinessEvaluator:
    """根据事件独立性、内部一致性和留出条件评估步态就绪状态。"""

    def __init__(self, config: VerifierConfig) -> None:
        self.config = config

    @staticmethod
    def _condition_key(event: GaitEnrollmentEvent) -> tuple[str, str]:
        """把摄像头/视角作为步态条件覆盖度的稳定键。"""

        return (event.camera_id, event.view_angle or "unknown-view")

    @staticmethod
    def _pairwise_min_neighbour_similarity(events: Sequence[GaitEnrollmentEvent]) -> float | None:
        """返回每个事件到最近同身份事件的相似度中的最小值。"""

        if len(events) < 2:
            return None
        values: list[float] = []
        for index, event in enumerate(events):
            neighbours = [
                float(np.dot(event.vector, other.vector))
                for other_index, other in enumerate(events)
                if index != other_index and event.vector.shape == other.vector.shape
            ]
            if not neighbours:
                return None
            values.append(max(neighbours))
        return min(values) if values else None

    def _holdout_passed(
        self,
        events: Sequence[GaitEnrollmentEvent],
    ) -> bool | None:
        """执行简单的 leave-one-event-out genuine 留出检查。"""

        if len(events) < self.config.gait_ready_min_events:
            return None
        threshold = self.config.gait_holdout_min_similarity
        for index, event in enumerate(events):
            neighbours = [
                float(np.dot(event.vector, other.vector))
                for other_index, other in enumerate(events)
                if index != other_index and event.vector.shape == other.vector.shape
            ]
            if not neighbours or max(neighbours) < threshold:
                return False
        return True

    def _open_set_passed(
        self,
        identity_id: str,
        events: Sequence[GaitEnrollmentEvent],
        all_events: Iterable[GaitEnrollmentEvent],
        all_gait_prototypes: Iterable[Prototype],
    ) -> bool | None:
        """检查已有其他身份的步态原型是否超过开放集相似度上限。"""

        impostors: list[np.ndarray] = [
            event.vector
            for event in all_events
            if event.identity_id != identity_id
        ]
        impostors.extend(
            prototype.vector
            for prototype in all_gait_prototypes
            if prototype.identity_id != identity_id and prototype.modality == "gait"
        )
        if not impostors:
            return None
        threshold = self.config.open_set_max_impostor_similarity
        for event in events:
            if any(
                event.vector.shape == impostor.shape
                and float(np.dot(event.vector, impostor)) >= threshold
                for impostor in impostors
            ):
                return False
        return True

    def evaluate(
        self,
        identity_id: str,
        events: Sequence[GaitEnrollmentEvent],
        gait_prototypes: Sequence[Prototype] = (),
        *,
        all_events: Iterable[GaitEnrollmentEvent] = (),
        all_gait_prototypes: Iterable[Prototype] = (),
    ) -> GaitReadinessReport:
        """返回一个不改变图库的步态就绪报告。"""

        accepted_event_count = len(events)
        prototype_count = len(gait_prototypes)
        stable_sample_count = sum(max(0, int(event.sample_count)) for event in events)
        independent_sessions = len({event.capture_session_id for event in events})
        coverage_count = len({self._condition_key(event) for event in events})
        reasons: list[str] = []

        if not events:
            return GaitReadinessReport(
                identity_id=identity_id,
                state=GaitReadinessState.NOT_STARTED,
                accepted_prototype_count=prototype_count,
                reasons=("no_gait_event",),
            )

        for event in events:
            if event.stability < self.config.gait_event_min_similarity:
                reasons.append("event_stability_below_gate")
                return GaitReadinessReport(
                    identity_id=identity_id,
                    state=GaitReadinessState.CONFLICT,
                    accepted_event_count=accepted_event_count,
                    accepted_prototype_count=prototype_count,
                    stable_sample_count=stable_sample_count,
                    independent_session_count=independent_sessions,
                    coverage_count=coverage_count,
                    reasons=tuple(dict.fromkeys(reasons)),
                )

        min_similarity = self._pairwise_min_neighbour_similarity(events)
        if min_similarity is not None and min_similarity < self.config.gait_event_min_similarity:
            reasons.append("independent_gait_events_conflict")
            state = GaitReadinessState.CONFLICT
        elif accepted_event_count < self.config.gait_provisional_min_events:
            reasons.append("need_more_independent_gait_events")
            state = GaitReadinessState.LEARNING
        else:
            state = GaitReadinessState.PROVISIONAL

        holdout_passed = self._holdout_passed(events)
        open_set_passed = self._open_set_passed(
            identity_id,
            events,
            all_events,
            all_gait_prototypes,
        )

        if state != GaitReadinessState.CONFLICT and accepted_event_count >= self.config.gait_ready_min_events:
            if prototype_count < min(2, self.config.gait_ready_min_coverage):
                reasons.append("gait_prototype_diversity_pending")
            if coverage_count < self.config.gait_ready_min_coverage:
                reasons.append("gait_condition_coverage_pending")
            if holdout_passed is not True:
                reasons.append("gait_holdout_pending")
            if open_set_passed is not True:
                reasons.append("gait_open_set_validation_pending")
            if (
                coverage_count >= self.config.gait_ready_min_coverage
                and prototype_count >= min(2, self.config.gait_ready_min_coverage)
                and holdout_passed is True
                and open_set_passed is True
            ):
                state = GaitReadinessState.READY
                reasons.append("gait_ready")

        if not reasons:
            reasons.append("gait_provisional")
        return GaitReadinessReport(
            identity_id=identity_id,
            state=state,
            accepted_event_count=accepted_event_count,
            accepted_prototype_count=prototype_count,
            stable_sample_count=stable_sample_count,
            independent_session_count=independent_sessions,
            coverage_count=coverage_count,
            minimum_inter_event_similarity=min_similarity,
            holdout_passed=holdout_passed,
            open_set_passed=open_set_passed,
            reasons=tuple(dict.fromkeys(reasons)),
        )


__all__ = ["GaitReadinessEvaluator"]
