"""候选人状态转换和证据独立性规则。

未知证据只有来自相互独立的事件时，才适合用于晋升。本模块定义自动建号
和人工建号共同使用的小型状态机及来源分组规则。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .types import CandidateRecord, Observation, VerificationState


ALLOWED_TRANSITIONS: dict[VerificationState, frozenset[VerificationState]] = {
    VerificationState.UNKNOWN: frozenset({VerificationState.ISOLATED_CANDIDATE}),
    VerificationState.ISOLATED_CANDIDATE: frozenset(
        {
            VerificationState.PROVISIONAL_IDENTITY,
            VerificationState.SUSPENDED,
            VerificationState.REVOKED,
        }
    ),
    VerificationState.PROVISIONAL_IDENTITY: frozenset(
        {
            VerificationState.CONFIRMED_IDENTITY,
            VerificationState.SUSPENDED,
            VerificationState.REVOKED,
        }
    ),
    VerificationState.CONFIRMED_IDENTITY: frozenset(
        {
            VerificationState.SUSPENDED,
            VerificationState.MERGED,
            VerificationState.REVOKED,
        }
    ),
    VerificationState.SUSPENDED: frozenset(
        {VerificationState.PROVISIONAL_IDENTITY, VerificationState.REVOKED}
    ),
    VerificationState.MERGED: frozenset(),
    VerificationState.REVOKED: frozenset(),
}


@dataclass(frozen=True)
class IndependenceResult:
    """说明观测中发现的独立证据分组。"""
    independent: bool
    group_key: str
    independent_event_count: int
    reason: str


def _group_key(observation: Observation) -> str:
    """选择当前可用的最强来源分组键。"""
    if observation.challenge_id:
        return f"challenge:{observation.challenge_id}"
    if observation.capture_session_id and observation.capture_session_id != "unknown-session":
        return f"session:{observation.capture_session_id}"
    return f"event:{observation.event_id}"


def _provenance_ids(observation: Observation) -> set[str]:
    """返回用于检测重复证据的事件 ID 和来源事件 ID。"""
    return {observation.event_id, *observation.source_event_ids}


def check_independence(previous: Iterable[Observation], current: Observation) -> IndependenceResult:
    """保守判断 ``current`` 是否来自新的证据来源。

    同一采集会话中的两次摄像头观测不能计数两次。来源事件重叠也会阻止独立
    性判定，因为它可能只是同一条跟踪轨迹的摄像头交接。挑战拥有自己的分
    组，但仍受 ``challenge.py`` 中挑战校验器的约束。
    """

    # 先检查事件 ID 和显式来源 ID，再进行会话分组；这样可以捕获本来会被误
    # 判为独立的摄像头交接。
    previous = tuple(previous)
    current_ids = _provenance_ids(current)
    previous_ids = set().union(*(_provenance_ids(item) for item in previous)) if previous else set()
    if current_ids & previous_ids:
        group_key = _group_key(current)
        return IndependenceResult(False, group_key, count_independent_groups(previous), "source_event_overlap")

    groups = {_group_key(item) for item in previous}
    group_key = _group_key(current)
    if group_key in groups:
        return IndependenceResult(False, group_key, len(groups), "same_capture_group")

    count = len(groups) + 1
    if current.challenge_id:
        reason = "new_active_challenge"
    elif current.capture_session_id == "unknown-session":
        reason = "new_unattributed_event"
    else:
        reason = "new_capture_session"
    return IndependenceResult(True, group_key, count, reason)


def count_independent_groups(observations: Iterable[Observation]) -> int:
    """返回观测集合中不同来源分组的数量。"""
    return len({_group_key(item) for item in observations})


def transition(candidate: CandidateRecord, target: VerificationState) -> None:
    """应用一次合法状态转换，并拒绝不可能的跳转。"""
    if candidate.state == target:
        return
    allowed = ALLOWED_TRANSITIONS.get(candidate.state, frozenset())
    if target not in allowed:
        raise ValueError(f"invalid candidate transition: {candidate.state.value} -> {target.value}")
    candidate.state = target


def mark_evidence(
    candidate: CandidateRecord,
    *,
    observation: Observation,
    evidence_id: str,
    identity_id: str | None,
    score: float | None,
    independent: IndependenceResult,
    high_quality: bool,
    conflict: bool,
    auto_provisional: bool,
) -> None:
    """应用一条证据，同时保持状态机不变量。"""

    candidate.updated_at = observation.timestamp
    if observation.event_id not in candidate.event_ids:
        candidate.event_ids.append(observation.event_id)
    if evidence_id not in candidate.evidence_ids:
        candidate.evidence_ids.append(evidence_id)
    candidate.independent_event_count = max(
        candidate.independent_event_count,
        independent.independent_event_count,
    )
    if independent.independent:
        candidate.metadata.setdefault("independence_reasons", []).append(independent.reason)
    if high_quality:
        candidate.high_quality_evidence_count += 1
    if conflict:
        candidate.conflict_count += 1
        # 单帧分支冲突先记录为隔离证据，不立即把候选人推进到终止状态。
        # 自动控制器会丢弃当前未完成窗口并等待新的无冲突轨迹；只有显式
        # 复核/策略层才应把候选人置为 SUSPENDED。
        candidate.metadata["needs_manual_review"] = True
        return

    if identity_id is not None:
        if candidate.proposed_identity is None:
            candidate.proposed_identity = identity_id
        elif candidate.proposed_identity != identity_id:
            candidate.conflict_count += 1
            candidate.metadata["needs_manual_review"] = True
            return

    if (
        auto_provisional
        and candidate.state == VerificationState.ISOLATED_CANDIDATE
        and candidate.independent_event_count >= 2
        and candidate.high_quality_evidence_count >= 1
        and score is not None
    ):
        transition(candidate, VerificationState.PROVISIONAL_IDENTITY)
