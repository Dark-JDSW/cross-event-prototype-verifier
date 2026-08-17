"""自动步态建号和外观请求编排。

控制器是帧跟踪器与验证器之间的策略接口：它为未知轨迹累积稳定步态序列，
请求验证器创建仅含步态的身份，并自动把一次性外观令牌返回给同一条轨迹。
GUI 只渲染结果状态，从不直接持有身份状态转换。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Sequence

import numpy as np

from .engine import CrossEventVerifier
from .types import (
    Decision,
    DecisionKind,
    FeatureBundle,
    GaitQualityBand,
    Observation,
    VerificationState,
)


class AutomationStage(str, Enum):
    """一条轨迹自动建号状态的用户可见阶段。"""
    DISABLED = "disabled"
    WAIT_MORE_DATA = "wait_more_data"
    WAITING_STRONG_GAIT = "waiting_strong_gait"
    COLLECTING_GAIT = "collecting_gait"
    WAITING_INDEPENDENT_EVENT = "waiting_independent_event"
    AMBIGUOUS = "ambiguous"
    GAIT_UNSTABLE = "gait_unstable"
    APPEARANCE_PENDING = "appearance_pending"
    APPEARANCE_ABSORBED = "appearance_absorbed"
    IDENTIFIED = "identified"
    VISUAL_CONFIRMED = "visual_confirmed"
    GAIT_LEARNING = "gait_learning"
    GAIT_PROVISIONAL = "gait_provisional"
    GAIT_READY = "gait_ready"
    GAIT_CONFLICT = "gait_conflict"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class AutomationPolicy:
    """把帧级步态转换为建号证据的阈值。

    控制器既需要轨迹达到最小预热长度，也需要稳定的嵌入窗口。单个看起来很
    强的帧不足以创建 ID。生产的 OSNet-first 控制器复用其中的窗口与稳定性
    参数，但不会用它们创建视觉身份；视觉身份编号由 OSNet 另行确认。
    """

    enabled: bool = True
    minimum_track_frames: int = 16
    minimum_stable_gait_samples: int = 8
    gait_sample_window: int = 16
    minimum_independent_gait_events: int = 2
    minimum_sample_similarity: float = 0.86
    minimum_gait_stability: float = 0.94

    def __post_init__(self) -> None:
        """校验采集窗口是否能够满足样本策略。"""
        if self.minimum_track_frames < 1:
            raise ValueError("minimum_track_frames must be positive")
        if self.minimum_stable_gait_samples < 2:
            raise ValueError("minimum_stable_gait_samples must be at least two")
        if self.gait_sample_window < self.minimum_stable_gait_samples:
            raise ValueError(
                "gait_sample_window cannot be smaller than minimum_stable_gait_samples"
            )
        if self.minimum_independent_gait_events < 1:
            raise ValueError("minimum_independent_gait_events must be positive")
        for name, value in {
            "minimum_sample_similarity": self.minimum_sample_similarity,
            "minimum_gait_stability": self.minimum_gait_stability,
        }.items():
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1]")


@dataclass(frozen=True)
class AutomationStatus:
    """供 GUI 渲染的可序列化状态快照。"""
    stage: AutomationStage
    message: str
    progress: float = 0.0
    identity_id: str | None = None
    request_id: str | None = None
    auto_registered: bool = False
    gait_quality_band: str | None = None
    readiness_state: str | None = None
    gait_event_count: int = 0
    gait_prototype_count: int = 0
    visual_identity_confirmed: bool = False


@dataclass
class _TrackAutomationState:
    """每条轨迹可变的建号窗口和待处理令牌绑定。"""
    gait_samples: deque[np.ndarray] = field(default_factory=deque)
    pending_request_id: str | None = None
    identity_id: str | None = None
    gait_event_proposals: dict[str, "_GaitProposal"] = field(default_factory=dict)
    gait_window_start_timestamp: float | None = None


@dataclass(frozen=True)
class _GaitProposal:
    """已准备好交给验证器建号门控的稳定步态中心。"""
    vector: np.ndarray
    stability: float
    model_version: str = "unconfigured"
    feature_schema: str = "unconfigured-v1"
    calibration_version: str = "heuristic-default-v1"
    artifact_sha256: str = "unverified"
    preprocess_version: str = "unversioned-v1"
    joint_format: str = "unknown"
    sequence_length: int | None = None
    tta_mode: str = "unknown"
    coordinate_contract: str = "unknown"
    embedding_dimensions: dict[str, int] = field(default_factory=dict)


class AutomaticVerificationController:
    """实现完整自动证据循环的深模块。

    它只负责输入源本地的证据累积和令牌绑定。身份状态转换仍属于
    ``CrossEventVerifier``；这种分离让 GUI、摄像头工作线程和测试共享同一套
    建号协议。
    """

    def __init__(
        self,
        verifier: CrossEventVerifier,
        policy: AutomationPolicy | None = None,
    ) -> None:
        """创建绑定一个验证器和采集策略的控制器。"""
        self.verifier = verifier
        self.policy = policy or AutomationPolicy()
        self._registration_enabled = self.policy.enabled
        self._states: dict[str, _TrackAutomationState] = {}
        self._manual_request_id: str | None = None
        self._hydrate_event_proposals()

    def _event_contract_matches(self, record: dict[str, object]) -> bool:
        """判断持久化事件是否仍属于当前视觉/校准协议。"""

        config = self.verifier.config
        # Standalone/demo controllers intentionally leave the deployment
        # contract unconfigured.  Their vector dimensions are still checked
        # by SQLite; there is no concrete protocol to compare against here.
        if config.model_version == "unconfigured":
            return True
        scalar_fields = (
            "model_version",
            "feature_schema",
            "artifact_sha256",
            "preprocess_version",
            "joint_format",
            "sequence_length",
            "tta_mode",
            "coordinate_contract",
            "calibration_version",
        )
        for field_name in scalar_fields:
            expected = getattr(config, field_name)
            actual = record.get(field_name)
            if field_name == "sequence_length":
                actual = int(actual) if actual is not None else None
            elif actual is not None:
                actual = str(actual)
            if actual != expected:
                return False
        expected_dimensions = dict(config.embedding_dimensions)
        if expected_dimensions and dict(record.get("embedding_dimensions", {})) != expected_dimensions:
            return False
        expected_gait_dimension = expected_dimensions.get("gait")
        vector = np.asarray(record.get("vector"), dtype=np.float32).reshape(-1)
        return expected_gait_dimension is None or vector.size == expected_gait_dimension

    def _hydrate_event_proposals(self) -> None:
        """从 SQLite 恢复跨进程/跨会话的稳定步态事件。"""

        records_by_candidate: dict[str, list[dict[str, object]]] = {}
        for record in self.verifier.load_gait_event_proposals():
            records_by_candidate.setdefault(str(record["candidate_id"]), []).append(record)

        for candidate_id, records in records_by_candidate.items():
            candidate = self.verifier.get_candidate(candidate_id)
            if candidate is not None and candidate.state in {
                VerificationState.CONFIRMED_IDENTITY,
                VerificationState.MERGED,
                VerificationState.REVOKED,
            }:
                # A successful promotion/enrollment consumes its event
                # proposals.  Clean up rows left by an older process version
                # or by an interrupted post-commit cleanup before hydrating
                # them into a live controller.
                self.verifier.delete_gait_event_proposals(candidate_id)
                continue
            if not all(self._event_contract_matches(record) for record in records):
                # Never combine an old-model event with a new-model event.  The
                # candidate must restart its evidence collection under one
                # coherent protocol, and the stale rows should not reappear on
                # the next process restart.
                self.verifier.delete_gait_event_proposals(candidate_id)
                self.verifier.store.audit(
                    "gait_event_contract_rejected",
                    candidate_id,
                    {
                        "event_count": len(records),
                        "model_version": self.verifier.config.model_version,
                        "feature_schema": self.verifier.config.feature_schema,
                        "calibration_version": self.verifier.config.calibration_version,
                    },
                )
                continue
            state = self._states.setdefault(candidate_id, _TrackAutomationState())
            for record in records:
                state.gait_event_proposals[str(record["event_key"])] = _GaitProposal(
                    np.asarray(record["vector"], dtype=np.float32),
                    float(record["stability"]),
                    str(record["model_version"]),
                    str(record["feature_schema"]),
                    str(record.get("calibration_version", "heuristic-default-v1")),
                    str(record.get("artifact_sha256", "unverified")),
                    str(record.get("preprocess_version", "unversioned-v1")),
                    str(record.get("joint_format", "unknown")),
                    (
                        int(record["sequence_length"])
                        if record.get("sequence_length") is not None
                        else None
                    ),
                    str(record.get("tta_mode", "unknown")),
                    str(record.get("coordinate_contract", "unknown")),
                    dict(record.get("embedding_dimensions", {})),
                )

    @property
    def registration_enabled(self) -> bool:
        """返回未知轨迹是否可以创建新的正式步态 ID。"""
        return self._registration_enabled

    @property
    def manual_request_id(self) -> str | None:
        """返回可选的操作员提供的外观令牌。"""
        return self._manual_request_id

    def set_registration_enabled(self, enabled: bool) -> None:
        """只启用/禁用新的自动 ID；令牌响应仍保持安全。"""

        self._registration_enabled = bool(enabled)

    def update_policy(self, policy: AutomationPolicy) -> None:
        """替换采集阈值，并只丢弃尚未完成的步态窗口。"""

        if policy == self.policy:
            return
        self.policy = policy
        self.reset_gait_samples()

    def reset_gait_samples(self) -> None:
        """丢弃未完成的建号证据，同时保留仍有效的令牌。"""

        for state in self._states.values():
            state.gait_samples.clear()
            state.gait_window_start_timestamp = None
            state.gait_event_proposals.clear()
        for candidate_id in tuple(self._states):
            self.verifier.delete_gait_event_proposals(candidate_id)

    def reset_tracks(
        self,
        *,
        preserve_candidate_ids: Sequence[str] = (),
        discard_unpreserved: bool = False,
    ) -> None:
        """清除短期 Track 窗口，不默认销毁持久化事件证据。

        输入源和 ByteTrack 的生命周期是短的，而 ``gait_event_proposals``
        属于采集任务生命周期。过去在换源时删除未显式保留的候选，导致 GUI
        兼容的旧版步态优先路径仍会显示独立事件进度；生产 OSNet-first 路径
        使用身份级 `gait_enrollment_events`，同一会话不会重复计数。只有显式
        取消/作废流程才应设置 ``discard_unpreserved=True``。
        """

        keep = {value for value in preserve_candidate_ids if value}
        for candidate_id in tuple(self._states):
            state = self._states[candidate_id]
            state.gait_samples.clear()
            state.gait_window_start_timestamp = None
            if discard_unpreserved and candidate_id not in keep:
                del self._states[candidate_id]
                self.verifier.delete_gait_event_proposals(candidate_id)

    def discard_candidate(self, candidate_id: str) -> None:
        """显式作废一个采集任务及其跨会话步态事件。"""

        self._states.pop(candidate_id, None)
        self.verifier.delete_gait_event_proposals(candidate_id)

    def discard_candidates(self, candidate_ids: Sequence[str]) -> None:
        """同步清除验证器维护任务撤销的候选人状态。"""

        for candidate_id in candidate_ids:
            self.discard_candidate(candidate_id)

    @staticmethod
    def _gait_event_key(observation: Observation) -> str:
        """选择独立步态事件的来源键。"""

        if observation.challenge_id:
            return f"challenge:{observation.challenge_id}"
        if observation.capture_session_id and observation.capture_session_id != "unknown-session":
            return f"session:{observation.capture_session_id}"
        return f"event:{observation.event_id}"

    def bind_request(
        self,
        candidate_id: str,
        request_id: str,
        *,
        identity_id: str | None = None,
    ) -> None:
        """把步态签发的外观请求绑定到恰好一条轨迹键。"""
        state = self._states.setdefault(candidate_id, _TrackAutomationState())
        state.pending_request_id = request_id
        state.identity_id = identity_id

    def set_manual_request(
        self,
        request_id: str | None,
        *,
        candidate_id: str | None = None,
    ) -> None:
        """将操作员提供的令牌绑定到一条轨迹或下一条轨迹。"""

        value = request_id.strip() if request_id else None
        if candidate_id is None:
            self._manual_request_id = value
            return
        state = self._states.setdefault(candidate_id, _TrackAutomationState())
        state.pending_request_id = value

    @staticmethod
    def _unit_mean(samples: deque[np.ndarray]) -> np.ndarray | None:
        """返回滚动步态样本窗口的归一化中心。"""
        if not samples:
            return None
        vector = np.mean(np.stack(tuple(samples), axis=0), axis=0)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            return None
        return (vector / norm).astype(np.float32)

    def _live_token(
        self,
        state: _TrackAutomationState,
        explicit_request_id: str | None,
    ) -> str | None:
        """解析显式/轨迹/全局令牌，并丢弃已过期句柄。"""
        request_id = (
            explicit_request_id
            or state.pending_request_id
            or self._manual_request_id
        )
        if not request_id:
            return None
        request = self.verifier.get_appearance_request(request_id)
        if request is not None and request.status == "pending":
            return request_id
        if state.pending_request_id == request_id:
            state.pending_request_id = None
        if self._manual_request_id == request_id:
            self._manual_request_id = None
        return None

    def _collect_gait(
        self,
        state: _TrackAutomationState,
        observation: Observation,
    ) -> tuple[_GaitProposal | None, AutomationStatus]:
        """为一条轨迹累积步态向量并检查稳定性。

        向量会与滚动单位中心比较。突然不匹配时只重置该轨迹的未完成窗口，
        防止把两个人或两个轨迹身份的样本平均到一起。
        """
        features = observation.features.normalized()
        quality = observation.quality
        hard_reasons = quality.gait_hard_veto_reasons(
            self.verifier.config.minimum_frames
        )
        if any(
            reason in {
                "invalid_box",
                "track_id_switch",
                "track_gap",
                "legs_invisible",
                "box_truncated",
            }
            for reason in hard_reasons
        ):
            # A broken track cannot contribute to the same stable window as
            # the frames before the break.  Partial quality, by contrast,
            # remains recoverable and is handled below without becoming a
            # negative identity signal.
            state.gait_samples.clear()
            state.gait_window_start_timestamp = None
        gait_quality = quality.gait_availability(
            self.verifier.config.minimum_frames,
            self.verifier.config.minimum_gait_cycles,
        )
        quality_band = quality.gait_quality_band(
            minimum_frames=self.verifier.config.minimum_frames,
            minimum_gait_cycles=self.verifier.config.minimum_gait_cycles,
            partial_threshold=self.verifier.config.partial_gait_quality,
            strong_threshold=self.verifier.config.strong_gait_quality,
        )
        required_frames = max(
            self.policy.minimum_track_frames,
            self.verifier.config.minimum_frames,
        )
        if not features.has_gait:
            # A frame without a gait feature breaks the stable sample window;
            # do not join the next valid vector to samples from before the
            # missing-pose interval.
            state.gait_samples.clear()
            state.gait_window_start_timestamp = None
            return None, AutomationStatus(
                AutomationStage.WAITING_STRONG_GAIT,
                "等待步态序列",
                gait_quality_band=GaitQualityBand.INVALID.value,
            )
        observed_frames = (
            quality.valid_pose_frames
            if quality.valid_pose_frames is not None
            else quality.frame_count
        )
        if observed_frames < required_frames:
            return None, AutomationStatus(
                AutomationStage.WAITING_STRONG_GAIT,
                f"步态预热 {observed_frames}/{required_frames} 帧",
                progress=observed_frames / required_frames,
                gait_quality_band=quality_band.value,
            )
        if quality_band == GaitQualityBand.INVALID:
            return None, AutomationStatus(
                AutomationStage.WAIT_MORE_DATA,
                f"步态质量 INVALID（Q={gait_quality:.2f}），等待更多完整序列",
                progress=float(np.clip(gait_quality / max(self.verifier.config.strong_gait_quality, 1e-8), 0.0, 1.0)),
                gait_quality_band=quality_band.value,
            )
        if quality_band == GaitQualityBand.PARTIAL:
            return None, AutomationStatus(
                AutomationStage.WAIT_MORE_DATA,
                f"步态质量 PARTIAL（Q={gait_quality:.2f}），仅保留候选证据",
                progress=(
                    gait_quality / self.verifier.config.strong_gait_quality
                    if self.verifier.config.strong_gait_quality > 0
                    else 0.0
                ),
                gait_quality_band=quality_band.value,
            )

        if not state.gait_samples:
            state.gait_window_start_timestamp = observation.timestamp
        event_key = self._gait_event_key(observation)
        if event_key in state.gait_event_proposals:
            return None, AutomationStatus(
                AutomationStage.WAITING_INDEPENDENT_EVENT,
                (
                    f"已记录独立步态事件 {len(state.gait_event_proposals)}/"
                    f"{self.policy.minimum_independent_gait_events}，等待新采集事件"
                ),
                progress=min(
                    1.0,
                    len(state.gait_event_proposals)
                    / self.policy.minimum_independent_gait_events,
                ),
                gait_quality_band=quality_band.value,
            )

        vector = np.asarray(features.gait, dtype=np.float32).reshape(-1)
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
                    return None, AutomationStatus(
                        AutomationStage.GAIT_UNSTABLE,
                        f"步态波动，重新采集（相似度 {similarity:.2f}）",
                        progress=1.0 / self.policy.minimum_stable_gait_samples,
                        gait_quality_band=quality_band.value,
                    )

        state.gait_samples.append(vector.copy())
        while len(state.gait_samples) > self.policy.gait_sample_window:
            state.gait_samples.popleft()
        sample_count = len(state.gait_samples)
        progress = min(1.0, sample_count / self.policy.minimum_stable_gait_samples)
        if sample_count < self.policy.minimum_stable_gait_samples:
            return None, AutomationStatus(
                AutomationStage.COLLECTING_GAIT,
                (
                    f"采集稳定步态 {sample_count}/"
                    f"{self.policy.minimum_stable_gait_samples}"
                ),
                progress=progress,
                gait_quality_band=quality_band.value,
            )

        centroid = self._unit_mean(state.gait_samples)
        if centroid is None:
            state.gait_samples.clear()
            state.gait_window_start_timestamp = None
            return None, AutomationStatus(
                AutomationStage.GAIT_UNSTABLE,
                "步态聚合失败，重新采集",
            )
        similarities = [float(np.dot(item, centroid)) for item in state.gait_samples]
        stability = float(np.mean(similarities))
        required_stability = max(
            self.policy.minimum_gait_stability,
            self.verifier.config.strong_gait_probability,
        )
        if stability < required_stability:
            return None, AutomationStatus(
                AutomationStage.GAIT_UNSTABLE,
                f"步态稳定度 {stability:.2f}/{required_stability:.2f}",
                progress=float(np.clip(stability / required_stability, 0.0, 1.0)),
                gait_quality_band=quality_band.value,
            )
        return (
            _GaitProposal(centroid, stability),
            AutomationStatus(
                AutomationStage.COLLECTING_GAIT,
                f"强步态已确认（稳定度 {stability:.2f}）",
                progress=1.0,
                gait_quality_band=quality_band.value,
            ),
        )

    def _record_gait_event_proposal(
        self,
        state: _TrackAutomationState,
        observation: Observation,
        proposal: _GaitProposal,
        *,
        candidate_id: str,
    ) -> _GaitProposal | None:
        """记录一个独立事件，并在事件数足够时聚合跨事件步态。"""

        event_key = self._gait_event_key(observation)
        proposal = replace(
            proposal,
            model_version=observation.model_version,
            feature_schema=observation.feature_schema,
            calibration_version=observation.calibration_version,
            artifact_sha256=observation.artifact_sha256,
            preprocess_version=observation.preprocess_version,
            joint_format=observation.joint_format,
            sequence_length=observation.sequence_length,
            tta_mode=observation.tta_mode,
            coordinate_contract=observation.coordinate_contract,
            embedding_dimensions=dict(observation.embedding_dimensions),
        )
        incompatible = any(
            item.model_version != proposal.model_version
            or item.feature_schema != proposal.feature_schema
            or item.calibration_version != proposal.calibration_version
            or item.artifact_sha256 != proposal.artifact_sha256
            or item.preprocess_version != proposal.preprocess_version
            or item.joint_format != proposal.joint_format
            or item.sequence_length != proposal.sequence_length
            or item.tta_mode != proposal.tta_mode
            or item.coordinate_contract != proposal.coordinate_contract
            or item.embedding_dimensions != proposal.embedding_dimensions
            or item.vector.shape != proposal.vector.shape
            for item in state.gait_event_proposals.values()
        )
        if incompatible:
            # Event vectors from a different pose/model contract cannot be
            # averaged.  Discard the stale persisted window and start again.
            state.gait_event_proposals.clear()
            self.verifier.delete_gait_event_proposals(candidate_id)
        if event_key not in state.gait_event_proposals:
            state.gait_event_proposals[event_key] = proposal
            try:
                self.verifier.save_gait_event_proposal(
                    candidate_id=candidate_id,
                    event_key=event_key,
                    vector=proposal.vector,
                    stability=proposal.stability,
                    observation=observation,
                    start_timestamp=(
                        state.gait_window_start_timestamp
                        if state.gait_window_start_timestamp is not None
                        else observation.timestamp
                    ),
                )
            except Exception:
                # Keep the in-memory proposal set aligned with SQLite when a
                # single event write fails; callers can retry the same event.
                state.gait_event_proposals.pop(event_key, None)
                raise
        if len(state.gait_event_proposals) < self.policy.minimum_independent_gait_events:
            return None
        vectors = np.stack(
            [item.vector for item in state.gait_event_proposals.values()],
            axis=0,
        )
        centroid = np.mean(vectors, axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm <= 1e-8:
            return None
        return _GaitProposal(
            (centroid / norm).astype(np.float32),
            min(item.stability for item in state.gait_event_proposals.values()),
            proposal.model_version,
            proposal.feature_schema,
            proposal.calibration_version,
            proposal.artifact_sha256,
            proposal.preprocess_version,
            proposal.joint_format,
            proposal.sequence_length,
            proposal.tta_mode,
            proposal.coordinate_contract,
            dict(proposal.embedding_dimensions),
        )

    def verify(
        self,
        observation: Observation,
        *,
        candidate_id: str,
    ) -> tuple[Decision, AutomationStatus]:
        """验证一条轨迹，并推进其自动建号协议。"""

        state = self._states.setdefault(candidate_id, _TrackAutomationState())
        request_id = self._live_token(state, observation.appearance_request_id)
        effective_observation = (
            replace(observation, appearance_request_id=request_id)
            if request_id != observation.appearance_request_id
            else observation
        )
        decision = self.verifier.verify(
            effective_observation,
            candidate_id=candidate_id,
            # 在线自动路径必须额外防范“单一身份图库吸附陌生人”。人工/直接
            # API 调用仍可使用兼容的 gait 锚点语义。
            open_set_guard=True,
        )
        return self._advance(
            effective_observation,
            candidate_id=candidate_id,
            state=state,
            request_id=request_id,
            decision=decision,
        )

    def verify_batch(
        self,
        observations: Sequence[Observation],
        *,
        candidate_ids: Sequence[str],
    ) -> tuple[tuple[Decision, AutomationStatus], ...]:
        """使用全局一对一身份指派验证一帧图像。"""

        if len(observations) != len(candidate_ids):
            raise ValueError("candidate_ids must align with observations")
        prepared: list[
            tuple[Observation, str, _TrackAutomationState, str | None]
        ] = []
        for observation, candidate_id in zip(observations, candidate_ids):
            state = self._states.setdefault(candidate_id, _TrackAutomationState())
            request_id = self._live_token(state, observation.appearance_request_id)
            effective = (
                replace(observation, appearance_request_id=request_id)
                if request_id != observation.appearance_request_id
                else observation
            )
            prepared.append((effective, candidate_id, state, request_id))
        decisions = self.verifier.verify_batch(
            [item[0] for item in prepared],
            candidate_ids=[item[1] for item in prepared],
            open_set_guard=True,
        )
        return tuple(
            self._advance(
                effective,
                candidate_id=candidate_id,
                state=state,
                request_id=request_id,
                decision=decision,
            )
            for (effective, candidate_id, state, request_id), decision in zip(
                prepared,
                decisions,
            )
        )

    def _advance(
        self,
        effective_observation: Observation,
        *,
        candidate_id: str,
        state: _TrackAutomationState,
        request_id: str | None,
        decision: Decision,
    ) -> tuple[Decision, AutomationStatus]:
        """验证器做出安全决策后推进自动化流程。

        正式匹配和外观响应优先处理。只有真正未知且无冲突的决策才会进入步态
        采集；这种顺序可以防止仅仅像旧 ID 的人物过早被登记成重复身份。
        """

        if decision.kind == DecisionKind.APPEARANCE_RESPONSE_ACCEPTED:
            consumed = decision.appearance_request_id or request_id
            if state.pending_request_id == consumed:
                state.pending_request_id = None
            if self._manual_request_id == consumed:
                self._manual_request_id = None
            state.identity_id = decision.identity_id
            state.gait_samples.clear()
            state.gait_window_start_timestamp = None
            return decision, AutomationStatus(
                AutomationStage.APPEARANCE_ABSORBED,
                f"{decision.identity_id} 外观已自动吸收并直接放行",
                progress=1.0,
                identity_id=decision.identity_id,
                request_id=consumed,
            )

        if decision.appearance_request_id and decision.identity_id:
            state.pending_request_id = decision.appearance_request_id
            state.identity_id = decision.identity_id
            state.gait_samples.clear()
            state.gait_window_start_timestamp = None
            return decision, AutomationStatus(
                AutomationStage.APPEARANCE_PENDING,
                f"{decision.identity_id} 步态已确认，等待强外观自动响应",
                progress=1.0,
                identity_id=decision.identity_id,
                request_id=decision.appearance_request_id,
            )

        if request_id:
            # 令牌仍然有效，但本帧没有通过强外观质量/相似度门控。继续自动重试。
            return decision, AutomationStatus(
                AutomationStage.APPEARANCE_PENDING,
                "外观请求已绑定，等待高质量正面画面",
                progress=0.8,
                identity_id=state.identity_id,
                request_id=request_id,
            )

        if any(
            reason in {"calibration_contract_mismatch", "feature_contract_mismatch"}
            for reason in decision.reasons
        ):
            return decision, AutomationStatus(
                AutomationStage.BLOCKED,
                "模型/校准协议与当前图库不一致，已暂停自动注册",
            )

        if decision.kind == DecisionKind.FORMAL_MATCH and decision.identity_id:
            state.identity_id = decision.identity_id
            state.gait_samples.clear()
            state.gait_window_start_timestamp = None
            return decision, AutomationStatus(
                AutomationStage.IDENTIFIED,
                f"已由强步态确认 {decision.identity_id}",
                progress=1.0,
                identity_id=decision.identity_id,
            )

        if not self._registration_enabled:
            return decision, AutomationStatus(
                AutomationStage.DISABLED,
                "自动注册已关闭",
            )
        if decision.state == VerificationState.SUSPENDED:
            state.gait_samples.clear()
            state.gait_window_start_timestamp = None
            return decision, AutomationStatus(
                AutomationStage.BLOCKED,
                "候选人已挂起，自动注册已暂停",
            )
        if any("conflict" in reason for reason in decision.reasons):
            # 冲突帧只代表当前证据不可判定，不应把整条轨迹永久终止。清掉
            # 可能混入不同分支/身份的窗口，从下一帧重新建立干净序列。
            state.gait_samples.clear()
            state.gait_window_start_timestamp = None
            return decision, AutomationStatus(
                AutomationStage.GAIT_UNSTABLE,
                "当前证据冲突，已丢弃本窗口并重新采集",
                progress=0.0,
            )
        if decision.identity_id is not None:
            # 延迟匹配可能对应一个步态尚未足够强的已有人员。等待比创建重复 ID 更安全。
            return decision, AutomationStatus(
                AutomationStage.WAITING_STRONG_GAIT,
                f"疑似 {decision.identity_id}，等待强步态确认",
                identity_id=decision.identity_id,
            )

        ambiguous = decision.kind == DecisionKind.AMBIGUOUS
        proposal, status = self._collect_gait(state, effective_observation)
        if proposal is None:
            if ambiguous and status.stage not in {
                AutomationStage.WAIT_MORE_DATA,
                AutomationStage.WAITING_INDEPENDENT_EVENT,
            }:
                return decision, replace(
                    status,
                    stage=AutomationStage.AMBIGUOUS,
                    message=f"步态 Top-2 歧义：{status.message}",
                )
            return decision, status

        event_proposal = self._record_gait_event_proposal(
            state,
            effective_observation,
            proposal,
            candidate_id=candidate_id,
        )
        stable_sample_count = len(state.gait_samples)
        state.gait_samples.clear()
        state.gait_window_start_timestamp = None
        if event_proposal is None:
            return decision, AutomationStatus(
                AutomationStage.WAITING_INDEPENDENT_EVENT,
                (
                    f"已记录独立步态事件 {len(state.gait_event_proposals)}/"
                    f"{self.policy.minimum_independent_gait_events}，等待下一独立事件"
                ),
                progress=min(
                    1.0,
                    len(state.gait_event_proposals)
                    / self.policy.minimum_independent_gait_events,
                ),
                gait_quality_band=status.gait_quality_band,
            )
        proposal = event_proposal

        enrollment_observation = replace(
            effective_observation,
            # The appearance vector is used only as a negative open-set signal
            # while deciding whether a second identity may be enrolled. The
            # enrollment API still writes gait only and issues a fresh,
            # one-shot appearance request, so this does not absorb appearance
            # into the new formal identity.
            features=FeatureBundle(
                gait=proposal.vector,
                appearance=effective_observation.features.appearance,
            ),
            appearance_request_id=None,
            metadata={
                **dict(effective_observation.metadata),
                "automation": "stable_gait_enrollment",
                "gait_sample_count": stable_sample_count,
                "gait_event_count": len(state.gait_event_proposals),
                "gait_stability": proposal.stability,
            },
        )
        try:
            enrolled = self.verifier.enroll_gait_identity(
                enrollment_observation,
                candidate_id=candidate_id,
                gait_confidence=proposal.stability,
            )
        except ValueError as error:
            # 开放集拒绝是可恢复的证据结果，不是候选人生命周期终止。清掉
            # 本窗口后让轨迹重新收集，避免下一帧反复拿同一批向量重试，也不
            # 把 GUI 永久留在 BLOCKED。真正不可恢复的候选状态（例如挂起）
            # 仍保留 BLOCKED 语义。
            state.gait_samples.clear()
            state.gait_window_start_timestamp = None
            if "open-set novel" in str(error):
                state.gait_event_proposals.clear()
                self.verifier.delete_gait_event_proposals(candidate_id)
                reason = str(error).split(";", 1)[-1].strip()
                stage = (
                    AutomationStage.AMBIGUOUS
                    if "gait_top2_ambiguous" in reason
                    else AutomationStage.GAIT_UNSTABLE
                )
                if "gait_near_duplicate" in reason:
                    message = "步态与已有图库过近（formal 原型近重复），已清空窗口并重新采集"
                elif "missing_appearance_negative_evidence" in reason:
                    message = "缺少可靠外观负证据，已清空窗口并重新采集"
                elif stage == AutomationStage.AMBIGUOUS:
                    message = "步态 Top-2 歧义，已清空窗口并等待新的独立事件"
                else:
                    message = "步态与已有图库过近，已清空窗口并重新采集"
                return decision, AutomationStatus(
                    stage,
                    message,
                    progress=0.0,
                )
            return decision, AutomationStatus(
                AutomationStage.BLOCKED,
                f"自动注册暂缓：{error}",
            )

        state.gait_samples.clear()
        state.gait_window_start_timestamp = None
        state.gait_event_proposals.clear()
        self.verifier.delete_gait_event_proposals(candidate_id)
        state.identity_id = enrolled.identity_id
        state.pending_request_id = enrolled.appearance_request_id
        return enrolled, AutomationStatus(
            AutomationStage.APPEARANCE_PENDING,
            f"已自动注册 {enrolled.identity_id}（仅步态），等待外观自动吸收",
            progress=1.0,
            identity_id=enrolled.identity_id,
            request_id=enrolled.appearance_request_id,
            auto_registered=True,
        )


__all__ = [
    "AutomationPolicy",
    "AutomationStage",
    "AutomationStatus",
    "AutomaticVerificationController",
]
