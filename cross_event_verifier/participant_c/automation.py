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

from ..participant_a.engine import CrossEventVerifier
from ..types import Decision, DecisionKind, FeatureBundle, Observation, VerificationState


class AutomationStage(str, Enum):
    """一条轨迹自动建号状态的用户可见阶段。"""
    DISABLED = "disabled"
    WAITING_STRONG_GAIT = "waiting_strong_gait"
    COLLECTING_GAIT = "collecting_gait"
    GAIT_UNSTABLE = "gait_unstable"
    APPEARANCE_PENDING = "appearance_pending"
    APPEARANCE_ABSORBED = "appearance_absorbed"
    IDENTIFIED = "identified"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class AutomationPolicy:
    """把帧级步态转换为建号证据的阈值。

    控制器既需要轨迹达到最小预热长度，也需要稳定的嵌入窗口。单个看起来很
    强的帧不足以创建 ID。
    """

    enabled: bool = True
    minimum_track_frames: int = 16
    minimum_stable_gait_samples: int = 8
    gait_sample_window: int = 16
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


@dataclass
class _TrackAutomationState:
    """每条轨迹可变的建号窗口和待处理令牌绑定。"""
    gait_samples: deque[np.ndarray] = field(default_factory=deque)
    pending_request_id: str | None = None
    identity_id: str | None = None


@dataclass(frozen=True)
class _GaitProposal:
    """已准备好交给验证器建号门控的稳定步态中心。"""
    vector: np.ndarray
    stability: float


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

    def reset_tracks(self) -> None:
        """清除输入源本地的 Track 状态，同时保留持久请求。"""

        self._states.clear()

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
        gait_quality = quality.gait_availability(
            self.verifier.config.minimum_frames,
            self.verifier.config.minimum_gait_cycles,
        )
        required_frames = max(
            self.policy.minimum_track_frames,
            self.verifier.config.minimum_frames,
        )
        if not features.has_gait:
            return None, AutomationStatus(
                AutomationStage.WAITING_STRONG_GAIT,
                "等待步态序列",
            )
        if quality.frame_count < required_frames:
            return None, AutomationStatus(
                AutomationStage.WAITING_STRONG_GAIT,
                f"步态预热 {quality.frame_count}/{required_frames} 帧",
                progress=quality.frame_count / required_frames,
            )
        if gait_quality < self.verifier.config.strong_gait_quality:
            return None, AutomationStatus(
                AutomationStage.WAITING_STRONG_GAIT,
                (
                    f"等待强步态 Pg={gait_quality:.2f}/"
                    f"{self.verifier.config.strong_gait_quality:.2f}"
                ),
                progress=(
                    gait_quality / self.verifier.config.strong_gait_quality
                    if self.verifier.config.strong_gait_quality > 0
                    else 0.0
                ),
            )

        vector = np.asarray(features.gait, dtype=np.float32).reshape(-1)
        reference = self._unit_mean(state.gait_samples)
        if reference is not None:
            if reference.shape != vector.shape:
                state.gait_samples.clear()
            else:
                similarity = float(np.dot(reference, vector))
                if similarity < self.policy.minimum_sample_similarity:
                    state.gait_samples.clear()
                    state.gait_samples.append(vector.copy())
                    return None, AutomationStatus(
                        AutomationStage.GAIT_UNSTABLE,
                        f"步态波动，重新采集（相似度 {similarity:.2f}）",
                        progress=1.0 / self.policy.minimum_stable_gait_samples,
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
            )

        centroid = self._unit_mean(state.gait_samples)
        if centroid is None:
            state.gait_samples.clear()
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
            )
        return (
            _GaitProposal(centroid, stability),
            AutomationStatus(
                AutomationStage.COLLECTING_GAIT,
                f"强步态已确认（稳定度 {stability:.2f}）",
                progress=1.0,
            ),
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

        if decision.kind == DecisionKind.FORMAL_MATCH and decision.identity_id:
            state.identity_id = decision.identity_id
            state.gait_samples.clear()
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
        if decision.state == VerificationState.SUSPENDED or any(
            "conflict" in reason for reason in decision.reasons
        ):
            state.gait_samples.clear()
            return decision, AutomationStatus(
                AutomationStage.BLOCKED,
                "证据冲突，已阻止自动注册",
            )
        if decision.identity_id is not None:
            # 延迟匹配可能对应一个步态尚未足够强的已有人员。等待比创建重复 ID 更安全。
            return decision, AutomationStatus(
                AutomationStage.WAITING_STRONG_GAIT,
                f"疑似 {decision.identity_id}，等待强步态确认",
                identity_id=decision.identity_id,
            )

        proposal, status = self._collect_gait(state, effective_observation)
        if proposal is None:
            return decision, status

        enrollment_observation = replace(
            effective_observation,
            features=FeatureBundle(gait=proposal.vector),
            appearance_request_id=None,
            metadata={
                **dict(effective_observation.metadata),
                "automation": "stable_gait_enrollment",
                "gait_sample_count": len(state.gait_samples),
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
            return decision, AutomationStatus(
                AutomationStage.BLOCKED,
                f"自动注册暂缓：{error}",
            )

        state.gait_samples.clear()
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
