"""短期主动挑战的签发与校验。

挑战属于可选的交互证据。管理器把提示绑定到候选人 ID 和过期时间；它不会
单独提高较弱视觉匹配的结果，从而将用户交互与生物特征分数路径分开。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from uuid import uuid4

from .types import Observation


@dataclass
class Challenge:
    """一个绑定候选人的、可持久化的挑战描述。"""
    challenge_id: str
    candidate_id: str
    prompt: str
    expected_action: str
    issued_at: float
    expires_at: float
    camera_id: str | None = None
    minimum_frames: int = 12
    minimum_gait_cycles: float = 1.0
    consumed: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ChallengeValidation:
    """包含明确原因、适合写入审计日志的校验结果。"""
    valid: bool
    reason: str
    challenge_id: str | None = None


class ChallengeManager:
    """将挑战状态保存在识别模型和图库之外。"""

    def __init__(self, *, ttl_seconds: float = 60.0) -> None:
        """创建一个具有正令牌生命周期的内存注册表。"""
        self.ttl_seconds = max(float(ttl_seconds), 1.0)
        self._challenges: dict[str, Challenge] = {}

    def issue(
        self,
        candidate_id: str,
        reasons: tuple[str, ...] | list[str] = (),
        *,
        camera_id: str | None = None,
        now: float | None = None,
    ) -> Challenge:
        """根据质量失败原因创建具有相应预期动作的提示。"""
        now = time.time() if now is None else float(now)
        reason_set = set(reasons)
        if "too_short" in reason_set or "too_few_gait_cycles" in reason_set:
            prompt = "请沿指定路线自然行走一段，保持正常速度。"
            action = "natural_walk"
        elif "occluded" in reason_set:
            prompt = "请移动到无遮挡区域后自然行走。"
            action = "move_to_clear_area"
        elif "conflict" in reason_set or "view_insufficient" in reason_set:
            prompt = "请从另一方向自然走过镜头。"
            action = "alternate_view"
        else:
            prompt = "请自然行走，然后在随机提示后短暂停下并继续。"
            action = "walk_stop_continue"
        challenge = Challenge(
            challenge_id=f"challenge-{uuid4().hex}",
            candidate_id=candidate_id,
            prompt=prompt,
            expected_action=action,
            issued_at=now,
            expires_at=now + self.ttl_seconds,
            camera_id=camera_id,
        )
        self._challenges[challenge.challenge_id] = challenge
        return challenge

    def get(self, challenge_id: str) -> Challenge | None:
        """按 ID 返回挑战，但不消费它。"""
        return self._challenges.get(challenge_id)

    def validate(
        self,
        candidate_id: str,
        observation: Observation,
        *,
        now: float | None = None,
    ) -> ChallengeValidation:
        """检查绑定关系、重放/过期状态、动作和序列要求。"""
        challenge_id = observation.challenge_id
        if not challenge_id:
            return ChallengeValidation(False, "missing_challenge_id")
        challenge = self._challenges.get(challenge_id)
        if challenge is None:
            return ChallengeValidation(False, "unknown_challenge", challenge_id)
        now = time.time() if now is None else float(now)
        if challenge.candidate_id != candidate_id:
            return ChallengeValidation(False, "candidate_mismatch", challenge_id)
        if challenge.consumed:
            return ChallengeValidation(False, "challenge_already_consumed", challenge_id)
        if now > challenge.expires_at:
            return ChallengeValidation(False, "challenge_expired", challenge_id)
        if challenge.camera_id and observation.camera_id != challenge.camera_id:
            return ChallengeValidation(False, "camera_mismatch", challenge_id)

        response = dict(observation.challenge_response)
        if response.get("is_replay") is True:
            return ChallengeValidation(False, "replay_flagged", challenge_id)
        if response.get("action") != challenge.expected_action:
            return ChallengeValidation(False, "unexpected_action", challenge_id)
        if observation.quality.frame_count < challenge.minimum_frames:
            return ChallengeValidation(False, "challenge_sequence_too_short", challenge_id)
        if observation.quality.gait_cycles < challenge.minimum_gait_cycles:
            return ChallengeValidation(False, "challenge_gait_incomplete", challenge_id)
        if response.get("response_time_ms", 0) and float(response["response_time_ms"]) > 15_000:
            return ChallengeValidation(False, "response_timeout", challenge_id)

        challenge.consumed = True
        return ChallengeValidation(True, "challenge_valid", challenge_id)

    def active_for_candidate(self, candidate_id: str) -> tuple[Challenge, ...]:
        """返回某候选人当前所有尚未消费的挑战。"""
        return tuple(
            challenge
            for challenge in self._challenges.values()
            if challenge.candidate_id == candidate_id and not challenge.consumed
        )
