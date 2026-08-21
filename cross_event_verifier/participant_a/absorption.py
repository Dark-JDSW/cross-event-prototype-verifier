"""一次性步态授权外观吸收请求。

外观信息很有用，但容易受到衣着、光照或附近其他人的污染。因此，验证器
只有在强步态证据确认人物后，才签发一个短期令牌。本管理器负责令牌的生命
周期，并检查身份、候选人、事件和过期约束；它不负责判断步态本身是否足够强。
"""

from __future__ import annotations

import time
from uuid import uuid4

from ..types import AppearanceAbsorptionRequest


class AppearanceAbsorptionManager:
    """签发、校验、消费和恢复一次性授权。

    管理器有意只保存在内存中。参与者 C 通过 SQLite 持久化数据类字段，并在
    验证器启动时恢复它们，使该策略类不依赖具体的存储实现。
    """

    def __init__(self, ttl_seconds: float = 90.0) -> None:
        """创建一个具有有限正生命周期的令牌注册表。"""
        self.ttl_seconds = max(float(ttl_seconds), 1.0)
        self._requests: dict[str, AppearanceAbsorptionRequest] = {}

    def issue(
        self,
        *,
        identity_id: str,
        issued_by_event_id: str,
        gait_probability: float,
        gait_quality: float,
        candidate_id: str | None = None,
        now: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> AppearanceAbsorptionRequest:
        """签发一个绑定到强步态事件和身份的待处理令牌。"""
        now = time.time() if now is None else float(now)
        request = AppearanceAbsorptionRequest(
            request_id=f"appearance-request-{uuid4().hex}",
            identity_id=identity_id,
            issued_by_event_id=issued_by_event_id,
            candidate_id=candidate_id,
            gait_probability=float(gait_probability),
            gait_quality=float(gait_quality),
            issued_at=now,
            expires_at=now + self.ttl_seconds,
            metadata=dict(metadata or {}),
        )
        self._requests[request.request_id] = request
        return request

    def get(self, request_id: str | None) -> AppearanceAbsorptionRequest | None:
        """查找请求，但不改变其状态。"""
        return self._requests.get(request_id) if request_id else None

    def pending(self) -> tuple[AppearanceAbsorptionRequest, ...]:
        """返回尚未消费且尚未过期的请求。"""
        return tuple(request for request in self._requests.values() if request.status == "pending")

    def pending_for_identity(
        self,
        identity_id: str,
        *,
        now: float | None = None,
    ) -> AppearanceAbsorptionRequest | None:
        """返回某身份最新的有效请求（如果存在）。

        视频流可能连续产生许多强步态观测。复用一个待处理令牌既能保持授权
        的一次性语义，也不会用重复请求淹没下游的外观采集界面。
        """

        now = time.time() if now is None else float(now)
        candidates = [
            request
            for request in self._requests.values()
            if (
                request.identity_id == identity_id
                and request.status == "pending"
                and now <= request.expires_at
            )
        ]
        return max(candidates, key=lambda item: item.issued_at, default=None)

    def validate(
        self,
        request_id: str | None,
        *,
        identity_id: str,
        event_id: str,
        candidate_id: str | None,
        now: float | None = None,
    ) -> tuple[bool, str, AppearanceAbsorptionRequest | None]:
        """校验响应并返回 ``(accepted, reason, request)``。

        令牌必须仍处于待处理状态、尚未过期、绑定到相同身份和候选人，并且
        必须来自不同事件。
        """
        if not request_id:
            return False, "missing_appearance_request", None
        request = self._requests.get(request_id)
        if request is None:
            return False, "unknown_appearance_request", None
        now = time.time() if now is None else float(now)
        if request.status != "pending":
            return False, f"appearance_request_{request.status}", request
        if now > request.expires_at:
            request.status = "expired"
            return False, "appearance_request_expired", request
        if request.identity_id != identity_id:
            return False, "appearance_request_identity_mismatch", request
        if request.candidate_id is not None and request.candidate_id != candidate_id:
            return False, "appearance_request_candidate_mismatch", request
        if request.issued_by_event_id == event_id:
            return False, "appearance_request_same_event", request
        return True, "appearance_request_valid", request

    def consume(self, request_id: str, response_event_id: str) -> AppearanceAbsorptionRequest:
        """将已校验请求原子地标记为由响应事件消费。"""
        request = self._requests.get(request_id)
        if request is None:
            raise KeyError(request_id)
        if request.status != "pending":
            raise ValueError(f"appearance request is not pending: {request.status}")
        request.status = "consumed"
        request.response_event_id = response_event_id
        return request

    def restore(self, request: AppearanceAbsorptionRequest) -> None:
        """在验证器恢复持久化状态时恢复一个请求。"""
        self._requests[request.request_id] = request

    def all(self) -> tuple[AppearanceAbsorptionRequest, ...]:
        """返回所有请求，包括已消费和已过期的审计记录。"""
        return tuple(self._requests.values())
