"""Track-local short action recognition from the existing COCO-17 pose stream."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from .actions import ActionType
from .vision import Box


@dataclass(frozen=True)
class ActionEvent:
    """An identity-authorized, completed short action."""

    identity_id: str
    track_id: int
    action: ActionType
    confidence: float
    timestamp: float
    camera_id: str


@dataclass(frozen=True)
class TransientActionConfig:
    """Time and geometry thresholds for the lightweight detector."""

    keypoint_confidence: float = 0.35
    state_ttl_seconds: float = 4.0
    pending_event_seconds: float = 1.0
    cooldown_seconds: float = 2.0
    hand_hold_seconds: float = 0.55
    wave_window_seconds: float = 1.4
    sit_hold_seconds: float = 0.75
    squat_max_seconds: float = 2.5
    bent_knee_degrees: float = 125.0
    straight_knee_degrees: float = 150.0
    wave_amplitude_torso: float = 0.55

    def __post_init__(self) -> None:
        if not 0.0 <= self.keypoint_confidence <= 1.0:
            raise ValueError("keypoint_confidence must be in [0, 1]")
        for name in (
            "state_ttl_seconds",
            "pending_event_seconds",
            "cooldown_seconds",
            "hand_hold_seconds",
            "wave_window_seconds",
            "sit_hold_seconds",
            "squat_max_seconds",
            "wave_amplitude_torso",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.bent_knee_degrees >= self.straight_knee_degrees:
            raise ValueError("bent knee threshold must be below straight knee threshold")


@dataclass(frozen=True)
class _DetectedAction:
    action: ActionType
    confidence: float
    timestamp: float


@dataclass
class _TrackActionState:
    segment_id: int | None
    last_seen: float
    identity_id: str | None = None
    posture: str = "unknown"
    low_started: float | None = None
    seated: bool = False
    hand_started: dict[int, float] = field(default_factory=dict)
    hand_emitted: set[int] = field(default_factory=set)
    wrist_history: dict[int, deque[tuple[float, float]]] = field(
        default_factory=lambda: {9: deque(), 10: deque()}
    )
    pending: deque[_DetectedAction] = field(default_factory=deque)
    last_emitted: dict[ActionType, float] = field(default_factory=dict)
    live_action: ActionType | None = None
    live_until: float = 0.0


def _angle(first: np.ndarray, middle: np.ndarray, last: np.ndarray) -> float:
    left = first[:2] - middle[:2]
    right = last[:2] - middle[:2]
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-8:
        return math.nan
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


class TransientActionManager:
    """Recognize short actions in RAM and release them only to a known identity."""

    def __init__(self, config: TransientActionConfig | None = None) -> None:
        self.config = config or TransientActionConfig()
        self._tracks: dict[int, _TrackActionState] = {}

    def reset(self) -> None:
        self._tracks.clear()

    def observe(
        self,
        track_id: int,
        pose: np.ndarray | Sequence[Sequence[float]] | None,
        box: Box,
        *,
        timestamp: float,
        segment_id: int | None = None,
    ) -> None:
        """Consume one pose before identity verification for this frame."""

        now = float(timestamp)
        state = self._tracks.get(track_id)
        if state is None or (
            segment_id is not None
            and state.segment_id is not None
            and segment_id != state.segment_id
        ):
            state = _TrackActionState(segment_id=segment_id, last_seen=now)
            self._tracks[track_id] = state
        state.segment_id = segment_id
        state.last_seen = now
        values = np.asarray(pose, dtype=np.float32) if pose is not None else np.empty((0, 3))
        if values.shape != (17, 3) or not np.isfinite(values).all():
            self._clear_gestures(state)
            return
        self._update_posture(state, values, now)
        self._update_hands(state, values, box, now)
        self._trim_pending(state, now)

    def authorize(
        self,
        track_id: int,
        identity_id: str | None,
        *,
        timestamp: float,
        camera_id: str,
    ) -> tuple[ActionEvent, ...]:
        """Bind the Track and release recent detections only for a confirmed ID."""

        state = self._tracks.get(track_id)
        if state is None:
            return ()
        now = float(timestamp)
        normalized = str(identity_id).strip() if identity_id else None
        if state.identity_id is not None and normalized != state.identity_id:
            # An identity loss or switch is a hard authorization boundary.
            state = _TrackActionState(
                segment_id=state.segment_id,
                last_seen=state.last_seen,
                identity_id=normalized,
            )
            self._tracks[track_id] = state
            return ()
        state.identity_id = normalized
        self._trim_pending(state, now)
        if normalized is None:
            return ()
        events: list[ActionEvent] = []
        while state.pending:
            detected = state.pending.popleft()
            last = state.last_emitted.get(detected.action, -math.inf)
            if detected.timestamp - last < self.config.cooldown_seconds:
                continue
            state.last_emitted[detected.action] = detected.timestamp
            events.append(
                ActionEvent(
                    identity_id=normalized,
                    track_id=int(track_id),
                    action=detected.action,
                    confidence=detected.confidence,
                    timestamp=detected.timestamp,
                    camera_id=str(camera_id),
                )
            )
        return tuple(events)

    def current_action(self, track_id: int, *, timestamp: float) -> ActionType | None:
        state = self._tracks.get(track_id)
        if state is None:
            return None
        if state.seated:
            return ActionType.SITTING
        return state.live_action if float(timestamp) <= state.live_until else None

    def expire(self, active_track_ids: set[int], *, timestamp: float) -> None:
        now = float(timestamp)
        for track_id, state in tuple(self._tracks.items()):
            if (
                track_id not in active_track_ids
                or now - state.last_seen > self.config.state_ttl_seconds
            ):
                del self._tracks[track_id]

    def _valid(self, pose: np.ndarray, indexes: Sequence[int]) -> bool:
        return bool(np.all(pose[list(indexes), 2] >= self.config.keypoint_confidence))

    def _detect(self, state: _TrackActionState, action: ActionType, confidence: float, now: float) -> None:
        state.pending.append(_DetectedAction(action, float(np.clip(confidence, 0.0, 1.0)), now))
        state.live_action = action
        state.live_until = now + 0.8

    def _update_posture(self, state: _TrackActionState, pose: np.ndarray, now: float) -> None:
        knee_angles = [
            _angle(pose[hip], pose[knee], pose[ankle])
            for hip, knee, ankle in ((11, 13, 15), (12, 14, 16))
            if self._valid(pose, (hip, knee, ankle))
        ]
        knee_angles = [value for value in knee_angles if np.isfinite(value)]
        if not knee_angles:
            return
        knee_angle = float(np.median(knee_angles))
        standing = knee_angle >= self.config.straight_knee_degrees
        bent = knee_angle <= self.config.bent_knee_degrees
        thighs_horizontal = []
        for hip, knee in ((11, 13), (12, 14)):
            if not self._valid(pose, (hip, knee)):
                continue
            delta = pose[knee, :2] - pose[hip, :2]
            length = float(np.linalg.norm(delta))
            if length > 1e-6:
                thighs_horizontal.append(abs(float(delta[1])) / length <= 0.50)
        seated_geometry = bent and bool(thighs_horizontal) and any(thighs_horizontal)

        if bent:
            if state.posture != "low":
                state.posture = "low"
                state.low_started = now
                state.seated = False
            if (
                seated_geometry
                and not state.seated
                and state.low_started is not None
                and now - state.low_started >= self.config.sit_hold_seconds
            ):
                state.seated = True
                self._detect(state, ActionType.SIT_DOWN, 0.82, now)
            return

        if standing:
            if state.posture == "low" and state.low_started is not None:
                duration = now - state.low_started
                if state.seated:
                    self._detect(state, ActionType.STAND_UP, 0.86, now)
                elif duration <= self.config.squat_max_seconds:
                    confidence = 0.78 + 0.12 * (1.0 - duration / self.config.squat_max_seconds)
                    self._detect(state, ActionType.SQUAT, confidence, now)
            state.posture = "standing"
            state.low_started = None
            state.seated = False

    def _update_hands(self, state: _TrackActionState, pose: np.ndarray, box: Box, now: float) -> None:
        if not self._valid(pose, (5, 6, 11, 12)):
            self._clear_gestures(state)
            return
        shoulder_center = pose[[5, 6], :2].mean(axis=0)
        hip_center = pose[[11, 12], :2].mean(axis=0)
        torso = max(float(np.linalg.norm(hip_center - shoulder_center)), (box[3] - box[1]) * 0.08, 1.0)
        for wrist, shoulder in ((9, 5), (10, 6)):
            raised = self._valid(pose, (wrist, shoulder)) and (
                pose[wrist, 1] < pose[shoulder, 1] - 0.10 * torso
            )
            history = state.wrist_history[wrist]
            if not raised:
                state.hand_started.pop(wrist, None)
                state.hand_emitted.discard(wrist)
                history.clear()
                continue
            state.hand_started.setdefault(wrist, now)
            history.append((now, float(pose[wrist, 0] / torso)))
            while history and history[0][0] < now - self.config.wave_window_seconds:
                history.popleft()
            if self._is_wave(history):
                self._detect(state, ActionType.WAVE, 0.90, now)
                history.clear()
                state.hand_emitted.add(wrist)
            elif (
                wrist not in state.hand_emitted
                and now - state.hand_started[wrist] >= self.config.hand_hold_seconds
            ):
                self._detect(state, ActionType.HAND_UP, 0.84, now)
                state.hand_emitted.add(wrist)

    def _is_wave(self, history: deque[tuple[float, float]]) -> bool:
        if len(history) < 4:
            return False
        x = np.asarray([sample[1] for sample in history], dtype=np.float32)
        if float(np.ptp(x)) < self.config.wave_amplitude_torso:
            return False
        delta = np.diff(x)
        signs = np.sign(delta[np.abs(delta) >= 0.08])
        return bool(len(signs) >= 3 and np.count_nonzero(signs[1:] != signs[:-1]) >= 2)

    @staticmethod
    def _clear_gestures(state: _TrackActionState) -> None:
        state.hand_started.clear()
        state.hand_emitted.clear()
        for history in state.wrist_history.values():
            history.clear()

    def _trim_pending(self, state: _TrackActionState, now: float) -> None:
        cutoff = now - self.config.pending_event_seconds
        while state.pending and state.pending[0].timestamp < cutoff:
            state.pending.popleft()


__all__ = ["ActionEvent", "TransientActionConfig", "TransientActionManager"]
