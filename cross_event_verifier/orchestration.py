"""Pure helpers shared by identity enrollment orchestrators.

The helpers in this module only manage source local Track state and rolling
sample windows.  They never decide identities, write galleries, promote
candidates, or access verifier storage.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

import numpy as np

from .actions import ActionPrediction, ActionRouter, ActionType
from .aggregation import weighted_unit_mean
from .types import Observation


class GaitWindowState(Protocol):
    """Minimum mutable state needed by the shared gait window helpers."""

    gait_samples: deque[np.ndarray]
    gait_sample_weights: deque[float]
    gait_window_start_timestamp: float | None
    subtracklet_id: str | None
    action_segment_id: str | None
    action_type: ActionType | None
    action_gate_active: bool


StateT = TypeVar("StateT")


class TrackStateRegistry(Generic[StateT]):
    """Own candidate to Track state association without identity semantics."""

    def __init__(self, factory: Callable[[], StateT]) -> None:
        self.states: dict[str, StateT] = {}
        self._factory = factory

    def get(self, candidate_id: str) -> StateT:
        return self.states.setdefault(candidate_id, self._factory())

    def discard(
        self,
        candidate_id: str,
        *,
        on_discard: Callable[[str, StateT | None], None] | None = None,
    ) -> None:
        state = self.states.pop(candidate_id, None)
        if on_discard is not None:
            on_discard(candidate_id, state)

    def discard_many(
        self,
        candidate_ids: Sequence[str],
        *,
        on_discard: Callable[[str, StateT | None], None] | None = None,
    ) -> None:
        for candidate_id in candidate_ids:
            self.discard(candidate_id, on_discard=on_discard)

    def reset_tracks(
        self,
        *,
        preserve_candidate_ids: Sequence[str],
        discard_unpreserved: bool,
        reset_state: Callable[[StateT], None],
        on_discard: Callable[[str, StateT | None], None] | None = None,
    ) -> None:
        keep = {value for value in preserve_candidate_ids if value}
        for candidate_id in tuple(self.states):
            if discard_unpreserved and candidate_id not in keep:
                self.discard(candidate_id, on_discard=on_discard)
                continue
            reset_state(self.states[candidate_id])


def metadata_identifier(metadata: Mapping[str, Any], key: str) -> str | None:
    """Read a nonempty metadata identifier without interpreting its meaning."""

    value = metadata.get(key)
    return str(value) if value not in (None, "") else None


def clear_gait_window(state: GaitWindowState) -> None:
    """Clear aligned gait vectors, weights, and optional quality values."""

    state.gait_samples.clear()
    state.gait_sample_weights.clear()
    qualities = getattr(state, "gait_sample_qualities", None)
    if qualities is not None:
        qualities.clear()


def sync_metadata_boundary(
    state: StateT,
    observation: Observation,
    *,
    metadata_key: str,
    state_attribute: str,
    on_change: Callable[[StateT], None],
) -> bool:
    """Update one Track boundary marker and report a real marker change."""

    current = metadata_identifier(observation.metadata, metadata_key)
    if current is None:
        return False
    previous = getattr(state, state_attribute)
    changed = previous is not None and previous != current
    if changed:
        on_change(state)
    setattr(state, state_attribute, current)
    return changed


def sync_action_segment(state: GaitWindowState, observation: Observation) -> bool:
    """Clear only the gait window when an action segment changes."""

    def reset(item: GaitWindowState) -> None:
        clear_gait_window(item)
        item.gait_window_start_timestamp = None

    return sync_metadata_boundary(
        state,
        observation,
        metadata_key="action_segment_id",
        state_attribute="action_segment_id",
        on_change=reset,
    )


def resolve_action(
    state: GaitWindowState,
    observation: Observation,
) -> ActionPrediction | None:
    """Parse action metadata and isolate gait samples across action changes."""

    prediction = ActionRouter.from_metadata(observation.metadata)
    if prediction is None:
        return None
    if state.action_gate_active and state.action_type != prediction.action_type:
        clear_gait_window(state)
        state.gait_window_start_timestamp = None
    state.action_type = prediction.action_type
    state.action_gate_active = True
    return prediction


def independent_event_key(
    observation: Observation,
    *,
    challenge_first: bool,
) -> str:
    """Build an event key while preserving each legacy strategy's precedence."""

    session_id = observation.capture_session_id
    has_session = bool(session_id and session_id != "unknown-session")
    if challenge_first and observation.challenge_id:
        return f"challenge:{observation.challenge_id}"
    if has_session:
        return f"session:{session_id}"
    if observation.challenge_id:
        return f"challenge:{observation.challenge_id}"
    return f"event:{observation.event_id}"


@dataclass(frozen=True)
class GaitWindowUpdate:
    """Result of appending one aligned gait sample to a rolling window."""

    centroid: np.ndarray | None
    sample_count: int
    sample_mass: float
    similarity: float | None = None
    restarted_for_instability: bool = False
    restarted_for_shape: bool = False


def gait_window_centroid(state: GaitWindowState) -> np.ndarray | None:
    return weighted_unit_mean(state.gait_samples, tuple(state.gait_sample_weights))


def append_gait_sample(
    state: GaitWindowState,
    vector: np.ndarray,
    *,
    weight: float,
    timestamp: float,
    maximum_samples: int,
    minimum_similarity: float,
    quality: float | None = None,
) -> GaitWindowUpdate:
    """Append one sample while keeping all window deques aligned."""

    if not state.gait_samples:
        state.gait_window_start_timestamp = timestamp
    sample = np.asarray(vector, dtype=np.float32).reshape(-1)
    reference = gait_window_centroid(state)
    similarity: float | None = None
    restarted_for_shape = False
    restarted_for_instability = False
    if reference is not None:
        if reference.shape != sample.shape:
            clear_gait_window(state)
            state.gait_window_start_timestamp = timestamp
            restarted_for_shape = True
        else:
            similarity = float(np.dot(reference, sample))
            if similarity < minimum_similarity:
                clear_gait_window(state)
                state.gait_window_start_timestamp = timestamp
                restarted_for_instability = True

    state.gait_samples.append(sample.copy())
    state.gait_sample_weights.append(float(weight))
    qualities = getattr(state, "gait_sample_qualities", None)
    if qualities is not None:
        if quality is None:
            raise ValueError("quality is required for a quality-aware gait window")
        qualities.append(float(quality))
    while len(state.gait_samples) > maximum_samples:
        state.gait_samples.popleft()
        state.gait_sample_weights.popleft()
        if qualities is not None:
            qualities.popleft()

    return GaitWindowUpdate(
        centroid=gait_window_centroid(state),
        sample_count=len(state.gait_samples),
        sample_mass=float(sum(state.gait_sample_weights)),
        similarity=similarity,
        restarted_for_instability=restarted_for_instability,
        restarted_for_shape=restarted_for_shape,
    )


def weighted_window_quality(state: GaitWindowState) -> float | None:
    """Return the aligned weighted quality average for a quality-aware window."""

    qualities = getattr(state, "gait_sample_qualities", None)
    if qualities is None:
        return None
    quality_values = np.asarray(tuple(qualities), dtype=np.float32)
    weight_values = np.asarray(tuple(state.gait_sample_weights), dtype=np.float32)
    if (
        quality_values.shape[0] != len(state.gait_samples)
        or weight_values.shape[0] != len(state.gait_samples)
    ):
        return None
    total_weight = float(weight_values.sum())
    if total_weight <= 1e-8:
        return None
    return float(np.clip(np.dot(weight_values, quality_values) / total_weight, 0.0, 1.0))
