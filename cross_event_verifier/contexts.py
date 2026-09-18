"""Optional, read-only facts downstream of identity verification.

Providers produce facts; the composer merges them. No verifier, gallery, model,
transport or calibration implementation is passed to a provider.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

from .transient_actions import ActionEvent
from .types import Decision


def _freeze(value: object) -> object:
    """Copy JSON-shaped extension data; reject live implementation objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("context keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    raise TypeError("context values must be finite JSON-shaped facts")


@dataclass(frozen=True, kw_only=True)
class ContextFacts:
    """Shared optional facts. Positions/distances are metres in coordinate_frame.

    velocity is (vx, vy) in metres/second; facing_direction is degrees in that
    frame. Missing calibration means unknown, never an invented origin/zone.
    Experimental facts use namespaced attributes, e.g. occupancy.room_count.
    """

    zone_id: str | None = None
    world_position: tuple[float, float] | None = None
    velocity: tuple[float, float] | None = None
    facing_direction: float | None = None
    nearest_device_id: str | None = None
    distance_to_device: float | None = None
    coordinate_frame: str | None = None
    calibration_id: str | None = None
    attributes: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("attributes", "metadata"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        for name in ("world_position", "velocity"):
            value = getattr(self, name)
            if value is not None:
                if len(value) != 2 or not all(math.isfinite(v) for v in value):
                    raise ValueError(f"{name} must be a finite pair")
                object.__setattr__(self, name, tuple(float(v) for v in value))
        for name in ("facing_direction", "distance_to_device"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.distance_to_device is not None and self.distance_to_device < 0:
            raise ValueError("distance_to_device must be nonnegative")


@dataclass(frozen=True, kw_only=True)
class ContextRequest:
    """Correlation envelope copied after Decision; no mutable core references."""

    identity_id: str | None = None
    track_id: int | None = None
    camera_id: str | None = None
    timestamp: float | None = None
    capture_session_id: str | None = None
    action: ActionEvent | None = None


@dataclass(frozen=True, kw_only=True)
class ContextContribution(ContextFacts):
    """One producer's facts, correlated to a track, camera, session and time."""

    track_id: int | None = None
    camera_id: str | None = None
    timestamp: float | None = None
    capture_session_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class SpatialContext(ContextContribution):
    """Spatial facts without identity authority; trajectory goes in attributes."""


@dataclass(frozen=True, kw_only=True)
class DeviceContext(ContextContribution):
    """Device/environment facts without device clients or identity authority."""


@dataclass(frozen=True, kw_only=True)
class InteractionContext(ContextFacts, ContextRequest):
    """Immutable composed snapshot. Errors make policies fail closed by default."""

    errors: tuple[str, ...] = ()


class ContextProvider(Protocol):
    def provide(self, request: ContextRequest) -> ContextContribution | None:
        """Return facts for this envelope or None; never perform identity writes."""
        ...


class ContextComposer:
    """Deterministic merge: reject stale/mismatched facts and conflicting values.

    Each contribution is atomic. Failed providers are recorded and skipped.
    Provider ordering never silently overwrites facts. No plugin is enabled by
    default; call this only after verification, outside the core transaction.
    """

    def __init__(self, providers: Sequence[ContextProvider] | None = None, *,
                 max_age_seconds: float = 1.0) -> None:
        if not math.isfinite(max_age_seconds) or max_age_seconds < 0:
            raise ValueError("max_age_seconds must be finite and nonnegative")
        self.providers = tuple(providers or ())
        self.max_age_seconds = max_age_seconds

    def compose(self, decision: Decision, action: ActionEvent | None = None, *,
                track_id: int | None = None, camera_id: str | None = None,
                timestamp: float | None = None, capture_session_id: str | None = None,
                spatial: SpatialContext | None = None,
                device: DeviceContext | None = None) -> InteractionContext:
        identity = decision.identity_id if decision.open_set_label == "KNOWN" else None
        request = ContextRequest(
            identity_id=identity,
            track_id=track_id if track_id is not None else (action.track_id if action else None),
            camera_id=camera_id if camera_id is not None else (action.camera_id if action else None),
            timestamp=timestamp if timestamp is not None else (action.timestamp if action else None),
            capture_session_id=capture_session_id,
        )
        if request.timestamp is not None and not math.isfinite(request.timestamp):
            raise ValueError("timestamp must be finite")
        errors: list[str] = []
        if action is not None:
            if (identity is None or action.identity_id != identity
                    or action.track_id != request.track_id
                    or action.camera_id != request.camera_id
                    or not math.isfinite(action.timestamp)
                    or not 0 <= request.timestamp - action.timestamp <= self.max_age_seconds):
                errors.append("action: identity, source or timestamp mismatch")
                action = None
        request = ContextRequest(**{f.name: getattr(request, f.name)
                                    for f in fields(ContextRequest) if f.name != "action"},
                                 action=action)
        values = {f.name: getattr(ContextFacts(), f.name) for f in fields(ContextFacts)}

        def merge(source: str, contribution: ContextContribution | None) -> None:
            if contribution is None:
                return
            try:
                if not isinstance(contribution, ContextContribution):
                    raise TypeError("provider must return ContextContribution or None")
                for key in ("track_id", "camera_id", "capture_session_id"):
                    if getattr(contribution, key) != getattr(request, key):
                        raise ValueError(f"{key} mismatch")
                if (request.timestamp is None or contribution.timestamp is None
                        or not math.isfinite(contribution.timestamp)
                        or not 0 <= request.timestamp - contribution.timestamp <= self.max_age_seconds):
                    raise ValueError("missing, stale or future timestamp")
                candidate = dict(values)
                for f in fields(ContextFacts):
                    incoming, current = getattr(contribution, f.name), candidate[f.name]
                    if f.name in ("attributes", "metadata"):
                        if any(k in current and current[k] != v for k, v in incoming.items()):
                            raise ValueError(f"conflicting {f.name}")
                        candidate[f.name] = {**current, **incoming}
                    elif incoming is not None:
                        if current is not None and current != incoming:
                            raise ValueError(f"conflicting {f.name}")
                        candidate[f.name] = incoming
                values.update(candidate)
            except Exception as error:
                errors.append(f"{source}: {type(error).__name__}: {error}")

        merge("spatial", spatial)
        merge("device", device)
        for index, provider in enumerate(self.providers):
            source = f"provider[{index}]"
            try:
                contribution = provider.provide(request)
            except Exception as error:
                errors.append(f"{source}: {type(error).__name__}: {error}")
                continue
            merge(source, contribution)
        return InteractionContext(**values, **{f.name: getattr(request, f.name)
                                              for f in fields(ContextRequest)},
                                  errors=tuple(errors))
