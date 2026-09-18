"""Optional spatial producer boundary; reuse existing calibrated vision facts."""
from __future__ import annotations

from typing import Protocol

from .contexts import ContextRequest, SpatialContext
from .vision import VisionTrack


class SpatialProvider(Protocol):
    def locate(self, track: VisionTrack, *, camera_id: str, timestamp: float,
               capture_session_id: str | None = None) -> SpatialContext | None:
        ...


class VisionMetadataSpatialProvider:
    """Expose existing production geometry without recalculating Homography.

    camera_zone is an image-space bin, NOT a physical room/zone. Movement
    direction is NOT facing direction. Keep both namespaced and distinct.
    """

    def locate(self, track: VisionTrack, *, camera_id: str, timestamp: float,
               capture_session_id: str | None = None) -> SpatialContext | None:
        data = track.metadata
        if not data.get("ground_calibrated") or data.get("world_position") is None:
            return None
        return SpatialContext(
            track_id=track.track_id, camera_id=camera_id, timestamp=timestamp,
            capture_session_id=capture_session_id,
            world_position=data["world_position"],
            coordinate_frame=data.get("ground_coordinate_frame"),
            calibration_id=data.get("ground_calibration_id"),
            attributes={f"spatial.{key}": data[key] for key in (
                "ground_speed", "world_direction_degrees", "camera_zone",
                "camera_view_timeline", "ground_trajectory_quality",
            ) if key in data},
        )


class SpatialContextProvider:
    """Adapt a precomputed spatial snapshot to the common provider protocol."""

    def __init__(self, context: SpatialContext | None = None) -> None:
        self.context = context

    def provide(self, request: ContextRequest) -> SpatialContext | None:
        # Composer owns source/session/freshness validation.
        return self.context


class IdentityPriorProvider(Protocol):
    """Reserved, NOT wired into scoring. 0.0 means neutral, range [-1, 1].

    Future integration must be opt-in with weight zero by default, bounded,
    audited and gated by the existing biometric/open-set rules. A prior never
    returns a Decision or writes a gallery. Existing camera prior is unchanged.
    """

    def prior(self, request: ContextRequest, *, candidate_identity_id: str) -> float:
        ...
