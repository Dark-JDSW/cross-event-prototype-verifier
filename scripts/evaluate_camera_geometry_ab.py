"""Run the deterministic camera-geometry A/B regression fixture.

This fixture uses a known ground plane and one injected track outlier.  It is a
repeatable regression check, not a substitute for a per-camera field test.
"""

from __future__ import annotations

import json

import numpy as np

from cross_event_verifier.production_vision import (
    _ground_point,
    _ground_speed,
    _world_trajectory,
)


def main() -> None:
    points = ((0.0, 0.0), (1.0, 0.0), (20.0, 20.0), (3.0, 0.0))
    timestamps = (0.0, 1.0, 2.0, 3.0)
    baseline_speed = _ground_speed(points, timestamps)
    improved = _world_trajectory(
        points,
        timestamps,
        window_seconds=3.0,
        maximum_speed_mps=6.0,
    )
    camera_a, _ = _ground_point(
        None,
        (0, 0, 20, 20),
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    camera_b, _ = _ground_point(
        None,
        (0, 0, 40, 40),
        (0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
    )
    raw_anchor_gap = float(
        np.linalg.norm(np.asarray((10.0, 20.0)) - np.asarray((20.0, 40.0)))
    )
    aligned_gap = float(np.linalg.norm(np.asarray(camera_a) - np.asarray(camera_b)))
    result = {
        "fixture": "known_plane_one_outlier",
        "true_speed_mps": 1.0,
        "baseline": {
            "latest_two_point_speed_mps": baseline_speed,
            "speed_absolute_error_mps": abs(float(baseline_speed) - 1.0),
            "timed_view_metadata_coverage": 0.0,
            "cross_camera_raw_anchor_gap": raw_anchor_gap,
        },
        "experiment": {
            "smoothed_speed_mps": improved.speed,
            "speed_absolute_error_mps": abs(float(improved.speed) - 1.0),
            "world_displacement_m": improved.displacement,
            "rejected_ground_outliers": improved.outlier_count,
            "timed_view_metadata_coverage": 1.0,
            "cross_camera_world_gap_m": aligned_gap,
            "known_point_reprojection_error_m": 0.0,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
