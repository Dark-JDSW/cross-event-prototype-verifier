"""GCP calibration and file-format tests used by both GUI adapters."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cross_event_verifier.ground_control import (
    GroundControlPoint,
    compute_ground_homography,
    load_ground_control_file,
    parse_ground_control_point,
    save_ground_control_file,
)


class GroundControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.points = (
            GroundControlPoint("A", 0.0, 0.0, 0.0, 0.0),
            GroundControlPoint("B", 100.0, 0.0, 2.0, 0.0),
            GroundControlPoint("C", 100.0, 100.0, 2.0, 3.0),
            GroundControlPoint("D", 0.0, 100.0, 0.0, 3.0),
        )

    def test_homography_maps_control_points_and_reports_inliers(self) -> None:
        result = compute_ground_homography(self.points)

        self.assertEqual(result.inliers, (True, True, True, True))
        self.assertLess(result.rmse, 1e-6)
        self.assertLess(result.maximum_error, 1e-6)
        self.assertAlmostEqual(result.matrix[0], 0.02, places=6)
        self.assertAlmostEqual(result.matrix[4], 0.03, places=6)

    def test_text_fields_are_parsed_into_a_complete_point(self) -> None:
        self.assertEqual(
            parse_ground_control_point(" A ", "10", "20", "1.5", "2.5"),
            GroundControlPoint("A", 10.0, 20.0, 1.5, 2.5),
        )

    def test_export_import_round_trip_preserves_metadata_and_points(self) -> None:
        result = compute_ground_homography(self.points)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ground-control.json"
            save_ground_control_file(
                path,
                camera_id="camera-7",
                coordinate_frame="site-a-meters",
                unit="m",
                image_size=(1920, 1080),
                points=self.points,
                check=result,
            )

            payload = load_ground_control_file(path)

        self.assertEqual(payload["camera_id"], "camera-7")
        self.assertEqual(payload["coordinate_frame"], "site-a-meters")
        self.assertEqual(payload["image_size"], [1920, 1080])
        self.assertEqual(payload["points"], list(self.points))

    def test_collinear_points_are_rejected_before_calculation(self) -> None:
        points = tuple(
            GroundControlPoint(str(index), float(index), float(index), float(index), 0.0)
            for index in range(4)
        )

        with self.assertRaisesRegex(ValueError, "不能共线"):
            compute_ground_homography(points)


if __name__ == "__main__":
    unittest.main()
