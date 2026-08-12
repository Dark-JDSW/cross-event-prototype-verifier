"""使用轻量检测器/模型替身接口的生产适配器测试。"""

import unittest
from unittest.mock import patch

import numpy as np

from cross_event_verifier.participant_a.gait_graph import gait_graph_multi_input
from cross_event_verifier.participant_b.production_vision import (
    ProductionVisionAdapter,
    ProductionVisionConfig,
)
from cross_event_verifier.types import Prototype
from cross_event_verifier.participant_c.vector_index import NumpyVectorIndex
from cross_event_verifier.participant_b.vision import OpenCvDemoAdapter


class _FakeDetector:
    def __init__(self, _config):
        pass

    def track(self, _frame):
        item = type("Detection", (), {})()
        item.track_id = 7
        item.box = (20, 10, 100, 190)
        item.confidence = 0.93
        return (item,)

    def reset(self):
        pass


class _FakeTruncatedDetector(_FakeDetector):
    def track(self, _frame):
        detections = super().track(_frame)
        detections[0].box = (0, 10, 100, 190)
        return detections


class _FakePose:
    providers = ("FakeExecutionProvider",)

    def __init__(self, _path, **_kwargs):
        pass

    def extract(self, _frame, boxes):
        points = np.zeros((17, 3), dtype=np.float32)
        points[:, 0] = np.linspace(35, 85, 17)
        points[:, 1] = np.linspace(20, 180, 17)
        points[:, 2] = 0.95
        return [points.copy() for _ in boxes]


class _FakeAppearance:
    def __init__(self, _path, _device):
        pass

    def extract(self, _frame, boxes):
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = 1.0
        return [vector.copy() for _ in boxes]


class _FakeGait:
    def __init__(self, *_args, **_kwargs):
        pass

    def encode(self, _poses):
        vector = np.zeros(384, dtype=np.float32)
        vector[0] = 1.0
        return vector


class ProductionVisionTests(unittest.TestCase):
    def test_demo_backend_cannot_auto_register(self) -> None:
        self.assertFalse(OpenCvDemoAdapter.supports_automatic_registration)

    def test_production_config_rejects_cpu_device(self) -> None:
        with self.assertRaises(ValueError):
            ProductionVisionConfig(device="cpu")

    def test_production_adapter_applies_hot_thresholds_without_model_reload(self) -> None:
        with patch(
            "cross_event_verifier.participant_b.production_vision.production_readiness",
            return_value=(True, ()),
        ):
            adapter = ProductionVisionAdapter()
        adapter.update_runtime_parameters(
            {
                "detector_confidence": 0.31,
                "output_confidence": 0.52,
                "keypoint_confidence": 0.50,
            }
        )
        values = adapter.runtime_parameters()
        self.assertAlmostEqual(values["detector_confidence"], 0.31)
        self.assertAlmostEqual(values["output_confidence"], 0.52)
        self.assertAlmostEqual(values["keypoint_confidence"], 0.50)

        with self.assertRaises(ValueError):
            adapter.update_runtime_parameters(
                {
                    "detector_confidence": 0.80,
                    "output_confidence": 0.40,
                }
            )
        self.assertAlmostEqual(adapter.config.detector_confidence, 0.31)

    def test_gait_graph_input_contains_joints_velocity_and_bones(self) -> None:
        sequence = np.zeros((25, 17, 3), dtype=np.float32)
        sequence[:, :, 0] = np.linspace(10, 90, 17)
        sequence[:, :, 1] = np.linspace(5, 180, 17)
        sequence[:, :, 2] = 0.9
        sequence[1:, 15, 0] += 3.0
        transformed = gait_graph_multi_input(sequence)
        self.assertEqual(transformed.shape, (25, 17, 3, 5))
        self.assertGreater(float(np.abs(transformed[:, :, 1]).sum()), 0.0)
        self.assertTrue(np.isfinite(transformed).all())

    def test_production_adapter_returns_deep_features_after_temporal_warmup(self) -> None:
        config = ProductionVisionConfig(
            minimum_pose_frames=8,
            gait_sequence_length=8,
            appearance_stride=1,
        )
        frame = np.zeros((200, 120, 3), dtype=np.uint8)
        with (
            patch(
                "cross_event_verifier.participant_b.production_vision.production_readiness",
                return_value=(True, ()),
            ),
            patch("cross_event_verifier.participant_b.production_vision._YoloByteTracker", _FakeDetector),
            patch("cross_event_verifier.participant_b.production_vision._RtmposeEstimator", _FakePose),
            patch(
                "cross_event_verifier.participant_b.production_vision._OsnetAppearanceExtractor",
                _FakeAppearance,
            ),
            patch("cross_event_verifier.participant_b.production_vision.TemporalGaitEncoder", _FakeGait),
            patch(
                "cross_event_verifier.participant_b.production_vision._walking_metrics",
                return_value=(0.95, 1.0),
            ),
        ):
            adapter = ProductionVisionAdapter(config)
            tracks = ()
            for _ in range(8):
                tracks = adapter.process(frame)

        self.assertEqual(len(tracks), 1)
        track = tracks[0]
        self.assertEqual(track.track_id, 7)
        self.assertEqual(track.features.normalized().appearance.shape, (512,))
        self.assertEqual(track.features.normalized().gait.shape, (384,))
        self.assertGreater(track.quality.gait_availability(), 0.7)

    def test_truncated_person_cannot_supply_strong_cached_evidence(self) -> None:
        config = ProductionVisionConfig(
            minimum_pose_frames=8,
            gait_sequence_length=8,
            appearance_stride=1,
        )
        frame = np.zeros((200, 120, 3), dtype=np.uint8)
        with (
            patch(
                "cross_event_verifier.participant_b.production_vision.production_readiness",
                return_value=(True, ()),
            ),
            patch(
                "cross_event_verifier.participant_b.production_vision._YoloByteTracker",
                _FakeTruncatedDetector,
            ),
            patch("cross_event_verifier.participant_b.production_vision._RtmposeEstimator", _FakePose),
            patch(
                "cross_event_verifier.participant_b.production_vision._OsnetAppearanceExtractor",
                _FakeAppearance,
            ),
            patch("cross_event_verifier.participant_b.production_vision.TemporalGaitEncoder", _FakeGait),
            patch(
                "cross_event_verifier.participant_b.production_vision._walking_metrics",
                return_value=(0.95, 1.0),
            ),
        ):
            adapter = ProductionVisionAdapter(config)
            for _ in range(8):
                tracks = adapter.process(frame)

        track = tracks[0]
        self.assertIsNotNone(track.features.gait)
        self.assertFalse(track.quality.box_valid)
        self.assertEqual(track.quality.gait_branch_quality, 0.0)
        self.assertEqual(track.quality.gait_availability(), 0.0)
        self.assertIn("box_truncated", track.quality.reasons)

    def test_numpy_index_survives_embedding_dimension_migration(self) -> None:
        index = NumpyVectorIndex()
        index.rebuild(
            [
                Prototype("old", "gait", np.asarray([1.0, 0.0, 0.0])),
                Prototype("new", "gait", np.asarray([1.0, 0.0, 0.0, 0.0])),
            ]
        )
        hits = index.search(np.asarray([1.0, 0.0, 0.0, 0.0]))
        self.assertEqual([item.identity_id for item in hits], ["new"])


if __name__ == "__main__":
    unittest.main()
