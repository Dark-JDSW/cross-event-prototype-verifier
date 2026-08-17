"""使用轻量检测器/模型替身接口的生产适配器测试。"""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from cross_event_verifier.gait_graph import (
    TemporalGaitEncoder,
    _coco_adjacency,
    _fill_missing,
    gait_graph_multi_input,
)
from cross_event_verifier.production_vision import (
    ProductionVisionAdapter,
    ProductionVisionConfig,
    _RtmposeEstimator,
    _canonical_pose,
)
from cross_event_verifier.types import Prototype
from cross_event_verifier.vector_index import NumpyVectorIndex
from cross_event_verifier.vision import OpenCvDemoAdapter


class _FakeDetector:
    def __init__(self, _config):
        self.calls = 0

    def track(self, _frame):
        self.calls += 1
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
        self.batch_sizes = []

    def extract(self, _frame, boxes):
        self.batch_sizes.append(len(boxes))
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = 1.0
        return [vector.copy() for _ in boxes]


class _FakeGait:
    def __init__(self, *_args, **_kwargs):
        self.batch_sizes = []

    def encode_batch(self, pose_sequences):
        self.batch_sizes.append(len(pose_sequences))
        output = []
        for index, _poses in enumerate(pose_sequences):
            vector = np.zeros(384, dtype=np.float32)
            vector[index % len(vector)] = 1.0
            output.append(vector)
        return output


class _FakeTwoPersonDetector(_FakeDetector):
    def track(self, _frame):
        first = super().track(_frame)[0]
        second = type("Detection", (), {})()
        second.track_id = 8
        second.box = (130, 10, 210, 190)
        second.confidence = 0.91
        return first, second


class _RecordingBatchModel:
    def __init__(self) -> None:
        self.batch_sizes = []

    def __call__(self, batch):
        self.batch_sizes.append(int(batch.shape[0]))
        output = torch.zeros((batch.shape[0], 128), dtype=torch.float32)
        for index in range(batch.shape[0]):
            output[index, index % output.shape[1]] = float(index + 1)
        return output


class ProductionVisionTests(unittest.TestCase):
    def test_simcc_confidence_uses_the_less_reliable_axis(self) -> None:
        simcc_x = np.zeros((1, 1, 4), dtype=np.float32)
        simcc_y = np.zeros((1, 1, 4), dtype=np.float32)
        simcc_x[0, 0, 1] = 0.90
        simcc_y[0, 0, 2] = 0.10

        _, _, confidence = _RtmposeEstimator._decode_simcc(simcc_x, simcc_y)

        self.assertAlmostEqual(float(confidence[0, 0]), 0.10, places=6)

    def test_demo_backend_cannot_auto_register(self) -> None:
        self.assertFalse(OpenCvDemoAdapter.supports_automatic_registration)

    def test_production_config_rejects_cpu_device(self) -> None:
        with self.assertRaises(ValueError):
            ProductionVisionConfig(device="cpu")

    def test_gait_pose_keeps_full_frame_coordinates(self) -> None:
        """GaitGraph2 receives GREW-style image coordinates, not box coordinates."""
        points = np.zeros((17, 3), dtype=np.float32)
        points[:, 0] = 300.0
        points[:, 1] = 200.0
        points[:, 2] = 0.95

        canonical = _canonical_pose(points, (100, 100, 200, 300), 0.45)

        self.assertIsNotNone(canonical)
        np.testing.assert_allclose(canonical[:, 0], 300.0)
        np.testing.assert_allclose(canonical[:, 1], 200.0)

    def test_production_adapter_applies_hot_thresholds_without_model_reload(self) -> None:
        with patch(
            "cross_event_verifier.production_vision.production_readiness",
            return_value=(True, ()),
        ):
            adapter = ProductionVisionAdapter()
        adapter.update_runtime_parameters(
            {
                "detector_confidence": 0.31,
                "output_confidence": 0.52,
                "detector_inference_stride": 3,
                "keypoint_confidence": 0.50,
                "appearance_stride": 6,
                "gait_inference_stride": 5,
            }
        )
        values = adapter.runtime_parameters()
        self.assertAlmostEqual(values["detector_confidence"], 0.31)
        self.assertAlmostEqual(values["output_confidence"], 0.52)
        self.assertEqual(values["detector_inference_stride"], 3)
        self.assertAlmostEqual(values["keypoint_confidence"], 0.50)
        self.assertEqual(values["appearance_stride"], 6)
        self.assertEqual(values["gait_inference_stride"], 5)

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

    def test_gait_graph_interpolates_partial_joint_gaps_without_fabricating_confidence(self) -> None:
        sequence = np.zeros((3, 17, 3), dtype=np.float32)
        sequence[:, 1:, 0] = 20.0
        sequence[:, 1:, 1] = 30.0
        sequence[:, 1:, 2] = 0.9
        sequence[0, 0] = (0.0, 10.0, 0.9)
        sequence[2, 0] = (10.0, 20.0, 0.9)

        filled = _fill_missing(sequence)

        self.assertAlmostEqual(float(filled[1, 0, 0]), 5.0, places=6)
        self.assertAlmostEqual(float(filled[1, 0, 1]), 15.0, places=6)
        self.assertEqual(float(filled[1, 0, 2]), 0.0)

    def test_gait_graph_adjacency_matches_opengait_partition_normalization(self) -> None:
        """The official graph normalizes over all reachable hops before masking."""
        adjacency = _coco_adjacency(3)

        # In the official COCO graph the nose can reach seven joints within
        # three hops, so the self-loop in hop 0 is 1/7, not 1/3 (the immediate
        # one-hop degree). This is a checkpoint compatibility invariant.
        self.assertAlmostEqual(float(adjacency[0, 0, 0]), 1.0 / 7.0, places=6)

    def test_temporal_gait_encoder_batches_valid_sequences_and_preserves_alignment(self) -> None:
        encoder = TemporalGaitEncoder.__new__(TemporalGaitEncoder)
        encoder.device = torch.device("cpu")
        encoder.sequence_length = 25
        encoder.use_tta = True
        encoder.model = _RecordingBatchModel()
        valid = np.zeros((25, 17, 3), dtype=np.float32)
        valid[:, :, 2] = 0.95
        invalid = np.zeros((8, 17, 3), dtype=np.float32)

        output = encoder.encode_batch([valid, invalid, valid.copy()])

        self.assertEqual(encoder.model.batch_sizes, [6])
        self.assertEqual(len(output), 3)
        self.assertIsNone(output[1])
        self.assertEqual(output[0].shape, (384,))
        self.assertEqual(output[2].shape, (384,))
        self.assertAlmostEqual(float(np.linalg.norm(output[0])), 1.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(output[2])), 1.0, places=6)

    def test_production_adapter_batches_due_gait_tracks_and_honors_stride(self) -> None:
        config = ProductionVisionConfig(
            minimum_pose_frames=8,
            gait_sequence_length=8,
            detector_inference_stride=2,
            appearance_stride=30,
            gait_inference_stride=3,
        )
        frame = np.zeros((200, 240, 3), dtype=np.uint8)
        with (
            patch(
            "cross_event_verifier.production_vision.production_readiness",
                return_value=(True, ()),
            ),
            patch(
                "cross_event_verifier.production_vision._YoloByteTracker",
                _FakeTwoPersonDetector,
            ),
            patch(
                "cross_event_verifier.production_vision._RtmposeEstimator",
                _FakePose,
            ),
            patch(
                "cross_event_verifier.production_vision._OsnetAppearanceExtractor",
                _FakeAppearance,
            ),
            patch(
                "cross_event_verifier.production_vision.TemporalGaitEncoder",
                _FakeGait,
            ),
            patch(
                "cross_event_verifier.production_vision._walking_metrics",
                return_value=(0.95, 1.0),
            ),
        ):
            adapter = ProductionVisionAdapter(config)
            gait_frames = []
            overlapping_frames = []
            for frame_index in range(1, 15):
                detector_calls = adapter.detector.calls if adapter.detector else 0
                gait_calls = len(adapter.gait.batch_sizes) if adapter.gait else 0
                tracks = adapter.process(frame)
                detector_ran = adapter.detector.calls > detector_calls
                gait_ran = len(adapter.gait.batch_sizes) > gait_calls
                if gait_ran:
                    gait_frames.append(frame_index)
                if detector_ran and gait_ran:
                    overlapping_frames.append(frame_index)

        self.assertEqual(len(tracks), 2)
        self.assertEqual(adapter.gait.batch_sizes, [2, 2])
        self.assertEqual(gait_frames, [8, 12])
        self.assertEqual(overlapping_frames, [])
        self.assertIsNotNone(tracks[0].features.gait)
        self.assertIsNotNone(tracks[1].features.gait)

    def test_production_adapter_reuses_detections_and_skips_empty_appearance_batches(self) -> None:
        config = ProductionVisionConfig(
            detector_inference_stride=2,
            appearance_stride=6,
        )
        frame = np.zeros((200, 120, 3), dtype=np.uint8)
        with (
            patch(
            "cross_event_verifier.production_vision.production_readiness",
                return_value=(True, ()),
            ),
            patch(
                "cross_event_verifier.production_vision._YoloByteTracker",
                _FakeDetector,
            ),
            patch(
                "cross_event_verifier.production_vision._RtmposeEstimator",
                _FakePose,
            ),
            patch(
                "cross_event_verifier.production_vision._OsnetAppearanceExtractor",
                _FakeAppearance,
            ),
            patch(
                "cross_event_verifier.production_vision.TemporalGaitEncoder",
                _FakeGait,
            ),
        ):
            adapter = ProductionVisionAdapter(config)
            appearance_frames = []
            overlapping_frames = []
            for frame_index in range(1, 9):
                detector_calls = adapter.detector.calls if adapter.detector else 0
                appearance_calls = (
                    len(adapter.appearance.batch_sizes) if adapter.appearance else 0
                )
                tracks = adapter.process(frame)
                detector_ran = adapter.detector.calls > detector_calls
                appearance_ran = len(adapter.appearance.batch_sizes) > appearance_calls
                if appearance_ran:
                    appearance_frames.append(frame_index)
                if frame_index > 1 and detector_ran and appearance_ran:
                    overlapping_frames.append(frame_index)

        self.assertEqual(len(tracks), 1)
        self.assertEqual(adapter.detector.calls, 4)
        self.assertEqual(adapter.appearance.batch_sizes, [1, 1])
        self.assertEqual(appearance_frames, [1, 8])
        self.assertEqual(overlapping_frames, [])

    def test_embedding_refreshes_are_not_starved_when_detection_runs_every_frame(self) -> None:
        config = ProductionVisionConfig(
            detector_inference_stride=1,
            minimum_pose_frames=8,
            gait_sequence_length=8,
            appearance_stride=2,
            gait_inference_stride=2,
        )
        frame = np.zeros((200, 120, 3), dtype=np.uint8)
        with (
            patch(
            "cross_event_verifier.production_vision.production_readiness",
                return_value=(True, ()),
            ),
            patch(
                "cross_event_verifier.production_vision._YoloByteTracker",
                _FakeDetector,
            ),
            patch(
                "cross_event_verifier.production_vision._RtmposeEstimator",
                _FakePose,
            ),
            patch(
                "cross_event_verifier.production_vision._OsnetAppearanceExtractor",
                _FakeAppearance,
            ),
            patch(
                "cross_event_verifier.production_vision.TemporalGaitEncoder",
                _FakeGait,
            ),
            patch(
                "cross_event_verifier.production_vision._walking_metrics",
                return_value=(0.95, 1.0),
            ),
        ):
            adapter = ProductionVisionAdapter(config)
            for _ in range(16):
                adapter.process(frame)

        self.assertGreaterEqual(len(adapter.appearance.batch_sizes), 3)
        self.assertGreaterEqual(len(adapter.gait.batch_sizes), 2)

    def test_production_adapter_returns_deep_features_after_temporal_warmup(self) -> None:
        config = ProductionVisionConfig(
            minimum_pose_frames=8,
            gait_sequence_length=8,
            appearance_stride=1,
        )
        frame = np.zeros((200, 120, 3), dtype=np.uint8)
        with (
            patch(
            "cross_event_verifier.production_vision.production_readiness",
                return_value=(True, ()),
            ),
            patch("cross_event_verifier.production_vision._YoloByteTracker", _FakeDetector),
            patch("cross_event_verifier.production_vision._RtmposeEstimator", _FakePose),
            patch(
                "cross_event_verifier.production_vision._OsnetAppearanceExtractor",
                _FakeAppearance,
            ),
            patch("cross_event_verifier.production_vision.TemporalGaitEncoder", _FakeGait),
            patch(
                "cross_event_verifier.production_vision._walking_metrics",
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
            "cross_event_verifier.production_vision.production_readiness",
                return_value=(True, ()),
            ),
            patch(
                "cross_event_verifier.production_vision._YoloByteTracker",
                _FakeTruncatedDetector,
            ),
            patch("cross_event_verifier.production_vision._RtmposeEstimator", _FakePose),
            patch(
                "cross_event_verifier.production_vision._OsnetAppearanceExtractor",
                _FakeAppearance,
            ),
            patch("cross_event_verifier.production_vision.TemporalGaitEncoder", _FakeGait),
            patch(
                "cross_event_verifier.production_vision._walking_metrics",
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
