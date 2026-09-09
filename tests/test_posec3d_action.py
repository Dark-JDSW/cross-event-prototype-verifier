import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cross_event_verifier.posec3d_action import (
    PoseC3DActionClassifier,
    PoseC3DPrediction,
)


LABEL_MAP = Path(__file__).resolve().parents[1] / "models" / "hmdb51_label_map.txt"


def _poses(length: int = 48) -> np.ndarray:
    values = np.zeros((length, 17, 3), dtype=np.float32)
    values[:, :, 0] = np.linspace(20.0, 100.0, 17)
    values[:, :, 1] = np.linspace(20.0, 220.0, 17)
    values[:, :, 2] = 0.95
    return values


class PoseC3DActionTests(unittest.TestCase):
    def _classifier(self, scores: np.ndarray, captured: dict[str, object]):
        def inferencer(_model, pose_results, image_shape):
            captured["pose_results"] = pose_results
            captured["image_shape"] = image_shape
            return SimpleNamespace(pred_score=scores)

        return PoseC3DActionClassifier(
            Path("checkpoint.pth"),
            Path("config.py"),
            LABEL_MAP,
            model=object(),
            inferencer=inferencer,
        )

    def test_walk_index_and_input_contract(self) -> None:
        scores = np.zeros(51, dtype=np.float32)
        scores[49] = 4.0
        captured: dict[str, object] = {}
        classifier = self._classifier(scores, captured)

        prediction = classifier.predict(_poses(), image_shape=(480, 640))

        self.assertEqual(prediction.top_index, 49)
        self.assertEqual(prediction.top_label, "walk")
        self.assertTrue(prediction.semantic_walk)
        self.assertEqual(captured["image_shape"], (480, 640))
        pose_results = captured["pose_results"]
        assert isinstance(pose_results, list)
        self.assertEqual(len(pose_results), 48)
        self.assertEqual(pose_results[0]["keypoints"].shape, (1, 17, 2))
        self.assertEqual(pose_results[0]["keypoint_scores"].shape, (1, 17))

    def test_non_walk_is_not_semantic_walk(self) -> None:
        scores = np.zeros(51, dtype=np.float32)
        scores[32] = 4.0  # run in the official HMDB51 label map
        classifier = self._classifier(scores, {})

        prediction = classifier.predict(_poses(), image_shape=(480, 640))

        self.assertEqual(prediction.top_label, "run")
        self.assertFalse(prediction.semantic_walk)

    def test_short_or_gap_window_fails_closed(self) -> None:
        scores = np.zeros(51, dtype=np.float32)
        scores[49] = 4.0
        classifier = self._classifier(scores, {})

        short = classifier.predict(_poses(12), image_shape=(480, 640))
        broken = _poses()
        broken[10, :, 2] = 0.0
        invalid = classifier.predict(broken, image_shape=(480, 640))

        self.assertTrue(short.reason.startswith("PoseC3D 需要至少"))
        self.assertEqual(invalid.reason, "PoseC3D 输入包含无姿态帧")
        self.assertEqual(short.top_index, -1)
        self.assertFalse(PoseC3DPrediction.invalid("test").semantic_walk)


if __name__ == "__main__":
    unittest.main()
