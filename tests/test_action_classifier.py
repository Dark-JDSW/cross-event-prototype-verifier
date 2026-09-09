import tempfile
import unittest
from pathlib import Path

import numpy as np

from cross_event_verifier import ActionQuality, ActionType
from cross_event_verifier.action_classifier import (
    ActionClassifierModel,
    PoseActionClassifier,
    extract_pose_action_features,
)


def _pose_sequence(length: int = 24, *, walking: bool = False) -> np.ndarray:
    values = np.zeros((length, 17, 3), dtype=np.float32)
    values[:, :, 2] = 0.95
    values[:, 5, :2] = (90, 50)
    values[:, 6, :2] = (110, 50)
    values[:, 11, :2] = (95, 100)
    values[:, 12, :2] = (105, 100)
    values[:, 13, :2] = (95, 145)
    values[:, 14, :2] = (105, 145)
    values[:, 15, :2] = (95, 195)
    values[:, 16, :2] = (105, 195)
    if walking:
        phase = np.where(np.arange(length) % 2 == 0, 18.0, -18.0)
        values[:, 15, 0] += phase
        values[:, 16, 0] -= phase
    return values


class ActionClassifierTests(unittest.TestCase):
    def test_incomplete_pose_window_is_not_called_walk(self) -> None:
        prediction = PoseActionClassifier(minimum_frames=12).predict(
            _pose_sequence(4, walking=True),
            walking_ratio=0.95,
            gait_cycles=2.0,
        )
        self.assertEqual(prediction.action_type, ActionType.UNKNOWN)
        self.assertEqual(prediction.quality, ActionQuality.PARTIAL)

    def test_kinematic_classifier_routes_a_complete_walking_window(self) -> None:
        prediction = PoseActionClassifier(minimum_frames=12).predict(
            _pose_sequence(walking=True),
            walking_ratio=0.90,
            gait_cycles=1.0,
        )
        self.assertEqual(prediction.action_type, ActionType.WALK)
        self.assertEqual(prediction.quality, ActionQuality.STRONG)
        self.assertGreaterEqual(prediction.completion, 0.5)

    def test_feature_extractor_has_stable_contract_and_no_nan(self) -> None:
        features = extract_pose_action_features(_pose_sequence())
        self.assertEqual(features.shape, (11,))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(
            extract_pose_action_features(np.zeros((0, 17, 3), dtype=np.float32)).shape,
            (11,),
        )

    def test_centroid_model_round_trips_without_pickle(self) -> None:
        sequences = [
            _pose_sequence(walking=True),
            _pose_sequence(walking=True),
            _pose_sequence(),
            _pose_sequence(),
        ]
        model = ActionClassifierModel.fit(
            sequences,
            [ActionType.WALK, ActionType.WALK, ActionType.OTHER, ActionType.OTHER],
            minimum_samples_per_class=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "action.npz"
            model.save(path)
            loaded = ActionClassifierModel.load(path)
        self.assertEqual(loaded.classes, model.classes)
        self.assertEqual(
            loaded.predict(_pose_sequence(walking=True)).action_type,
            ActionType.WALK,
        )

    def test_centroid_fit_requires_multiple_examples_per_class(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2 samples"):
            ActionClassifierModel.fit(
                [_pose_sequence(walking=True), _pose_sequence()],
                [ActionType.WALK, ActionType.OTHER],
            )

    def test_action_type_accepts_gui_branch_labels(self) -> None:
        self.assertEqual(ActionType.parse("running"), ActionType.RUNNING)
        self.assertEqual(ActionType.parse("pushing_pulling"), ActionType.PUSHING_PULLING)
        self.assertEqual(ActionType.parse("sit"), ActionType.SITTING)


if __name__ == "__main__":
    unittest.main()
