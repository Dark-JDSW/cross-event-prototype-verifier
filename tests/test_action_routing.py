import unittest

import numpy as np

from cross_event_verifier import (
    ActionQuality,
    ActionRouter,
    ActionType,
    AutomationPolicy,
    CrossEventVerifier,
    FeatureBundle,
    Observation,
    TrackQuality,
    conservative_walk_prediction,
)
from cross_event_verifier.appearance_first import AppearanceFirstGaitEnrollmentController
from cross_event_verifier.automation import AutomationStage


def _quality() -> TrackQuality:
    return TrackQuality(
        detection_confidence=0.95,
        box_height=220,
        keypoint_visibility=0.95,
        leg_visibility=0.95,
        gait_branch_quality=0.95,
        contour_area=5000.0,
        frame_count=30,
        valid_pose_frames=30,
        valid_leg_frames=30,
        gait_cycles=2.0,
        walking_ratio=0.90,
        occlusion=0.02,
    )


def _observation(action_type: str | None) -> Observation:
    metadata = {}
    if action_type is not None:
        metadata = {
            "action_type": action_type,
            "action_confidence": 0.95,
            "action_quality": "STRONG",
            "action_completion": 1.0,
            "action_source": "test-classifier",
            "action_model_version": "test-action-v1",
        }
    return Observation(
        camera_id="camera-a",
        capture_session_id="session-a",
        track_id="track-a",
        features=FeatureBundle(
            appearance=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            gait=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        ),
        quality=_quality(),
        metadata=metadata,
    )


class ActionRoutingTests(unittest.TestCase):
    def test_explicit_prediction_is_conservative(self) -> None:
        router = ActionRouter()
        self.assertIsNone(router.from_metadata({}))
        squat = router.from_metadata(
            {
                "action_type": "squat",
                "action_confidence": "0.95",
                "action_quality": "STRONG",
                "action_completion": 1.0,
            }
        )
        self.assertIsNotNone(squat)
        assert squat is not None
        self.assertEqual(squat.action_type, ActionType.SQUAT)
        self.assertFalse(router.allows_walk(squat))
        self.assertEqual(
            router.quarantine_reason(squat),
            "action_squat_routed_to_quarantine",
        )

    def test_conservative_heuristic_only_emits_walk_when_complete(self) -> None:
        walk = conservative_walk_prediction(
            walking_ratio=0.9,
            gait_cycles=2.0,
            valid_pose_frames=30,
            valid_leg_frames=29,
            minimum_frames=25,
        )
        unknown = conservative_walk_prediction(
            walking_ratio=0.1,
            gait_cycles=0.0,
            valid_pose_frames=30,
            valid_leg_frames=29,
            minimum_frames=25,
        )
        self.assertEqual(walk.action_type, ActionType.WALK)
        self.assertEqual(walk.quality, ActionQuality.STRONG)
        self.assertTrue(ActionRouter().allows_walk(walk))
        self.assertEqual(unknown.action_type, ActionType.UNKNOWN)
        self.assertFalse(ActionRouter().allows_walk(unknown))

    def test_non_walk_action_cannot_write_legacy_gait_bank(self) -> None:
        verifier = CrossEventVerifier()
        controller = AppearanceFirstGaitEnrollmentController(
            verifier,
            AutomationPolicy(
                minimum_track_frames=1,
                minimum_stable_gait_samples=2,
                gait_sample_window=2,
            ),
        )
        controller.register_visual_identity(
            "P1",
            _observation(None),
            state_key="camera-a:session-a:track-a",
        )
        try:
            decision, status = controller.verify(
                _observation("SQUAT"),
                candidate_id="camera-a:session-a:track-a",
            )
            self.assertEqual(status.stage, AutomationStage.ACTION_QUARANTINE)
            self.assertEqual(status.action_type, ActionType.SQUAT.value)
            self.assertIn("action_quarantine", decision.reasons)
            self.assertEqual(verifier.load_gait_enrollment_events("P1"), [])
        finally:
            verifier.close()

    def test_explicit_walk_preserves_gaitgraph_enrollment(self) -> None:
        verifier = CrossEventVerifier()
        controller = AppearanceFirstGaitEnrollmentController(
            verifier,
            AutomationPolicy(
                minimum_track_frames=1,
                minimum_stable_gait_samples=2,
                gait_sample_window=2,
                minimum_sample_similarity=0.80,
                minimum_gait_stability=0.80,
            ),
        )
        controller.register_visual_identity(
            "P1",
            _observation(None),
            state_key="camera-a:session-a:track-a",
        )
        try:
            for _ in range(2):
                _, status = controller.verify(
                    _observation("WALK"),
                    candidate_id="camera-a:session-a:track-a",
                )
            self.assertNotEqual(status.stage, AutomationStage.ACTION_QUARANTINE)
            events = verifier.load_gait_enrollment_events("P1")
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].action_type, ActionType.WALK.value)
        finally:
            verifier.close()

    def test_appearance_prototype_absorption_is_not_gated_by_walk(self) -> None:
        verifier = CrossEventVerifier()
        verifier.register_identity(
            "P1",
            FeatureBundle(appearance=np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        )
        try:
            decision = verifier.enroll_appearance_prototype(
                "P1",
                _observation("SQUAT"),
                stability=1.0,
                sample_count=1,
            )
            self.assertEqual(decision.identity_id, "P1")
            self.assertIn("appearance_prototype_absorbed", decision.reasons)
        finally:
            verifier.close()

    def test_gait_prototype_api_keeps_the_walk_gate(self) -> None:
        verifier = CrossEventVerifier()
        verifier.register_identity(
            "P1",
            FeatureBundle(appearance=np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        )
        try:
            with self.assertRaisesRegex(
                ValueError,
                "gait prototype enrollment requires a strong WALK action prediction",
            ):
                verifier.enroll_gait_prototype(
                    "P1",
                    _observation("SQUAT"),
                    event_key="session:session-a",
                    stability=1.0,
                    sample_count=1,
                )
        finally:
            verifier.close()

    def test_non_walk_track_quarantines_after_appearance_update(self) -> None:
        verifier = CrossEventVerifier()
        controller = AppearanceFirstGaitEnrollmentController(
            verifier,
            AutomationPolicy(minimum_track_frames=1),
        )
        controller.register_visual_identity(
            "P1",
            _observation(None),
            state_key="camera-a:session-a:track-a",
        )
        try:
            statuses = []
            for _ in range(verifier.config.appearance_identity_min_samples):
                _, status = controller.verify(
                    _observation("SQUAT"),
                    candidate_id="camera-a:session-a:track-a",
                )
                statuses.append(status)
            self.assertEqual(statuses[-1].stage, AutomationStage.ACTION_QUARANTINE)
            self.assertEqual(verifier.load_gait_enrollment_events("P1"), [])
        finally:
            verifier.close()


if __name__ == "__main__":
    unittest.main()
