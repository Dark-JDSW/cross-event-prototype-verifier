from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from cross_event_verifier import (
    AutomationPolicy,
    CrossEventVerifier,
    DecisionKind,
    FeatureBundle,
    GaitReadinessState,
    Observation,
    TrackQuality,
    VerifierConfig,
)
from cross_event_verifier.appearance_first import (
    AppearanceFirstGaitEnrollmentController,
)


def _quality() -> TrackQuality:
    return TrackQuality(
        detection_confidence=0.95,
        box_height=220,
        sharpness=1.0,
        keypoint_visibility=0.95,
        leg_visibility=0.95,
        gait_branch_quality=0.95,
        contour_area=5000.0,
        frame_count=30,
        valid_pose_frames=30,
        valid_leg_frames=30,
        gait_cycles=2.0,
        walking_ratio=0.95,
        occlusion=0.02,
    )


def _observation(session: str, gait: np.ndarray, timestamp: float) -> Observation:
    return Observation(
        event_id=f"{session}-{timestamp}",
        camera_id="cam-a",
        capture_session_id=session,
        track_id="7",
        timestamp=timestamp,
        features=FeatureBundle(
            appearance=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            gait=gait,
        ),
        quality=_quality(),
    )


def _appearance_only_observation(
    appearance: np.ndarray,
    timestamp: float,
    *,
    gait: np.ndarray | None = None,
    valid_pose_frames: int | None = None,
) -> Observation:
    quality = _quality()
    if valid_pose_frames is not None:
        quality = replace(quality, valid_pose_frames=valid_pose_frames)
    return Observation(
        event_id=f"appearance-{timestamp}",
        camera_id="cam-a",
        capture_session_id="appearance-session",
        track_id="7",
        timestamp=timestamp,
        features=FeatureBundle(appearance=appearance, gait=gait),
        quality=quality,
    )


class AppearanceFirstTests(unittest.TestCase):
    def _controller(self) -> tuple[CrossEventVerifier, AppearanceFirstGaitEnrollmentController]:
        verifier = CrossEventVerifier(
            VerifierConfig(
                appearance_identity_min_samples=2,
                gait_provisional_min_events=2,
                gait_ready_min_events=3,
                gait_ready_min_coverage=1,
                gait_event_min_similarity=0.70,
                gait_holdout_min_similarity=0.70,
            )
        )
        controller = AppearanceFirstGaitEnrollmentController(
            verifier,
            AutomationPolicy(
                minimum_track_frames=1,
                minimum_stable_gait_samples=2,
                gait_sample_window=3,
                minimum_sample_similarity=0.80,
                minimum_gait_stability=0.80,
            ),
        )
        return verifier, controller

    def test_osnet_creates_visual_identity_before_gait_event(self) -> None:
        verifier, controller = self._controller()
        first = _observation("event-a", np.array([0.0, 1.0, 0.0]), 1.0)
        second = _observation("event-a", np.array([0.0, 1.0, 0.0]), 2.0)
        third = _observation("event-a", np.array([0.0, 1.0, 0.0]), 3.0)

        decision1, status1 = controller.verify(first, candidate_id="cam:event-a:track-7")
        self.assertIsNone(decision1.identity_id)
        self.assertEqual(status1.readiness_state, None)

        decision2, status2 = controller.verify(second, candidate_id="cam:event-a:track-7")
        self.assertEqual(decision2.identity_id, "P1")
        self.assertEqual(decision2.kind.value, "visual_identity_created")
        self.assertEqual(status2.stage.value, "visual_confirmed")
        self.assertEqual(verifier.memory.formal_prototypes("P1", "appearance")[0].modality, "appearance")
        self.assertEqual(verifier.memory.formal_prototypes("P1", "gait"), ())

        decision3, status3 = controller.verify(third, candidate_id="cam:event-a:track-7")
        self.assertEqual(decision3.identity_id, "P1")
        self.assertEqual(len(verifier.load_gait_enrollment_events("P1")), 1)
        self.assertEqual(status3.gait_event_count, 1)
        self.assertEqual(status3.readiness_state, GaitReadinessState.LEARNING.value)

        # Same session's later windows cannot produce a second independent event.
        _, repeated = controller.verify(
            _observation("event-a", np.array([0.0, 1.0, 0.0]), 4.0),
            candidate_id="cam:event-a:track-7",
        )
        self.assertEqual(repeated.gait_event_count, 1)
        verifier.close()

    def test_second_independent_event_becomes_provisional(self) -> None:
        verifier, controller = self._controller()
        key_a = "cam:event-a:track-7"
        for timestamp in (1.0, 2.0, 3.0):
            controller.verify(
                _observation("event-a", np.array([0.0, 1.0, 0.0]), timestamp),
                candidate_id=key_a,
            )
        controller.reset_tracks(discard_unpreserved=True)
        key_b = "cam:event-b:track-7"
        for timestamp in (4.0, 5.0, 6.0):
            decision, status = controller.verify(
                _observation("event-b", np.array([0.0, 0.98, 0.20]), timestamp),
                candidate_id=key_b,
            )
        self.assertEqual(decision.identity_id, "P1")
        self.assertEqual(status.stage.value, "gait_provisional")
        self.assertEqual(status.gait_event_count, 2)
        report = verifier.evaluate_gait_readiness("P1")
        self.assertEqual(report.state, GaitReadinessState.PROVISIONAL)
        self.assertEqual(report.independent_session_count, 2)
        verifier.close()

    def test_same_session_continues_learning_samples_after_event_counted(self) -> None:
        """同一事件不重复计数，但后续窗口仍应更新该事件的步态原型。"""

        verifier, controller = self._controller()
        key = "cam:event-a:track-7"
        for timestamp in (1.0, 2.0, 3.0):
            controller.verify(
                _observation("event-a", np.array([0.0, 1.0, 0.0]), timestamp),
                candidate_id=key,
            )
        before = verifier.load_gait_enrollment_events("P1")[0]

        statuses = []
        for timestamp in (4.0, 5.0, 6.0):
            _, status = controller.verify(
                _observation("event-a", np.array([0.25, 0.9682458, 0.0]), timestamp),
                candidate_id=key,
            )
            statuses.append(status)

        after = verifier.load_gait_enrollment_events("P1")[0]
        self.assertEqual(len(verifier.load_gait_enrollment_events("P1")), 1)
        self.assertGreater(after.sample_count, before.sample_count)
        self.assertGreater(float(np.dot(after.vector, before.vector)), 0.99)
        self.assertLess(float(np.dot(after.vector, np.array([0.25, 0.9682458, 0.0]))), 1.0)
        self.assertTrue(
            any("继续吸收本会话步态样本" in status.message for status in statuses)
        )
        verifier.close()

    def test_short_gait_cannot_reidentify_a_visual_track(self) -> None:
        """短步态即使与正式原型相同，也不能越权绑定视觉身份。"""

        verifier = CrossEventVerifier()
        verifier.register_identity(
            "P1",
            FeatureBundle(
                appearance=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                gait=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            ),
        )
        controller = AppearanceFirstGaitEnrollmentController(
            verifier,
            AutomationPolicy(minimum_track_frames=1),
        )
        decision, _ = controller.verify(
            _appearance_only_observation(
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                1.0,
                gait=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                valid_pose_frames=25,
            ),
            candidate_id="cam:appearance-session:track-7",
        )
        self.assertIsNone(decision.identity_id)
        self.assertNotIn("gait_reidentified_visual_identity", decision.reasons)
        verifier.close()

    def test_gait_identity_quality_gate_rejects_resampled_short_window(self) -> None:
        """25 个真实帧不能因为重采样到 60 帧而成为强步态查询。"""

        verifier = CrossEventVerifier()
        verifier.register_identity(
            "P1",
            FeatureBundle(gait=np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        )
        decision = verifier.match_gait_identity(
            _appearance_only_observation(
                np.zeros(3, dtype=np.float32),
                1.0,
                gait=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                valid_pose_frames=25,
            ),
            require_identity_quality=True,
        )
        self.assertNotEqual(decision.kind, DecisionKind.FORMAL_MATCH)
        self.assertIn("gait_identity_sequence_immature", decision.reasons)
        verifier.close()

    def test_strong_appearance_conflict_vetoes_bound_gait(self) -> None:
        """高质量外观反对已绑定身份时，步态不得覆盖该冲突。"""

        verifier = CrossEventVerifier()
        verifier.register_identity(
            "P1",
            FeatureBundle(
                appearance=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                gait=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            ),
        )
        controller = AppearanceFirstGaitEnrollmentController(verifier)
        state_key = "cam:appearance-session:track-7"
        controller.register_visual_identity(
            "P1",
            _appearance_only_observation(
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                1.0,
            ),
            state_key=state_key,
        )
        decision, _ = controller.verify(
            _appearance_only_observation(
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                2.0,
                gait=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                valid_pose_frames=45,
            ),
            candidate_id=state_key,
        )
        self.assertEqual(decision.kind, DecisionKind.CONFLICT)
        self.assertIn("appearance_conflicts_with_bound_identity", decision.reasons)
        self.assertIsNone(decision.identity_id)
        verifier.close()

    def test_bound_identity_absorbs_a_new_view_as_an_appearance_prototype(self) -> None:
        """连续 Track 的新视角应扩展 P1，而不是生成 P2。"""

        verifier = CrossEventVerifier()
        verifier.register_identity(
            "P1",
            FeatureBundle(appearance=np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        )
        controller = AppearanceFirstGaitEnrollmentController(
            verifier,
            AutomationPolicy(
                minimum_track_frames=1,
                gait_sample_window=8,
            ),
        )
        state_key = "cam:appearance-session:track-7"
        controller.register_visual_identity(
            "P1",
            _appearance_only_observation(
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                1.0,
            ),
            state_key=state_key,
        )
        side_view = np.array([0.5, 0.8660254, 0.0], dtype=np.float32)
        for timestamp in range(2, 10):
            decision, _ = controller.verify(
                _appearance_only_observation(side_view, float(timestamp)),
                candidate_id=state_key,
            )
        self.assertEqual(decision.identity_id, "P1")
        self.assertEqual(verifier.formal_identities, ("P1",))
        appearance_prototypes = verifier.memory.formal_prototypes("P1", "appearance")
        self.assertGreaterEqual(len(appearance_prototypes), 2)
        verifier.close()

    def test_simultaneous_bound_tracks_cannot_share_an_identity(self) -> None:
        """两个并行可靠 Track 不能依靠缓存身份共享一个 P。"""

        verifier = CrossEventVerifier()
        verifier.register_identity(
            "P1",
            FeatureBundle(appearance=np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        )
        controller = AppearanceFirstGaitEnrollmentController(verifier)
        for track_id in ("track-7", "track-8"):
            controller.register_visual_identity(
                "P1",
                _appearance_only_observation(
                    np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    1.0,
                ),
                state_key=f"cam:appearance-session:{track_id}",
            )
        results = controller.verify_batch(
            [
                _appearance_only_observation(np.zeros(3, dtype=np.float32), 2.0),
                _appearance_only_observation(np.zeros(3, dtype=np.float32), 2.0),
            ],
            candidate_ids=[
                "cam:appearance-session:track-7",
                "cam:appearance-session:track-8",
            ],
        )
        self.assertTrue(all(item[0].kind == DecisionKind.CONFLICT for item in results))
        self.assertTrue(all(item[0].identity_id is None for item in results))
        verifier.close()


if __name__ == "__main__":
    unittest.main()
