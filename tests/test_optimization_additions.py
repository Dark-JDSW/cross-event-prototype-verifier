import unittest
from unittest.mock import patch

import numpy as np

from cross_event_verifier import (
    AutomationPolicy,
    CrossEventVerifier,
    Decision,
    DecisionKind,
    FeatureBundle,
    VerificationState,
    evaluate_open_set_protocol,
)
from cross_event_verifier.aggregation import weighted_unit_mean
from cross_event_verifier.automation import (
    AutomationStage,
    AutomationStatus,
    AutomaticVerificationController,
    _TrackAutomationState,
)
from cross_event_verifier.appearance_first import AppearanceFirstGaitEnrollmentController
from cross_event_verifier.pipeline import VideoVerifierPipeline
from cross_event_verifier.storage import SqliteStore
from cross_event_verifier.types import (
    CleanSubTracklet,
    GaitEnrollmentEvent,
    Observation,
    TrackQuality,
)
from cross_event_verifier.vision import VisionTrack


class _Vision:
    supports_automatic_registration = True

    def process(self, _frame):
        return ()


def _event(vector, *, event_key="event-1", event_id="event-1"):
    return GaitEnrollmentEvent(
        identity_id="P1",
        event_key=event_key,
        event_id=event_id,
        camera_id="camera-a",
        capture_session_id="session-a",
        track_id="track-a",
        vector=np.asarray(vector, dtype=np.float32),
        stability=0.95,
        quality=0.90,
        sample_count=8,
    )


class OptimizationAdditionTests(unittest.TestCase):
    def test_clean_subtracklet_is_propagated_to_pipeline_observation(self):
        class Vision:
            def process(self, _frame):
                return (
                    VisionTrack(
                        track_id=7,
                        box=(10, 10, 80, 180),
                        detection_confidence=0.95,
                        features=FeatureBundle(gait=np.array([1.0, 0.0])),
                        quality=TrackQuality(
                            detection_confidence=0.95,
                            box_height=170,
                            frame_count=25,
                            valid_pose_frames=25,
                            valid_leg_frames=25,
                            gait_cycles=1.0,
                            walking_ratio=0.8,
                            gait_branch_quality=0.9,
                        ),
                        clean_subtracklet=CleanSubTracklet(
                            source_track_id=7,
                            segment_id=2,
                            frame_start=20,
                            frame_end=44,
                            frame_count=25,
                            valid_pose_frames=25,
                            valid_leg_frames=25,
                            boundary_reasons=("track_gap",),
                        ),
                    ),
                )

            def reset(self):
                pass

        verifier = CrossEventVerifier()
        pipeline = VideoVerifierPipeline(verifier, Vision(), appearance_first=True)
        captured: list[Observation] = []

        def capture(observations, *, candidate_ids):
            captured.extend(observations)
            return tuple(
                (
                    Decision(
                        kind=DecisionKind.UNKNOWN,
                        state=VerificationState.UNKNOWN,
                    ),
                    AutomationStatus(AutomationStage.DISABLED, "test"),
                )
                for _ in candidate_ids
            )

        try:
            with patch.object(pipeline.automation, "verify_batch", side_effect=capture):
                pipeline.process_frame(np.zeros((200, 100, 3), dtype=np.uint8))
            self.assertEqual(captured[0].metadata["subtracklet_id"], "7:2")
            self.assertEqual(
                captured[0].metadata["subtracklet_boundary_reasons"],
                ("track_gap",),
            )
        finally:
            verifier.close()

    def test_appearance_first_drops_unfinished_window_at_subtracklet_boundary(self):
        verifier = CrossEventVerifier()
        verifier.register_identity("P1", FeatureBundle(appearance=np.array([1.0, 0.0])))
        controller = AppearanceFirstGaitEnrollmentController(
            verifier,
            AutomationPolicy(
                minimum_track_frames=1,
                minimum_stable_gait_samples=3,
                gait_sample_window=3,
            ),
        )
        state = controller._state("candidate")
        state.identity_id = "P1"
        state.subtracklet_id = "7:0"
        state.appearance_samples.append(np.array([1.0, 0.0]))
        state.gait_samples.append(np.array([1.0, 0.0]))
        state.gait_sample_weights.append(1.0)
        state.gait_sample_qualities.append(0.9)
        try:
            controller.verify(
                Observation(
                    camera_id="cam-a",
                    capture_session_id="session-a",
                    track_id="7",
                    timestamp=2.0,
                    features=FeatureBundle(gait=np.array([1.0, 0.0])),
                    quality=TrackQuality(
                        detection_confidence=0.95,
                        box_height=200,
                        frame_count=25,
                        valid_pose_frames=25,
                        valid_leg_frames=25,
                        gait_cycles=1.0,
                        walking_ratio=0.8,
                        gait_branch_quality=0.9,
                    ),
                    metadata={"subtracklet_id": "7:1"},
                ),
                candidate_id="candidate",
            )
            self.assertEqual(len(state.appearance_samples), 0)
            self.assertEqual(len(state.gait_samples), 1)
            self.assertEqual(state.subtracklet_id, "7:1")
        finally:
            verifier.close()

    def test_legacy_automation_drops_unfinished_window_at_subtracklet_boundary(self):
        verifier = CrossEventVerifier()
        controller = AutomaticVerificationController(verifier)
        state = controller._states.setdefault("candidate", _TrackAutomationState())
        state.subtracklet_id = "7:0"
        state.gait_samples.append(np.array([1.0, 0.0]))
        state.gait_sample_weights.append(1.0)
        state.gait_window_start_timestamp = 1.0
        try:
            controller._sync_subtracklet(
                state,
                Observation(metadata={"subtracklet_id": "7:1"}),
            )
            self.assertEqual(len(state.gait_samples), 0)
            self.assertEqual(len(state.gait_sample_weights), 0)
            self.assertIsNone(state.gait_window_start_timestamp)
            self.assertEqual(state.subtracklet_id, "7:1")
        finally:
            verifier.close()

    def test_quality_weighted_centroid_favours_strong_sample(self):
        centroid = weighted_unit_mean(
            ([1.0, 0.0], [0.0, 1.0]),
            (1.0, 0.1),
        )
        assert centroid is not None
        self.assertGreater(float(centroid[0]), float(centroid[1]))

    def test_quality_weighted_window_enrolls_using_aggregate_quality(self):
        verifier = CrossEventVerifier()
        verifier.register_identity(
            'P1',
            FeatureBundle(
                appearance=np.array([1.0, 0.0]),
                gait=np.array([1.0, 0.0]),
            ),
        )
        controller = AppearanceFirstGaitEnrollmentController(
            verifier,
            AutomationPolicy(minimum_stable_gait_samples=3, gait_sample_window=3),
        )
        state = controller._state('P1')
        state.identity_id = 'P1'
        statuses = []
        try:
            for index, branch_quality in enumerate((0.90, 0.90, 0.60)):
                quality = TrackQuality(
                    detection_confidence=0.95,
                    box_height=200,
                    frame_count=25,
                    valid_pose_frames=25,
                    valid_leg_frames=25,
                    gait_cycles=1.0,
                    gait_branch_quality=branch_quality,
                    leg_visibility=0.9,
                    walking_ratio=0.8,
                )
                _, status = controller.verify(
                    Observation(
                        event_id=f'event-{index}',
                        camera_id='cam-a',
                        capture_session_id='session-a',
                        track_id='track-1',
                        timestamp=float(index),
                        features=FeatureBundle(gait=np.array([1.0, 0.0])),
                        quality=quality,
                    ),
                    candidate_id='P1',
                )
                statuses.append(status)
            events = verifier.load_gait_enrollment_events('P1')
            self.assertEqual(len(events), 1)
            self.assertGreaterEqual(events[0].quality, verifier.config.strong_gait_quality)
            self.assertEqual(statuses[-1].gait_quality_band, 'strong')
        finally:
            verifier.close()

    def test_quality_gate_failure_is_recoverable(self):
        verifier = CrossEventVerifier()
        verifier.register_identity(
            "P1",
            FeatureBundle(appearance=np.array([1.0, 0.0])),
        )
        controller = AppearanceFirstGaitEnrollmentController(
            verifier,
            AutomationPolicy(minimum_stable_gait_samples=3, gait_sample_window=3),
        )
        state = controller._state("P1")
        state.identity_id = "P1"
        try:
            with patch.object(
                verifier,
                "enroll_gait_prototype",
                side_effect=ValueError(
                    "gait prototype enrollment requires STRONG gait quality: partial"
                ),
            ):
                for index in range(3):
                    _, status = controller.verify(
                        Observation(
                            event_id=f"deferred-{index}",
                            camera_id="cam-a",
                            capture_session_id="session-a",
                            track_id="track-1",
                            timestamp=float(index),
                            features=FeatureBundle(gait=np.array([1.0, 0.0])),
                            quality=TrackQuality(
                                detection_confidence=0.95,
                                box_height=200,
                                frame_count=25,
                                valid_pose_frames=25,
                                valid_leg_frames=25,
                                gait_cycles=1.0,
                                gait_branch_quality=0.9,
                                leg_visibility=0.9,
                                walking_ratio=0.8,
                            ),
                        ),
                        candidate_id="P1",
                    )
            self.assertEqual(verifier.load_gait_enrollment_events("P1"), [])
            self.assertIn("暂缓写入", status.message)
        finally:
            verifier.close()

    def test_open_set_protocol_reports_fnir_and_fpir(self):
        report = evaluate_open_set_protocol(
            {"P1": ([1.0, 0.0],), "P2": ([0.0, 1.0],)},
            {"P1": ([1.0, 0.0],), "P2": ([0.0, 1.0],)},
            ([1.0, 1.0],),
            target_fpir=0.0,
        )
        self.assertEqual(report.known_correct_count, 2)
        self.assertEqual(report.fnir, 0.0)
        self.assertEqual(report.fpir, 0.0)

    def test_event_revisions_and_derived_model_are_persisted(self):
        store = SqliteStore(":memory:")
        first = _event([1.0, 0.0])
        second = _event([0.9, 0.1], event_id="event-2")
        self.assertTrue(store.save_gait_enrollment_event(first))
        self.assertTrue(store.save_gait_enrollment_event(second, replace_existing=True))
        revisions = store.load_gait_enrollment_event_revisions("P1", "event-1")
        self.assertEqual([item["revision"] for item in revisions], [1, 2])
        model_id = store.save_derived_gait_model(
            identity_id="P1",
            vector=[1.0, 0.0],
            quality=0.9,
            support_count=1,
            source_event_ids=["event-2"],
            contract_event=second,
        )
        self.assertEqual(store.load_derived_gait_models("P1")[0]["model_id"], model_id)

    def test_decision_exposes_open_set_label(self):
        decision = Decision(
            kind=DecisionKind.DEFERRED,
            state=VerificationState.ISOLATED_CANDIDATE,
            reasons=("margin",),
        )
        self.assertEqual(decision.open_set_label, "AMBIGUOUS")


if __name__ == "__main__":
    unittest.main()
