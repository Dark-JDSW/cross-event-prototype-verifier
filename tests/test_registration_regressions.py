"""回归测试：开放集注册不能把疑似匹配当成已知身份。"""

import unittest
from dataclasses import replace
import tempfile
from pathlib import Path

import numpy as np

from cross_event_verifier import CrossEventVerifier, FeatureBundle, Observation
from cross_event_verifier.config import VerifierConfig
from cross_event_verifier.automation import (
    AutomationPolicy,
    AutomationStage,
    AutomaticVerificationController,
)
from cross_event_verifier.types import DecisionKind, TrackQuality
from cross_event_verifier.storage import SqliteStore


def good_quality(*, gait_branch_quality: float = 0.95) -> TrackQuality:
    return TrackQuality(
        detection_confidence=0.95,
        box_height=200,
        sharpness=0.95,
        occlusion=0.02,
        contour_area=2400,
        frame_count=30,
        gait_cycles=2,
        walking_ratio=0.95,
        gait_branch_quality=gait_branch_quality,
        leg_visibility=0.90,
    )


class RegistrationRegressionTests(unittest.TestCase):
    def test_deferred_gait_does_not_claim_existing_identity(self) -> None:
        verifier = CrossEventVerifier()
        verifier.register_identity(
            "P1",
            FeatureBundle(gait=np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        )
        # 相似度足够高会进入旧实现的 deferred，但没有达到强步态确认的
        # Top-2/开放集语义；它不能直接对外宣称 P1。
        gait = np.array([0.5, np.sqrt(0.75), 0.0], dtype=np.float32)
        decision = verifier.verify(
            Observation(
                event_id="deferred-unknown",
                camera_id="cam-a",
                capture_session_id="session-a",
                track_id="track-1",
                features=FeatureBundle(gait=gait),
                quality=good_quality(),
            ),
            candidate_id="candidate-1",
        )
        self.assertIsNone(decision.identity_id)
        verifier.close()

    def test_low_quality_branch_disagreement_is_not_hard_conflict(self) -> None:
        verifier = CrossEventVerifier()
        verifier.register_identity(
            "P1",
            FeatureBundle(
                appearance=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                gait=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            ),
        )
        verifier.register_identity(
            "P2",
            FeatureBundle(
                appearance=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                gait=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            ),
        )
        decision = verifier.verify(
            Observation(
                event_id="weak-disagreement",
                camera_id="cam-a",
                capture_session_id="session-a",
                track_id="track-2",
                features=FeatureBundle(
                    appearance=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    gait=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                ),
                quality=good_quality(gait_branch_quality=0.45),
            ),
            candidate_id="candidate-2",
        )
        self.assertNotIn("appearance_gait_conflict", decision.reasons)
        self.assertNotEqual(decision.state.value, "suspended")
        verifier.close()

    def test_enrollment_rejects_gait_that_is_not_open_set_novel(self) -> None:
        verifier = CrossEventVerifier(VerifierConfig())
        verifier.register_identity(
            "P1",
            FeatureBundle(gait=np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        )
        gait = np.array([0.5, np.sqrt(0.75), 0.0], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "not open-set novel"):
            verifier.enroll_gait_identity(
                Observation(
                    event_id="unsafe-enrollment",
                    camera_id="cam-a",
                    capture_session_id="session-a",
                    track_id="track-3",
                    features=FeatureBundle(gait=gait),
                    quality=good_quality(),
                ),
                candidate_id="candidate-3",
                gait_confidence=0.95,
            )
        self.assertEqual(verifier.formal_identities, ("P1",))
        verifier.close()

    def test_multi_gallery_rejects_high_absolute_impostor_even_with_margin(self) -> None:
        verifier = CrossEventVerifier(VerifierConfig())
        verifier.register_identity("P1", FeatureBundle(gait=[1.0, 0.0, 0.0]))
        verifier.register_identity("P2", FeatureBundle(gait=[0.0, 1.0, 0.0]))
        query = np.array([0.98, np.sqrt(1.0 - 0.98**2), 0.0], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "not open-set novel"):
            verifier.enroll_gait_identity(
                Observation(
                    event_id="clear-winner-enrollment",
                    camera_id="cam-a",
                    capture_session_id="session-a",
                    track_id="track-clear-winner",
                    features=FeatureBundle(gait=query),
                    quality=good_quality(),
                ),
                candidate_id="candidate-clear-winner",
                gait_confidence=0.95,
            )

        self.assertEqual(verifier.formal_identities, ("P1", "P2"))
        rejected = [
            item
            for item in verifier.store.audit_log()
            if item["action"] == "gait_open_set_rejected"
        ]
        self.assertEqual(
            rejected[-1]["payload"]["reason"],
            "gait_open_set_similarity_too_high",
        )
        verifier.close()

    def test_multi_gallery_ambiguous_gait_does_not_enroll(self) -> None:
        verifier = CrossEventVerifier(VerifierConfig())
        verifier.register_identity("P1", FeatureBundle(gait=[1.0, 0.0, 0.0]))
        verifier.register_identity("P2", FeatureBundle(gait=[0.0, 1.0, 0.0]))
        query = np.array([1.0, 1.0, 0.0], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "not open-set novel"):
            verifier.enroll_gait_identity(
                Observation(
                    event_id="ambiguous-enrollment",
                    camera_id="cam-a",
                    capture_session_id="session-a",
                    track_id="track-ambiguous",
                    features=FeatureBundle(gait=query),
                    quality=good_quality(),
                ),
                candidate_id="candidate-ambiguous",
                gait_confidence=0.95,
            )

        self.assertEqual(verifier.formal_identities, ("P1", "P2"))
        rejected = [
            item
            for item in verifier.store.audit_log()
            if item["action"] == "gait_open_set_rejected"
        ]
        self.assertEqual(rejected[-1]["payload"]["reason"], "gait_top2_ambiguous")
        verifier.close()

    def test_multi_gallery_ambiguous_gait_is_explicitly_marked(self) -> None:
        verifier = CrossEventVerifier(VerifierConfig())
        verifier.register_identity("P1", FeatureBundle(gait=[1.0, 0.0, 0.0]))
        verifier.register_identity("P2", FeatureBundle(gait=[0.0, 1.0, 0.0]))
        decision = verifier.verify(
            Observation(
                event_id="ambiguous-decision",
                camera_id="cam-a",
                capture_session_id="session-a",
                track_id="track-ambiguous-decision",
                features=FeatureBundle(gait=[1.0, 1.0, 0.0]),
                quality=good_quality(),
            ),
            candidate_id="candidate-ambiguous-decision",
        )

        self.assertEqual(decision.kind, DecisionKind.AMBIGUOUS)
        self.assertIn("gait_top2_ambiguous", decision.reasons)
        self.assertIsNone(decision.identity_id)
        verifier.close()

    def test_automatic_path_does_not_absorb_a_single_gallery_identity(self) -> None:
        verifier = CrossEventVerifier()
        verifier.register_identity(
            "P1",
            FeatureBundle(gait=np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        )
        controller = AutomaticVerificationController(
            verifier,
            AutomationPolicy(
                minimum_track_frames=1,
                minimum_stable_gait_samples=3,
                gait_sample_window=4,
                minimum_independent_gait_events=1,
            ),
        )
        gait = np.array([0.5, np.sqrt(0.75), 0.0], dtype=np.float32)
        decisions = []
        status = None
        for index in range(3):
            decision, status = controller.verify(
                Observation(
                    event_id=f"auto-deferred-{index}",
                    camera_id="cam-a",
                    capture_session_id="session-a",
                    track_id="track-4",
                    features=FeatureBundle(gait=gait),
                    quality=good_quality(),
                ),
                candidate_id="candidate-4",
            )
            decisions.append(decision)
        assert status is not None
        self.assertTrue(all(item.identity_id is None for item in decisions))
        self.assertEqual(verifier.formal_identities, ("P1",))
        # Open-set novelty failure is recoverable: the controller clears the
        # stale window and starts a fresh collection instead of terminating the
        # candidate in BLOCKED.
        self.assertEqual(status.stage, AutomationStage.GAIT_UNSTABLE)
        self.assertIn("缺少可靠外观负证据", status.message)
        self.assertEqual(
            len(controller._states["candidate-4"].gait_samples), 0
        )
        verifier.close()

    def test_single_gallery_can_enroll_clear_appearance_negative(self) -> None:
        verifier = CrossEventVerifier()
        verifier.register_identity(
            "P1",
            FeatureBundle(
                appearance=[1, 0, 0],
                gait=[0, 1, 0],
            ),
        )
        controller = AutomaticVerificationController(
            verifier,
            AutomationPolicy(
                minimum_track_frames=1,
                minimum_stable_gait_samples=3,
                gait_sample_window=4,
            ),
        )
        # The gait is not a near-duplicate of P1 and the strong appearance
        # branch clearly rejects P1. This is the controlled bootstrap path for
        # adding P2 when the gallery has only one gait identity.
        gait = np.array([0.5, np.sqrt(0.75), 0.0], dtype=np.float32)
        results = []
        status = None
        for index in range(3):
            result, status = controller.verify(
                Observation(
                    event_id=f"auto-negative-{index}",
                    camera_id="cam-a",
                    capture_session_id="session-a",
                    track_id="track-6",
                    features=FeatureBundle(
                        appearance=[-1, 0, 0],
                        gait=gait,
                    ),
                    quality=good_quality(),
                ),
                candidate_id="candidate-6",
            )
            results.append(result)
        assert status is not None
        self.assertEqual(verifier.formal_identities, ("P1",))
        self.assertEqual(status.stage, AutomationStage.WAITING_INDEPENDENT_EVENT)
        self.assertFalse(status.auto_registered)
        self.assertIsNone(results[-1].identity_id)

        for index in range(3, 6):
            result, status = controller.verify(
                Observation(
                    event_id=f"auto-negative-independent-{index}",
                    camera_id="cam-a",
                    capture_session_id="session-b",
                    track_id="track-6",
                    features=FeatureBundle(
                        appearance=[-1, 0, 0],
                        gait=gait,
                    ),
                    quality=good_quality(),
                ),
                candidate_id="candidate-6",
            )
            results.append(result)
        assert status is not None
        self.assertEqual(verifier.formal_identities, ("P1", "P2"))
        self.assertEqual(status.stage, AutomationStage.APPEARANCE_PENDING)
        self.assertTrue(status.auto_registered)
        self.assertEqual(results[-1].identity_id, "P2")
        verifier.close()

    def test_automatic_event_proposals_survive_verifier_restart(self) -> None:
        policy = AutomationPolicy(
            minimum_track_frames=1,
            minimum_stable_gait_samples=3,
            gait_sample_window=4,
            minimum_independent_gait_events=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "registration.sqlite3")
            first_verifier = CrossEventVerifier(store=SqliteStore(path))
            first_controller = AutomaticVerificationController(first_verifier, policy)
            for index in range(3):
                first_controller.verify(
                    Observation(
                        event_id=f"restart-a-{index}",
                        camera_id="cam-a",
                        capture_session_id="event-a",
                        track_id="track-7",
                        features=FeatureBundle(gait=[0, 1, 0]),
                        quality=good_quality(),
                    ),
                    candidate_id="enrollment-restart",
                )
            self.assertEqual(first_verifier.formal_identities, ())
            first_verifier.close()

            second_verifier = CrossEventVerifier(store=SqliteStore(path))
            second_controller = AutomaticVerificationController(second_verifier, policy)
            result = None
            for index in range(3):
                result, status = second_controller.verify(
                    Observation(
                        event_id=f"restart-b-{index}",
                        camera_id="cam-a",
                        capture_session_id="event-b",
                        track_id="track-7",
                        features=FeatureBundle(gait=[0, 1, 0]),
                        quality=good_quality(),
                    ),
                    candidate_id="enrollment-restart",
                )
            self.assertIsNotNone(result)
            self.assertEqual(result.identity_id, "P1")
            self.assertTrue(status.auto_registered)
            self.assertEqual(second_verifier.formal_identities, ("P1",))
            self.assertEqual(
                second_verifier.store.load_gait_event_proposals(
                    "enrollment-restart"
                ),
                [],
            )
            second_verifier.close()

    def test_restart_discards_event_proposals_from_an_old_model_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "contract.sqlite3")
            first = CrossEventVerifier(
                store=SqliteStore(path),
                config=VerifierConfig(
                    model_version="model-a",
                    feature_schema="schema-a",
                ),
            )
            observation = Observation(
                event_id="old-contract-event",
                camera_id="cam-a",
                capture_session_id="session-a",
                track_id="track-1",
                timestamp=100.0,
                features=FeatureBundle(gait=[0.0, 1.0, 0.0]),
                quality=good_quality(),
                model_version="model-a",
                feature_schema="schema-a",
            )
            first.save_gait_event_proposal(
                candidate_id="contract-candidate",
                event_key="session:session-a",
                vector=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                stability=0.99,
                observation=observation,
            )
            first.close()

            second = CrossEventVerifier(
                store=SqliteStore(path),
                config=VerifierConfig(
                    model_version="model-b",
                    feature_schema="schema-b",
                ),
            )
            AutomaticVerificationController(second)
            self.assertEqual(
                second.store.load_gait_event_proposals("contract-candidate"),
                [],
            )
            second.close()

    def test_automatic_registration_blocks_incompatible_gallery_contract(self) -> None:
        verifier = CrossEventVerifier(
            VerifierConfig(
                model_version="model-a",
                feature_schema="gaitgraph2-rtmpose-v1",
            )
        )
        verifier.register_identity("P1", FeatureBundle(gait=[0, 1, 0]))
        controller = AutomaticVerificationController(
            verifier,
            AutomationPolicy(
                minimum_track_frames=1,
                minimum_stable_gait_samples=3,
                gait_sample_window=4,
                minimum_independent_gait_events=1,
            ),
        )
        decision, status = controller.verify(
            Observation(
                event_id="contract-switch",
                model_version="model-b",
                feature_schema="gaitgraph2-hrnet-v2",
                features=FeatureBundle(gait=[1, 0, 0]),
                quality=good_quality(),
            ),
            candidate_id="contract-candidate",
        )
        self.assertEqual(decision.kind, DecisionKind.NEED_MORE_DATA)
        self.assertEqual(status.stage, AutomationStage.BLOCKED)
        self.assertEqual(verifier.formal_identities, ("P1",))
        verifier.close()

    def test_partial_gait_waits_without_clearing_as_identity_negative(self) -> None:
        verifier = CrossEventVerifier()
        controller = AutomaticVerificationController(
            verifier,
            AutomationPolicy(minimum_track_frames=1, minimum_stable_gait_samples=3),
        )
        partial = good_quality(gait_branch_quality=0.55)
        partial = replace(partial, walking_ratio=0.05)
        _, status = controller.verify(
            Observation(
                event_id="partial-gait",
                camera_id="cam-a",
                capture_session_id="session-partial",
                track_id="track-partial",
                features=FeatureBundle(gait=[1.0, 0.0, 0.0]),
                quality=partial,
            ),
            candidate_id="candidate-partial",
        )

        self.assertEqual(status.stage, AutomationStage.WAIT_MORE_DATA)
        self.assertIn("PARTIAL", status.message)
        verifier.close()

    def test_hard_conflict_restarts_collection_instead_of_terminating_candidate(self) -> None:
        verifier = CrossEventVerifier()
        verifier.register_identity(
            "P1",
            FeatureBundle(appearance=[1, 0, 0], gait=[0, 1, 0]),
        )
        verifier.register_identity(
            "P2",
            FeatureBundle(appearance=[0, 1, 0], gait=[0, 0, 1]),
        )
        controller = AutomaticVerificationController(
            verifier,
            AutomationPolicy(minimum_track_frames=1),
        )
        decision, status = controller.verify(
            Observation(
                event_id="hard-conflict",
                camera_id="cam-a",
                capture_session_id="session-a",
                track_id="track-5",
                features=FeatureBundle(appearance=[1, 0, 0], gait=[0, 0, 1]),
                quality=good_quality(),
            ),
            candidate_id="candidate-5",
        )
        self.assertIn("appearance_gait_conflict", decision.reasons)
        self.assertEqual(status.stage, AutomationStage.GAIT_UNSTABLE)
        candidate = verifier.get_candidate("candidate-5")
        assert candidate is not None
        self.assertNotEqual(candidate.state.value, "suspended")
        verifier.close()


if __name__ == "__main__":
    unittest.main()
