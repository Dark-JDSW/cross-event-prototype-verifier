"""证据策略、建号和持久化的端到端领域测试。"""

import unittest
import tempfile
from pathlib import Path
import os
import subprocess
import sys
import tempfile

import numpy as np

from cross_event_verifier import (
    CrossEventVerifier,
    DecisionKind,
    FeatureBundle,
    GaitQualityBand,
    Observation,
    TrackQuality,
    VerificationState,
)
from cross_event_verifier.assignment import gated_global_assignment
from cross_event_verifier.config import VerifierConfig
from cross_event_verifier.fusion import fuse_calibrated_scores
from cross_event_verifier.storage import SqliteStore


def good_quality() -> TrackQuality:
    return TrackQuality(
        detection_confidence=0.95,
        box_height=200,
        sharpness=0.95,
        occlusion=0.05,
        keypoint_visibility=0.90,
        contour_area=2500,
        frame_count=24,
        gait_cycles=2,
        walking_ratio=0.95,
    )


class CoreTests(unittest.TestCase):
    def test_verify_batch_commits_one_sqlite_transaction(self) -> None:
        store = SqliteStore(":memory:")
        verifier = CrossEventVerifier(store=store)
        statements: list[str] = []
        store.connection.set_trace_callback(statements.append)

        def observation(track_id: str) -> Observation:
            return Observation(
                event_id=f"event-{track_id}",
                camera_id="camera-a",
                capture_session_id="session-a",
                track_id=track_id,
                features=FeatureBundle(
                    appearance=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                    gait=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                ),
                quality=good_quality(),
            )

        try:
            verifier.verify_batch(
                [observation("1"), observation("2")],
                candidate_ids=["candidate-1", "candidate-2"],
            )
        finally:
            verifier.close()

        commits = [item for item in statements if item.strip().upper() == "COMMIT"]
        self.assertEqual(commits, ["COMMIT"])

    def test_partial_gait_quality_is_not_invalidated_by_low_walking_ratio(self) -> None:
        quality = TrackQuality(
            box_height=200,
            frame_count=30,
            gait_cycles=1,
            walking_ratio=0.12,
            gait_branch_quality=0.55,
            leg_visibility=0.82,
        )

        self.assertEqual(
            quality.gait_quality_band(
                minimum_frames=8,
                minimum_gait_cycles=1,
                partial_threshold=0.35,
                strong_threshold=0.70,
            ),
            GaitQualityBand.PARTIAL,
        )
        self.assertEqual(quality.gait_hard_veto_reasons(), ())

    def test_gait_quality_hard_veto_covers_track_integrity_and_missing_legs(self) -> None:
        quality = TrackQuality(
            frame_count=30,
            gait_cycles=2,
            gait_branch_quality=0.95,
            leg_visibility=0.0,
            id_switches=1,
        )

        self.assertEqual(
            quality.gait_quality_band(
                minimum_frames=8,
                minimum_gait_cycles=1,
                partial_threshold=0.35,
                strong_threshold=0.70,
            ),
            GaitQualityBand.INVALID,
        )
        self.assertIn("track_id_switch", quality.gait_hard_veto_reasons())
        self.assertIn("legs_invisible", quality.gait_hard_veto_reasons())

    def test_gait_quality_does_not_apply_walking_penalty_twice(self) -> None:
        quality = TrackQuality(
            frame_count=30,
            gait_cycles=1,
            walking_ratio=0.20,
            gait_branch_quality=0.80,
        )

        self.assertAlmostEqual(
            quality.gait_availability(minimum_frames=8, minimum_gait_cycles=1),
            0.80,
            places=6,
        )

    def test_gait_weight_is_bounded_when_both_modalities_exist(self) -> None:
        from cross_event_verifier.calibration import ScoreCalibrator

        result = fuse_calibrated_scores(
            appearance_similarity=0.70,
            gait_similarity=0.99,
            appearance_quality=0.8,
            gait_quality=1.0,
            appearance_stability=1.0,
            gait_stability=1.0,
            spatial_probability=0.5,
            appearance_calibrator=ScoreCalibrator(midpoint=0.5),
            gait_calibrator=ScoreCalibrator(midpoint=0.5),
            maximum_gait_weight=0.35,
        )
        self.assertLessEqual(result.gait_weight, 0.35 + 1e-8)
        self.assertGreater(result.fused_probability, 0.5)

    def test_gated_assignment_does_not_let_bad_edge_consume_column(self) -> None:
        scores = np.array([[0.79, 0.91], [0.88, 0.80]], dtype=np.float32)
        appearance = np.array([[0.50, 0.80], [0.80, 0.50]], dtype=np.float32)
        assigned = gated_global_assignment(
            scores,
            appearance,
            accept_threshold=0.75,
            appearance_floor=0.70,
            margin_threshold=0.0,
        )
        self.assertEqual(assigned, {0: 1, 1: 0})

    def test_gated_assignment_has_explicit_unknown_for_infeasible_rows(self) -> None:
        assigned = gated_global_assignment(
            np.asarray([[0.20, 0.30]], dtype=np.float32),
            accept_threshold=0.80,
            appearance_floor=0.0,
        )
        self.assertEqual(assigned, {})

    def test_formal_match_is_separate_from_unknown_candidate(self) -> None:
        verifier = CrossEventVerifier()
        verifier.register_identity("P1", FeatureBundle([1, 0, 0], [0, 1, 0]))
        decision = verifier.verify(
            Observation(
                event_id="known-1",
                camera_id="cam-a",
                capture_session_id="session-a",
                features=FeatureBundle([1, 0.01, 0], [0, 1, 0.01]),
                quality=good_quality(),
            )
        )
        self.assertEqual(decision.kind, DecisionKind.FORMAL_MATCH)
        self.assertEqual(decision.identity_id, "P1")
        self.assertEqual(len(verifier.memory.quarantine), 0)

    def test_two_independent_unknown_events_can_be_promoted_explicitly(self) -> None:
        config = VerifierConfig(minimum_frames=2, minimum_gait_cycles=1)
        verifier = CrossEventVerifier(config)
        first = verifier.verify(
            Observation(
                event_id="unknown-a",
                camera_id="cam-a",
                capture_session_id="session-a",
                track_id="track-1",
                features=FeatureBundle([1, 0, 0], [0, 1, 0]),
                quality=good_quality(),
            ),
            candidate_id="C1",
        )
        second = verifier.verify(
            Observation(
                event_id="unknown-b",
                camera_id="cam-b",
                capture_session_id="session-b",
                track_id="track-7",
                features=FeatureBundle([0.99, 0.02, 0], [0, 0.99, 0.02]),
                quality=good_quality(),
            ),
            candidate_id="C1",
        )
        self.assertEqual(first.kind, DecisionKind.CANDIDATE_CREATED)
        self.assertEqual(second.kind, DecisionKind.CANDIDATE_UPDATED)
        candidate = verifier.get_candidate("C1")
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.independent_event_count, 2)
        self.assertEqual(candidate.state, VerificationState.ISOLATED_CANDIDATE)

        result = verifier.promote_candidate("C1", identity_id="P-new")
        self.assertEqual(result.identity_id, "P-new")
        self.assertEqual(verifier.formal_identities, ("P-new",))
        self.assertEqual(verifier.memory.candidate_ids(), ())
        self.assertEqual(verifier.get_candidate("C1").state, VerificationState.CONFIRMED_IDENTITY)

    def test_same_capture_session_is_not_independent(self) -> None:
        verifier = CrossEventVerifier(VerifierConfig(minimum_frames=2))
        for event_id, camera in (("e1", "cam-a"), ("e2", "cam-b")):
            verifier.verify(
                Observation(
                    event_id=event_id,
                    camera_id=camera,
                    capture_session_id="same-session",
                    features=FeatureBundle([1, 0, 0]),
                    quality=good_quality(),
                ),
                candidate_id="C1",
            )
        self.assertEqual(verifier.get_candidate("C1").independent_event_count, 1)

    def test_sqlite_hydrates_formal_gallery_and_vector_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "verifier.sqlite3")
            first = CrossEventVerifier(store=SqliteStore(path))
            first.register_identity("P1", FeatureBundle([1, 0, 0]))
            first.close()

            second = CrossEventVerifier(store=SqliteStore(path))
            self.assertEqual(second.formal_identities, ("P1",))
            hits = second.search_vectors("appearance", [1, 0, 0], k=1)
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0].identity_id, "P1")
            second.close()

    def test_embedding_contract_blocks_cross_model_gallery_match_and_write(self) -> None:
        config = VerifierConfig(
            model_version="production-model-a",
            feature_schema="gaitgraph2-rtmpose-v1",
        )
        verifier = CrossEventVerifier(config)
        verifier.register_identity("P1", FeatureBundle(appearance=[1, 0, 0]))

        incompatible = Observation(
            event_id="incompatible-model",
            model_version="production-model-b",
            feature_schema="gaitgraph2-hrnet-v2",
            features=FeatureBundle(appearance=[1, 0, 0]),
            quality=good_quality(),
        )
        self.assertEqual(verifier.rank(incompatible), ())
        incompatible_decision = verifier.verify(incompatible)
        self.assertEqual(incompatible_decision.kind, DecisionKind.NEED_MORE_DATA)
        self.assertIn("feature_contract_mismatch", incompatible_decision.reasons)
        wrong_dimension = Observation(
            event_id="incompatible-dimension",
            model_version="production-model-a",
            feature_schema="gaitgraph2-rtmpose-v1",
            features=FeatureBundle(appearance=[1, 0]),
            quality=good_quality(),
        )
        self.assertEqual(verifier.rank(wrong_dimension), ())
        with self.assertRaises(ValueError):
            verifier.memory.add_formal(
                "P1",
                FeatureBundle(appearance=[1, 0, 0]),
                model_version="production-model-b",
                feature_schema="gaitgraph2-hrnet-v2",
            )
        verifier.close()

    def test_calibration_status_is_explicit_and_strict_mode_rejects_defaults(self) -> None:
        default_verifier = CrossEventVerifier()
        self.assertFalse(default_verifier.calibration_status["ready"])
        default_verifier.register_identity("P1", FeatureBundle(appearance=[1, 0, 0]))
        mismatched_observation = Observation(
            event_id="calibration-mismatch",
            calibration_version="target-camera-v1",
            features=FeatureBundle(appearance=[1, 0, 0]),
            quality=good_quality(),
        )
        self.assertEqual(default_verifier.rank(mismatched_observation), ())
        mismatched_decision = default_verifier.verify(mismatched_observation)
        self.assertEqual(mismatched_decision.kind, DecisionKind.NEED_MORE_DATA)
        self.assertIn("calibration_contract_mismatch", mismatched_decision.reasons)
        default_verifier.close()

        with self.assertRaises(ValueError):
            CrossEventVerifier(VerifierConfig(require_calibrated_scores=True))

        from cross_event_verifier.calibration import ScoreCalibrator

        similarities = np.concatenate(
            [np.linspace(-0.2, 0.2, 16), np.linspace(0.6, 0.95, 16)]
        )
        labels = np.asarray([0] * 16 + [1] * 16, dtype=np.int64)
        fitted = ScoreCalibrator.fit(
            similarities,
            labels,
            name="target-camera-v1",
            minimum_pairs=32,
        )
        strict = CrossEventVerifier(
            VerifierConfig(
                require_calibrated_scores=True,
                calibration_version="target-camera-v1",
            ),
            appearance_calibrator=fitted,
            gait_calibrator=fitted,
        )
        self.assertTrue(strict.calibration_status["ready"])
        strict.close()

    def test_embedding_and_calibration_contracts_survive_sqlite_round_trip(self) -> None:
        store = SqliteStore(":memory:")
        observation = Observation(
            event_id="contract-round-trip",
            model_version="model-a",
            feature_schema="gaitgraph2-rtmpose-v1",
            calibration_version="target-camera-v1",
            threshold_version="threshold-v4",
            features=FeatureBundle(appearance=[1, 0, 0]),
            quality=good_quality(),
        )
        store.save_observation(observation, candidate_id="C1")
        restored_observation = store.observations_for_candidate("C1")[0]
        self.assertEqual(restored_observation.feature_schema, "gaitgraph2-rtmpose-v1")
        self.assertEqual(restored_observation.calibration_version, "target-camera-v1")

        verifier = CrossEventVerifier(
            VerifierConfig(model_version="model-a", feature_schema="gaitgraph2-rtmpose-v1"),
            store=store,
        )
        verifier.register_identity("P1", FeatureBundle(appearance=[1, 0, 0]))
        prototype = verifier.memory.formal_prototypes("P1", "appearance")[0]
        self.assertEqual(prototype.model_version, "model-a")
        self.assertEqual(prototype.feature_schema, "gaitgraph2-rtmpose-v1")
        verifier.close()

    def test_maintenance_does_not_revoke_merged_candidate(self) -> None:
        verifier = CrossEventVerifier(
            VerifierConfig(
                candidate_ttl_seconds=1.0,
                quarantine_max_candidates=1,
            )
        )
        candidate = verifier._get_or_create_candidate("merged-candidate")
        candidate.state = VerificationState.MERGED
        candidate.updated_at = 0.0
        verifier.store.save_candidate(candidate)

        removed = verifier.maintenance(now=100.0)

        self.assertEqual(removed, ())
        self.assertEqual(
            verifier.get_candidate("merged-candidate").state,
            VerificationState.MERGED,
        )
        verifier.close()

    def test_strong_gait_requests_appearance_and_strong_response_is_absorbed(self) -> None:
        verifier = CrossEventVerifier()
        verifier.register_identity("P1", FeatureBundle([1, 0, 0], [0, 1, 0]))
        original_appearance = verifier.memory.formal_prototypes("P1", "appearance")[0].vector.copy()
        gait_decision = verifier.verify(
            Observation(
                event_id="gait-anchor",
                camera_id="cam-a",
                capture_session_id="gait-session",
                # 虽然存在外观向量，但在响应步态签发的请求前，不得吸收该外观。
                features=FeatureBundle(appearance=[0, 0, 1], gait=[0, 1, 0]),
                quality=good_quality(),
            )
        )
        self.assertEqual(gait_decision.kind, DecisionKind.FORMAL_MATCH)
        self.assertIsNotNone(gait_decision.appearance_request_id)
        self.assertIn("appearance_absorption_requested", gait_decision.reasons)
        self.assertTrue(
            np.array_equal(
                verifier.memory.formal_prototypes("P1", "appearance")[0].vector,
                original_appearance,
            )
        )

        response = verifier.verify(
            Observation(
                event_id="appearance-response",
                camera_id="cam-a",
                capture_session_id="appearance-session",
                appearance_request_id=gait_decision.appearance_request_id,
                features=FeatureBundle(appearance=[1, 0.01, 0]),
                quality=TrackQuality(
                    detection_confidence=0.95,
                    box_height=200,
                    frame_count=1,
                    sharpness=0.95,
                    occlusion=0.05,
                ),
            )
        )
        self.assertEqual(response.kind, DecisionKind.APPEARANCE_RESPONSE_ACCEPTED)
        self.assertEqual(response.identity_id, "P1")
        request = verifier.appearance_absorption.get(gait_decision.appearance_request_id)
        self.assertEqual(request.status, "consumed")
        self.assertEqual(len(verifier.memory.quarantine), 0)

    def test_unrequested_appearance_is_not_formal_or_absorbed(self) -> None:
        verifier = CrossEventVerifier()
        verifier.register_identity("P1", FeatureBundle([1, 0, 0], [0, 1, 0]))
        before = tuple(verifier.memory.formal_prototypes("P1", "appearance"))
        decision = verifier.verify(
            Observation(
                event_id="appearance-without-token",
                camera_id="cam-a",
                capture_session_id="appearance-session",
                features=FeatureBundle(appearance=[1, 0.01, 0]),
                quality=good_quality(),
            ),
            candidate_id="appearance-candidate",
        )
        self.assertEqual(decision.kind, DecisionKind.CANDIDATE_CREATED)
        self.assertIn("appearance_is_absorbable_only", decision.reasons)
        after = tuple(verifier.memory.formal_prototypes("P1", "appearance"))
        self.assertEqual(len(after), len(before))
        self.assertTrue(np.array_equal(after[0].vector, before[0].vector))

    def test_appearance_request_rejects_a_different_track(self) -> None:
        verifier = CrossEventVerifier()
        verifier.register_identity("P1", FeatureBundle([1, 0, 0], [0, 1, 0]))
        issued = verifier.verify(
            Observation(
                event_id="track-bound-gait",
                camera_id="cam-a",
                capture_session_id="session-a",
                track_id="track-a",
                features=FeatureBundle(gait=[0, 1, 0]),
                quality=good_quality(),
            )
        )
        response = verifier.verify(
            Observation(
                event_id="track-bound-response",
                camera_id="cam-a",
                capture_session_id="session-b",
                track_id="track-b",
                appearance_request_id=issued.appearance_request_id,
                features=FeatureBundle(appearance=[1, 0, 0]),
                quality=TrackQuality(
                    detection_confidence=0.95,
                    box_height=200,
                    sharpness=0.95,
                    occlusion=0.05,
                ),
            )
        )
        self.assertNotEqual(response.kind, DecisionKind.APPEARANCE_RESPONSE_ACCEPTED)
        self.assertIn("appearance_request_track_mismatch", response.reasons)
        verifier.close()

    def test_strong_gait_can_authorize_a_new_clothing_appearance(self) -> None:
        verifier = CrossEventVerifier()
        verifier.register_identity("P1", FeatureBundle([1, 0, 0], [0, 1, 0]))
        gait = verifier.verify(
            Observation(
                event_id="new-clothes-gait",
                camera_id="cam-a",
                capture_session_id="new-clothes-session",
                features=FeatureBundle(appearance=[0, 0, 1], gait=[0, 1, 0]),
                quality=good_quality(),
            )
        )
        response = verifier.verify(
            Observation(
                event_id="new-clothes-response",
                camera_id="cam-a",
                capture_session_id="new-clothes-session",
                appearance_request_id=gait.appearance_request_id,
                features=FeatureBundle(appearance=[0, 0, 1], gait=[0, 1, 0]),
                quality=good_quality(),
            )
        )

        self.assertEqual(response.kind, DecisionKind.APPEARANCE_RESPONSE_ACCEPTED)
        self.assertIn(
            "strong_gait_authorized_appearance_refresh",
            response.reasons,
        )
        appearances = verifier.memory.formal_prototypes("P1", "appearance")
        self.assertEqual(len(appearances), 2)
        self.assertGreater(float(np.dot(appearances[-1].vector, [0, 0, 1])), 0.99)
        verifier.close()

    def test_package_runs_without_videotracker_directory(self) -> None:
        """本包是独立安装项目，不应导入相邻项目。"""

        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            package_target = Path(directory) / "cross_event_verifier"
            import shutil

            shutil.copytree(project_root / "cross_event_verifier", package_target)
            env = os.environ.copy()
            env["PYTHONPATH"] = directory
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from cross_event_verifier import CrossEventVerifier; print(CrossEventVerifier().__class__.__name__)",
                ],
                cwd=directory,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "CrossEventVerifier")

    def test_appearance_request_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "requests.sqlite3")
            first = CrossEventVerifier(store=SqliteStore(path))
            first.register_identity("P1", FeatureBundle([1, 0, 0], [0, 1, 0]))
            gait = first.verify(
                Observation(
                    event_id="persisted-gait",
                    camera_id="cam-a",
                    capture_session_id="s-a",
                    features=FeatureBundle(gait=[0, 1, 0]),
                    quality=good_quality(),
                )
            )
            request_id = gait.appearance_request_id
            first.close()

            second = CrossEventVerifier(store=SqliteStore(path))
            self.assertEqual(second.appearance_absorption.get(request_id).status, "pending")
            response = second.verify(
                Observation(
                    event_id="persisted-appearance",
                    camera_id="cam-a",
                    capture_session_id="s-b",
                    appearance_request_id=request_id,
                    features=FeatureBundle(appearance=[1, 0.01, 0]),
                    quality=good_quality(),
                )
            )
            self.assertEqual(response.kind, DecisionKind.APPEARANCE_RESPONSE_ACCEPTED)
            second.close()


if __name__ == "__main__":
    unittest.main()
