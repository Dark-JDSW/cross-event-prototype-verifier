"""使用虚拟视觉而非真实摄像头资产的 GUI/管线接口测试。"""

import unittest
import tkinter as tk
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from cross_event_verifier import AutomationPolicy, CrossEventVerifier, FeatureBundle
from cross_event_verifier.participant_c.gui import VerifierWindow, _frame_to_photo
from cross_event_verifier.participant_c.media import FrameWorker, ParameterUpdateMessage, SourceSpec
from cross_event_verifier.participant_c.pipeline import VideoVerifierPipeline
from cross_event_verifier.types import TrackQuality
from cross_event_verifier.participant_b.vision import OpenCvDemoAdapter, VisionTrack


def stable_quality() -> TrackQuality:
    return TrackQuality(
        detection_confidence=0.95,
        box_height=180,
        sharpness=0.95,
        occlusion=0.02,
        contour_area=2400,
        frame_count=24,
        gait_cycles=2,
        walking_ratio=0.95,
        gait_branch_quality=0.95,
    )


class FakeVision:
    """证明 GUI 管线不持有模型推理的测试适配器。"""

    def __init__(self) -> None:
        self.latest = {}

    def reset(self) -> None:
        self.latest.clear()

    def process(self, frame_bgr: np.ndarray) -> tuple[VisionTrack, ...]:
        item = VisionTrack(
            track_id=7,
            box=(20, 10, 90, 190),
            detection_confidence=0.95,
            features=FeatureBundle(
                appearance=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                gait=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            ),
            quality=stable_quality(),
        )
        self.latest[7] = item
        return (item,)


class TwoPersonVision(FakeVision):
    def process(self, frame_bgr: np.ndarray) -> tuple[VisionTrack, ...]:
        first = super().process(frame_bgr)[0]
        second = VisionTrack(
            track_id=8,
            box=(100, 10, 170, 190),
            detection_confidence=first.detection_confidence,
            features=first.features,
            quality=first.quality,
        )
        self.latest[8] = second
        return first, second


class NovelGaitVision(FakeVision):
    """步态有意与图库正交的稳定轨迹。"""

    def process(self, frame_bgr: np.ndarray) -> tuple[VisionTrack, ...]:
        item = VisionTrack(
            track_id=7,
            box=(20, 10, 90, 190),
            detection_confidence=0.95,
            features=FeatureBundle(
                # 外观有意类似 P1，但不能覆盖明确拒绝所有正式身份的步态。
                appearance=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                gait=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            ),
            quality=stable_quality(),
        )
        self.latest[7] = item
        return (item,)


class StrongThenWeakVision(FakeVision):
    """最新帧暂时失去步态质量的跟踪人物。"""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def process(self, frame_bgr: np.ndarray) -> tuple[VisionTrack, ...]:
        item = super().process(frame_bgr)[0]
        self.calls += 1
        if self.calls >= 2:
            item = replace(
                item,
                quality=replace(
                    item.quality,
                    gait_branch_quality=0.30,
                    walking_ratio=0.30,
                ),
            )
            self.latest[7] = item
        return (item,)


class KnownAndNovelVision(FakeVision):
    def process(self, frame_bgr: np.ndarray) -> tuple[VisionTrack, ...]:
        known = super().process(frame_bgr)[0]
        novel = VisionTrack(
            track_id=8,
            box=(100, 10, 170, 190),
            detection_confidence=0.95,
            features=FeatureBundle(
                appearance=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                gait=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            ),
            quality=stable_quality(),
        )
        self.latest[8] = novel
        return known, novel


class GuiPipelineTests(unittest.TestCase):
    def test_camera_frame_converts_to_tk_photo(self) -> None:
        root = tk.Tk()
        root.withdraw()
        try:
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            photo = _frame_to_photo(frame, master=root)
            self.assertEqual((photo.width(), photo.height()), (160, 120))
        finally:
            root.destroy()

    def test_gui_has_a_realtime_parameter_page(self) -> None:
        with patch(
            "cross_event_verifier.participant_c.gui.build_vision_adapter",
            return_value=FakeVision(),
        ):
            window = VerifierWindow(":memory:", "demo")
        window.root.withdraw()
        try:
            self.assertEqual(len(window.notebook.tabs()), 2)
            self.assertIn(
                "verifier.gait_novelty_threshold",
                window.parameter_entries,
            )
            self.assertEqual(
                str(window.parameter_entries["vision.detector_confidence"].cget("state")),
                "disabled",
            )
        finally:
            window.close()

    def test_pipeline_accepts_swappable_vision_adapter_and_enrolls_track(self) -> None:
        verifier = CrossEventVerifier()
        pipeline = VideoVerifierPipeline(verifier, FakeVision(), camera_id="cam-test")
        frame = np.zeros((220, 120, 3), dtype=np.uint8)

        before = pipeline.process_frame(frame, timestamp=100.0)
        self.assertEqual(before.tracks[0].decision.kind.value, "candidate_created")
        self.assertEqual(pipeline.register_current_track("P-test"), 7)
        self.assertEqual(verifier.formal_identities, ("P-test",))

        after = pipeline.process_frame(frame, timestamp=101.0)
        self.assertEqual(after.tracks[0].decision.identity_id, "P-test")
        self.assertEqual(
            after.tracks[0].decision.kind.value,
            "appearance_response_accepted",
        )
        self.assertEqual(
            after.tracks[0].automation.stage.value,
            "appearance_absorbed",
        )
        self.assertEqual(
            len(verifier.memory.formal_prototypes("P-test", "appearance")),
            1,
        )
        repeated = pipeline.process_frame(frame, timestamp=102.0)
        self.assertEqual(repeated.tracks[0].decision.kind.value, "formal_match")
        self.assertIsNone(repeated.tracks[0].decision.appearance_request_id)
        self.assertIsNone(pipeline.appearance_request_id)
        verifier.close()

    def test_manual_registration_uses_recent_strong_gait_snapshot(self) -> None:
        verifier = CrossEventVerifier()
        pipeline = VideoVerifierPipeline(
            verifier,
            StrongThenWeakVision(),
            camera_id="cam-manual",
        )
        frame = np.zeros((220, 120, 3), dtype=np.uint8)

        pipeline.process_frame(frame, timestamp=110.0)
        pipeline.process_frame(frame, timestamp=111.0)

        # 最新帧暂时较弱，但该 Track 已经有强步态快照。手工登记不能仅因最后
        # 一帧被遮挡就丢弃这份有效证据。
        self.assertEqual(pipeline.register_current_track("P-manual"), 7)
        self.assertIn("P-manual", verifier.formal_identities)
        verifier.close()

    def test_unknown_person_is_registered_from_stable_gait_then_absorbs_appearance(self) -> None:
        verifier = CrossEventVerifier()
        pipeline = VideoVerifierPipeline(
            verifier,
            FakeVision(),
            camera_id="cam-auto",
            automation_policy=AutomationPolicy(
                minimum_track_frames=1,
                minimum_stable_gait_samples=3,
                gait_sample_window=4,
            ),
        )
        frame = np.zeros((220, 120, 3), dtype=np.uint8)

        first = pipeline.process_frame(frame, timestamp=200.0)
        second = pipeline.process_frame(frame, timestamp=201.0)
        enrolled = pipeline.process_frame(frame, timestamp=202.0)

        self.assertEqual(first.tracks[0].automation.stage.value, "collecting_gait")
        self.assertEqual(second.tracks[0].automation.stage.value, "collecting_gait")
        self.assertEqual(enrolled.tracks[0].decision.kind.value, "appearance_requested")
        self.assertTrue(enrolled.tracks[0].automation.auto_registered)
        self.assertEqual(enrolled.tracks[0].decision.identity_id, "P1")
        self.assertEqual(
            len(verifier.memory.formal_prototypes("P1", "gait")),
            1,
        )
        self.assertEqual(
            len(verifier.memory.formal_prototypes("P1", "appearance")),
            0,
        )

        absorbed = pipeline.process_frame(frame, timestamp=203.0)
        self.assertEqual(
            absorbed.tracks[0].decision.kind.value,
            "appearance_response_accepted",
        )
        self.assertEqual(absorbed.tracks[0].decision.identity_id, "P1")
        self.assertEqual(
            len(verifier.memory.formal_prototypes("P1", "appearance")),
            1,
        )
        request = verifier.get_appearance_request(
            enrolled.tracks[0].decision.appearance_request_id
        )
        self.assertIsNotNone(request)
        self.assertEqual(request.status, "consumed")
        verifier.close()

    def test_novel_gait_is_registered_when_gallery_already_has_an_identity(self) -> None:
        verifier = CrossEventVerifier()
        verifier.register_identity(
            "P1",
            FeatureBundle(
                appearance=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                gait=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            ),
        )
        pipeline = VideoVerifierPipeline(
            verifier,
            NovelGaitVision(),
            camera_id="cam-novel",
            automation_policy=AutomationPolicy(
                minimum_track_frames=1,
                minimum_stable_gait_samples=3,
                gait_sample_window=4,
            ),
        )
        frame = np.zeros((220, 120, 3), dtype=np.uint8)

        first = pipeline.process_frame(frame, timestamp=250.0)
        pipeline.process_frame(frame, timestamp=251.0)
        enrolled = pipeline.process_frame(frame, timestamp=252.0)

        self.assertIsNone(first.tracks[0].decision.identity_id)
        self.assertIn(
            "high_quality_gait_rejects_formal_gallery",
            first.tracks[0].decision.reasons,
        )
        self.assertEqual(verifier.formal_identities, ("P1", "P2"))
        self.assertEqual(enrolled.tracks[0].decision.identity_id, "P2")
        self.assertTrue(enrolled.tracks[0].automation.auto_registered)
        verifier.close()

    def test_known_person_does_not_starve_simultaneous_novel_registration(self) -> None:
        verifier = CrossEventVerifier()
        verifier.register_identity(
            "P1",
            FeatureBundle(
                appearance=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                gait=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            ),
        )
        pipeline = VideoVerifierPipeline(
            verifier,
            KnownAndNovelVision(),
            automation_policy=AutomationPolicy(
                minimum_track_frames=1,
                minimum_stable_gait_samples=3,
                gait_sample_window=4,
            ),
        )
        frame = np.zeros((220, 190, 3), dtype=np.uint8)
        result = None
        for timestamp in (270.0, 271.0, 272.0):
            result = pipeline.process_frame(frame, timestamp=timestamp)

        assert result is not None
        by_track = {item.track_id: item for item in result.tracks}
        self.assertEqual(by_track[7].decision.identity_id, "P1")
        self.assertEqual(by_track[8].decision.identity_id, "P2")
        self.assertTrue(by_track[8].automation.auto_registered)
        self.assertEqual(verifier.formal_identities, ("P1", "P2"))
        verifier.close()

    def test_runtime_parameter_update_is_validated_as_one_transaction(self) -> None:
        verifier = CrossEventVerifier()
        pipeline = VideoVerifierPipeline(verifier, FakeVision())

        state = pipeline.update_runtime_parameters(
            {
                "verifier.gait_novelty_threshold": "0.42",
                "automation.minimum_stable_gait_samples": "4",
                "automation.gait_sample_window": "6",
                "calibration.gait_midpoint": "0.72",
            }
        )
        self.assertEqual(state.revision, 1)
        self.assertAlmostEqual(verifier.config.gait_novelty_threshold, 0.42)
        self.assertEqual(pipeline.automation.policy.minimum_stable_gait_samples, 4)
        self.assertAlmostEqual(verifier.gait_calibrator.midpoint, 0.72)
        self.assertEqual(verifier.config.threshold_version, "runtime-v1")

        with self.assertRaises(ValueError):
            pipeline.update_runtime_parameters(
                {
                    "verifier.accept_threshold": "0.50",
                    "verifier.defer_threshold": "0.60",
                }
            )
        self.assertAlmostEqual(verifier.config.accept_threshold, 0.82)
        self.assertAlmostEqual(verifier.config.defer_threshold, 0.62)
        self.assertEqual(pipeline.runtime_parameter_state().revision, 1)
        verifier.close()

    def test_worker_reports_parameter_updates_through_the_gui_queue(self) -> None:
        verifier = CrossEventVerifier()
        pipeline = VideoVerifierPipeline(verifier, FakeVision())
        worker = FrameWorker(pipeline)
        worker.set_runtime_parameters(
            {"verifier.gait_novelty_threshold": "0.40"}
        )
        message = worker.messages.get_nowait()
        self.assertIsInstance(message, ParameterUpdateMessage)
        self.assertTrue(message.success)
        self.assertIsNotNone(message.state)
        self.assertAlmostEqual(verifier.config.gait_novelty_threshold, 0.40)
        verifier.close()

    def test_novelty_threshold_takes_effect_without_restarting_pipeline(self) -> None:
        verifier = CrossEventVerifier()
        verifier.register_identity(
            "P1",
            FeatureBundle(
                appearance=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                gait=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            ),
        )
        pipeline = VideoVerifierPipeline(
            verifier,
            NovelGaitVision(),
            automation_policy=AutomationPolicy(
                minimum_track_frames=1,
                minimum_stable_gait_samples=3,
                gait_sample_window=4,
            ),
        )
        pipeline.update_runtime_parameters(
            {"verifier.gait_novelty_threshold": "0.001"}
        )
        frame = np.zeros((220, 120, 3), dtype=np.uint8)
        before = pipeline.process_frame(frame, timestamp=260.0)
        self.assertEqual(before.tracks[0].decision.identity_id, "P1")

        pipeline.update_runtime_parameters(
            {"verifier.gait_novelty_threshold": "0.35"}
        )
        first = pipeline.process_frame(frame, timestamp=261.0)
        second = pipeline.process_frame(frame, timestamp=262.0)
        enrolled = pipeline.process_frame(frame, timestamp=263.0)
        self.assertIsNone(first.tracks[0].decision.identity_id)
        self.assertIsNone(second.tracks[0].decision.identity_id)
        self.assertEqual(enrolled.tracks[0].decision.identity_id, "P2")
        self.assertTrue(enrolled.tracks[0].automation.auto_registered)
        verifier.close()

    def test_runtime_change_does_not_mix_partial_gait_evidence(self) -> None:
        verifier = CrossEventVerifier()
        pipeline = VideoVerifierPipeline(
            verifier,
            FakeVision(),
            automation_policy=AutomationPolicy(
                minimum_track_frames=1,
                minimum_stable_gait_samples=3,
                gait_sample_window=4,
            ),
        )
        frame = np.zeros((220, 120, 3), dtype=np.uint8)
        first = pipeline.process_frame(frame, timestamp=280.0)
        self.assertEqual(first.tracks[0].automation.stage.value, "collecting_gait")

        pipeline.update_runtime_parameters(
            {"verifier.minimum_matching_quality": "0.39"}
        )
        after_one = pipeline.process_frame(frame, timestamp=281.0)
        after_two = pipeline.process_frame(frame, timestamp=282.0)
        self.assertFalse(after_one.tracks[0].automation.auto_registered)
        self.assertFalse(after_two.tracks[0].automation.auto_registered)
        self.assertEqual(verifier.formal_identities, ())

        enrolled = pipeline.process_frame(frame, timestamp=283.0)
        self.assertTrue(enrolled.tracks[0].automation.auto_registered)
        self.assertEqual(verifier.formal_identities, ("P1",))
        verifier.close()

    def test_source_spec_keeps_camera_and_file_inputs_explicit(self) -> None:
        camera = SourceSpec("camera", 0, "camera-0")
        video = SourceSpec("file", "sample.mp4", "sample.mp4")
        self.assertEqual(camera.kind, "camera")
        self.assertEqual(video.value, "sample.mp4")

    def test_pipeline_keeps_identity_assignment_one_to_one_for_two_people(self) -> None:
        verifier = CrossEventVerifier()
        verifier.register_identity(
            "P1",
            FeatureBundle(
                appearance=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                gait=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            ),
        )
        pipeline = VideoVerifierPipeline(
            verifier,
            TwoPersonVision(),
            automation_policy=AutomationPolicy(enabled=False),
        )
        result = pipeline.process_frame(
            np.zeros((220, 190, 3), dtype=np.uint8),
            timestamp=300.0,
        )
        assigned = [
            track.decision.identity_id
            for track in result.tracks
            if track.decision.identity_id == "P1"
        ]
        self.assertEqual(assigned, ["P1"])
        verifier.close()

    def test_opencv_adapter_has_foreground_fallback(self) -> None:
        adapter = OpenCvDemoAdapter(max_processing_dimension=320)
        adapter.hog = None
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        self.assertIsInstance(adapter.process(frame), tuple)


if __name__ == "__main__":
    unittest.main()
