"""Regression tests for the local action-annotation video overlay."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "cross_event_training_v1"
    / "tools"
    / "action_branch_gui.py"
)
_SPEC = spec_from_file_location("action_branch_gui", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class ActionBranchOverlayTests(unittest.TestCase):
    def test_preferred_saved_annotation_restores_the_current_directional_record(self) -> None:
        annotations = [
            {
                "schema_version": "mevid-action-branch-v1",
                "source_video_id": "video-1",
                "source_video_path": "E:/videos/video-1.mp4",
                "track_id": 5,
                "gait_branch_id": "gait_walking",
                "start_frame": 100,
                "end_frame": 200,
            },
            {
                "schema_version": "mevid-action-branch-v3",
                "source_video_id": "video-1",
                "source_video_path": "E:/videos/video-1.mp4",
                "track_id": 5,
                "action_label": "walking",
                "walking_view": "front",
                "walking_direction": "approaching",
                "gait_branch_id": "gait_walking_front_approaching",
                "start_frame": 110,
                "end_frame": 210,
                "annotation_status": "human_confirmed",
                "annotation_note": "current annotation",
            },
        ]

        restored = _MODULE.preferred_saved_annotation(
            annotations, "video-1", "E:/videos/video-1.mp4", "5"
        )

        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored["schema_version"], "mevid-action-branch-v3")
        self.assertEqual(restored["walking_direction"], "approaching")

    def test_walking_view_creates_an_isolated_directional_branch(self) -> None:
        self.assertEqual(
            _MODULE.branch_id_for_action("walking", "side", "left_to_right"),
            "gait_walking_side_left_to_right",
        )
        self.assertEqual(
            _MODULE.branch_id_for_action("walking", "back", "leaving"),
            "gait_walking_back_leaving",
        )
        self.assertEqual(_MODULE.branch_id_for_action("walking", "transition", "leaving"), "unknown")
        self.assertEqual(_MODULE.branch_id_for_action("walking", "side", "unknown"), "unknown")
        self.assertEqual(_MODULE.branch_id_for_action("running", "side"), "gait_running")

    def test_display_mode_selects_all_candidate_tracks(self) -> None:
        tracks = [{"track_id": 3}, {"track_id": 5}, {"track_id": 8}]

        self.assertEqual(_MODULE.tracks_to_display(tracks, "5", False), [{"track_id": 5}])
        self.assertEqual(_MODULE.tracks_to_display(tracks, "5", True), tracks)
        self.assertEqual(_MODULE.TRACK_DISPLAY_MODES[1], "显示所有候选追踪框")

    def test_track_range_uses_complete_track_metadata_not_preview_sampling(self) -> None:
        track = {
            "first_frame_index": 384,
            "last_frame_index": 736,
            "preview": [{"frame_index": 728, "bbox_xyxy": [874, 649, 1108, 1072]}],
        }

        self.assertEqual(_MODULE.track_observation_range(track), (384, 736))

    def test_track_box_is_hidden_outside_its_observed_frame_range(self) -> None:
        """A long video must not keep drawing a retired local track forever."""

        preview = [
            {"frame_index": 384, "bbox_xyxy": [1081, 285, 1202, 532]},
            {"frame_index": 728, "bbox_xyxy": [874, 649, 1108, 1072]},
        ]

        observed = (384, 736)
        self.assertIsNone(_MODULE.interpolated_box(preview, 383, observed))
        self.assertEqual(
            _MODULE.interpolated_box(preview, 736, observed),
            [874.0, 649.0, 1108.0, 1072.0],
        )
        self.assertIsNone(_MODULE.interpolated_box(preview, 737, observed))

    def test_track_box_interpolates_while_the_track_is_observed(self) -> None:
        preview = [
            {"frame_index": 100, "bbox_xyxy": [0, 0, 10, 10]},
            {"frame_index": 110, "bbox_xyxy": [10, 20, 30, 40]},
        ]

        self.assertEqual(_MODULE.interpolated_box(preview, 105), [5.0, 10.0, 20.0, 25.0])


if __name__ == "__main__":
    unittest.main()
