from pathlib import Path

from datasets.cross_event_training_v1.tools.action_branch_gui import (
    annotation_key,
    branch_id_for_action,
    interpolated_box,
    load_queue,
    read_jsonl,
    tracks_to_display,
    write_jsonl_atomic,
)


def test_branch_id_keeps_concrete_actions_separate() -> None:
    assert branch_id_for_action("walking") == "gait_walking"
    assert branch_id_for_action("running") == "gait_running"
    assert branch_id_for_action("unknown") == "unknown"


def test_annotation_key_includes_track_branch_and_frame_span() -> None:
    assert annotation_key("video-a", 3, "gait_walking", 10, 80) == "video-a|track:3|branch:gait_walking|frames:10-80"


def test_default_overlay_only_displays_selected_track() -> None:
    tracks = [{"track_id": 1}, {"track_id": 3}, {"track_id": 5}]
    assert tracks_to_display(tracks, "5", show_all=False) == [{"track_id": 5}]
    assert tracks_to_display(tracks, "5", show_all=True) == tracks


def test_interpolated_box_moves_between_detection_frames() -> None:
    preview = [
        {"frame_index": 0, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]},
        {"frame_index": 10, "bbox_xyxy": [10.0, 20.0, 30.0, 40.0]},
    ]
    assert interpolated_box(preview, 5) == [5.0, 10.0, 20.0, 25.0]
    assert interpolated_box(preview, -1) == [0.0, 0.0, 10.0, 10.0]
    assert interpolated_box(preview, 11) == [10.0, 20.0, 30.0, 40.0]


def test_queue_and_annotation_jsonl_round_trip(tmp_path: Path) -> None:
    queue = tmp_path / "fast_screen.jsonl"
    queue.write_text('{"source_video_id":"video-a","source_video_path":"E:/videos/video-a.avi","recommended_track_id":2}\n', encoding="utf-8")
    assert load_queue(queue)[0]["recommended_track_id"] == 2

    output = tmp_path / "annotations.jsonl"
    row = {"annotation_key": "video-a|track:2|branch:gait_walking|frames:0-25", "action_label": "walking"}
    write_jsonl_atomic(output, [row])
    assert read_jsonl(output) == [row]
