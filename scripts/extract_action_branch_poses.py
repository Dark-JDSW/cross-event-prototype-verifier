"""Extract project RTMPose rows for manually selected action branches.

The action GUI only records a video, source-local track ID and frame span. This
script replays those videos through the fixed production detector/tracker/pose
chain and writes one protocol-compatible ``.npz`` per annotation. It processes
each source video once, even when several people or action spans were labelled.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cross_event_verifier.production_vision import (  # noqa: E402
    ProductionVisionAdapter,
    ProductionVisionConfig,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain JSON objects")
        rows.append(value)
    return rows


def _annotation_id(row: dict[str, Any]) -> str:
    key = str(row.get("annotation_key") or "")
    if not key:
        raise ValueError("each action annotation needs annotation_key")
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def _validate_row(row: dict[str, Any]) -> tuple[Path, int, int, int]:
    path = Path(str(row.get("source_video_path") or ""))
    if not path.is_file():
        raise FileNotFoundError(f"annotated video does not exist: {path}")
    try:
        track_id = int(row["track_id"])
        start = int(row["start_frame"])
        end = int(row["end_frame"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid action annotation fields: {row}") from error
    if track_id < 0 or start < 0 or end < start:
        raise ValueError(f"invalid track/frame span in annotation: {row}")
    return path, track_id, start, end


def _source_frame_stride(row: dict[str, Any]) -> int:
    """Return the frame cadence used by the screen that supplied the track ID."""

    try:
        return max(1, int(row.get("source_frame_stride", 1)))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid source_frame_stride in action annotation: {row}") from error


def _source_image_size(row: dict[str, Any], *, default: int) -> int:
    """Return the detector image size recorded by the screening pass."""

    try:
        value = int(row.get("source_image_size", default))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid source_image_size in action annotation: {row}") from error
    if value < 320:
        raise ValueError(f"source_image_size must be at least 320: {row}")
    return value


def _directional_gait_training_eligible(row: dict[str, Any]) -> bool:
    """Reject generic or mixed-direction walking samples from gait training."""

    if str(row.get("action_label", "")).strip().lower() != "walking":
        return False
    walking_view = str(row.get("walking_view", "unknown")).strip().lower()
    if walking_view not in {"side", "oblique", "front", "back"}:
        return False
    walking_direction = str(row.get("walking_direction", "unknown")).strip().lower()
    if walking_direction not in {"left_to_right", "right_to_left", "approaching", "leaving"}:
        return False
    return row.get("gait_branch_id") == f"gait_walking_{walking_view}_{walking_direction}"


def extract(
    annotations_path: Path,
    output_dir: Path,
    *,
    config: ProductionVisionConfig | None = None,
) -> dict[str, object]:
    """Replay labelled videos and write pose arrays plus a manifest."""

    annotations = _read_jsonl(annotations_path)
    if not annotations:
        raise ValueError(f"no action annotations found in {annotations_path}")
    groups: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in annotations:
        path, _track_id, _start, _end = _validate_row(row)
        groups[path].append(row)

    output_dir = Path(output_dir)
    pose_dir = output_dir / "poses"
    pose_dir.mkdir(parents=True, exist_ok=True)
    base_config = config or ProductionVisionConfig()
    adapters: dict[int, ProductionVisionAdapter] = {}
    manifest: list[dict[str, object]] = []

    for video_path, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        source_strides = {
            _source_frame_stride(row)
            for row in rows
            if row.get("source_frame_stride") not in (None, "")
        }
        if len(source_strides) != 1:
            raise ValueError(
                "annotations for one source video must use one source_frame_stride: "
                f"{video_path} has {sorted(source_strides)}"
            )
        # Older hand-written annotations may predate the cadence metadata. If
        # newer annotations for the same video carry it, use that shared
        # value rather than treating the old row as a conflicting stride.
        source_stride = source_strides.pop() if source_strides else 1
        default_image_size = 640 if source_stride > 1 else base_config.image_size
        source_image_sizes = {
            _source_image_size(row, default=default_image_size) for row in rows
            if row.get("source_image_size") not in (None, "")
        }
        if len(source_image_sizes) != 1:
            raise ValueError(
                "annotations for one source video must use one source_image_size: "
                f"{video_path} has {sorted(source_image_sizes)}"
            )
        source_image_size = source_image_sizes.pop()
        # ``fast_screen`` assigned track IDs only on its sampled frames.  Feed
        # the same sampled stream to ByteTrack during extraction; otherwise a
        # long video can receive different local IDs merely because the replay
        # cadence changed.  Low-light enhancement is disabled here because
        # fast_screen also tracks the original BGR frames.
        adapter = adapters.get(source_stride)
        if adapter is None:
            replay_config = replace(
                base_config,
                detector_inference_stride=1,
                image_size=source_image_size,
                low_light_threshold=0.0,
            )
            adapter = ProductionVisionAdapter(replay_config)
            adapter.preload()
            adapter.warmup()
            adapters[source_stride] = adapter
        adapter.reset()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open annotated video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        max_end = max(int(row["end_frame"]) for row in rows)
        buffers: dict[str, dict[str, list[Any]]] = {}
        for row in rows:
            annotation_id = _annotation_id(row)
            buffers[annotation_id] = {
                "keypoints_xy": [],
                "keypoint_scores": [],
                "valid_mask": [],
                "boxes_xyxy": [],
                "frame_indices": [],
                "timestamp": [],
                "track_id": [],
                "missing_mask": [],
            }

        frame_index = 0
        try:
            while frame_index <= max_end:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = frame_index / fps if fps > 0.0 else float(frame_index)
                if frame_index % source_stride:
                    frame_index += 1
                    continue
                adapter.process_at(frame, timestamp=timestamp)
                for row in rows:
                    start = int(row["start_frame"])
                    end = int(row["end_frame"])
                    if not start <= frame_index <= end:
                        continue
                    annotation_id = _annotation_id(row)
                    snapshot = adapter.pose_snapshot(int(row["track_id"]))
                    buffer = buffers[annotation_id]
                    if snapshot is None:
                        pose = np.zeros((17, 2), dtype=np.float32)
                        scores = np.zeros(17, dtype=np.float32)
                        box = np.zeros(4, dtype=np.float32)
                        missing = True
                    else:
                        pose = np.asarray(snapshot["keypoints_xy"], dtype=np.float32)
                        scores = np.asarray(snapshot["keypoint_scores"], dtype=np.float32)
                        box = np.asarray(snapshot["boxes_xyxy"], dtype=np.float32).reshape(4)
                        missing = bool(snapshot["missing"])
                    buffer["keypoints_xy"].append(pose)
                    buffer["keypoint_scores"].append(scores)
                    buffer["valid_mask"].append(scores > 0.0)
                    buffer["boxes_xyxy"].append(box)
                    buffer["frame_indices"].append(frame_index)
                    buffer["timestamp"].append(timestamp)
                    buffer["track_id"].append(int(row["track_id"]))
                    buffer["missing_mask"].append(missing)
                frame_index += 1
        finally:
            capture.release()

        for row in rows:
            annotation_id = _annotation_id(row)
            buffer = buffers[annotation_id]
            count = len(buffer["frame_indices"])
            path = pose_dir / f"{annotation_id}.npz"
            if count:
                np.savez_compressed(
                    path,
                    keypoints_xy=np.asarray(buffer["keypoints_xy"], dtype=np.float32),
                    keypoint_scores=np.asarray(buffer["keypoint_scores"], dtype=np.float32),
                    valid_mask=np.asarray(buffer["valid_mask"], dtype=bool),
                    boxes_xyxy=np.asarray(buffer["boxes_xyxy"], dtype=np.float32),
                    frame_indices=np.asarray(buffer["frame_indices"], dtype=np.int64),
                    timestamp=np.asarray(buffer["timestamp"], dtype=np.float64),
                    track_id=np.asarray(buffer["track_id"], dtype=np.int64),
                    missing_mask=np.asarray(buffer["missing_mask"], dtype=bool),
                )
            manifest.append(
                {
                    "annotation_key": row.get("annotation_key"),
                    "annotation_id": annotation_id,
                    "source_video_id": row.get("source_video_id") or video_path.stem,
                    "source_video_path": str(video_path),
                    "track_id": int(row["track_id"]),
                    "action_label": row.get("action_label", "unknown"),
                    "walking_view": row.get("walking_view", "unknown"),
                    "walking_direction": row.get("walking_direction", "unknown"),
                    "gait_branch_id": row.get("gait_branch_id", "unknown"),
                    "directional_gait_training_eligible": _directional_gait_training_eligible(row),
                    "start_frame": int(row["start_frame"]),
                    "end_frame": int(row["end_frame"]),
                    "fps": fps,
                    "source_frame_stride": source_stride,
                    "source_image_size": source_image_size,
                    "pose_sampling_stride": source_stride,
                    "pose_path": str(path.relative_to(output_dir)) if count else None,
                    "frame_count": count,
                    "status": "ok" if count else "no_matching_track",
                    "pose_protocol": "rtmpose-production-v1",
                }
            )

    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest),
        encoding="utf-8",
    )
    report = {
        "schema_version": "action-branch-pose-extraction-v1",
        "annotation_count": len(annotations),
        "source_video_count": len(groups),
        "matched_annotation_count": sum(row["status"] == "ok" for row in manifest),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract(args.annotations, args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
