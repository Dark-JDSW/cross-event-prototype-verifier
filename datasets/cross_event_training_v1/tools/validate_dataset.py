"""Validate metadata or production completeness of a built manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


BUCKETS = {
    "train-dev": 40,
    "calibration-known": 10,
    "calibration-unknown": 10,
    "locked-test-known": 10,
    "locked-test-unknown": 10,
}
REQUIRED = {
    "event_id", "identity_id", "capture_session_id", "camera_id", "source_video_id",
    "video_path", "video_sha256", "fps", "frame_count", "frame_start", "frame_end",
    "capture_period", "clothing_condition_id", "load_condition", "view_conditions",
    "walking_direction", "split", "consent_id", "source_track_key", "tracks_path", "poses_dir",
}


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"events.jsonl:{line_number} is not an object")
            missing = sorted(REQUIRED - row.keys())
            if missing:
                raise ValueError(f"events.jsonl:{line_number} missing fields: {', '.join(missing)}")
            events.append(row)
    return events


def pending_report(root: Path, reason: str) -> dict[str, Any]:
    return {
        "status": "pending_external_data",
        "dataset_root": str(root),
        "event_count": 0,
        "identity_count": 0,
        "identity_counts_by_split": {},
        "production_training_eligible": False,
        "reason": reason,
    }


def validate_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        raise ValueError("events.jsonl is empty")
    event_ids = [event["event_id"] for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("duplicate event_id")
    identities: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identity_splits: dict[str, str] = {}
    for event in events:
        if event["split"] not in BUCKETS:
            raise ValueError(f"unknown split: {event['split']}")
        identity = event["identity_id"]
        if identity in identity_splits and identity_splits[identity] != event["split"]:
            raise ValueError(f"identity appears in multiple splits: {identity}")
        identity_splits[identity] = event["split"]
        if len(event["video_sha256"]) != 64:
            raise ValueError(f"invalid video_sha256: {event['event_id']}")
        if event["frame_start"] > event["frame_end"] or event["frame_end"] >= event["frame_count"]:
            raise ValueError(f"invalid event frame range: {event['event_id']}")
        if not event["view_conditions"]:
            raise ValueError(f"missing view condition: {event['event_id']}")
        identities[identity].append(event)

    summary_by_identity: dict[str, dict[str, int]] = {}
    for identity, rows in identities.items():
        summary = {
            "events": len(rows),
            "sessions": len({row["capture_session_id"] for row in rows}),
            "periods": len({row["capture_period"] for row in rows}),
            "views": len({view for row in rows for view in row["view_conditions"]}),
            "clothing": len({row["clothing_condition_id"] for row in rows}),
        }
        summary_by_identity[identity] = summary
        minimums = {"events": 12, "sessions": 4, "periods": 2, "views": 3, "clothing": 2}
        failures = [f"{key}={summary[key]}<{value}" for key, value in minimums.items() if summary[key] < value]
        if failures:
            raise ValueError(f"identity {identity} fails minimums: {', '.join(failures)}")

    split_counts = {split: sum(1 for value in identity_splits.values() if value == split) for split in BUCKETS}
    for split, minimum in BUCKETS.items():
        if split_counts[split] < minimum:
            raise ValueError(f"split {split} has {split_counts[split]} identities; {minimum} required")
    if len(identity_splits) < 60:
        raise ValueError(f"only {len(identity_splits)} distinct identities; at least 60 required")
    if len(events) < 720:
        raise ValueError(f"only {len(events)} events; at least 720 required")
    return {
        "event_count": len(events),
        "identity_count": len(identity_splits),
        "identity_counts_by_split": split_counts,
        "identity_metrics": dict(sorted(summary_by_identity.items())),
    }


def validate_production(root: Path, events: list[dict[str, Any]]) -> None:
    missing_tracks: list[str] = []
    missing_pose_dirs: list[str] = []
    bad_tracks: list[str] = []
    bad_poses: list[str] = []
    for event in events:
        tracks_path = root / event["tracks_path"]
        pose_dir = root / event["poses_dir"]
        if not tracks_path.is_file():
            missing_tracks.append(event["event_id"])
        else:
            try:
                track_payload = json.loads(tracks_path.read_text(encoding="utf-8"))
                if track_payload.get("event_id") != event["event_id"] or not track_payload.get("tracks"):
                    bad_tracks.append(event["event_id"])
            except (OSError, json.JSONDecodeError, AttributeError):
                bad_tracks.append(event["event_id"])
        if not pose_dir.is_dir() or not any(pose_dir.glob("*.npz")):
            missing_pose_dirs.append(event["event_id"])
        else:
            for pose_path in pose_dir.glob("*.npz"):
                if not pose_is_valid(pose_path):
                    bad_poses.append(str(pose_path))
    if missing_tracks or missing_pose_dirs or bad_tracks or bad_poses:
        details = []
        if missing_tracks:
            details.append(f"missing tracks={len(missing_tracks)}")
        if missing_pose_dirs:
            details.append(f"missing pose npz={len(missing_pose_dirs)}")
        if bad_tracks:
            details.append(f"invalid tracks={len(bad_tracks)}")
        if bad_poses:
            details.append(f"invalid pose npz={len(bad_poses)}")
        raise ValueError("production stage incomplete: " + ", ".join(details))


def pose_is_valid(path: Path) -> bool:
    try:
        import numpy as np

        required = {
            "keypoints_xy",
            "keypoint_scores",
            "valid_mask",
            "boxes_xyxy",
            "frame_indices",
            "timestamp",
            "track_id",
            "missing_mask",
        }
        with np.load(path, allow_pickle=False) as payload:
            if not required.issubset(payload.files):
                return False
            length = int(payload["keypoints_xy"].shape[0])
            if payload["keypoints_xy"].shape[1:] != (17, 2):
                return False
            if payload["keypoint_scores"].shape != (length, 17):
                return False
            if payload["valid_mask"].shape != (length, 17):
                return False
            if payload["boxes_xyxy"].shape != (length, 4):
                return False
            if payload["frame_indices"].shape != (length,):
                return False
            if payload["timestamp"].shape != (length,):
                return False
            if payload["track_id"].shape != (length,):
                return False
            if payload["missing_mask"].shape != (length,):
                return False
            for name in ("keypoints_xy", "keypoint_scores", "boxes_xyxy", "timestamp"):
                if not np.isfinite(payload[name]).all():
                    return False
            return length > 0
    except (ImportError, OSError, ValueError, KeyError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("metadata", "production"), default="metadata")
    args = parser.parse_args()
    events_path = args.dataset_root / "events.jsonl"
    if not events_path.is_file():
        if args.stage == "metadata":
            print(json.dumps(pending_report(args.dataset_root, "events.jsonl is not present yet"), ensure_ascii=False, indent=2))
            return 0
        raise FileNotFoundError(f"Missing {events_path}")
    events = read_events(events_path)
    if not events and args.stage == "metadata":
        print(json.dumps(pending_report(args.dataset_root, "events.jsonl has no verified events yet"), ensure_ascii=False, indent=2))
        return 0
    report = validate_events(events)
    if args.stage == "production":
        validate_production(args.dataset_root, events)
        report["production_training_eligible"] = True
    else:
        report["production_training_eligible"] = False
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
