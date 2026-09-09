"""Build a strict cross-event manifest from MEVID annotation indices.

The builder intentionally requires a separate source_metadata.jsonl file because
the public MEVID five-column track index does not prove raw-video timestamps,
walking activity, consent, or RTMPose provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


BUCKETS = (
    ("train-dev", "train", 40),
    ("calibration-known", "train", 10),
    ("calibration-unknown", "test", 10),
    ("locked-test-known", "test", 10),
    ("locked-test-unknown", "test", 10),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain JSON objects")
            rows.append(value)
    return rows


def read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_mevid_tracks(root: Path, split: str) -> list[dict[str, Any]]:
    info_path = root / f"track_{split}_info.txt"
    names_path = root / f"{split}_name.txt"
    if not info_path.is_file() or not names_path.is_file():
        raise FileNotFoundError(f"Missing MEVID files for {split}: {info_path} and {names_path}")

    names = read_names(names_path)
    tracks: list[dict[str, Any]] = []
    for index, line in enumerate(info_path.read_text(encoding="utf-8").splitlines()):
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{info_path}:{index + 1} must have five integer columns")
        start, end, identity, outfit, camera = (int(value) for value in values)
        if start < 0 or end < start or end >= len(names):
            raise ValueError(f"Invalid name range in {info_path}:{index + 1}")
        tracks.append(
            {
                "track_key": f"{split}:{index}",
                "annotation_split": split,
                "track_index": index,
                "identity_id": f"MEVID-{identity:04d}",
                "outfit_id": outfit,
                "camera_id": f"G{camera}",
                "chip_frame_count": end - start + 1,
                "first_chip_name": names[start],
                "last_chip_name": names[end],
            }
        )
    return tracks


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    values = read_jsonl(path)
    metadata: dict[str, dict[str, Any]] = {}
    for row in values:
        key = row.get("track_key")
        if not isinstance(key, str) or not key:
            raise ValueError("Every source metadata row needs a non-empty track_key")
        if key in metadata:
            raise ValueError(f"Duplicate source metadata track_key: {key}")
        metadata[key] = row
    return metadata


def require_metadata(row: dict[str, Any], track: dict[str, Any]) -> None:
    required = (
        "source_video_id",
        "source_video_path",
        "source_video_sha256",
        "source_video_frame_count",
        "raw_frame_start",
        "raw_frame_end",
        "fps",
        "capture_session_id",
        "capture_period",
        "camera_id",
        "view_condition",
        "walking_verified",
        "walking_direction",
        "load_condition",
        "consent_id",
        "license_status",
        "manual_review_id",
        "metadata_version",
    )
    missing = [key for key in required if key not in row or row[key] in (None, "")]
    if missing:
        raise ValueError(f"{track['track_key']} is missing source metadata: {', '.join(missing)}")
    if row["track_key"] != track["track_key"]:
        raise ValueError(f"Metadata key mismatch for {track['track_key']}")
    if row["annotation_split"] != track["annotation_split"] or int(row["track_index"]) != track["track_index"]:
        raise ValueError(f"Metadata index mismatch for {track['track_key']}")
    if row["license_status"] != "approved_for_this_use":
        raise ValueError(f"{track['track_key']} does not have approved_for_this_use license_status")
    if not row["walking_verified"]:
        raise ValueError(f"{track['track_key']} must be manually verified as walking before construction")
    if int(row["raw_frame_end"]) < int(row["raw_frame_start"]):
        raise ValueError(f"Invalid raw frame range for {track['track_key']}")
    if int(row["raw_frame_end"]) >= int(row["source_video_frame_count"]):
        raise ValueError(f"Raw frame range exceeds source video for {track['track_key']}")
    if len(str(row["source_video_sha256"])) != 64:
        raise ValueError(f"source_video_sha256 must be a 64-character digest for {track['track_key']}")


def metrics(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "events": len(rows),
        "sessions": len({row["capture_session_id"] for row in rows}),
        "periods": len({row["capture_period"] for row in rows}),
        "views": len({view for row in rows for view in row["view_conditions"]}),
        "clothing": len({row["clothing_condition_id"] for row in rows}),
    }


def is_eligible(rows: list[dict[str, Any]], minimums: dict[str, int]) -> bool:
    summary = metrics(rows)
    return all(summary[key] >= value for key, value in minimums.items())


def make_event(track: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    require_metadata(metadata, track)
    event_id = f"{track['identity_id']}_{metadata['source_video_id']}_track{track['track_index']:05d}"
    return {
        "event_id": event_id,
        "identity_id": track["identity_id"],
        "capture_session_id": metadata["capture_session_id"],
        "camera_id": metadata["camera_id"],
        "source_video_id": metadata["source_video_id"],
        "video_path": metadata["source_video_path"],
        "video_sha256": metadata["source_video_sha256"].lower(),
        "fps": float(metadata["fps"]),
        "frame_count": int(metadata["source_video_frame_count"]),
        "frame_start": int(metadata["raw_frame_start"]),
        "frame_end": int(metadata["raw_frame_end"]),
        "capture_period": metadata["capture_period"],
        "clothing_condition_id": f"outfit_{track['outfit_id']:03d}",
        "load_condition": metadata["load_condition"],
        "view_conditions": [metadata["view_condition"]],
        "walking_direction": metadata["walking_direction"],
        "consent_id": metadata["consent_id"],
        "source_track_key": track["track_key"],
        "source_id": "mevid-v1",
        "chip_frame_count": track["chip_frame_count"],
        "tracks_path": f"tracks/{event_id}.json",
        "poses_dir": f"poses/{event_id}",
        "occlusion_level": metadata.get("occlusion_level", "unknown"),
        "adverse_quality_condition": metadata.get("adverse_quality_condition", "unknown"),
        "manual_review_id": metadata["manual_review_id"],
        "metadata_version": metadata["metadata_version"],
        "event_status": "metadata_complete_pose_pending",
    }


def choose_identities(
    events_by_identity: dict[str, list[dict[str, Any]]],
    source_split: str,
    count: int,
    minimums: dict[str, int],
    already_used: set[str],
) -> list[str]:
    candidates = []
    for identity, rows in events_by_identity.items():
        if identity in already_used or {row["annotation_split"] for row in rows} != {source_split}:
            continue
        if is_eligible(rows, minimums):
            candidates.append((identity, metrics(rows)))
    candidates.sort(key=lambda item: (-item[1]["events"], item[0]))
    if len(candidates) < count:
        raise RuntimeError(
            f"Only {len(candidates)} eligible {source_split} identities remain; {count} are required. "
            "Do not lower the requirement or manufacture events."
        )
    return [identity for identity, _ in candidates[:count]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mevid-root", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-events", type=int, default=12)
    parser.add_argument("--min-sessions", type=int, default=4)
    parser.add_argument("--min-periods", type=int, default=2)
    parser.add_argument("--min-views", type=int, default=3)
    parser.add_argument("--min-clothing", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    minimums = {
        "events": args.min_events,
        "sessions": args.min_sessions,
        "periods": args.min_periods,
        "views": args.min_views,
        "clothing": args.min_clothing,
    }

    tracks = read_mevid_tracks(args.mevid_root, "train") + read_mevid_tracks(args.mevid_root, "test")
    metadata = load_metadata(args.source_metadata)
    events_by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_events: list[dict[str, Any]] = []

    for track in tracks:
        row = metadata.get(track["track_key"])
        if row is None:
            continue
        event = make_event(track, row)
        events_by_identity[event["identity_id"]].append({**event, "annotation_split": track["annotation_split"]})
        all_events.append({**event, "annotation_split": track["annotation_split"]})

    selected: dict[str, str] = {}
    used: set[str] = set()
    for bucket, source_split, count in BUCKETS:
        for identity in choose_identities(events_by_identity, source_split, count, minimums, used):
            selected[identity] = bucket
            used.add(identity)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    selected_events = [
        {key: value for key, value in event.items() if key != "annotation_split"}
        | {"split": selected[event["identity_id"]]}
        for event in all_events
        if event["identity_id"] in selected
    ]
    selected_events.sort(key=lambda event: event["event_id"])

    with (output / "events.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for event in selected_events:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    identity_split = {
        "schema_version": "cross-event-rgb-gait-v1",
        "identity_to_split": dict(sorted(selected.items())),
        "selection_rule": "identity/session/event disjoint; source metadata verified; deterministic event-count descending selection",
    }
    events_path = output / "events.jsonl"
    manifest_sha256 = hashlib.sha256(events_path.read_bytes()).hexdigest()
    (output / "identity_split.json").write_text(
        json.dumps(identity_split, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "status": "metadata_complete_pose_pending",
        "source_id": "mevid-v1",
        "event_count": len(selected_events),
        "identity_count": len(selected),
        "events_by_split": {
            bucket: sum(1 for event in selected_events if event["split"] == bucket) for bucket, _, _ in BUCKETS
        },
        "identity_metrics": {
            identity: metrics(rows) for identity, rows in sorted(events_by_identity.items()) if identity in selected
        },
        "minimums": minimums,
        "pose_and_track_files_present": False,
        "production_training_eligible": False,
        "events_jsonl_sha256": manifest_sha256,
    }
    (output / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
