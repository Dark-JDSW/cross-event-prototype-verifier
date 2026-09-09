"""Download official MEVID videos and export machine-screened candidates.

This tool deliberately keeps machine-generated candidates separate from the
production manifest.  Identity binding, walking confirmation, and consent
review must happen before candidates are promoted into ``tracks/`` and
``poses/``.  The ``screen`` command is an automated pre-screen only; it never
creates a production event or claims that a human review happened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


DEFAULT_URL_LIST_URL = (
    "https://raw.githubusercontent.com/Kitware/MEVID/main/"
    "mevid-v1-video-URLS.txt"
)
ALLOWED_VIEWS = {"side", "oblique", "front", "back", "transition", "unknown"}
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}
SCREEN_SCHEMA_VERSION = "mevid-machine-screen-v1"
SCREEN_MINIMUMS = {
    "observed_frames": 25,
    "pose_coverage": 0.90,
    "leg_coverage": 0.85,
    "walking_ratio": 0.55,
    "gait_cycles": 0.75,
    "action_confidence": 0.70,
    "action_completion": 0.50,
    "max_occlusion": 0.20,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, timeout: float, retries: int) -> dict[str, Any]:
    """Download one HTTPS resource with a resumable ``.part`` file."""

    if urlparse(url).scheme.lower() != "https":
        raise ValueError(f"only HTTPS URLs are accepted: {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return {
            "url": url,
            "path": str(destination),
            "status": "existing",
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
        }

    partial = destination.with_name(destination.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            offset = partial.stat().st_size if partial.is_file() else 0
            headers = {"User-Agent": "cross-event-training-dataset/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = Request(url, headers=headers)
            with urlopen(request, timeout=timeout) as response:
                status = int(getattr(response, "status", 200) or 200)
                append = offset > 0 and status == 206
                if offset and not append:
                    offset = 0
                mode = "ab" if append else "wb"
                with partial.open(mode) as stream:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        stream.write(block)
            os.replace(partial, destination)
            return {
                "url": url,
                "path": str(destination),
                "status": "downloaded",
                "attempt": attempt,
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        except (OSError, TimeoutError, ValueError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(1.0)
    raise RuntimeError(f"download failed after {retries} attempts: {url}: {last_error}")


def _read_urls(path: Path) -> list[str]:
    urls = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        if urlparse(value).scheme.lower() != "https":
            raise ValueError(f"{path}:{line_number} is not an HTTPS URL")
        urls.append(value)
    return urls


def download_command(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    url_list_path = (
        args.url_list.resolve()
        if args.url_list is not None
        else output_dir.parent / "mevid-v1-video-URLS.txt"
    )
    if not url_list_path.is_file():
        _download(args.url_list_url, url_list_path, args.timeout, args.retries)
    urls = _read_urls(url_list_path)
    if args.contains:
        urls = [url for url in urls if args.contains in url]
    if args.limit is not None:
        urls = urls[: args.limit]

    records: list[dict[str, Any]] = []
    for index, url in enumerate(urls, 1):
        name = Path(unquote(urlparse(url).path)).name
        if not name:
            raise ValueError(f"cannot derive a filename from {url}")
        try:
            record = _download(url, output_dir / name, args.timeout, args.retries)
        except RuntimeError as error:
            record = {"url": url, "status": "error", "error": str(error)}
        record["index"] = index
        records.append(record)
        print(json.dumps(record, ensure_ascii=False))

    report = {
        "source": "mevid-v1",
        "url_list": str(url_list_path),
        "requested_count": len(urls),
        "downloaded_or_existing": sum(
            record["status"] in {"downloaded", "existing"} for record in records
        ),
        "failed": sum(record["status"] == "error" for record in records),
        "limited": args.limit is not None,
        "records": records,
    }
    (output_dir / "download_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if report["failed"] == 0 else 2


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            events.append(value)
    return events


def _view_name(value: Any) -> str:
    value = str(value or "unknown").lower()
    if value == "frontal":
        value = "front"
    return value if value in ALLOWED_VIEWS else "unknown"


def _frame_record(track: Any, frame_index: int, timestamp: float) -> dict[str, Any]:
    quality = track.quality
    subtracklet = getattr(track, "clean_subtracklet", None)
    metadata = dict(getattr(track, "metadata", {}) or {})
    return {
        "frame_index": int(frame_index),
        "timestamp": float(timestamp),
        "source_track_id": int(track.track_id),
        "segment_id": int(getattr(subtracklet, "segment_id", 0)),
        "bbox_xyxy": [float(value) for value in track.box],
        "detection_score": float(track.detection_confidence),
        "valid_pose": bool(float(getattr(quality, "keypoint_visibility", 0.0)) > 0.0),
        "valid_leg": bool(float(getattr(quality, "leg_visibility", 0.0) or 0.0) >= 0.45),
        "keypoint_visibility": float(getattr(quality, "keypoint_visibility", 0.0)),
        "leg_visibility": float(getattr(quality, "leg_visibility", 0.0) or 0.0),
        "gait_quality": float(getattr(quality, "gait_branch_quality", 0.0) or 0.0),
        "walking_ratio": float(getattr(quality, "walking_ratio", 0.0)),
        "gait_cycles": float(getattr(quality, "gait_cycles", 0.0)),
        "posture_condition": _view_name(getattr(quality, "view_angle", None)),
        "posture_confidence": float(getattr(quality, "keypoint_visibility", 0.0)),
        "occlusion": float(getattr(quality, "occlusion", 0.0)),
        "feet_visible": bool(getattr(quality, "leg_visibility", 0.0) >= 0.45),
        "track_id_switch": bool(getattr(quality, "id_switches", 0)),
        "track_gap": bool(getattr(quality, "track_gap_frames", 0)),
        "detector_gap": bool(getattr(quality, "detector_gap_frames", 0)),
        "box_truncated": "box_truncated" in getattr(quality, "reasons", ()),
        "action_type": str(metadata.get("action_type", "UNKNOWN")),
        "action_confidence": float(metadata.get("action_confidence", 0.0)),
        "action_quality": str(metadata.get("action_quality", "INVALID")),
        "action_completion": float(metadata.get("action_completion", 0.0)),
        "missing_mask": False,
    }


def _finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(sum(finite) / len(finite)) if finite else 0.0


def _track_screen_summary(track_key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one source-local track without making an identity decision."""

    observed_frames = len(rows)
    frame_start = min(row["frame_index"] for row in rows)
    frame_end = max(row["frame_index"] for row in rows)
    span_frames = frame_end - frame_start + 1
    positive_deltas = [
        row["timestamp"] - previous["timestamp"]
        for previous, row in zip(rows, rows[1:])
        if row["timestamp"] > previous["timestamp"]
    ]
    fps = _finite_mean([1.0 / delta for delta in positive_deltas])
    pose_coverage = sum(bool(row["valid_pose"]) for row in rows) / max(observed_frames, 1)
    leg_coverage = sum(bool(row["valid_leg"]) for row in rows) / max(observed_frames, 1)
    best_action = max(
        rows,
        key=lambda row: (
            float(row["action_confidence"]),
            float(row["action_completion"]),
            float(row["walking_ratio"]),
        ),
    )
    reasons: list[str] = []
    if observed_frames < SCREEN_MINIMUMS["observed_frames"]:
        reasons.append("too_few_observed_frames")
    if pose_coverage < SCREEN_MINIMUMS["pose_coverage"]:
        reasons.append("pose_coverage_below_machine_gate")
    if leg_coverage < SCREEN_MINIMUMS["leg_coverage"]:
        reasons.append("leg_coverage_below_machine_gate")
    if max(float(row["occlusion"]) for row in rows) > SCREEN_MINIMUMS["max_occlusion"]:
        reasons.append("occlusion_above_machine_gate")
    if any(row["track_id_switch"] for row in rows):
        reasons.append("track_id_switch")
    if any(row["track_gap"] or row["detector_gap"] for row in rows):
        reasons.append("track_or_detector_gap")
    if any(row["box_truncated"] for row in rows):
        reasons.append("box_truncated")
    if max(float(row["walking_ratio"]) for row in rows) < SCREEN_MINIMUMS["walking_ratio"]:
        reasons.append("walking_ratio_below_machine_gate")
    if max(float(row["gait_cycles"]) for row in rows) < SCREEN_MINIMUMS["gait_cycles"]:
        reasons.append("gait_cycles_below_machine_gate")
    if best_action["action_type"] != "WALK":
        reasons.append("walking_action_not_confirmed")
    if best_action["action_quality"] != "STRONG":
        reasons.append("walking_action_quality_not_strong")
    if float(best_action["action_confidence"]) < SCREEN_MINIMUMS["action_confidence"]:
        reasons.append("walking_action_confidence_below_machine_gate")
    if float(best_action["action_completion"]) < SCREEN_MINIMUMS["action_completion"]:
        reasons.append("walking_action_completion_below_machine_gate")
    sample_stride = max(1, len(rows) // 36)
    preview_rows = rows[::sample_stride]
    if rows[-1] not in preview_rows:
        preview_rows.append(rows[-1])

    return {
        "track_key": track_key,
        "track_id": int(rows[0]["source_track_id"]),
        "segment_id": int(rows[0]["segment_id"]),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "span_frames": span_frames,
        "observed_frames": observed_frames,
        "timestamp_span_seconds": float(rows[-1]["timestamp"] - rows[0]["timestamp"]),
        "estimated_fps": fps,
        "pose_coverage": float(pose_coverage),
        "leg_coverage": float(leg_coverage),
        "mean_detection_score": _finite_mean([float(row["detection_score"]) for row in rows]),
        "mean_keypoint_visibility": _finite_mean([float(row["keypoint_visibility"]) for row in rows]),
        "max_occlusion": max(float(row["occlusion"]) for row in rows),
        "max_gait_quality": max(float(row["gait_quality"]) for row in rows),
        "max_walking_ratio": max(float(row["walking_ratio"]) for row in rows),
        "max_gait_cycles": max(float(row["gait_cycles"]) for row in rows),
        "best_action": {
            "type": best_action["action_type"],
            "confidence": float(best_action["action_confidence"]),
            "quality": best_action["action_quality"],
            "completion": float(best_action["action_completion"]),
        },
        "view_candidates": sorted({row["posture_condition"] for row in rows}),
        "preview_boxes": [
            {
                "frame_index": int(row["frame_index"]),
                "bbox_xyxy": row["bbox_xyxy"],
                "track_key": track_key,
            }
            for row in preview_rows
        ],
        "machine_reasons": reasons,
        "machine_eligible": not reasons,
    }


def _screen_video(video_path: Path, adapter: Any, cv2: Any, np: Any) -> dict[str, Any]:
    """Run the fixed production chain and return an auditable video decision."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {
            "machine_status": "needs_manual_review",
            "machine_reasons": ["video_open_failed"],
            "candidate_tracks": [],
        }
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    declared_frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    track_rows: dict[str, list[dict[str, Any]]] = {}
    processed_frames = 0
    try:
        adapter.reset()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            timestamp = timestamp_ms / 1000.0 if timestamp_ms > 0 else processed_frames / fps
            for track in adapter.process_at(frame, timestamp=timestamp):
                row = _frame_record(track, processed_frames, timestamp)
                track_key = f"{row['source_track_id']}:{row['segment_id']}"
                track_rows.setdefault(track_key, []).append(row)
            processed_frames += 1
    finally:
        capture.release()

    summaries = [
        _track_screen_summary(track_key, rows)
        for track_key, rows in sorted(track_rows.items())
    ]
    eligible = [summary for summary in summaries if summary["machine_eligible"]]
    reasons: list[str] = []
    if not summaries:
        reasons.append("no_detected_track")
    if len(summaries) != 1:
        reasons.append("multiple_or_zero_tracks_need_selection")
    if len(eligible) != 1:
        reasons.append("no_unique_machine_eligible_track")
    status = "auto_pass" if len(summaries) == 1 and len(eligible) == 1 else "needs_manual_review"
    recommended = eligible[0] if len(eligible) == 1 else (
        max(summaries, key=lambda summary: (summary["max_gait_quality"], summary["pose_coverage"]))
        if summaries else None
    )
    if recommended is not None and not recommended["machine_eligible"]:
        reasons.extend(recommended["machine_reasons"])
    return {
        "schema_version": SCREEN_SCHEMA_VERSION,
        "source_video_id": video_path.stem,
        "source_video_path": str(video_path.resolve()),
        "fps": fps,
        "declared_frame_count": declared_frame_count,
        "processed_frame_count": processed_frames,
        "candidate_track_count": len(summaries),
        "recommended_track_key": recommended["track_key"] if recommended else None,
        "candidate_tracks": summaries,
        "machine_status": status,
        "machine_reasons": sorted(set(reasons)),
    }


def screen_command(args: argparse.Namespace) -> int:
    try:
        import cv2
        import numpy as np
        from cross_event_verifier.production_vision import ProductionVisionAdapter
    except ImportError as error:
        raise RuntimeError(
            "machine screening requires the project's production dependencies "
            "(opencv-python, torch, ultralytics, and onnxruntime-gpu)"
        ) from error

    video_dir = args.video_dir.resolve()
    if not video_dir.is_dir():
        raise FileNotFoundError(f"video directory does not exist: {video_dir}")
    iterator = video_dir.rglob("*") if args.recursive else video_dir.iterdir()
    paths = sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if args.contains:
        paths = [path for path in paths if args.contains in path.name]
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise ValueError(f"no supported video files found in {video_dir}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = ProductionVisionAdapter()
    adapter.preload()
    records: list[dict[str, Any]] = []
    for index, path in enumerate(paths, 1):
        try:
            record = _screen_video(path, adapter, cv2, np)
        except Exception as error:
            record = {
                "schema_version": SCREEN_SCHEMA_VERSION,
                "source_video_id": path.stem,
                "source_video_path": str(path.resolve()),
                "candidate_tracks": [],
                "candidate_track_count": 0,
                "machine_status": "needs_manual_review",
                "machine_reasons": ["screening_error"],
                "error": str(error),
            }
        record["index"] = index
        records.append(record)
        print(json.dumps(record, ensure_ascii=False))

    all_path = output_dir / "machine_screen.jsonl"
    review_path = output_dir / "review_queue.jsonl"
    passed_path = output_dir / "auto_passed.jsonl"
    for path, values in (
        (all_path, records),
        (review_path, [row for row in records if row["machine_status"] == "needs_manual_review"]),
        (passed_path, [row for row in records if row["machine_status"] == "auto_pass"]),
    ):
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in values),
            encoding="utf-8",
        )
    report = {
        "schema_version": SCREEN_SCHEMA_VERSION,
        "video_dir": str(video_dir),
        "output_dir": str(output_dir),
        "requested_count": len(paths),
        "auto_passed": sum(row["machine_status"] == "auto_pass" for row in records),
        "needs_manual_review": sum(
            row["machine_status"] == "needs_manual_review" for row in records
        ),
        "machine_gate": SCREEN_MINIMUMS,
        "status": "complete",
    }
    (output_dir / "screen_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def process_command(args: argparse.Namespace) -> int:
    try:
        import cv2
        import numpy as np
        from cross_event_verifier.production_vision import ProductionVisionAdapter
    except ImportError as error:
        raise RuntimeError(
            "batch processing requires the project's production dependencies "
            "(opencv-python, torch, ultralytics, and onnxruntime-gpu)"
        ) from error

    events_path = args.dataset_root / "events.jsonl"
    events = _read_events(events_path)
    if not events:
        raise ValueError(f"no events found in {events_path}")
    candidate_root = args.candidate_root.resolve()
    candidate_root.mkdir(parents=True, exist_ok=True)
    adapter = ProductionVisionAdapter()
    adapter.preload()
    report: dict[str, Any] = {
        "status": "candidate_review_required",
        "dataset_root": str(args.dataset_root.resolve()),
        "candidate_root": str(candidate_root),
        "event_count": len(events),
        "processed_events": 0,
        "candidate_track_count": 0,
        "errors": [],
    }

    for event in events:
        event_id = str(event["event_id"])
        video_path = Path(str(event["video_path"]))
        if not video_path.is_absolute():
            video_path = args.dataset_root / video_path
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            report["errors"].append({"event_id": event_id, "error": f"cannot open {video_path}"})
            continue
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(fps) or fps <= 0:
                fps = float(event["fps"])
            start = int(event["frame_start"])
            end = int(event["frame_end"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            adapter.reset()
            observations: dict[int, list[dict[str, Any]]] = {}
            poses: dict[int, list[dict[str, Any]]] = {}
            frame_index = start
            while frame_index <= end:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                timestamp = timestamp_ms / 1000.0 if timestamp_ms > 0 else frame_index / fps
                outputs = adapter.process_at(frame, timestamp=timestamp)
                for track in outputs:
                    track_id = int(track.track_id)
                    state = adapter.states.get(track_id)
                    if state is None or not state.poses:
                        continue
                    pose = np.asarray(state.poses[-1], dtype=np.float32)
                    if pose.shape != (17, 3):
                        continue
                    observations.setdefault(track_id, []).append(
                        _frame_record(track, frame_index, timestamp)
                    )
                    poses.setdefault(track_id, []).append(
                        {
                            "xy": pose[:, :2].copy(),
                            "scores": pose[:, 2].copy(),
                            "box": np.asarray(track.box, dtype=np.float32),
                            "frame": frame_index,
                            "timestamp": timestamp,
                        }
                    )
                frame_index += 1
        finally:
            cap.release()

        event_root = candidate_root / event_id
        pose_root = event_root / "poses"
        pose_root.mkdir(parents=True, exist_ok=True)
        candidates = []
        for track_id in sorted(observations):
            frame_rows = observations[track_id]
            pose_rows = poses[track_id]
            xy = np.asarray([row["xy"] for row in pose_rows], dtype=np.float32)
            scores = np.asarray([row["scores"] for row in pose_rows], dtype=np.float32)
            boxes = np.asarray([row["box"] for row in pose_rows], dtype=np.float32)
            frames = np.asarray([row["frame"] for row in pose_rows], dtype=np.int64)
            timestamps = np.asarray([row["timestamp"] for row in pose_rows], dtype=np.float64)
            np.savez_compressed(
                pose_root / f"track_{track_id}.npz",
                keypoints_xy=xy,
                keypoint_scores=scores,
                valid_mask=scores > 0.0,
                boxes_xyxy=boxes,
                frame_indices=frames,
                timestamp=timestamps,
                track_id=np.full(frames.shape, track_id, dtype=np.int64),
                missing_mask=np.zeros(frames.shape, dtype=np.bool_),
            )
            candidates.append({"track_id": track_id, "frames": frame_rows})

        (event_root / "tracks.json").write_text(
            json.dumps(
                {
                    "event_id": event_id,
                    "source_track_key": event.get("source_track_key"),
                    "selection_required": True,
                    "tracks": candidates,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        report["processed_events"] += 1
        report["candidate_track_count"] += len(candidates)

    (candidate_root / "batch_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not report["errors"] else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="download MEVID source videos")
    download.add_argument("--url-list", type=Path)
    download.add_argument("--url-list-url", default=DEFAULT_URL_LIST_URL)
    download.add_argument("--output-dir", type=Path, required=True)
    download.add_argument("--contains")
    download.add_argument("--limit", type=int)
    download.add_argument("--timeout", type=float, default=60.0)
    download.add_argument("--retries", type=int, default=3)
    download.set_defaults(handler=download_command)

    process = subparsers.add_parser("process", help="export reviewable candidate tracks and poses")
    process.add_argument("--dataset-root", type=Path, required=True)
    process.add_argument("--candidate-root", type=Path, required=True)
    process.set_defaults(handler=process_command)

    screen = subparsers.add_parser(
        "screen",
        help="screen raw videos and split auto-passed videos from the review queue",
    )
    screen.add_argument("--video-dir", type=Path, required=True)
    screen.add_argument("--output-dir", type=Path, required=True)
    screen.add_argument("--contains")
    screen.add_argument("--limit", type=int)
    screen.add_argument("--recursive", action="store_true")
    screen.set_defaults(handler=screen_command)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
