"""Local, resumable first-pass screening for the downloaded MEVID videos.

The first pass intentionally uses only YOLO/ByteTrack and RTMPose on sampled
frames.  It does not load OSNet or GPGait++, does not claim identity,
consent, or walking confirmation, and never promotes results into the
production dataset.  Each completed video is flushed to a worker shard so an
interrupted run can resume without losing the completed work.

Optional review frames are written locally for a small, user-selected subset.
This tool never uploads them; a user may choose to share a few images with a
multimodal service separately.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any


VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}
FAST_SCHEMA_VERSION = "mevid-local-fast-screen-v1"


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def _read_shard(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
            if not isinstance(value, dict) or not value.get("source_video_path"):
                raise ValueError(f"invalid fast-screen record in {path}:{line_number}")
            records[str(value["source_video_path"])] = value
    return records


def _source_paths(video_dir: Path, *, recursive: bool, contains: str | None) -> list[Path]:
    iterator = video_dir.rglob("*") if recursive else video_dir.iterdir()
    paths = [
        path.resolve()
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in VIDEO_SUFFIXES
        and (contains is None or contains in path.name)
    ]
    return sorted(paths, key=lambda path: str(path).lower())


def _nvidia_memory() -> tuple[int, int] | None:
    """Return (total MiB, used MiB) when nvidia-smi is available."""

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        first = result.stdout.strip().splitlines()[0]
        total, used = (int(part.strip()) for part in first.split(",", 1))
        return total, used
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _summarize_track(track_id: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pose_rows = [row for row in rows if row["pose_valid"]]
    scores = [row["mean_keypoint_score"] for row in rows if row["pose_valid"]]
    return {
        "track_id": int(track_id),
        "sampled_observations": len(rows),
        "first_frame_index": int(rows[0]["frame_index"]),
        "last_frame_index": int(rows[-1]["frame_index"]),
        "timestamp_span_seconds": round(
            _safe_float(rows[-1]["timestamp"]) - _safe_float(rows[0]["timestamp"]), 6
        ),
        "pose_observations": len(pose_rows),
        "pose_coverage": round(len(pose_rows) / max(len(rows), 1), 6),
        "mean_keypoint_score": round(sum(scores) / max(len(scores), 1), 6),
        "mean_detection_confidence": round(
            sum(_safe_float(row["detection_confidence"]) for row in rows)
            / max(len(rows), 1),
            6,
        ),
        "preview": rows[:: max(1, len(rows) // 12)],
    }


def _screen_one(
    video_path: Path,
    *,
    cv2: Any,
    np: Any,
    detector: Any,
    pose: Any,
    sample_fps: float,
    minimum_pose_coverage: float,
    image_size: int,
) -> dict[str, Any]:
    from cross_event_verifier.production_vision import ProductionVisionError

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {
            "schema_version": FAST_SCHEMA_VERSION,
            "source_video_id": video_path.stem,
            "source_video_path": str(video_path),
            "machine_status": "needs_manual_review",
            "machine_reasons": ["video_open_failed"],
            "source_image_size": int(image_size),
            "candidate_tracks": [],
        }

    fps = _safe_float(capture.get(cv2.CAP_PROP_FPS), 30.0)
    if fps <= 0:
        fps = 30.0
    declared_frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_stride = max(1, int(round(fps / sample_fps)))
    sampled_frames = 0
    decoded_frames = 0
    track_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    started = time.perf_counter()
    detector.reset()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index = decoded_frames
            decoded_frames += 1
            if frame_index % frame_stride:
                continue
            timestamp = frame_index / fps
            detections = detector.track(frame)
            boxes = [item.box for item in detections]
            poses = pose.extract(frame, boxes) if boxes else []
            if len(poses) != len(boxes):
                raise ProductionVisionError(
                    f"pose/detection count mismatch: poses={len(poses)}, boxes={len(boxes)}"
                )
            sampled_frames += 1
            for detection, keypoints in zip(detections, poses):
                array = None if keypoints is None else np.asarray(keypoints)
                pose_valid = bool(
                    array is not None
                    and array.shape == (17, 3)
                    and np.isfinite(array).all()
                    and np.any(array[:, 2] > 0.0)
                )
                mean_score = (
                    float(np.clip(array[:, 2], 0.0, 1.0).mean()) if pose_valid else 0.0
                )
                track_rows[int(detection.track_id)].append(
                    {
                        "frame_index": frame_index,
                        "timestamp": timestamp,
                        "bbox_xyxy": [float(value) for value in detection.box],
                        "track_id": int(detection.track_id),
                        "detection_confidence": float(detection.confidence),
                        "pose_valid": pose_valid,
                        "mean_keypoint_score": mean_score,
                    }
                )
    finally:
        capture.release()

    summaries = [
        _summarize_track(track_id, rows)
        for track_id, rows in sorted(track_rows.items())
    ]
    viable = [
        summary
        for summary in summaries
        if summary["sampled_observations"] >= 25
        and summary["pose_coverage"] >= minimum_pose_coverage
    ]
    reasons: list[str] = [
        "walking_not_verified_in_fast_stage",
        "identity_not_verified_in_fast_stage",
        "consent_not_verified_in_fast_stage",
    ]
    if not summaries:
        reasons.append("no_detected_track")
    if len(viable) != 1:
        reasons.append("zero_or_multiple_viable_tracks")
    status = "candidate_stage2" if len(viable) == 1 else "needs_manual_review"
    recommended = viable[0] if len(viable) == 1 else (
        max(summaries, key=lambda item: (item["pose_coverage"], item["sampled_observations"]))
        if summaries
        else None
    )
    return {
        "schema_version": FAST_SCHEMA_VERSION,
        "source_video_id": video_path.stem,
        "source_video_path": str(video_path),
        "fps": fps,
        "source_image_size": int(image_size),
        "frame_stride": frame_stride,
        "requested_sample_fps": sample_fps,
        "declared_frame_count": declared_frame_count,
        "decoded_frame_count": decoded_frames,
        "sampled_frame_count": sampled_frames,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "candidate_track_count": len(summaries),
        "viable_track_count": len(viable),
        "recommended_track_id": recommended["track_id"] if recommended else None,
        "candidate_tracks": summaries,
        "machine_status": status,
        "machine_reasons": reasons,
        "machine_only": True,
        "production_training_eligible": False,
        "local_review_frames_available": False,
    }


def _review_frame_indices(record: dict[str, Any], maximum: int) -> list[int]:
    values: set[int] = set()
    for track in record.get("candidate_tracks", []):
        for row in track.get("preview", []):
            try:
                values.add(int(row["frame_index"]))
            except (KeyError, TypeError, ValueError):
                continue
    ordered = sorted(values)
    if len(ordered) <= maximum:
        return ordered
    positions = [0, len(ordered) // 2, len(ordered) - 1]
    return sorted({ordered[position] for position in positions[:maximum]})


def _export_review_frames(
    record: dict[str, Any],
    output_dir: Path,
    *,
    cv2: Any,
    maximum: int,
) -> None:
    """Export a few local, annotated JPEGs; this function performs no upload."""

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(str(record["source_video_path"]))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return
    frame_indices = _review_frame_indices(record, maximum)
    frame_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for track in record.get("candidate_tracks", []):
        track_id = track.get("track_id")
        for row in track.get("preview", []):
            try:
                frame_rows[int(row["frame_index"])].append(
                    {**row, "track_id": track_id}
                )
            except (KeyError, TypeError, ValueError):
                continue
    exported: list[str] = []
    try:
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            for row in frame_rows.get(frame_index, []):
                x1, y1, x2, y2 = (int(round(float(value))) for value in row["bbox_xyxy"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 3)
                cv2.putText(
                    frame,
                    f"track {row['track_id']}",
                    (max(0, x1), max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 220, 0),
                    2,
                    cv2.LINE_AA,
                )
            height, width = frame.shape[:2]
            if width > 960:
                scaled_height = max(1, int(round(height * 960 / width)))
                frame = cv2.resize(frame, (960, scaled_height), interpolation=cv2.INTER_AREA)
            target = output_dir / (
                f"{record['source_video_id']}__frame-{frame_index:06d}.jpg"
            )
            if cv2.imwrite(str(target), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82]):
                exported.append(str(target.resolve()))
    finally:
        capture.release()
    record["local_review_frames"] = exported
    record["local_review_frames_available"] = bool(exported)


def _worker(
    worker_id: int,
    paths: list[Path],
    output_dir: Path,
    sample_fps: float,
    minimum_pose_coverage: float,
    device: str,
    image_size: int,
) -> None:
    import cv2
    import numpy as np
    from cross_event_verifier.production_vision import (
        ProductionVisionConfig,
        _RtmposeEstimator,
        _YoloByteTracker,
    )

    shard_path = output_dir / f"fast_worker_{worker_id:02d}.jsonl"
    # Read every existing shard, not just this worker's shard.  This keeps a
    # resumed run lossless when the worker count changes (for example, from
    # three to four); repartitioning must not cause already completed videos
    # to be processed again.
    completed: dict[str, dict[str, Any]] = {}
    for existing_shard in output_dir.glob("fast_worker_*.jsonl"):
        completed.update(_read_shard(existing_shard))
    config = ProductionVisionConfig(
        image_size=image_size,
        detector_inference_stride=1,
        appearance_stride=1,
        gait_inference_stride=1,
        device=device,
        require_cuda=True,
    )
    detector = _YoloByteTracker(config)
    pose = _RtmposeEstimator(
        config.pose_path,
        require_cuda=True,
        cuda_device_id=config.cuda_device_index,
    )
    with shard_path.open("a", encoding="utf-8") as stream:
        for path in paths:
            key = str(path.resolve())
            if key in completed:
                continue
            try:
                record = _screen_one(
                    path,
                    cv2=cv2,
                    np=np,
                    detector=detector,
                    pose=pose,
                    sample_fps=sample_fps,
                    minimum_pose_coverage=minimum_pose_coverage,
                    image_size=image_size,
                )
            except Exception as error:  # one bad file must not lose the batch
                record = {
                    "schema_version": FAST_SCHEMA_VERSION,
                    "source_video_id": path.stem,
                    "source_video_path": key,
                    "machine_status": "needs_manual_review",
                    "machine_reasons": ["screening_error"],
                    "error": f"{type(error).__name__}: {error}",
                    "candidate_tracks": [],
                    "source_image_size": int(image_size),
                    "machine_only": True,
                    "production_training_eligible": False,
                    "local_review_frames_available": False,
                }
            record["worker_id"] = worker_id
            record["completed_at_epoch"] = time.time()
            stream.write(_json_line(record))
            stream.flush()
            completed[key] = record
            print(
                json.dumps(
                    {
                        "worker_id": worker_id,
                        "source_video_id": record["source_video_id"],
                        "machine_status": record["machine_status"],
                        "elapsed_seconds": record.get("elapsed_seconds"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


def _merge(
    output_dir: Path,
    paths: list[Path],
    workers: int,
    sample_fps: float,
    *,
    review_frames_dir: Path | None = None,
    review_limit: int = 0,
) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for shard in sorted(output_dir.glob("fast_worker_*.jsonl")):
        records.update(_read_shard(shard))
    ordered = [records[str(path.resolve())] for path in paths if str(path.resolve()) in records]
    if review_frames_dir is not None and review_limit:
        import cv2

        review_records = [
            row for row in ordered if row["machine_status"] == "needs_manual_review"
        ][:review_limit]
        for row in review_records:
            _export_review_frames(
                row,
                review_frames_dir,
                cv2=cv2,
                maximum=3,
            )
    all_path = output_dir / "fast_screen.jsonl"
    candidate_path = output_dir / "fast_candidates.jsonl"
    review_path = output_dir / "fast_review_queue.jsonl"
    for target, values in (
        (all_path, ordered),
        (candidate_path, [row for row in ordered if row["machine_status"] == "candidate_stage2"]),
        (review_path, [row for row in ordered if row["machine_status"] == "needs_manual_review"]),
    ):
        target.write_text("".join(_json_line(row) for row in values), encoding="utf-8")
    report = {
        "schema_version": FAST_SCHEMA_VERSION,
        "video_dir": str(paths[0].parent) if paths else None,
        "output_dir": str(output_dir),
        "requested_count": len(paths),
        "completed_count": len(ordered),
        "candidate_count": sum(row["machine_status"] == "candidate_stage2" for row in ordered),
        "needs_manual_review": sum(row["machine_status"] == "needs_manual_review" for row in ordered),
        "sample_fps": sample_fps,
        "workers": workers,
        "status": "complete" if len(ordered) == len(paths) else "incomplete",
        "machine_only": True,
        "production_training_eligible": False,
        "multimodal_upload": "not_performed; local review frames are optional and user-selected",
        "review_frames_dir": str(review_frames_dir) if review_frames_dir else None,
        "review_frame_video_limit": review_limit,
    }
    (output_dir / "fast_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parse_workers(value: str) -> int:
    if value == "auto":
        return 0
    workers = int(value)
    if workers < 1:
        raise argparse.ArgumentTypeError("workers must be a positive integer or auto")
    return workers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--minimum-pose-coverage", type=float, default=0.80)
    parser.add_argument("--workers", type=_parse_workers, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--contains")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--review-frames-dir",
        type=Path,
        help="optional local output for up to three low-resolution frames per review video",
    )
    parser.add_argument(
        "--review-limit",
        type=int,
        default=0,
        help="maximum number of needs_manual_review videos to export locally (default: 0)",
    )
    args = parser.parse_args()

    if not 3.0 <= args.sample_fps <= 6.0:
        parser.error("sample-fps must be between 3 and 6")
    if not 0.0 < args.minimum_pose_coverage <= 1.0:
        parser.error("minimum-pose-coverage must be in (0, 1]")
    if args.image_size < 320:
        parser.error("image-size must be at least 320")
    if args.review_limit < 0:
        parser.error("review-limit cannot be negative")
    if args.review_limit and args.review_frames_dir is None:
        parser.error("review-limit requires --review-frames-dir")
    video_dir = args.video_dir.resolve()
    if not video_dir.is_dir():
        raise FileNotFoundError(f"video directory does not exist: {video_dir}")
    paths = _source_paths(video_dir, recursive=args.recursive, contains=args.contains)
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise ValueError(f"no supported videos found in {video_dir}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    memory = _nvidia_memory()
    if args.workers == 0:
        # Auto mode leaves a large safety margin for the desktop and decoder.
        # Bound parallel decoding/pose workers to avoid exhausting GPU memory.
        workers = 1 if memory is None else max(1, min(3, (memory[0] - memory[1]) // 1536))
    else:
        workers = args.workers
    workers = min(workers, len(paths))
    print(
        json.dumps(
            {
                "schema_version": FAST_SCHEMA_VERSION,
                "video_count": len(paths),
                "workers": workers,
                "sample_fps": args.sample_fps,
                "gpu_memory_mib": memory,
                "multimodal_upload": False,
                "production_training_eligible": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    partitions = [paths[index::workers] for index in range(workers)]
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_worker,
            args=(
                worker_id,
                partition,
                output_dir,
                args.sample_fps,
                args.minimum_pose_coverage,
                args.device,
                args.image_size,
            ),
        )
        for worker_id, partition in enumerate(partitions)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    if any(process.exitcode != 0 for process in processes):
        _merge(
            output_dir,
            paths,
            workers,
            args.sample_fps,
            review_frames_dir=args.review_frames_dir,
            review_limit=args.review_limit,
        )
        return 2
    report = _merge(
        output_dir,
        paths,
        workers,
        args.sample_fps,
        review_frames_dir=args.review_frames_dir,
        review_limit=args.review_limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
