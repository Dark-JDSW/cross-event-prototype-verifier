"""Build a non-destructive inventory and quality report for F:\\dataset assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


VIDEO_SUFFIXES = {".avi", ".mp4", ".mpg", ".mpeg", ".mov", ".mkv"}
ARCHIVE_SUFFIXES = {".rar", ".tar", ".zip", ".7z"}
ANNOTATION_SUFFIXES = {".json", ".csv", ".xml", ".xgtf", ".txt"}
HARD_NEGATIVE_NAMES = {
    "running",
    "run",
    "jogging",
    "jog",
    "boxing",
    "clapping",
    "digging",
    "throwing",
    "waving",
    "open",
    "close",
}


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _source(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "ucf_arg" in parts:
        return "UCF-ARG"
    if "pdestre" in parts:
        return "P-DESTRE"
    if "cmu_bvh" in parts:
        return "CMU-MoCap"
    return "unknown"


def _candidate_label(path: Path, source: str) -> str | None:
    words = {
        token.lower().replace("-", "_").replace(" ", "_")
        for token in path.parts
    }
    stem = path.stem.lower().replace("-", "_").replace(" ", "_")
    if "walking" in words or "walk" in words or "walk" in stem:
        return "WALK" if source != "CMU-MoCap" else "WALK_SYNTHETIC_CANDIDATE"
    if source == "UCF-ARG" and any(name in words or name in stem for name in HARD_NEGATIVE_NAMES):
        return "UNKNOWN"
    return None


def _probe_video(path: Path) -> tuple[dict[str, Any], list[str]]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return {}, ["video_unreadable"]
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    issues: list[str] = []
    if width <= 0 or height <= 0:
        issues.append("invalid_video_dimensions")
    if fps <= 0:
        issues.append("invalid_video_fps")
    if frames <= 0:
        issues.append("invalid_video_frame_count")
    return {"width": width, "height": height, "fps": fps, "frame_count": frames}, issues


def _probe_bvh(path: Path) -> tuple[dict[str, Any], list[str]]:
    frames: int | None = None
    frame_time: float | None = None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for _ in range(4096):
                line = handle.readline()
                if not line:
                    break
                lowered = line.strip().lower()
                if lowered.startswith("frames:"):
                    frames = int(lowered.split(":", 1)[1].strip())
                elif lowered.startswith("frame time:"):
                    frame_time = float(lowered.split(":", 1)[1].strip())
                if frames is not None and frame_time is not None:
                    break
    except (OSError, ValueError):
        return {}, ["bvh_header_unreadable"]
    issues: list[str] = []
    if frames is None or frames <= 0:
        issues.append("invalid_bvh_frame_count")
    if frame_time is None or frame_time <= 0:
        issues.append("invalid_bvh_frame_time")
    return {
        "frame_count": frames,
        "fps": (1.0 / frame_time) if frame_time and frame_time > 0 else None,
    }, issues


def _kind(path: Path) -> str:
    # macOS archive extraction can produce ``._name.ext`` AppleDouble sidecar
    # files next to the real media.  They are metadata, not corrupt videos;
    # keep them in the inventory but never send them through OpenCV.
    if path.name.startswith("._"):
        return "sidecar"
    suffix = path.suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix == ".bvh":
        return "bvh"
    if suffix in ARCHIVE_SUFFIXES:
        return "archive"
    if suffix in ANNOTATION_SUFFIXES:
        return "annotation"
    return "other"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"F:\dataset"))
    parser.add_argument("--require-pdestre", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"dataset root does not exist: {root}")

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    candidates: dict[tuple[str, int], list[Path]] = defaultdict(list)
    pdestre_video_count = 0

    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if reports in path.parents:
            continue
        size = path.stat().st_size
        source = _source(path)
        kind = _kind(path)
        metadata: dict[str, Any] = {}
        issues: list[str] = []
        if size <= 0:
            issues.append("zero_byte_file")
        if kind == "video":
            metadata, probe_issues = _probe_video(path)
            issues.extend(probe_issues)
            if source == "P-DESTRE":
                pdestre_video_count += 1
        elif kind == "bvh":
            metadata, probe_issues = _probe_bvh(path)
            issues.extend(probe_issues)
        if kind in {"video", "bvh", "archive"} and size > 0:
            candidates[(path.suffix.lower(), size)].append(path)
        records.append(
            {
                "path": _relative(path, root),
                "source": source,
                "kind": kind,
                "bytes": size,
                "candidate_label": _candidate_label(path, source),
                "camera_parameters": {
                    "azimuth_deg": None,
                    "pitch_deg": None,
                    "distance_m": None,
                    "status": "not_inferred_from_file_path",
                },
                "media": metadata,
                "issues": issues,
                "usable_for_derivation": not issues and kind in {"video", "bvh"},
            }
        )

    if args.require_pdestre and pdestre_video_count == 0:
        raise SystemExit("P-DESTRE extraction completed without any readable video asset")

    duplicate_groups: list[dict[str, Any]] = []
    duplicate_of: dict[str, str] = {}
    for paths in candidates.values():
        if len(paths) < 2:
            continue
        hashes: dict[str, list[Path]] = defaultdict(list)
        for path in paths:
            hashes[_sha256(path)].append(path)
        for digest, identical in hashes.items():
            if len(identical) < 2:
                continue
            canonical = _relative(identical[0], root)
            members = [_relative(path, root) for path in identical]
            duplicate_groups.append({"sha256": digest, "canonical": canonical, "members": members})
            for member in members[1:]:
                duplicate_of[member] = canonical

    for record in records:
        record["duplicate_of"] = duplicate_of.get(record["path"])
        if record["duplicate_of"] is not None:
            record["usable_for_derivation"] = False
            record["issues"].append("exact_duplicate")

    anomalies = [record for record in records if record["issues"]]
    labels = defaultdict(int)
    for record in records:
        if record["candidate_label"] and record["usable_for_derivation"]:
            labels[record["candidate_label"]] += 1
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "record_count": len(records),
        "usable_derivation_assets": sum(record["usable_for_derivation"] for record in records),
        "source_counts": {source: sum(record["source"] == source for record in records) for source in sorted({record["source"] for record in records})},
        "candidate_label_counts": dict(sorted(labels.items())),
        "anomaly_count": len(anomalies),
        "duplicate_group_count": len(duplicate_groups),
        "normalization_policy": "Raw assets are preserved. The catalogue provides one schema; no camera metadata, action labels, or repaired frames are fabricated.",
        "sidecar_policy": "AppleDouble sidecars are inventoried as metadata and excluded from media validation/derivation.",
        "next_derivation": "Create M=1/T=100/C=3 COCO-17 windows only from manually verified, video/person/event-disjoint tracks.",
    }

    with (reports / "external_walk_catalog.jsonl").open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (reports / "external_walk_catalog.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "source", "kind", "bytes", "candidate_label", "usable_for_derivation", "duplicate_of", "issues"])
        writer.writeheader()
        for record in records:
            writer.writerow({
                "path": record["path"],
                "source": record["source"],
                "kind": record["kind"],
                "bytes": record["bytes"],
                "candidate_label": record["candidate_label"],
                "usable_for_derivation": record["usable_for_derivation"],
                "duplicate_of": record["duplicate_of"] or "",
                "issues": ";".join(record["issues"]),
            })
    _write_json(reports / "external_walk_audit_summary.json", summary)
    _write_json(reports / "external_walk_anomalies.json", anomalies)
    _write_json(reports / "external_walk_duplicate_groups.json", duplicate_groups)
    (reports / "external_walk_processing_report.md").write_text(
        "# 外部行走数据处理报告\n\n"
        f"生成时间：{summary['generated_at']}\n\n"
        f"- 资产记录数：{summary['record_count']}\n"
        f"- 可用于后续派生的媒体：{summary['usable_derivation_assets']}\n"
        f"- 异常项：{summary['anomaly_count']}\n"
        f"- 精确重复组：{summary['duplicate_group_count']}\n\n"
        "## 处理边界\n\n"
        "原始文件未转码、未删除、未对缺失相机参数或动作标签进行猜测性补全。"
        "`external_walk_catalog.jsonl` 统一记录各源的格式和可用性；异常和精确重复仅从后续派生集排除。"
        "macOS AppleDouble (`._*`) 文件被识别为元数据 sidecar，不再误报为损坏视频。"
        "训练前仍须按原始视频、人物、事件和相机人工核验标签并构建无泄漏 split。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
