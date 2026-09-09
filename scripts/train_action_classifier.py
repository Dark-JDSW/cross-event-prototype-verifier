"""Train the small action gate from GUI annotations and extracted pose arrays.

Run ``extract_action_branch_poses.py`` first.  The script refuses to train on
unmatched/unknown annotations and holds out complete source videos, so adjacent
windows from one long video cannot leak into validation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cross_event_verifier.action_classifier import (  # noqa: E402
    ActionClassifierModel,
    write_training_manifest,
)
from cross_event_verifier.actions import ActionType  # noqa: E402


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


def train(
    manifest_path: Path,
    output_path: Path,
    *,
    validation_fraction: float = 0.20,
    minimum_samples_per_class: int = 3,
) -> dict[str, object]:
    rows = _read_jsonl(manifest_path)
    if not rows:
        raise ValueError(f"no extraction records found in {manifest_path}")
    base = manifest_path.parent
    usable: list[tuple[dict[str, Any], np.ndarray, np.ndarray | None, ActionType]] = []
    skipped: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("status") != "ok" or not row.get("pose_path"):
            skipped[str(row.get("status", "missing_pose"))] += 1
            continue
        action = ActionType.parse(row.get("action_label", "unknown"))
        if action == ActionType.UNKNOWN:
            skipped["unknown_action"] += 1
            continue
        pose_path = base / str(row["pose_path"])
        if not pose_path.is_file():
            skipped["missing_pose_file"] += 1
            continue
        with np.load(pose_path, allow_pickle=False) as data:
            required = {"keypoints_xy", "keypoint_scores"}
            if not required.issubset(data.files):
                raise ValueError(f"pose file lacks required arrays: {pose_path}")
            poses = np.concatenate(
                [
                    np.asarray(data["keypoints_xy"], dtype=np.float32),
                    np.asarray(data["keypoint_scores"], dtype=np.float32)[..., None],
                ],
                axis=2,
            )
            timestamp_values = None
            if "timestamp" in data.files:
                candidate_timestamps = np.asarray(data["timestamp"], dtype=np.float64).reshape(-1)
                if (
                    candidate_timestamps.shape != (poses.shape[0],)
                    or not np.isfinite(candidate_timestamps).all()
                ):
                    raise ValueError(
                        f"invalid timestamp tensor in {pose_path}: {candidate_timestamps.shape}"
                    )
                timestamp_values = candidate_timestamps
        if poses.ndim != 3 or poses.shape[1:] != (17, 3) or poses.shape[0] < 1:
            raise ValueError(f"invalid pose tensor in {pose_path}: {poses.shape}")
        usable.append((row, poses, timestamp_values, action))

    if not usable:
        raise ValueError("no usable labelled pose sequences; annotate and extract at least two actions")
    videos = sorted({str(row.get("source_video_id") or row.get("source_video_path")) for row, _poses, _timestamps, _action in usable})
    if len(videos) < 2 and validation_fraction > 0.0:
        raise ValueError("validation requires at least two independently labelled source videos")
    holdout_count = min(
        max(1, int(round(len(videos) * validation_fraction))) if validation_fraction > 0 else 0,
        max(0, len(videos) - 1),
    )
    held_out = set(videos[-holdout_count:]) if holdout_count else set()
    train_rows = [item for item in usable if str(item[0].get("source_video_id") or item[0].get("source_video_path")) not in held_out]
    validation_rows = [item for item in usable if str(item[0].get("source_video_id") or item[0].get("source_video_path")) in held_out]
    train_classes = {action for _row, _poses, _timestamps, action in train_rows}
    missing_validation_class = sorted(
        {action.value for _row, _poses, _timestamps, action in usable} - {action.value for action in train_classes}
    )
    if missing_validation_class:
        raise ValueError(
            "a labelled action appears only in held-out videos; add another source video for "
            + ", ".join(missing_validation_class)
        )
    model = ActionClassifierModel.fit(
        [poses for _row, poses, _timestamps, _action in train_rows],
        [action for _row, _poses, _timestamps, action in train_rows],
        timestamps=[timestamps for _row, _poses, timestamps, _action in train_rows],
        minimum_samples_per_class=minimum_samples_per_class,
    )
    output_path = Path(output_path)
    model.save(output_path)
    write_training_manifest(
        output_path.with_suffix(".json"),
        model=model,
        sample_count=len(train_rows),
    )
    correct = 0
    for _row, poses, timestamps, action in validation_rows:
        if model.predict(poses, timestamps).action_type == action:
            correct += 1
    report = {
        "schema_version": "pose-action-classifier-training-v1",
        "model_path": str(output_path),
        "model_version": model.model_version,
        "classes": [action.value for action in model.classes],
        "usable_annotation_count": len(usable),
        "train_annotation_count": len(train_rows),
        "validation_annotation_count": len(validation_rows),
        "train_source_video_count": len(videos) - len(held_out),
        "validation_source_video_count": len(held_out),
        "validation_accuracy": (correct / len(validation_rows) if validation_rows else None),
        "skipped": dict(skipped),
        "warning": (
            "This model is a routing gate, not a gait identity model. "
            "Keep action branches and formal WALK gallery protocols separate."
        ),
    }
    output_path.with_name(output_path.stem + "_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-samples-per-class", type=int, default=3)
    args = parser.parse_args()
    if not 0.0 <= args.validation_fraction < 1.0:
        raise SystemExit("--validation-fraction must be in [0, 1)")
    if args.minimum_samples_per_class < 2:
        raise SystemExit("--minimum-samples-per-class must be at least 2")
    print(
        json.dumps(
            train(
                args.manifest,
                args.output,
                validation_fraction=args.validation_fraction,
                minimum_samples_per_class=args.minimum_samples_per_class,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
