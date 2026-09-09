"""Collect a deduplicated, provenance-aware UAV-Human walk sample pool.

The source annotation file contains train-only oversampling clones.  This tool
keeps one original A152 sequence per ``frame_dir`` and records membership in
the existing CS-v1/CS-v2 splits so that diagnostic replay is not mistaken for
an independent test.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any


PROTOCOL_SPLITS = (
    "cs_v1_train",
    "cs_v1_val",
    "cs_v1_test",
    "cs_v2_train",
    "cs_v2_val",
    "cs_v2_test",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-pkl", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Optional MMAction2 DumpResults file keyed by video_id.",
    )
    return parser.parse_args()


def _deduplicate_walk_annotations(data: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for annotation in data.get("annotations", []):
        if annotation.get("action_id") != "A152":
            continue
        if annotation.get("is_train_oversample"):
            continue
        frame_dir = str(annotation.get("frame_dir", ""))
        if not frame_dir or frame_dir in seen:
            continue
        seen.add(frame_dir)
        rows.append(dict(annotation))
    return sorted(rows, key=lambda row: str(row["frame_dir"]))


def main() -> None:
    args = _parse_args()
    with args.input.open("rb") as handle:
        data = pickle.load(handle)

    predictions: dict[str, dict[str, Any]] = {}
    if args.predictions:
        with args.predictions.open("rb") as handle:
            for result in pickle.load(handle):
                scores = result["pred_score"].detach().cpu().reshape(-1).tolist()
                predicted = int(result["pred_label"].reshape(-1)[0])
                predictions[str(result["video_id"])] = {
                    "pred_label": predicted,
                    "pred_action": "walk" if predicted == 1 else "unknown",
                    "unknown_probability": float(scores[0]),
                    "walk_probability": float(scores[1]),
                    "correct": predicted == 1,
                }

    split_sets = {
        name: {str(value) for value in data.get("split", {}).get(name, [])}
        for name in PROTOCOL_SPLITS
    }
    rows = _deduplicate_walk_annotations(data)
    records: list[dict[str, Any]] = []
    grouped: dict[str, list[str]] = {
        "all_walk_source": [],
        "cs_v1_train": [],
        "cs_v1_val": [],
        "cs_v1_test": [],
        "cs_v2_train": [],
        "cs_v2_val": [],
        "cs_v2_test": [],
        "strict_cs_v1_test": [],
        "strict_cs_v2_test": [],
        "diagnostic_not_strict_cs_v2": [],
    }

    for row in rows:
        frame_dir = str(row["frame_dir"])
        memberships = [name for name, values in split_sets.items() if frame_dir in values]
        v1_split = next((name.removeprefix("cs_v1_") for name in memberships if name.startswith("cs_v1_")), None)
        v2_split = next((name.removeprefix("cs_v2_") for name in memberships if name.startswith("cs_v2_")), None)
        strict_v1 = v1_split == "test"
        strict_v2 = v2_split == "test"
        v2_train_or_val = v2_split in {"train", "val"}

        row.update(
            {
                "protocol_membership": memberships,
                "cs_v1_split": v1_split,
                "cs_v2_split": v2_split,
                "strict_test_cs_v1": strict_v1,
                "strict_test_cs_v2": strict_v2,
                "v2_train_or_val_leakage": v2_train_or_val,
                "evaluation_role": (
                    "strict_cs_v2_test"
                    if strict_v2
                    else "strict_cs_v1_test_but_not_cs_v2_independent"
                    if strict_v1
                    else "train_or_validation_replay"
                ),
            }
        )
        if frame_dir in predictions:
            row["diagnostic_prediction"] = predictions[frame_dir]
        records.append(row)
        grouped["all_walk_source"].append(frame_dir)
        for name in memberships:
            grouped[name].append(frame_dir)
        if strict_v1:
            grouped["strict_cs_v1_test"].append(frame_dir)
        if strict_v2:
            grouped["strict_cs_v2_test"].append(frame_dir)
        if not strict_v2:
            grouped["diagnostic_not_strict_cs_v2"].append(frame_dir)

    output = {
        "split": grouped,
        "annotations": records,
        "metadata": {
            "label": "walk",
            "action_id": "A152",
            "deduplicated_source_sequences": len(records),
            "source_pickle": str(args.input),
            "warning": (
                "Only strict_cs_v2_test is independent for a model trained on CS-v2. "
                "Other groups are diagnostic replay and may overlap training or validation."
            ),
            "prediction_source": str(args.predictions) if args.predictions else None,
        },
    }
    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_pkl.open("wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "label": "walk",
        "action_id": "A152",
        "source_pickle": str(args.input),
        "sample_count": len(records),
        "counts": {name: len(values) for name, values in grouped.items()},
        "samples": [
            {
                "frame_dir": row["frame_dir"],
                "source_file": row.get("source_file"),
                "person_id": row.get("person_id"),
                "setup_id": row.get("setup_id"),
                "video_id": row.get("video_id"),
                "session_id": row.get("session_id"),
                "total_frames": row.get("total_frames"),
                "cs_v1_split": row.get("cs_v1_split"),
                "cs_v2_split": row.get("cs_v2_split"),
                "strict_test_cs_v1": row.get("strict_test_cs_v1"),
                "strict_test_cs_v2": row.get("strict_test_cs_v2"),
                "evaluation_role": row.get("evaluation_role"),
                "diagnostic_prediction": row.get("diagnostic_prediction"),
            }
            for row in records
        ],
    }
    args.output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"pkl": str(args.output_pkl), "manifest": str(args.output_manifest), **manifest["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
