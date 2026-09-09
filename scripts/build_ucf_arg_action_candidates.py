"""Build a subject-disjoint UCF-ARG aerial WALK/UNKNOWN candidate manifest.

The manifest is intentionally video-level.  It does not pretend that a class
folder identifies a stable ByteTrack ID or a clean human pose sequence.  Feed
the selected videos through ``extract_action_branch_poses.py`` (or an
equivalent reviewed tracking pass) before deriving action-classifier windows.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import re


DEFAULT_UNKNOWN_ACTIONS = (
    "boxing",
    "clapping",
    "digging",
    "jogging",
    "openclosetrunk",
    "running",
    "throwing",
    "waving",
)
_PERSON_RE = re.compile(r"^person(?P<person>\d+)_", re.IGNORECASE)


def _subject_split(person_id: int) -> str:
    """Return the frozen 8/2/2 subject split used for candidate screening."""

    if 1 <= person_id <= 8:
        return "train"
    if 9 <= person_id <= 10:
        return "validation"
    if 11 <= person_id <= 12:
        return "test"
    raise ValueError(f"unexpected UCF-ARG person ID: {person_id}")


def build(
    root: Path,
    output: Path,
    *,
    include_carrying: bool = False,
) -> dict[str, object]:
    """Write a candidate manifest without creating overlapping windows."""

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"UCF-ARG aerial clip directory not found: {root}")
    unknown_actions = set(DEFAULT_UNKNOWN_ACTIONS)
    if include_carrying:
        unknown_actions.add("carrying")
    included_actions = {"walking", *unknown_actions}
    rows: list[dict[str, object]] = []
    excluded_actions: Counter[str] = Counter()
    for action_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        action = action_dir.name.strip().lower()
        files = sorted(action_dir.glob("*.avi"))
        if action not in included_actions:
            excluded_actions[action] += len(files)
            continue
        for video_path in files:
            match = _PERSON_RE.match(video_path.name)
            if match is None:
                raise ValueError(f"cannot infer subject from UCF-ARG file: {video_path}")
            person_id = int(match.group("person"))
            label = "walking" if action == "walking" else "unknown"
            rows.append(
                {
                    "schema_version": "ucf-arg-action-candidate-v1",
                    "source_dataset": "UCF-ARG",
                    "source_video_id": video_path.stem,
                    "source_video_path": str(video_path),
                    "source_action": action,
                    "action_label": label,
                    "label_status": "directory-label-candidate-requires-track-review",
                    "person_id": f"person{person_id:02d}",
                    "split": _subject_split(person_id),
                    "split_group": f"person{person_id:02d}",
                    "camera_view": "aerial",
                    "window_policy": "derive-only-after-track-review; never split overlapping windows across splits",
                    "pose_contract_target": "M=1,T=100,C=3,COCO-17,production-rtmpose",
                }
            )
    if not rows:
        raise ValueError(f"no eligible AVI files found in {root}")
    rows.sort(key=lambda item: (str(item["split"]), str(item["source_video_id"])))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    counts = Counter((str(row["split"]), str(row["action_label"])) for row in rows)
    return {
        "schema_version": "ucf-arg-action-candidate-v1",
        "output": str(output),
        "candidate_video_count": len(rows),
        "by_split_and_label": {
            f"{split}:{label}": counts[(split, label)]
            for split, label in sorted(counts)
        },
        "excluded_actions": dict(sorted(excluded_actions.items())),
        "carrying_policy": (
            "included_as_unknown" if include_carrying else "excluded_as_ambiguous"
        ),
        "next_step": (
            "Replay each selected source video through the production detector, "
            "ByteTrack and RTMPose; manually/semiautomatically choose one clean "
            "track span before generating fixed-Hz M=1/T=100/C=3 windows."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"F:\dataset\ucf_arg\extracted\aerial\aerial_clips"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"F:\dataset\ucf_arg\manifests\aerial_walk_unknown_candidates.jsonl"),
    )
    parser.add_argument(
        "--include-carrying",
        action="store_true",
        help="Treat carrying clips as UNKNOWN instead of retaining them for stress testing.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.root, args.output, include_carrying=args.include_carrying),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
