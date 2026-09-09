"""Run the production visual adapter on the small cross-clothing validation set.

The four clips contain one person only, so this is a positive cross-event
regression/diagnostic set.  The report deliberately does not manufacture
unknown probes or an FPIR value; a separate target-domain unknown split is
required for open-set calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cross_event_verifier.production_vision import (  # noqa: E402
    ProductionVisionAdapter,
    ProductionVisionConfig,
)


PAIR_DEFINITIONS = {
    "cross_clothing_sideway": (
        "P1_sideway_blue.mp4",
        "P1_sideway_gray.mp4",
    ),
    "cross_clothing_straight": (
        "P1_straight_blue.mp4",
        "P1_straight_gray.mp4",
    ),
    "same_clothing_blue_view_control": (
        "P1_sideway_blue.mp4",
        "P1_straight_blue.mp4",
    ),
    "same_clothing_gray_view_control": (
        "P1_sideway_gray.mp4",
        "P1_straight_gray.mp4",
    ),
    "cross_view_and_clothing": (
        "P1_sideway_blue.mp4",
        "P1_straight_gray.mp4",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unit_mean(values: list[np.ndarray]) -> np.ndarray | None:
    if not values:
        return None
    normalized = []
    for value in values:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-8 and np.isfinite(norm) and np.isfinite(vector).all():
            normalized.append(vector / norm)
    if not normalized:
        return None
    centroid = np.mean(np.stack(normalized), axis=0)
    norm = float(np.linalg.norm(centroid))
    return centroid / norm if norm > 1e-8 else None


def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None or left.shape != right.shape:
        return None
    return float(np.dot(left, right))


def evaluate(val_dir: Path) -> dict[str, object]:
    files = sorted(val_dir.glob("P1_*.mp4"))
    if not files:
        raise FileNotFoundError(f"no P1_*.mp4 files found in {val_dir}")
    adapter = ProductionVisionAdapter(ProductionVisionConfig())
    adapter.preload()
    adapter.warmup()
    events: dict[str, dict[str, object]] = {}
    started = time.perf_counter()
    for path in files:
        adapter.reset()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open validation video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = 0
        track_outputs = 0
        track_frames = 0
        fresh_appearance = 0
        gait_outputs = 0
        generations: set[tuple[int, int]] = set()
        segments: set[str] = set()
        appearance_vectors: list[np.ndarray] = []
        gait_vectors: list[np.ndarray] = []
        pose_coverages: list[float] = []
        qualities: list[float] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            tracks = adapter.process_at(
                frame,
                timestamp=frame_count / fps if fps > 0 else None,
            )
            frame_count += 1
            track_frames += int(bool(tracks))
            track_outputs += len(tracks)
            for track in tracks:
                metadata = track.metadata
                if metadata.get("appearance_fresh"):
                    fresh_appearance += 1
                    if track.features.has_appearance:
                        appearance_vectors.append(
                            np.asarray(track.features.appearance, dtype=np.float32)
                        )
                if metadata.get("appearance_generation") is not None:
                    generations.add(
                        (track.track_id, int(metadata["appearance_generation"]))
                    )
                if track.features.has_gait:
                    gait_outputs += 1
                    gait_vectors.append(np.asarray(track.features.gait, dtype=np.float32))
                if track.clean_subtracklet is not None:
                    segments.add(track.clean_subtracklet.subtracklet_id)
                pose_coverages.append(track.quality.real_pose_coverage())
                qualities.append(track.quality.overall())
        capture.release()
        events[path.name] = {
            "sha256": _sha256(path),
            "frames": frame_count,
            "fps": fps,
            "track_frames": track_frames,
            "track_outputs": track_outputs,
            "appearance_fresh_outputs": fresh_appearance,
            "appearance_generations": len(generations),
            "segments": sorted(segments),
            "gait_outputs": gait_outputs,
            "mean_pose_coverage": (
                float(np.mean(pose_coverages)) if pose_coverages else None
            ),
            "mean_quality": float(np.mean(qualities)) if qualities else None,
            "appearance_embedding": _unit_mean(appearance_vectors),
            "gait_embedding": _unit_mean(gait_vectors),
        }

    pair_scores = []
    for label, (left_name, right_name) in PAIR_DEFINITIONS.items():
        left = events.get(left_name)
        right = events.get(right_name)
        if left is None or right is None:
            continue
        pair_scores.append(
            {
                "pair": label,
                "left": left_name,
                "right": right_name,
                "appearance_cosine": _cosine(
                    left["appearance_embedding"], right["appearance_embedding"]
                ),
                "gait_cosine": _cosine(
                    left["gait_embedding"], right["gait_embedding"]
                ),
            }
        )
    for event in events.values():
        event.pop("appearance_embedding", None)
        event.pop("gait_embedding", None)
    return {
        "protocol": "val-cross-clothing-positive-regression-v1",
        "identity_count": 1,
        "unknown_probe_count": 0,
        "impostor_pair_count": 0,
        "open_set_metrics": {
            "fpir": None,
            "fnir": None,
            "status": "not_estimable_without_unknown_probes",
        },
        "elapsed_s": round(time.perf_counter() - started, 3),
        "events": events,
        "pair_scores": pair_scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-dir", type=Path, default=ROOT / "val")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.val_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
