"""Build metadata for downloaded CMU walking BVH files.

CMU BVH contains motion, not a camera.  The manifest therefore separates the
original capture metadata from an explicit synthetic UAV render profile.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DESCRIPTIONS = {
    "07_01.bvh": "walk",
    "07_04.bvh": "slow walk",
    "07_12.bvh": "brisk walk",
    "08_01.bvh": "walk",
    "08_04.bvh": "slow walk",
    "08_11.bvh": "slow walk/stride",
    "16_11.bvh": "walk, veer left",
    "16_15.bvh": "walk",
    "16_17.bvh": "walk, 90-degree left turn",
    "17_01.bvh": "walk with anger/frustration",
    "17_03.bvh": "walk stealthily",
    "17_08.bvh": "muscular/heavyset walk",
    "37_01.bvh": "slow walk",
    "69_01.bvh": "walk forward",
    "69_06.bvh": "walk and turn (repeated)",
}

TARGET_CAMERA = {
    "profile_id": "uav_oblique_downward_proxy_v1",
    "source_camera_available": False,
    "azimuth_deg": 45.0,
    "pitch_deg": -55.0,
    "distance_m": 25.0,
    "altitude_m": 20.5,
    "render_resolution": [1920, 1080],
    "render_fps": 30.0,
    "match_note": (
        "Synthetic camera profile chosen to approximate the project's oblique "
        "downward view. CMU provides no camera extrinsics; render validation is required."
    ),
}


def _parse_bvh_header(path: Path) -> tuple[int | None, float | None]:
    text = path.read_text(encoding="utf-8", errors="replace")
    frames_match = re.search(r"(?m)^Frames:\s*(\d+)", text)
    time_match = re.search(r"(?m)^Frame Time:\s*([0-9.eE+-]+)", text)
    frames = int(frames_match.group(1)) if frames_match else None
    frame_time = float(time_match.group(1)) if time_match else None
    return frames, frame_time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    samples = []
    for name, description in DESCRIPTIONS.items():
        path = args.input_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            continue
        frames, frame_time = _parse_bvh_header(path)
        samples.append(
            {
                "file": name,
                "relative_path": str(path.relative_to(args.input_dir)),
                "source_url": (
                    "https://raw.githubusercontent.com/una-dinosauria/"
                    f"cmu-mocap/master/data/{name[:2].zfill(3)}/{name}"
                ),
                "action": "walk",
                "description": description,
                "format": "BVH",
                "frames": frames,
                "frame_time_s": frame_time,
                "source_fps": (1.0 / frame_time if frame_time else None),
                "camera": dict(TARGET_CAMERA),
                "requires_render": True,
                "stgcn_ready": False,
            }
        )

    manifest = {
        "dataset": "CMU Graphics Lab Motion Capture Database walking subset",
        "source_repository": "https://github.com/una-dinosauria/cmu-mocap",
        "official_source": "https://mocap.cs.cmu.edu/",
        "license": (
            "CMU states the motions are free for all uses; commercial use is allowed, "
            "but direct resale/redistribution of the data is prohibited. Cite CMU and "
            "the NSF-supported database when publishing. Verify current terms before redistribution."
        ),
        "camera_policy": (
            "BVH has no original camera. The per-sample camera field is an explicit "
            "synthetic render target, not measured CMU metadata."
        ),
        "sample_count": len(samples),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sample_count": len(samples)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
