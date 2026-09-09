"""Create a conservative walk-segment manifest from UCF-ARG evaluation data.

The UCF VIPER annotations identify walk spans, but they do not publish camera
extrinsics.  This utility preserves those unknowns as null rather than
inventing values.  It does not cut or re-encode the source videos.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2


def local_name(tag: str) -> str:
    return tag.rsplit('}', 1)[-1]


def video_metadata(path: Path) -> dict[str, float | int | None]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f'Unable to inspect video: {path}')
    try:
        return {
            'width': int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': capture.get(cv2.CAP_PROP_FPS),
            'frame_count': int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        capture.release()


def walk_spans(xgtf_path: Path) -> list[dict[str, object]]:
    root = ET.parse(xgtf_path).getroot()
    spans: list[dict[str, object]] = []

    def visit(node: ET.Element, ancestors: list[ET.Element]) -> None:
        if local_name(node.tag) == 'lvalue' and node.attrib.get('value') == 'walking':
            frame_start, frame_end = (int(value) for value in node.attrib['framespan'].split(':'))
            context = [
                {'tag': local_name(parent.tag), 'attributes': dict(parent.attrib)}
                for parent in ancestors[-4:]
            ]
            spans.append({
                'frame_start_1based': frame_start,
                'frame_end_1based': frame_end,
                'frame_count': frame_end - frame_start + 1,
                'annotation_context': context,
            })
        for child in node:
            visit(child, [*ancestors, node])

    visit(root, [])
    return spans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path,
                        default=Path(r'F:\dataset\ucf_arg\extracted\evaluation\UCF-ARG_Evaluation'))
    parser.add_argument('--output', type=Path,
                        default=Path(r'F:\dataset\ucf_arg\manifests\evaluation_walk_segments.json'))
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    for annotation in sorted(args.root.glob('*.xgtf')):
        video = annotation.with_suffix('.avi')
        if not video.exists():
            raise FileNotFoundError(f'Missing matching AVI for {annotation.name}')
        source = video_metadata(video)
        for index, span in enumerate(walk_spans(annotation), start=1):
            records.append({
                'id': f'{annotation.stem}_walk_{index:03d}',
                'label': 'walk',
                'source_video': str(video),
                'annotation': str(annotation),
                'video': source,
                'segment': span,
                'camera': {
                    'view': 'aerial/rooftop/ground sequence not identified in archive metadata',
                    'azimuth_deg': None,
                    'pitch_deg': None,
                    'distance_m': None,
                    'camera_parameters_status': 'not published in UCF-ARG evaluation archive',
                },
                'quality_status': 'annotation-derived candidate; manually review track and occlusion before use',
                'stgcn_ready': False,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'dataset': 'UCF-ARG Evaluation',
        'source_url': 'https://vision.eecs.ucf.edu/data/UCF-ARG.html',
        'license_status': 'No explicit license displayed on official page; research only pending permission review.',
        'record_count': len(records),
        'records': records,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'Wrote {len(records)} walk candidates to {args.output}')


if __name__ == '__main__':
    main()
