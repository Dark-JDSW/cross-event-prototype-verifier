# Production model assets

The GUI production backend expects these local files:

| File | Purpose |
|---|---|
| `yolo11x.pt` | YOLO11 person detector used with ByteTrack |
| `rtmpose-s.onnx` | top-down COCO-17 RTMPose model |
| `osnet_ain_x1_0_dg.pth` | domain-generalized OSNet-AIN appearance encoder |
| `gaitgraph2_grew_state.pt` | tensor-only GaitGraph2 GREW state dictionary |
| `bytetrack-cross-event.yaml` | tracker thresholds |

The binary weights are intentionally ignored by Git. They must stay in this
directory after the sibling `videotracker` directory is removed. Run
`python -m cross_event_verifier download-models` after cloning, then run
`python -m cross_event_verifier doctor` to verify all assets and runtimes.

The downloader validates each upstream source before writing. OSNet-AIN and
GaitGraph2 are converted into tensor-only files, allowing runtime loading with
`torch.load(..., weights_only=True)`. See `manifest.json` and
`THIRD_PARTY_NOTICES.md` for source attribution and use restrictions.

Expected source hashes, direct-file hashes and converted tensor fingerprints
are recorded in `manifest.json`.
