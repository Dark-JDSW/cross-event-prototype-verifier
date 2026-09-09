# Production model assets

The GUI production backend expects these local files:

| File | Purpose |
|---|---|
| `yolo11l.pt` | YOLO11L person detector used with ByteTrack |
| `rtmpose-s.onnx` | top-down COCO-17 RTMPose model |
| `osnet_ain_x1_0_dg.pth` | domain-generalized OSNet-AIN appearance encoder |
| `gaitgraph2_grew_state.pt` | tensor-only GaitGraph2 GREW state dictionary |
| `posec3d_hmdb51_split1.pth` | optional official HMDB51 PoseC3D WALK semantic gate |
| `bytetrack-cross-event.yaml` | tracker thresholds |

PoseC3D is opt-in because its MMAction2 runtime is not part of the base
dependency set.  When enabled, keep `posec3d_hmdb51_split1.pth`,
`posec3d_hmdb51_runtime.py`, and `hmdb51_label_map.txt` together; the runtime
adapter accepts one tracked COCO-17 sequence and only treats label index 49
(`walk`) as a semantic WALK candidate.
Install MMAction2, MMEngine, and a compatible MMCV build from the official
MMAction2 installation matrix before enabling the flag; these packages are
intentionally not pinned in the base environment because their wheels are
coupled to the installed PyTorch/CUDA build.  Start the GUI with
`python -m cross_event_verifier gui --posec3d-checkpoint models/posec3d_hmdb51_split1.pth`.

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
