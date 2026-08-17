# Third-party notices

The standalone production vision backend uses or interoperates with the
following third-party projects and model assets. The top-level Apache-2.0
license does not replace their own terms.

## Ultralytics YOLO11

- Project: <https://github.com/ultralytics/ultralytics>
- Runtime/model: `ultralytics==8.3.163` in the validated environment, `models/yolo11l.pt`
- Upstream terms: AGPL-3.0 or an Ultralytics Enterprise license.

Ultralytics states that proprietary/private/commercial deployment requires an
appropriate Enterprise license unless the complete application is distributed
under the applicable AGPL terms. Confirm this before redistribution or
commercial deployment.

## ByteTrack

- Project: <https://github.com/ifzhang/ByteTrack>
- License: MIT
- Use: multi-object tracking configured by
  `models/bytetrack-cross-event.yaml` through Ultralytics.

## RTMPose / MMPose

- Project: <https://github.com/open-mmlab/mmpose>
- License: Apache-2.0
- Model source: <https://huggingface.co/ziq/rtm>
- Model: `models/rtmpose-s.onnx` (the repository's MIT-licensed ONNX export)
- Use: top-down COCO-17 pose estimation and quality measurement.

## deep-person-reid / OSNet-AIN

- Project: <https://github.com/KaiyangZhou/deep-person-reid>
- License: MIT
- Model: `models/osnet_ain_x1_0_dg.pth`
- Adaptation: `cross_event_verifier/osnet_ain.py` contains only the
  checkpoint-compatible inference architecture; upstream training, datasets,
  evaluation and Cython code are not bundled.

## OpenGait / GaitGraph2

- Project: <https://github.com/ShiqiYu/OpenGait>
- Checkpoint source:
  <https://huggingface.co/opengait/OpenGait/tree/main/GREW/GaitGraph2/GaitGraph2/checkpoints>
- Model: `models/gaitgraph2_grew_state.pt`
- Adaptation: `cross_event_verifier/gait_graph.py` contains a compact,
  inference-only checkpoint-compatible ResGCN implementation.

The OpenGait repository explicitly describes its code as academic-use only.
The tensor-only state dictionary included here was derived from the official
checkpoint to permit safe `weights_only=True` loading; that transformation does
not remove upstream usage restrictions. Obtain separate permission or replace
this gait backend before commercial deployment.
