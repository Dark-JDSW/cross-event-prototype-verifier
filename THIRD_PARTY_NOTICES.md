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

## FastPoseGait / GPGait++

- Project: <https://github.com/BNU-IVC/FastPoseGait>
- Pinned commit: `d3ce631971a01d806eeb3a154774d482c0951813`
- Model: `models/gpgaitpp_ccpg_state.pt`, official CCPG small, iteration 40000.
- Official source checkpoint: <https://drive.google.com/file/d/1Y1XqjMbDVikslNfahHWzmXAQmG-Hommo>
- Adaptation: `_gpgaitpp_network.py` contains the checkpoint-compatible inference
  primitives; `gpgaitpp_preprocessing.py` preserves the official inference transforms.
  Only CUDA-specific placement was generalized to the input device. Training,
  optimizers, distributed data loaders and dataset evaluation are excluded.
- Source file hashes, configuration and asset provenance: `models/gpgaitpp_source.json`.

The pinned upstream README explicitly limits its code to academic purposes and
prohibits commercial use. Conversion to a tensor-only checkpoint does not change
these terms. This application's Apache-2.0 license does not relicense upstream
code or weights; obtain appropriate permission for other uses.

The compatibility module `gait_graph.py` retains the earlier OpenGait/GaitGraph2
inference API for existing import callers. It is not selected by the production
adapter and its old checkpoint is no longer bundled. Its source remains subject
to OpenGait's academic-use terms: <https://github.com/ShiqiYu/OpenGait>.

## MMAction2 / PoseC3D-HMDB51

- Project: <https://github.com/open-mmlab/mmaction2>
- Code license: Apache-2.0
- Checkpoint source:
  <https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint_20220815-17eaa484.pth>
- Model: `models/posec3d_hmdb51_split1.pth`
- Runtime: the optional adapter in `cross_event_verifier/posec3d_action.py`
  uses the official MMAction2 inference API and the vendored
  `models/posec3d_hmdb51_runtime.py` pipeline.

The checkpoint is trained on HMDB51. Verify the dataset and checkpoint terms
separately before redistribution or commercial deployment; the MMAction2 code
license does not grant rights to the underlying dataset or weights.

## Bleak

- Project: <https://github.com/hbldh/bleak>
- License: MIT
- Use: optional Windows/Linux/macOS BLE GATT client used by
  `cross_event_verifier/ble_bridge.py` to send action events to the ESP32-S3
  firmware.

## ESP-IDF NimBLE firmware

- Project: <https://github.com/espressif/esp-idf>
- License: Apache-2.0 (see the upstream repository for the complete notices)
- Use: the standalone `firmware/esp32s3_ble` project uses the ESP-IDF NimBLE
  host and GATT server APIs; ESP-IDF must be installed separately.
