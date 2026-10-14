"""生产视频前端：YOLO、ByteTrack、RTMPose、ReID 和步态。

整个重量级视觉栈都隐藏在既有的 ``VisionAdapter`` 接口之后。GUI 和身份策略
代码只接收 ``VisionTrack`` 值，不导入模型框架，也不需要知道时序状态如何维护。

适配器为每条轨迹维护检测、姿态、嵌入及时序质量状态。一帧图像会依次经过
检测/跟踪、批量姿态推理、外观刷新、步态窗口更新和质量计算。模型权重既可以
在首帧到来时按需加载，也可以由 GUI 在后台显式分阶段预加载并执行一次无身份
语义的 Dummy warmup；两种入口共用同一套模型槽位。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from typing import Callable, Sequence

import cv2
import numpy as np

from .actions import ActionPrediction, ActionQuality, ActionRouter, ActionType
from .action_classifier import PoseActionClassifier
from .adapters import occlusion_scores
from .gpgaitpp import GPGaitPPEncoder
from .model_assets import tensor_state_fingerprint
from .osnet_ain import load_osnet_ain
from .posec3d_action import PoseC3DActionClassifier, PoseC3DPrediction
from .types import CleanSubTracklet, FeatureBundle, TrackQuality
from .vision import Box, VisionTrack


MODEL_DIRECTORY = Path(__file__).resolve().parents[1] / "models"
MODEL_MANIFEST = MODEL_DIRECTORY / "manifest.json"


def _sha256_file(path: Path) -> str:
    """以有限内存分块计算模型/配置文件的哈希。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ProductionVisionError(RuntimeError):
    """当生产视觉栈无法安全加载或运行时抛出。"""


@dataclass(frozen=True)
class ProductionVisionConfig:
    """生产视觉分支和质量策略的不可变配置。

    ``require_cuda`` 默认采用失败即关闭策略：缺少 GPU provider 时，不能把
    已校准的生产运行悄悄变成较慢的 CPU 运行。
    """
    detector_path: Path = MODEL_DIRECTORY / "yolo11l.pt"
    tracker_path: Path = MODEL_DIRECTORY / "bytetrack-cross-event.yaml"
    pose_path: Path = MODEL_DIRECTORY / "rtmpose-s.onnx"
    appearance_path: Path = MODEL_DIRECTORY / "osnet_ain_x1_0_dg.pth"
    gait_path: Path = MODEL_DIRECTORY / "gpgaitpp_ccpg_state.pt"
    action_classifier_path: Path | None = None
    # Optional official HMDB51 PoseC3D semantic gate.  The existing
    # ``action_classifier_path`` remains reserved for the project's .npz
    # centroid classifier.
    posec3d_checkpoint_path: Path | None = None
    posec3d_config_path: Path = MODEL_DIRECTORY / "posec3d_hmdb51_runtime.py"
    posec3d_label_map_path: Path = MODEL_DIRECTORY / "hmdb51_label_map.txt"
    posec3d_probability_threshold: float = 0.80
    posec3d_margin_threshold: float = 0.15
    posec3d_clip_length: int = 48
    posec3d_inference_stride: int = 3
    detector_confidence: float = 0.25
    output_confidence: float = 0.45
    detector_iou: float = 0.50
    image_size: int = 736
    maximum_people: int = 16
    keypoint_confidence: float = 0.45
    minimum_pose_frames: int = 25
    gait_sequence_length: int = 60
    detector_inference_stride: int = 1
    appearance_stride: int = 6
    appearance_change_suspect_similarity: float = 0.55
    gait_inference_stride: int = 3
    low_light_threshold: float = 100.0
    low_light_check_interval: int = 12
    low_light_hysteresis: float = 8.0
    low_light_min_hold_frames: int = 24
    state_retention_frames: int = 90
    state_retention_seconds: float = 3.0
    maximum_track_gap_seconds: float = 1.5
    # Optional image-to-ground homography.  It is deliberately inert until a
    # real per-camera calibration is supplied; gait model coordinates must
    # remain in image space before official GPGait++ preprocessing.
    ground_homography: tuple[float, ...] | None = None
    ground_calibration_id: str | None = None
    ground_coordinate_frame: str | None = None
    ground_trajectory_window_seconds: float = 2.0
    ground_max_speed_mps: float = 6.0
    action_stability_frames: int = 3
    action_unknown_end_frames: int = 6
    device: str | None = None
    # 生产推理有意采用“失败即关闭”策略。任何一个分支回退到 CPU，都会让实时
    # 管线表面看起来正常，却悄悄破坏吞吐量并使预期部署配置失效。
    require_cuda: bool = True

    def __post_init__(self) -> None:
        """校验置信度范围及时序长度不变量。"""
        for name in (
            "detector_confidence",
            "output_confidence",
            "detector_iou",
            "keypoint_confidence",
            "appearance_change_suspect_similarity",
            "posec3d_probability_threshold",
            "posec3d_margin_threshold",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.output_confidence < self.detector_confidence:
            raise ValueError("output_confidence cannot be below detector_confidence")
        if self.maximum_people < 1:
            raise ValueError("maximum_people must be positive")
        if self.minimum_pose_frames < 8:
            raise ValueError("minimum_pose_frames must be at least 8")
        if self.gait_sequence_length < self.minimum_pose_frames:
            raise ValueError("gait_sequence_length cannot be shorter than minimum_pose_frames")
        if self.detector_inference_stride < 1:
            raise ValueError("detector_inference_stride must be at least 1")
        if self.appearance_stride < 1:
            raise ValueError("appearance_stride must be at least 1")
        if self.gait_inference_stride < 1:
            raise ValueError("gait_inference_stride must be at least 1")
        if self.posec3d_clip_length < 8:
            raise ValueError("posec3d_clip_length must be at least 8")
        if self.posec3d_clip_length != PoseC3DActionClassifier.clip_length:
            raise ValueError(
                "posec3d_clip_length must remain 48 for the official HMDB51 checkpoint"
            )
        if self.posec3d_inference_stride < 1:
            raise ValueError("posec3d_inference_stride must be at least 1")
        if self.low_light_hysteresis < 0:
            raise ValueError("low_light_hysteresis cannot be negative")
        if self.low_light_min_hold_frames < 1:
            raise ValueError("low_light_min_hold_frames must be positive")
        if self.state_retention_seconds <= 0 or self.maximum_track_gap_seconds <= 0:
            raise ValueError("track retention/gap seconds must be positive")
        if self.maximum_track_gap_seconds >= self.state_retention_seconds:
            raise ValueError(
                "maximum_track_gap_seconds must be shorter than state_retention_seconds"
            )
        if self.ground_homography is not None:
            matrix = np.asarray(self.ground_homography, dtype=np.float64)
            if matrix.size != 9 or not np.isfinite(matrix).all():
                raise ValueError("ground_homography must contain nine finite values")
        if self.ground_trajectory_window_seconds <= 0:
            raise ValueError("ground_trajectory_window_seconds must be positive")
        if self.ground_max_speed_mps <= 0:
            raise ValueError("ground_max_speed_mps must be positive")
        for name in ("ground_calibration_id", "ground_coordinate_frame"):
            value = getattr(self, name)
            if value is not None and not str(value).strip():
                raise ValueError(f"{name} cannot be blank when supplied")
        if self.action_stability_frames < 1:
            raise ValueError("action_stability_frames must be positive")
        if self.action_unknown_end_frames < self.action_stability_frames:
            raise ValueError(
                "action_unknown_end_frames cannot be below action_stability_frames"
            )
        if self.require_cuda and self.torch_device.lower().startswith("cpu"):
            raise ValueError("生产视觉链路 require_cuda=True 时不能把 device 设为 CPU")
        if self.action_classifier_path is not None and not Path(self.action_classifier_path).is_file():
            raise ValueError(
                f"action classifier model does not exist: {self.action_classifier_path}"
            )
        if self.posec3d_checkpoint_path is not None:
            if not Path(self.posec3d_checkpoint_path).is_file():
                raise ValueError(
                    f"PoseC3D checkpoint does not exist: {self.posec3d_checkpoint_path}"
                )
            if not Path(self.posec3d_config_path).is_file():
                raise ValueError(
                    f"PoseC3D config does not exist: {self.posec3d_config_path}"
                )
            if not Path(self.posec3d_label_map_path).is_file():
                raise ValueError(
                    f"PoseC3D label map does not exist: {self.posec3d_label_map_path}"
                )
            if self.gait_sequence_length < self.posec3d_clip_length:
                raise ValueError(
                    "gait_sequence_length cannot be shorter than posec3d_clip_length"
                )

    @property
    def required_files(self) -> tuple[Path, ...]:
        """返回生产启动前所需的全部本地文件。"""
        paths = [
            self.detector_path,
            self.tracker_path,
            self.pose_path,
            self.appearance_path,
            self.gait_path,
        ]
        if self.posec3d_checkpoint_path is not None:
            paths.append(self.posec3d_checkpoint_path)
        return tuple(paths)

    @property
    def torch_device(self) -> str:
        """返回身份表征编码器使用的规范化 Torch 设备。"""

        if self.device is None:
            return "cuda:0"
        value = str(self.device).strip().lower()
        if value.isdigit():
            return f"cuda:{value}"
        if value == "cuda":
            return "cuda:0"
        return str(self.device)

    @property
    def cuda_device_index(self) -> int:
        """Return the CUDA ordinal shared by PyTorch and ONNX Runtime."""

        value = self.torch_device.strip().lower()
        if not value.startswith("cuda:") or not value.removeprefix("cuda:").isdigit():
            raise ValueError(f"production device must be cuda:N, got {self.torch_device!r}")
        return int(value.removeprefix("cuda:"))


def production_readiness(
    config: ProductionVisionConfig | None = None,
    *,
    verify_hashes: bool = False,
) -> tuple[bool, tuple[str, ...]]:
    """检查依赖、CUDA provider、模型资产以及可选哈希。

    ``doctor`` 和启动校验都会使用这份只读报告。所有发现的问题会一起返回，
    便于部署时一次性修复环境。
    """

    cfg = config or ProductionVisionConfig()
    issues: list[str] = []
    for module in ("torch", "ultralytics", "onnxruntime"):
        if importlib.util.find_spec(module) is None:
            issues.append(f"缺少 Python 依赖：{module}")
    if cfg.posec3d_checkpoint_path is not None:
        missing_posec3d = [
            module
            for module in ("mmaction", "mmengine", "mmcv")
            if importlib.util.find_spec(module) is None
        ]
        if missing_posec3d:
            issues.append(
                "PoseC3D 运行时缺少官方 MMAction2 依赖："
                + ", ".join(missing_posec3d)
            )
    if importlib.util.find_spec("ultralytics") is not None and MODEL_MANIFEST.is_file():
        try:
            runtime_contract = json.loads(
                MODEL_MANIFEST.read_text(encoding="utf-8")
            ).get("runtime_contract", {})
            expected_ultralytics = runtime_contract.get("ultralytics")
            if expected_ultralytics:
                import ultralytics

                actual_ultralytics = str(getattr(ultralytics, "__version__", "unknown"))
                if actual_ultralytics != str(expected_ultralytics):
                    issues.append(
                        "Ultralytics 运行时版本与模型清单不符："
                        f"expected={expected_ultralytics}, actual={actual_ultralytics}"
                    )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            issues.append(f"运行时模型契约无效：{error}")
    if cfg.require_cuda:
        try:
            import torch

            if not torch.cuda.is_available():
                issues.append("PyTorch CUDA 不可用，生产视觉链路拒绝 CPU 回退")
        except Exception as error:
            issues.append(f"PyTorch CUDA 检查失败：{error}")
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" not in providers:
                issues.append(
                    "ONNX Runtime 未提供 CUDAExecutionProvider；请安装 onnxruntime-gpu，"
                    f"当前为 {providers}"
                )
        except Exception as error:
            issues.append(f"ONNX Runtime CUDA 检查失败：{error}")
    for path in cfg.required_files:
        if not Path(path).is_file() or Path(path).stat().st_size <= 0:
            issues.append(f"缺少模型文件：{path}")
    if verify_hashes and not issues:
        if not MODEL_MANIFEST.is_file():
            issues.append(f"缺少模型清单：{MODEL_MANIFEST}")
        else:
            try:
                entries = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))["models"]
                for path in cfg.required_files:
                    # ByteTrack YAML 是配置而非二进制模型，因此有意不纳入权重清单。
                    expected = entries.get(path.name)
                    if expected is None:
                        if path.suffix.lower() != ".yaml":
                            issues.append(f"模型清单缺少条目：{path.name}")
                        continue
                    if expected.get("tensor_sha256"):
                        digest = tensor_state_fingerprint(path)
                        if digest != str(expected["tensor_sha256"]):
                            issues.append(f"模型 tensor 指纹不符：{path.name}")
                    else:
                        size = path.stat().st_size
                        if size != int(expected["bytes"]):
                            issues.append(f"模型大小不符：{path.name}")
                            continue
                        digest = _sha256_file(path)
                        if digest != str(expected["sha256"]):
                            issues.append(f"模型哈希不符：{path.name}")
            except (
                KeyError,
                TypeError,
                ValueError,
                OSError,
                RuntimeError,
                json.JSONDecodeError,
            ) as error:
                issues.append(f"模型清单无效：{error}")
    return not issues, tuple(issues)


@dataclass(frozen=True)
class _Detection:
    """YOLO/ByteTrack 返回的归一化检测记录。"""
    track_id: int
    box: Box
    confidence: float


class _LowLightEnhancer:
    """仅在采样亮度较低时启用的可选 CLAHE 预处理器。"""
    def __init__(
        self,
        threshold: float,
        interval: int,
        hysteresis: float = 8.0,
        min_hold_frames: int = 24,
    ) -> None:
        """配置亮度采样间隔和可复用的 CLAHE 算子。"""
        self.threshold = float(threshold)
        self.interval = max(1, int(interval))
        self.hysteresis = max(0.0, float(hysteresis))
        self.min_hold_frames = max(1, int(min_hold_frames))
        self.frame_index = 0
        self.enabled = False
        self._last_switch_frame = -self.min_hold_frames
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def reset(self) -> None:
        """清除已采样亮度，使下一帧重新评估光照。"""
        self.frame_index = 0
        self.enabled = False
        self._last_switch_frame = -self.min_hold_frames

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """根据采样间隔返回原始帧或 CLAHE 增强帧。"""
        if self.frame_index % self.interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(gray.mean())
            can_switch = (
                self.frame_index - self._last_switch_frame
                >= self.min_hold_frames
            )
            if not self.enabled and brightness < self.threshold and can_switch:
                self.enabled = True
                self._last_switch_frame = self.frame_index
            elif (
                self.enabled
                and brightness > self.threshold + self.hysteresis
                and can_switch
            ):
                self.enabled = False
                self._last_switch_frame = self.frame_index
        self.frame_index += 1
        if not self.enabled:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        luminance, channel_a, channel_b = cv2.split(lab)
        luminance = self.clahe.apply(luminance)
        return cv2.cvtColor(
            cv2.merge((luminance, channel_a, channel_b)),
            cv2.COLOR_LAB2BGR,
        )


class _YoloByteTracker:
    """延迟加载的 YOLO 人体检测器及持久化 ByteTrack 状态。"""
    def __init__(self, config: ProductionVisionConfig) -> None:
        """加载 YOLO，并执行配置中的 CUDA/失败即关闭策略。"""
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as error:  # pragma: no cover - checked by readiness
            raise ProductionVisionError(
                "YOLO/ByteTrack 需要安装 ultralytics、lap 和 CUDA PyTorch"
            ) from error
        self.config = config
        self._torch = torch
        if config.require_cuda and not torch.cuda.is_available():
            raise ProductionVisionError("YOLO/ByteTrack 生产模式要求 CUDA，不允许使用 CPU")
        self.device = "0" if config.device is None else str(config.device)
        if self.device.lower() == "cuda":
            self.device = "0"
        if config.require_cuda and self.device.lower() in {"cpu", "-1"}:
            raise ProductionVisionError("YOLO/ByteTrack 生产模式不允许使用 CPU device")
        self.half = self.device.lower() not in {"cpu", "-1"} and torch.cuda.is_available()
        self.model = YOLO(str(config.detector_path))
        self.overflowed = False

    def track(self, frame: np.ndarray) -> tuple[_Detection, ...]:
        """仅跟踪人物，并返回裁剪且通过置信度门控的检测框。"""
        try:
            results = self.model.track(
                frame,
                classes=[0],
                persist=True,
                tracker=str(self.config.tracker_path),
                conf=self.config.detector_confidence,
                iou=self.config.detector_iou,
                imgsz=self.config.image_size,
                half=self.half,
                device=self.device,
                # Ask for one extra result so a crowded frame is observable
                # instead of silently looking like an ordinary 16-person frame.
                max_det=self.config.maximum_people + 1,
                verbose=False,
            )
        except Exception as error:
            raise ProductionVisionError(f"YOLO/ByteTrack 推理失败：{error}") from error
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            self.overflowed = False
            return ()
        boxes = results[0].boxes.xyxy.detach().cpu().numpy()
        self.overflowed = len(boxes) > self.config.maximum_people
        boxes = boxes[: self.config.maximum_people]
        ids = results[0].boxes.id.detach().cpu().numpy()
        ids = ids[: self.config.maximum_people]
        confidences = results[0].boxes.conf.detach().cpu().numpy()
        confidences = confidences[: self.config.maximum_people]
        height, width = frame.shape[:2]
        output: list[_Detection] = []
        for raw_box, raw_id, raw_confidence in zip(boxes, ids, confidences):
            confidence = float(raw_confidence)
            if confidence < self.config.output_confidence:
                continue
            x1, y1, x2, y2 = (int(round(float(value))) for value in raw_box)
            x1, y1 = max(0, min(x1, width - 1)), max(0, min(y1, height - 1))
            x2, y2 = max(0, min(x2, width)), max(0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue
            output.append(_Detection(int(raw_id), (x1, y1, x2, y2), confidence))
        return tuple(output)

    def reset(self) -> None:
        """在摄像头/视频会话之间重置 Ultralytics 跟踪器状态。"""
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", ()) if predictor is not None else ()
        for tracker in trackers or ():
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()
        if predictor is not None and hasattr(predictor, "vid_path"):
            predictor.vid_path = [None] * max(1, len(trackers or ()))

    def warmup(self) -> None:
        """用空白画面预热 YOLO/ByteTrack，并丢弃预热产生的轨迹状态。"""

        frame = np.zeros(
            (self.config.image_size, self.config.image_size, 3),
            dtype=np.uint8,
        )
        self.track(frame)
        self.reset()


class _RtmposeEstimator:
    """使用批量 CUDA ONNX 推理的自顶向下 RTMPose-s 适配器。"""
    _MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
    _STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
    _INPUT_WIDTH = 192
    _INPUT_HEIGHT = 256
    _PADDING = 1.25
    _SIMCC_SPLIT = 2.0
    _BUCKETS = (1, 2, 4, 8, 16, 32)

    def __init__(
        self,
        path: Path,
        *,
        require_cuda: bool = True,
        cuda_device_id: int = 0,
    ) -> None:
        """创建以 CUDA 为必需主 provider 的 ONNX 会话。"""
        try:
            # PyTorch 自带 CUDA PyPI 构建的 ONNX Runtime 所需 CUDA/cuDNN DLL。
            # 先导入它，使 Windows 上的 ORT 能发现这些 DLL，而不依赖系统级 CUDA。
            import torch  # noqa: F401
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover - checked by readiness
            raise ProductionVisionError("RTMPose 需要安装 onnxruntime-gpu") from error
        available = ort.get_available_providers()
        if require_cuda:
            if "CUDAExecutionProvider" not in available:
                raise ProductionVisionError(
                    "RTMPose 生产模式要求 CUDAExecutionProvider，"
                    f"当前可用后端：{available}"
                )
            # CUDA 负责主图，CPU 作为明确的图分区回退以承接少量 shape/layout
            # 辅助节点。provider 列表本身不代表每个节点的位置，启动诊断会
            # 同时记录实际 provider 配置和端到端延迟。
            providers = [("CUDAExecutionProvider", {"device_id": int(cuda_device_id)})]
            if "CPUExecutionProvider" in available:
                providers.append("CPUExecutionProvider")
        else:
            providers = [
                provider
                for provider in (
                    "CUDAExecutionProvider",
                    "DmlExecutionProvider",
                    "CPUExecutionProvider",
                )
                if provider in available
            ]
            if not providers:
                raise ProductionVisionError("ONNX Runtime 没有可用执行后端")
        options = ort.SessionOptions()
        if "DmlExecutionProvider" in available and not require_cuda:
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.providers = tuple(self.session.get_providers())
        self.provider_options = dict(self.session.get_provider_options())
        self.last_latency_ms = 0.0

    @classmethod
    def _crop(
        cls,
        frame: np.ndarray,
        box: Box,
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]] | None:
        """将人物框变换为 RTMPose 所需的 192x256 仿射裁剪。"""
        x1, y1, x2, y2 = map(float, box)
        width, height = x2 - x1, y2 - y1
        if width < 2 or height < 2:
            return None
        center_x, center_y = x1 + width / 2.0, y1 + height / 2.0
        ratio = cls._INPUT_WIDTH / cls._INPUT_HEIGHT
        if width / height > ratio:
            height = width / ratio
        else:
            width = height * ratio
        width, height = width * cls._PADDING, height * cls._PADDING
        source = np.asarray(
            [
                [center_x - width / 2.0, center_y - height / 2.0],
                [center_x + width / 2.0, center_y - height / 2.0],
                [center_x - width / 2.0, center_y + height / 2.0],
            ],
            dtype=np.float32,
        )
        destination = np.asarray(
            [[0, 0], [cls._INPUT_WIDTH, 0], [0, cls._INPUT_HEIGHT]],
            dtype=np.float32,
        )
        matrix = cv2.getAffineTransform(source, destination)
        crop = cv2.warpAffine(
            frame,
            matrix,
            (cls._INPUT_WIDTH, cls._INPUT_HEIGHT),
            flags=cv2.INTER_LINEAR,
        )
        return crop, (center_x, center_y), (width, height)

    def extract(self, frame: np.ndarray, boxes: Sequence[Box]) -> list[np.ndarray | None]:
        """为每个检测框估计 17 个关键点，同时保持输入顺序。"""
        output: list[np.ndarray | None] = [None] * len(boxes)
        crops: list[np.ndarray] = []
        metadata: list[tuple[tuple[float, float], tuple[float, float]]] = []
        indexes: list[int] = []
        for index, box in enumerate(boxes):
            result = self._crop(frame, box)
            if result is None:
                continue
            crop, center, scale = result
            crops.append(crop)
            metadata.append((center, scale))
            indexes.append(index)
        if not crops:
            return output

        count = len(crops)
        bucket = next((value for value in self._BUCKETS if value >= count), count)
        batch = np.zeros(
            (bucket, 3, self._INPUT_HEIGHT, self._INPUT_WIDTH),
            dtype=np.float32,
        )
        for index, crop in enumerate(crops):
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
            rgb = (rgb - self._MEAN) / self._STD
            batch[index] = rgb.transpose(2, 0, 1)
        started = time.perf_counter()
        try:
            simcc_x, simcc_y = self.session.run(None, {self.input_name: batch})
        except Exception as error:
            raise ProductionVisionError(f"RTMPose 推理失败：{error}") from error
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        simcc_x = np.asarray(simcc_x)
        simcc_y = np.asarray(simcc_y)
        if (
            simcc_x.ndim != 3
            or simcc_y.ndim != 3
            or simcc_x.shape[0] < count
            or simcc_y.shape[0] < count
        ):
            raise ProductionVisionError(
                "RTMPose 输出张量形状无效："
                f"x={simcc_x.shape}, y={simcc_y.shape}"
            )
        simcc_x, simcc_y = simcc_x[:count], simcc_y[:count]
        positions_x, positions_y, confidence = self._decode_simcc(simcc_x, simcc_y)
        for row, destination_index in enumerate(indexes):
            (center_x, center_y), (width, height) = metadata[row]
            points = np.stack(
                [
                    positions_x[row] / self._INPUT_WIDTH * width + center_x - width / 2.0,
                    positions_y[row] / self._INPUT_HEIGHT * height + center_y - height / 2.0,
                    confidence[row],
                ],
                axis=1,
            ).astype(np.float32)
            output[destination_index] = points
        return output

    def warmup(self) -> None:
        """用一个空白人体框执行一次完整的 RTMPose 前向。"""

        frame = np.zeros(
            (self._INPUT_HEIGHT, self._INPUT_WIDTH, 3),
            dtype=np.uint8,
        )
        self.extract(frame, ((0, 0, self._INPUT_WIDTH, self._INPUT_HEIGHT),))

    @classmethod
    def _decode_simcc(
        cls,
        simcc_x: np.ndarray,
        simcc_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """按 MMPose ``get_simcc_maximum`` 语义解码坐标和置信度。

        SimCC 的关键点置信度是两个轴峰值中较小的一个。取平均会在一个轴
        明显不可靠时抬高分数，进而错误地通过关键点/下肢质量门。
        """

        positions_x = simcc_x.argmax(2) / cls._SIMCC_SPLIT
        positions_y = simcc_y.argmax(2) / cls._SIMCC_SPLIT
        confidence = np.minimum(simcc_x.max(2), simcc_y.max(2))
        return positions_x, positions_y, np.clip(confidence, 0.0, 1.0)


class _OsnetAppearanceExtractor:
    """批量处理 OSNet-AIN 裁剪图，并返回 L2 归一化外观向量。"""

    output_dimension = 512
    def __init__(self, path: Path, device: str | None) -> None:
        """在选定的 Torch 设备上加载一次 OSNet-AIN。"""
        try:
            import torch
        except ImportError as error:  # pragma: no cover - checked by readiness
            raise ProductionVisionError("OSNet-AIN 需要安装 PyTorch") from error
        self.torch = torch
        self.model, self.device = load_osnet_ain(path, device=device)

    @staticmethod
    def _tensor(crop: np.ndarray) -> np.ndarray:
        """将一张 BGR 裁剪图归一化为 OSNet 所需的 CHW RGB 张量布局。"""
        resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb -= np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        rgb /= np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        return rgb.transpose(2, 0, 1)

    def extract(self, frame: np.ndarray, boxes: Sequence[Box]) -> list[np.ndarray | None]:
        """裁剪每个检测框、运行 OSNet，并按检测框索引放回嵌入。"""
        output: list[np.ndarray | None] = [None] * len(boxes)
        tensors: list[np.ndarray] = []
        indexes: list[int] = []
        frame_height, frame_width = frame.shape[:2]
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            margin_x = int((x2 - x1) * 0.02)
            margin_y = int((y2 - y1) * 0.02)
            x1, x2 = max(0, x1 + margin_x), min(frame_width, x2 - margin_x)
            y1, y2 = max(0, y1 + margin_y), min(frame_height, y2 - margin_y)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            tensors.append(self._tensor(crop))
            indexes.append(index)
        if not tensors:
            return output
        batch = self.torch.from_numpy(np.stack(tensors))
        try:
            with self.torch.inference_mode():
                batch = batch.to(self.device)
                if self.device.type == "cuda":
                    with self.torch.autocast(device_type="cuda", dtype=self.torch.float16):
                        feature_tensor = self.model(batch)
                else:
                    feature_tensor = self.model(batch)
                features = feature_tensor.float().cpu().numpy()
        except Exception as error:
            raise ProductionVisionError(f"OSNet-AIN 推理失败：{error}") from error
        if features.ndim != 2 or features.shape != (len(indexes), self.output_dimension):
            raise ProductionVisionError(
                "OSNet-AIN 输出维度无效："
                f"actual={features.shape}, expected={(len(indexes), self.output_dimension)}"
            )
        for row, destination_index in enumerate(indexes):
            vector = features[row].astype(np.float32)
            norm = float(np.linalg.norm(vector))
            if norm > 1e-8:
                output[destination_index] = vector / norm
        return output

    def warmup(self) -> None:
        """用空白人物框执行一次 OSNet-AIN 前向，不写入任何身份数据。"""

        frame = np.zeros((256, 128, 3), dtype=np.uint8)
        self.extract(frame, ((0, 0, 128, 256),))


@dataclass
class _TrackState:
    """由一个 ByteTrack ID 持有的短期时序缓存。"""
    # ``frame_count`` and the quality counters below describe the current clean
    # subtracklet.  Lifetime counters remain audit-only and must never veto a
    # freshly established segment.
    frame_count: int = 0
    lifetime_frame_count: int = 0
    last_seen_frame: int = 0
    poses: deque[np.ndarray] = field(default_factory=deque)
    timestamps: deque[float] = field(default_factory=deque)
    boxes: deque[Box] = field(default_factory=lambda: deque(maxlen=20))
    real_boxes: deque[Box] = field(default_factory=lambda: deque(maxlen=2))
    real_box_frames: deque[int] = field(default_factory=lambda: deque(maxlen=2))
    valid_pose_frames: int = 0
    valid_leg_frames: int = 0
    detector_gap_frames: int = 0
    lifetime_detector_gap_frames: int = 0
    track_gap_frames: int = 0
    lifetime_track_gap_frames: int = 0
    last_seen_timestamp: float | None = None
    appearance: np.ndarray | None = None
    last_appearance_frame: int | None = None
    appearance_generation: int = 0
    appearance_similarity_to_previous: float | None = None
    appearance_change_suspected: bool = False
    gait: np.ndarray | None = None
    last_gait_frame: int | None = None
    id_switches: int = 0
    lifetime_id_switches: int = 0
    segment_id: int = 0
    segment_start_frame: int = 0
    segment_boundary_reasons: tuple[str, ...] = ()
    views: deque[str | None] = field(default_factory=lambda: deque(maxlen=60))
    ground_points: deque[tuple[float, float] | None] = field(
        default_factory=lambda: deque(maxlen=60)
    )
    camera_view_timeline: deque[_CameraViewSample] = field(
        default_factory=lambda: deque(maxlen=60)
    )
    action_type: ActionType = ActionType.UNKNOWN
    action_branch_generation: int = 0
    action_segment_id: int = 0
    action_boundary_reason: str | None = None
    action_candidate_type: ActionType | None = None
    action_candidate_frames: int = 0
    action_unknown_frames: int = 0
    posec3d_prediction: PoseC3DPrediction | None = None


@dataclass(frozen=True)
class _CameraViewSample:
    """One track-aligned camera/view observation for audit and replay."""

    frame_index: int
    timestamp: float
    view_angle: str | None
    camera_zone: str | None
    world_position: tuple[float, float] | None
    ground_anchor_source: str | None
    missing: bool = False


@dataclass(frozen=True)
class _WorldTrajectory:
    """Smoothed ground-plane motion derived from an aligned track window."""

    position: tuple[float, float] | None = None
    speed: float | None = None
    displacement: float | None = None
    direction_degrees: float | None = None
    quality: float = 0.0
    valid_samples: int = 0
    outlier_count: int = 0


def _box_valid(box: Box) -> bool:
    """在使用证据前拒绝过小或形状不合理的人物框。"""
    width, height = box[2] - box[0], box[3] - box[1]
    ratio = height / max(width, 1)
    return width >= 20 and height >= 40 and 0.85 <= ratio <= 5.0


def _canonical_pose(
    keypoints: np.ndarray | None,
    box: Box,
    confidence_floor: float,
) -> np.ndarray | None:
    """清理 RTMPose 点位并保留其原始全帧坐标。

    GPGait++ 的官方预处理接收原图坐标；身体对齐和尺度转换由编码器承担。
    此处只进行有限值与置信度清理，不提前按检测框归一化。``box`` 保留在签名
    中是为了让调用方继续显式传递检测上下文；它不应该参与模型输入变换。
    """
    if keypoints is None or np.asarray(keypoints).shape != (17, 3):
        return None
    points = np.asarray(keypoints, dtype=np.float32).copy()
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] >= confidence_floor)
    points[~valid] = 0.0
    if not bool(valid.any()):
        return None
    # Do not clip valid coordinates before official GPGait++ alignment.
    # coordinates, and a detector/pose mismatch should be handled by quality
    # gates rather than silently changing the model's coordinate distribution.
    return points


def _pose_visibility(points: np.ndarray | None, floor: float) -> tuple[float, float]:
    """返回姿态的整体可见度和下肢可见度比例。"""
    if points is None:
        return 0.0, 0.0
    confidence = np.asarray(points, dtype=np.float32)[:, 2]
    visible = confidence >= floor
    all_visibility = float(visible.mean() * np.clip(confidence[visible].mean() if visible.any() else 0.0, 0, 1))
    leg_indexes = np.asarray([11, 12, 13, 14, 15, 16])
    leg_visible = confidence[leg_indexes] >= floor
    leg_visibility = float(
        leg_visible.mean()
        * np.clip(confidence[leg_indexes][leg_visible].mean() if leg_visible.any() else 0.0, 0, 1)
    )
    return all_visibility, leg_visibility


def _walking_metrics(
    poses: Sequence[np.ndarray],
    box: Box | None = None,
    timestamps: Sequence[float] | None = None,
) -> tuple[float, float]:
    """根据下肢相位估计运动能量和步态周期数。"""
    if len(poses) < 8:
        return 0.0, 0.0
    values = np.asarray(poses, dtype=np.float32)
    confidence = values[:, :, 2]
    hip_valid = (confidence[:, 11] > 0) & (confidence[:, 12] > 0)
    if int(hip_valid.sum()) < max(6, len(values) // 2):
        return 0.0, 0.0
    hip_center = (values[:, 11, :2] + values[:, 12, :2]) * 0.5
    lower = values[:, [13, 14, 15, 16], :2] - hip_center[:, None, :]
    if box is None:
        metric_scale = np.asarray([128.0, 256.0], dtype=np.float32)
    else:
        x1, y1, x2, y2 = box
        metric_scale = np.asarray(
            [max(float(x2 - x1), 1.0), max(float(y2 - y1), 1.0)],
            dtype=np.float32,
        )
    lower /= metric_scale
    valid_lower = confidence[:, [13, 14, 15, 16]] > 0
    deltas = np.linalg.norm(np.diff(lower, axis=0), axis=2)
    delta_valid = valid_lower[1:] & valid_lower[:-1]
    if timestamps is not None and len(timestamps) == len(values):
        times = np.asarray(timestamps, dtype=np.float64)
        deltas_time = np.diff(times)
        positive = deltas_time[deltas_time > 1e-6]
        if positive.size:
            nominal = float(np.median(positive))
            # Preserve the observed time base: do not turn a detector/track
            # gap into a single high-speed gait step.
            scale = np.clip(deltas_time / max(nominal, 1e-6), 1.0, 8.0)
            deltas = deltas / scale[:, None]
            delta_valid &= (deltas_time <= nominal * 2.5)[:, None]
    energy = float(deltas[delta_valid].mean()) if delta_valid.any() else 0.0
    walking_ratio = float(np.clip((energy - 0.003) / 0.025, 0.0, 1.0))

    phase = (
        values[:, 15, 1]
        - values[:, 16, 1]
        + 0.5 * (values[:, 13, 1] - values[:, 14, 1])
    ) / metric_scale[1]
    phase_valid = (
        (confidence[:, 13] > 0)
        & (confidence[:, 14] > 0)
        & (confidence[:, 15] > 0)
        & (confidence[:, 16] > 0)
    )
    if int(phase_valid.sum()) < 8:
        return walking_ratio, 0.0
    signal = phase[phase_valid]
    if len(signal) >= 5:
        signal = np.convolve(signal, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    signal -= float(np.median(signal))
    amplitude = float(np.percentile(signal, 90) - np.percentile(signal, 10))
    if amplitude < 0.015:
        return walking_ratio, 0.0
    threshold = max(0.003, amplitude * 0.12)
    signs = np.where(signal > threshold, 1, np.where(signal < -threshold, -1, 0))
    signs = signs[signs != 0]
    crossings = int(np.count_nonzero(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
    return walking_ratio, float(np.clip(crossings / 2.0, 0.0, 3.0))


def _box_jitter(boxes: Sequence[Box]) -> float:
    """估计近期轨迹中心运动的归一化方差。"""
    if len(boxes) < 3:
        return 0.0
    values = np.asarray(boxes, dtype=np.float32)
    centers = (values[:, :2] + values[:, 2:]) * 0.5
    sizes = np.maximum(values[:, 2:] - values[:, :2], 1.0)
    movement = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    scale = np.sqrt((sizes[:-1, 0] * sizes[:-1, 1]).clip(min=1.0))
    normalized = movement / scale
    return float(np.clip(np.std(normalized) * 3.0, 0.0, 1.0))


def _pose_jitter(poses: Sequence[np.ndarray]) -> float:
    """Estimate upper-body relative jitter without mistaking walking for noise."""

    if len(poses) < 3:
        return 0.0
    values = np.asarray(poses, dtype=np.float32)
    confidence = values[:, :, 2]
    anchor_valid = (confidence[:, 11] > 0.0) & (confidence[:, 12] > 0.0)
    if int(anchor_valid.sum()) < 3:
        return 0.0
    anchors = (values[:, 11, :2] + values[:, 12, :2]) * 0.5
    shoulders = values[:, [5, 6], :2] - anchors[:, None, :]
    valid = (
        anchor_valid[:, None]
        & (confidence[:, [5, 6]] > 0.0)
        & np.isfinite(shoulders).all(axis=2)
    )
    delta = np.linalg.norm(np.diff(shoulders, axis=0), axis=2)
    paired = valid[:-1] & valid[1:]
    if not paired.any():
        return 0.0
    torso = np.linalg.norm(values[:, 5, :2] - anchors, axis=1)
    scale = float(np.median(torso[np.isfinite(torso) & (torso > 1e-3)]))
    if not np.isfinite(scale) or scale <= 1e-3:
        return 0.0
    return float(np.clip(np.median(delta[paired]) / scale, 0.0, 1.0))


def _lower_body_mean_score(poses: Sequence[np.ndarray]) -> float:
    """Return the observed RTMPose confidence for hips, knees and ankles."""

    if len(poses) == 0:
        return 0.0
    values = np.asarray(poses, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (17, 3):
        return 0.0
    scores = values[:, [11, 12, 13, 14, 15, 16], 2]
    finite = np.where(np.isfinite(scores), scores, 0.0)
    return float(np.clip(np.mean(finite), 0.0, 1.0))


def _longest_leg_gap_frames(poses: Sequence[np.ndarray]) -> int:
    """Return the longest consecutive slot without sufficient leg evidence."""

    longest = 0
    current = 0
    for pose in poses:
        values = np.asarray(pose, dtype=np.float32)
        visible = (
            values.shape == (17, 3)
            and np.isfinite(values[[11, 12, 13, 14, 15, 16], 2]).all()
            and float(np.mean(values[[11, 12, 13, 14, 15, 16], 2] > 0.0)) >= 0.5
        )
        if visible:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _camera_zone(box: Box, frame_shape: tuple[int, ...]) -> str:
    """Return a calibration-free 3x3 image zone for prototype auditing."""

    height, width = frame_shape[:2]
    center_x = (float(box[0]) + float(box[2])) * 0.5 / max(float(width), 1.0)
    center_y = (float(box[1]) + float(box[3])) * 0.5 / max(float(height), 1.0)
    column = min(2, max(0, int(center_x * 3.0)))
    row = min(2, max(0, int(center_y * 3.0)))
    return f"r{row}c{column}"


def _ground_point(
    points: np.ndarray | None,
    box: Box,
    homography: tuple[float, ...] | None,
) -> tuple[tuple[float, float] | None, str | None]:
    """Project a ground-contact proxy without modifying pose model inputs."""

    if homography is None:
        return None, None
    anchor: tuple[float, float]
    source: str
    if (
        points is not None
        and points.shape == (17, 3)
        and np.isfinite(points[[15, 16]]).all()
        and np.all(points[[15, 16], 2] > 0.0)
    ):
        ankles = points[[15, 16], :2]
        anchor = (float(ankles[:, 0].mean()), float(ankles[:, 1].mean()))
        source = "ankle_midpoint"
    else:
        anchor = ((float(box[0]) + float(box[2])) * 0.5, float(box[3]))
        source = "bbox_bottom"
    matrix = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    point = np.asarray([anchor[0], anchor[1], 1.0], dtype=np.float64)
    projected = matrix @ point
    if not np.isfinite(projected).all() or abs(float(projected[2])) <= 1e-8:
        return None, None
    return (
        (float(projected[0] / projected[2]), float(projected[1] / projected[2])),
        source,
    )


def _ground_speed(
    points: Sequence[tuple[float, float] | None],
    timestamps: Sequence[float],
) -> float | None:
    """Return the latest calibrated ground speed, or ``None`` without data."""

    for index in range(len(points) - 1, 0, -1):
        current, previous = points[index], points[index - 1]
        if current is None or previous is None:
            continue
        elapsed = float(timestamps[index] - timestamps[index - 1])
        if elapsed <= 1e-6:
            continue
        return float(np.linalg.norm(np.asarray(current) - np.asarray(previous)) / elapsed)
    return None


def _view_consensus(samples: Sequence[_CameraViewSample]) -> tuple[str | None, float]:
    """Return the most frequent recent view and its observed share."""

    labels = [item.view_angle for item in samples if item.view_angle is not None]
    if not labels:
        return None, 0.0
    counts = {label: labels.count(label) for label in set(labels)}
    latest_index = {
        label: max(index for index, item in enumerate(labels) if item == label)
        for label in counts
    }
    view = max(counts, key=lambda label: (counts[label], latest_index[label]))
    return view, float(counts[view] / len(labels))


def _world_trajectory(
    points: Sequence[tuple[float, float] | None],
    timestamps: Sequence[float],
    *,
    window_seconds: float,
    maximum_speed_mps: float,
) -> _WorldTrajectory:
    """Fit a short ground trajectory after rejecting impossible track jumps."""

    samples = [
        (float(timestamp), np.asarray(point, dtype=np.float64))
        for point, timestamp in zip(points, timestamps)
        if point is not None
        and np.isfinite(point).all()
        and np.isfinite(timestamp)
    ]
    if not samples:
        return _WorldTrajectory()
    latest_time = samples[-1][0]
    samples = [item for item in samples if item[0] >= latest_time - window_seconds]
    accepted = [samples[0]]
    outlier_count = 0
    for timestamp, point in samples[1:]:
        previous_time, previous_point = accepted[-1]
        elapsed = timestamp - previous_time
        if elapsed <= 1e-6:
            outlier_count += 1
            continue
        if float(np.linalg.norm(point - previous_point) / elapsed) > maximum_speed_mps:
            outlier_count += 1
            continue
        accepted.append((timestamp, point))
    if len(accepted) < 2:
        return _WorldTrajectory(
            position=tuple(float(item) for item in accepted[-1][1]),
            quality=0.0,
            valid_samples=len(accepted),
            outlier_count=outlier_count,
        )
    times = np.asarray([item[0] for item in accepted], dtype=np.float64)
    values = np.asarray([item[1] for item in accepted], dtype=np.float64)
    relative_times = times - times[0]
    span = float(relative_times[-1])
    if span <= 1e-6:
        return _WorldTrajectory(
            position=tuple(float(item) for item in values[-1]),
            quality=0.0,
            valid_samples=len(accepted),
            outlier_count=outlier_count,
        )
    slopes = np.asarray(
        [np.polyfit(relative_times, values[:, axis], 1)[0] for axis in range(2)],
        dtype=np.float64,
    )
    fitted_start = np.asarray(
        [np.polyfit(relative_times, values[:, axis], 1)[1] for axis in range(2)],
        dtype=np.float64,
    )
    displacement_vector = slopes * span
    displacement = float(np.linalg.norm(displacement_vector))
    valid_fraction = len(accepted) / max(len(samples), 1)
    return _WorldTrajectory(
        position=tuple(float(item) for item in fitted_start + displacement_vector),
        speed=float(np.linalg.norm(slopes)),
        displacement=displacement,
        direction_degrees=(
            float(np.degrees(np.arctan2(displacement_vector[1], displacement_vector[0])))
            if displacement > 1e-8
            else None
        ),
        quality=float(np.clip(valid_fraction * min(1.0, span / window_seconds), 0.0, 1.0)),
        valid_samples=len(accepted),
        outlier_count=outlier_count,
    )


def _abrupt_track_jump(previous: Box, current: Box) -> bool:
    """检测过大的位移，判断其是否不可能属于同一轨迹的延续。"""
    previous_center = np.asarray(
        [(previous[0] + previous[2]) * 0.5, (previous[1] + previous[3]) * 0.5]
    )
    current_center = np.asarray(
        [(current[0] + current[2]) * 0.5, (current[1] + current[3]) * 0.5]
    )
    diagonal = float(
        np.hypot(previous[2] - previous[0], previous[3] - previous[1])
    )
    return float(np.linalg.norm(current_center - previous_center)) > max(80.0, diagonal * 1.25)


def _view_angle(points: np.ndarray | None) -> str | None:
    """根据肩部到躯干的几何关系判断正面、斜面或侧面视角。"""
    if points is None or not np.all(points[[5, 6, 11, 12], 2] > 0):
        return None
    shoulder_width = abs(float(points[6, 0] - points[5, 0]))
    torso_height = abs(float(points[[11, 12], 1].mean() - points[[5, 6], 1].mean()))
    ratio = shoulder_width / max(torso_height, 1.0)
    if ratio < 0.45:
        return "side"
    if ratio > 0.85:
        return "frontal"
    return "oblique"


class ProductionVisionAdapter:
    """满足两方法 ``VisionAdapter`` 接口的深度生产适配器。

    一个实例拥有一个输入源会话的全部时序缓存。下游代码只看到稳定的轨迹 ID、
    归一化嵌入和质量元数据，不需要知道图像来自 YOLO/RTMPose 还是未来的模型实现。
    """

    supports_automatic_registration = True
    # The GPGait++ contract starts with raw full-frame
    # RTMPose coordinates followed by official Affine / RescaleCenter / HOD.  A
    # database produced by the old bbox-normalized path must not be reused as
    # if its gait prototypes were comparable to this stream.
    model_version = "yolo11l-bytetrack+rtmpose-s+osnet-ain+gpgaitpp-ccpg-v1"
    feature_schema = (
        "osnet-ain-512-imagenet-v1|"
        "gpgaitpp-ccpg-coco17-affine-hod6-bnneck-v1"
    )
    preprocess_version = "rtmpose-s-256x192-simcc2|fastposegait-d3ce631-affine-rescalecenter-hod6-skeleton-v1"
    joint_format = "COCO-17"
    tta_mode = "none"
    coordinate_contract = "raw-full-frame-image-coordinates->official-affine-scale225-neck"
    embedding_dimensions = {"appearance": 512, "gait": 4864}

    @property
    def artifact_sha256(self) -> str:
        """返回模型清单的稳定指纹，作为图库协议的一部分。

        启动时 ``production_readiness(..., verify_hashes=True)`` 校验真实
        文件；这里保存清单内容的摘要，避免每一帧重新读取大权重。
        """
        if not MODEL_MANIFEST.is_file():
            return "manifest-missing"
        try:
            payload = MODEL_MANIFEST.read_bytes()
            return hashlib.sha256(payload).hexdigest()
        except OSError:
            return "manifest-unreadable"

    @property
    def sequence_length(self) -> int:
        """返回当前 GPGait++ 窗口长度。"""
        return int(self.config.gait_sequence_length)
    _RUNTIME_PARAMETER_NAMES = (
        "detector_confidence",
        "output_confidence",
        "detector_iou",
        "detector_inference_stride",
        "keypoint_confidence",
        "minimum_pose_frames",
        "appearance_stride",
        "gait_inference_stride",
        "low_light_threshold",
    )

    def __init__(
        self,
        config: ProductionVisionConfig | None = None,
    ) -> None:
        """校验就绪状态，并准备延迟加载的模型槽位。"""
        self.config = config or ProductionVisionConfig()
        ready, issues = production_readiness(self.config, verify_hashes=True)
        if not ready:
            raise ProductionVisionError("；".join(issues))
        self.detector: _YoloByteTracker | None = None
        self.pose: _RtmposeEstimator | None = None
        self.appearance: _OsnetAppearanceExtractor | None = None
        self.gait: GPGaitPPEncoder | None = None
        self.posec3d: PoseC3DActionClassifier | None = None
        self.action_classifier = (
            PoseActionClassifier.from_path(self.config.action_classifier_path)
            if self.config.action_classifier_path is not None
            else PoseActionClassifier(minimum_frames=self.config.minimum_pose_frames)
        )
        self._pose_buffer_length = int(self.config.gait_sequence_length)
        self._action_router = ActionRouter()
        self.enhancer = _LowLightEnhancer(
            self.config.low_light_threshold,
            self.config.low_light_check_interval,
            self.config.low_light_hysteresis,
            self.config.low_light_min_hold_frames,
        )
        self.states: dict[int, _TrackState] = {}
        self.latest: dict[int, VisionTrack] = {}
        self.frame_index = 0
        self._cached_detections: tuple[_Detection, ...] = ()
        self._last_detector_frame: int | None = None
        self._detector_refreshed_this_frame = False
        self._current_timestamp: float | None = None
        self._last_frame_latency_ms = 0.0
        self._frame_latency_samples_ms: deque[float] = deque(maxlen=120)

    def _record_frame_latency(self, started: float) -> None:
        """记录最近端到端帧处理耗时，供启动/运行诊断使用。"""

        elapsed = (time.perf_counter() - started) * 1000.0
        self._last_frame_latency_ms = elapsed
        self._frame_latency_samples_ms.append(elapsed)

    @staticmethod
    def _notify_stage(
        callback: Callable[[str, float], None] | None,
        text: str,
        progress: float,
    ) -> None:
        """向后台预加载调用方报告一个可展示的阶段。"""

        if callback is not None:
            callback(text, max(0.0, min(1.0, float(progress))))

    def preload(
        self,
        on_stage: Callable[[str, float], None] | None = None,
    ) -> None:
        """按 YOLO、RTMPose、OSNet、可选 PoseC3D、GPGait++ 顺序加载模型。"""

        posec3d_ready = (
            self.config.posec3d_checkpoint_path is None or self.posec3d is not None
        )
        if all(
            component is not None
            for component in (self.detector, self.pose, self.appearance, self.gait)
        ) and posec3d_ready:
            return
        try:
            self._notify_stage(on_stage, "正在加载 YOLO/ByteTrack…", 0.16)
            self.detector = _YoloByteTracker(self.config)
            self._notify_stage(on_stage, "YOLO/ByteTrack 已加载", 0.30)

            self._notify_stage(on_stage, "正在加载 RTMPose…", 0.30)
            self.pose = _RtmposeEstimator(
                self.config.pose_path,
                require_cuda=self.config.require_cuda,
                cuda_device_id=(
                    self.config.cuda_device_index if self.config.require_cuda else 0
                ),
            )
            self._notify_stage(on_stage, "RTMPose 已加载", 0.44)

            self._notify_stage(on_stage, "正在加载 OSNet-AIN…", 0.44)
            self.appearance = _OsnetAppearanceExtractor(
                self.config.appearance_path,
                self.config.torch_device,
            )
            self._notify_stage(on_stage, "OSNet-AIN 已加载", 0.58)

            if self.config.posec3d_checkpoint_path is not None:
                self._notify_stage(on_stage, "正在加载 PoseC3D-HMDB51…", 0.58)
                self.posec3d = PoseC3DActionClassifier(
                    self.config.posec3d_checkpoint_path,
                    self.config.posec3d_config_path,
                    self.config.posec3d_label_map_path,
                    device=self.config.torch_device,
                    probability_threshold=self.config.posec3d_probability_threshold,
                    margin_threshold=self.config.posec3d_margin_threshold,
                )
                self._notify_stage(on_stage, "PoseC3D-HMDB51 已加载", 0.62)

            self._notify_stage(
                on_stage,
                "正在加载 GPGait++…",
                0.62 if self.posec3d is not None else 0.58,
            )
            self.gait = GPGaitPPEncoder(
                self.config.gait_path,
                device=self.config.torch_device,
                sequence_length=self.config.gait_sequence_length,
            )
            gait_dimension = getattr(self.gait, "output_dimension", None)
            if gait_dimension is not None and gait_dimension != 4864:
                raise ProductionVisionError(
                    "GPGait++ production contract requires a 4864-dimensional BN-neck embedding"
                )
            appearance_dimension = getattr(self.appearance, "output_dimension", None)
            if appearance_dimension is not None and appearance_dimension != 512:
                raise ProductionVisionError(
                    "OSNet-AIN production contract requires a 512-dimensional embedding"
                )
            self._notify_stage(on_stage, "GPGait++ 已加载", 0.70)
        except Exception:
            self.detector = None
            self.pose = None
            self.appearance = None
            self.gait = None
            self.posec3d = None
            raise

    def warmup(
        self,
        on_stage: Callable[[str, float], None] | None = None,
    ) -> None:
        """对身份模型和可选动作模型执行空白输入预热。"""

        self.preload(on_stage=on_stage)
        assert self.detector is not None
        assert self.pose is not None
        assert self.appearance is not None
        assert self.gait is not None

        self._notify_stage(on_stage, "正在预热 YOLO/ByteTrack…", 0.70)
        self.detector.warmup()
        self._notify_stage(on_stage, "YOLO/ByteTrack 预热完成", 0.77)

        self._notify_stage(on_stage, "正在预热 RTMPose…", 0.77)
        self.pose.warmup()
        self._notify_stage(on_stage, "RTMPose 预热完成", 0.84)

        self._notify_stage(on_stage, "正在预热 OSNet-AIN…", 0.84)
        self.appearance.warmup()
        self._notify_stage(on_stage, "OSNet-AIN 预热完成", 0.91)

        if self.posec3d is not None:
            self._notify_stage(on_stage, "正在预热 PoseC3D-HMDB51…", 0.91)
            self.posec3d.warmup()
            self._notify_stage(on_stage, "PoseC3D-HMDB51 预热完成", 0.94)

        self._notify_stage(
            on_stage,
            "正在预热 GPGait++…",
            0.94 if self.posec3d is not None else 0.91,
        )
        self.gait.warmup()
        self._notify_stage(on_stage, "GPGait++ 预热完成", 0.97)

    def _load(self) -> None:
        """兼容旧入口，只执行分阶段模型加载而不额外预热。"""

        self.preload()

    @property
    def backend_status(self) -> str:
        """描述已加载的后端和当前使用的 ONNX provider。"""
        if self.detector is None:
            return "生产模型将在开始采集时加载"
        pose_backend = (
            ",".join(getattr(self.pose, "providers", ()))
            if self.pose is not None
            else "none"
        )
        pose_latency = (
            float(getattr(self.pose, "last_latency_ms", 0.0))
            if self.pose is not None
            else 0.0
        )
        latency = (
            f", last RTMPose {pose_latency:.1f}ms"
            if pose_latency > 0
            else ""
        )
        posec3d_status = (
            f"PoseC3D({self.posec3d.model_version}) + "
            if self.posec3d is not None
            else ""
        )
        return (
            f"YOLO/ByteTrack + RTMPose({pose_backend}{latency}) + "
            f"PoseActionClassifier({self.action_classifier.model_version}) + "
            f"{posec3d_status}"
            "OSNet-AIN + GPGait++"
        )

    def diagnostics(self) -> dict[str, object]:
        """返回启动/运行诊断，明确 provider 列表不等于节点全在 GPU。"""
        if self.pose is None:
            return {"loaded": False, "providers": (), "node_placement": "not-observed"}
        return {
            "loaded": True,
            "providers": tuple(getattr(self.pose, "providers", ())),
            "provider_options": dict(getattr(self.pose, "provider_options", {})),
            "last_rtmpose_latency_ms": float(
                getattr(self.pose, "last_latency_ms", 0.0)
            ),
            "last_frame_latency_ms": self._last_frame_latency_ms,
            "p95_frame_latency_ms": (
                float(np.percentile(tuple(self._frame_latency_samples_ms), 95))
                if self._frame_latency_samples_ms
                else 0.0
            ),
            "node_placement": "ORT graph partitioning; inspect profiling for per-node placement",
            "action_classifier_model_version": self.action_classifier.model_version,
            "action_classifier_feature_schema": "pose-action-kinematics-v1",
            "posec3d_model_version": (
                self.posec3d.model_version if self.posec3d is not None else None
            ),
            "posec3d_enabled": self.posec3d is not None,
        }

    def pose_snapshot(self, track_id: int) -> dict[str, object] | None:
        """Return the latest real pose row for an emitted track.

        The snapshot is intended for the annotation-to-training tool.  It is
        read-only and copies arrays so a caller cannot mutate the live tracker
        state.  A missing/expired track returns ``None`` instead of fabricating
        a pose row.
        """

        state = self.states.get(int(track_id))
        track = self.latest.get(int(track_id))
        if state is None or track is None or not state.poses or not state.timestamps:
            return None
        pose = np.asarray(state.poses[-1], dtype=np.float32)
        if pose.shape != (17, 3):
            return None
        return {
            "keypoints_xy": pose[:, :2].copy(),
            "keypoint_scores": pose[:, 2].copy(),
            "valid_mask": (pose[:, 2] > 0.0).copy(),
            "boxes_xyxy": np.asarray(track.box, dtype=np.float32),
            "frame_index": int(self.frame_index),
            "timestamp": float(state.timestamps[-1]),
            "track_id": int(track.track_id),
            "missing": bool(not np.any(pose[:, 2] > 0.0)),
            "action_type": str(track.metadata.get("action_type", ActionType.UNKNOWN.value)),
        }

    def runtime_parameters(self) -> dict[str, int | float]:
        """返回已加载或未加载适配器支持的热更新参数。"""

        return {
            name: getattr(self.config, name)
            for name in self._RUNTIME_PARAMETER_NAMES
        }

    def update_runtime_parameters(self, changes: dict[str, int | float]) -> None:
        """校验并应用视觉阈值，不重新加载模型权重。"""

        unknown = sorted(set(changes) - set(self._RUNTIME_PARAMETER_NAMES))
        if unknown:
            raise ValueError(f"unsupported production vision parameters: {unknown}")
        candidate = replace(self.config, **changes)
        reset_gait = any(
            name in changes
            for name in ("keypoint_confidence", "minimum_pose_frames")
        )
        self.config = candidate
        if self.action_classifier.model is None:
            self.action_classifier.minimum_frames = int(candidate.minimum_pose_frames)
        if self.detector is not None:
            self.detector.config = candidate
        if any(
            name in changes
            for name in (
                "detector_confidence",
                "output_confidence",
                "detector_iou",
                "detector_inference_stride",
            )
        ):
            # 下一帧必须重新检测，不能让旧阈值产生的缓存框继续生效。
            self._cached_detections = ()
            self._last_detector_frame = None
        self.enhancer.threshold = candidate.low_light_threshold
        self.enhancer.interval = max(1, candidate.low_light_check_interval)
        if reset_gait:
            for state in self.states.values():
                self._start_new_subtracklet(
                    state,
                    self.frame_index,
                    ("runtime_parameter_change",),
                )

    def reset(self) -> None:
        """清除跟踪器状态、时序窗口和缓存的轨迹输出。"""
        if self.detector is not None:
            self.detector.reset()
        self.enhancer.reset()
        self.states.clear()
        self.latest.clear()
        self.frame_index = 0
        self._cached_detections = ()
        self._last_detector_frame = None
        self._detector_refreshed_this_frame = False
        self._current_timestamp = None
        self._last_frame_latency_ms = 0.0
        self._frame_latency_samples_ms.clear()

    def _detections_for_frame(self, frame_bgr: np.ndarray) -> tuple[_Detection, ...]:
        """按配置刷新检测，并在中间帧复用最近一次轨迹框。

        缓存为空时始终执行真实检测，以便新人物能够立即进入管线。已有检测最多
        只复用 ``detector_inference_stride - 1`` 帧；下一次 YOLO/ByteTrack
        返回空结果时会立刻清空缓存，避免长期保留已经离场的人物。
        """

        due = (
            not self._cached_detections
            or self._last_detector_frame is None
            or self.frame_index - self._last_detector_frame
            >= self.config.detector_inference_stride
        )
        self._detector_refreshed_this_frame = due
        if due:
            assert self.detector is not None
            self._cached_detections = self.detector.track(frame_bgr)
            self._last_detector_frame = self.frame_index
            return self._cached_detections
        # ByteTrack is stateful, but it only sees detector calls. On skipped
        # frames, extrapolate exclusively from its two latest *real*
        # observations. A predicted box must never become the next velocity
        # anchor, otherwise detector stride recursively amplifies crop drift.
        height, width = frame_bgr.shape[:2]
        predicted: list[_Detection] = []
        for detection in self._cached_detections:
            state = self.states.get(detection.track_id)
            box = detection.box
            if state is not None and len(state.real_boxes) >= 2:
                previous = np.asarray(state.real_boxes[-2], dtype=np.float32)
                current = np.asarray(state.real_boxes[-1], dtype=np.float32)
                previous_frame, current_frame = state.real_box_frames
                velocity = (current - previous) / max(current_frame - previous_frame, 1)
                raw = current + velocity * max(self.frame_index - current_frame, 0)
                x1, y1, x2, y2 = (int(round(float(value))) for value in raw)
                x1 = max(0, min(x1, max(width - 1, 0)))
                y1 = max(0, min(y1, max(height - 1, 0)))
                x2 = max(x1 + 1, min(x2, width))
                y2 = max(y1 + 1, min(y2, height))
                box = (x1, y1, x2, y2)
                state.detector_gap_frames += 1
                state.lifetime_detector_gap_frames += 1
            predicted.append(_Detection(detection.track_id, box, detection.confidence))
        return tuple(predicted)

    def _expire_states(
        self,
        active_ids: set[int],
        *,
        timestamp: float | None = None,
    ) -> None:
        """删除缺席时间超过保留策略的时序状态。"""
        for track_id, state in tuple(self.states.items()):
            if track_id in active_ids:
                continue
            frame_gap = self.frame_index - state.last_seen_frame
            time_gap = (
                float(timestamp - state.last_seen_timestamp)
                if timestamp is not None and state.last_seen_timestamp is not None
                else 0.0
            )
            if (
                time_gap > self.config.state_retention_seconds
                or frame_gap > self.config.state_retention_frames
            ):
                # A reappearing ByteTrack ID is not allowed to inherit old gait
                # evidence after a real track gap.
                del self.states[track_id]
                self.latest.pop(track_id, None)

    @staticmethod
    def _start_new_subtracklet(
        state: _TrackState,
        frame_index: int,
        reasons: Sequence[str],
    ) -> None:
        """关闭旧时序窗口，并从当前帧开始一个新的干净子轨迹。"""

        state.segment_id += 1
        state.segment_start_frame = int(frame_index)
        state.segment_boundary_reasons = tuple(dict.fromkeys(str(item) for item in reasons))
        state.poses.clear()
        state.timestamps.clear()
        state.boxes.clear()
        state.real_boxes.clear()
        state.real_box_frames.clear()
        state.views.clear()
        state.ground_points.clear()
        state.camera_view_timeline.clear()
        state.valid_pose_frames = 0
        state.valid_leg_frames = 0
        state.frame_count = 0
        state.detector_gap_frames = 0
        state.track_gap_frames = 0
        state.id_switches = 0
        state.gait = None
        state.last_gait_frame = None
        state.appearance_similarity_to_previous = None
        state.appearance_change_suspected = False
        # A new clean subtracklet may belong to another person. The previous
        # embedding is audit context only; never carry it into new evidence.
        state.appearance = None
        state.last_appearance_frame = None
        state.action_type = ActionType.UNKNOWN
        state.action_branch_generation += 1
        state.action_segment_id += 1
        state.action_boundary_reason = "subtracklet_boundary"
        state.action_candidate_type = None
        state.action_candidate_frames = 0
        state.action_unknown_frames = 0
        state.posec3d_prediction = None

    @staticmethod
    def _reset_action_window(state: _TrackState, *, reason: str) -> None:
        """Keep only the boundary frame after a confirmed action change.

        A gait encoder must not receive a window that straddles walking and a
        different action.  The current frame is retained as the first sample
        of the new branch; the following frames rebuild the minimum temporal
        context.  We intentionally keep real detector boxes so ByteTrack's
        motion anchor is not reset just because the action label changed.
        """

        current_pose = state.poses[-1].copy() if state.poses else None
        current_timestamp = state.timestamps[-1] if state.timestamps else None
        current_view = state.views[-1] if state.views else None
        current_box = state.boxes[-1] if state.boxes else None
        current_ground_point = state.ground_points[-1] if state.ground_points else None
        current_camera_view = (
            state.camera_view_timeline[-1] if state.camera_view_timeline else None
        )
        state.poses.clear()
        state.timestamps.clear()
        state.views.clear()
        state.boxes.clear()
        state.ground_points.clear()
        state.camera_view_timeline.clear()
        state.gait = None
        state.last_gait_frame = None
        state.posec3d_prediction = None
        state.action_segment_id += 1
        state.action_boundary_reason = reason
        if current_pose is not None and current_timestamp is not None:
            state.poses.append(current_pose)
            state.timestamps.append(float(current_timestamp))
            state.views.append(current_view)
            state.ground_points.append(current_ground_point)
            if current_camera_view is not None:
                state.camera_view_timeline.append(current_camera_view)
            state.valid_pose_frames = int(np.any(current_pose[:, 2] > 0.0))
            state.valid_leg_frames = int(
                np.mean(current_pose[[11, 12, 13, 14, 15, 16], 2] > 0.0) >= 0.45
            )
        else:
            state.valid_pose_frames = 0
            state.valid_leg_frames = 0
        if current_box is not None:
            state.boxes.append(current_box)

    def _advance_action_state(
        self,
        state: _TrackState,
        prediction: ActionPrediction,
    ) -> bool:
        """Commit only stable action changes and expose a fail-closed gate.

        The raw classifier may flicker for a few frames.  During a candidate
        transition no gait is emitted; a committed transition starts a fresh
        action segment, so the fixed-length gait window cannot straddle two
        actions.  This state belongs to the production adapter rather than to
        the classifier because it must reset its pose/embedding caches.
        """

        is_strong = prediction.quality == ActionQuality.STRONG
        if not is_strong or prediction.action_type == ActionType.UNKNOWN:
            state.action_candidate_type = None
            state.action_candidate_frames = 0
            state.action_unknown_frames += 1
            if (
                state.action_type != ActionType.UNKNOWN
                and len(state.poses) >= self.config.minimum_pose_frames
                and state.action_unknown_frames >= self.config.action_unknown_end_frames
            ):
                state.action_type = ActionType.UNKNOWN
                state.action_branch_generation += 1
                self._reset_action_window(state, reason="sustained_unknown")
            return False

        state.action_unknown_frames = 0
        if prediction.action_type == state.action_type:
            state.action_candidate_type = None
            state.action_candidate_frames = 0
            return True

        # The first strong action is produced from the classifier's own
        # minimum complete window.  There is no preceding committed action to
        # mix with, so preserve that evidence for startup latency while still
        # assigning an explicit segment for downstream aggregation.
        if state.action_type == ActionType.UNKNOWN:
            state.action_type = prediction.action_type
            state.action_branch_generation += 1
            state.action_segment_id += 1
            state.action_boundary_reason = "initial_action_confirmed"
            state.action_candidate_type = None
            state.action_candidate_frames = 0
            return True

        if prediction.action_type == state.action_candidate_type:
            state.action_candidate_frames += 1
        else:
            state.action_candidate_type = prediction.action_type
            state.action_candidate_frames = 1

        # The first concrete action after an UNKNOWN warm-up still gets a
        # clean temporal segment.  It has no write permission until enough
        # subsequent samples rebuild that segment.
        if state.action_candidate_frames < self.config.action_stability_frames:
            return False

        state.action_type = prediction.action_type
        state.action_branch_generation += 1
        state.action_candidate_type = None
        state.action_candidate_frames = 0
        self._reset_action_window(state, reason="stable_action_change")
        return True

    def _posec3d_prediction(
        self,
        state: _TrackState,
        frame_shape: tuple[int, int],
    ) -> PoseC3DPrediction | None:
        """Return a cached or fresh PoseC3D result for the current track window."""

        if self.posec3d is None:
            return None
        clip_length = self.config.posec3d_clip_length
        if len(state.poses) < clip_length:
            state.posec3d_prediction = None
            return PoseC3DPrediction.invalid("posec3d_window_short")
        window = np.asarray(tuple(state.poses)[-clip_length:], dtype=np.float32)
        if (
            window.shape != (clip_length, 17, 3)
            or not np.isfinite(window).all()
            or not np.any(window[:, :, 2] > 0.0, axis=1).all()
        ):
            state.posec3d_prediction = None
            return PoseC3DPrediction.invalid("posec3d_window_invalid")
        if (
            state.posec3d_prediction is not None
            and state.frame_count % self.config.posec3d_inference_stride != 0
        ):
            return state.posec3d_prediction
        prediction = self.posec3d.predict(window, image_shape=frame_shape)
        state.posec3d_prediction = prediction
        return prediction

    @staticmethod
    def _fuse_action_predictions(
        heuristic: ActionPrediction,
        semantic: PoseC3DPrediction | None,
        *,
        probability_threshold: float,
        margin_threshold: float,
        model_version: str,
    ) -> ActionPrediction:
        """Fuse semantic and kinematic evidence without creating a soft fallback."""

        if semantic is None:
            return heuristic
        semantic_ok = bool(
            semantic.reason == "ok"
            and semantic.top_label == "walk"
            and semantic.walk_probability >= probability_threshold
            and semantic.margin >= margin_threshold
        )
        if (
            not semantic_ok
            or heuristic.action_type != ActionType.WALK
            or heuristic.quality != ActionQuality.STRONG
        ):
            quality = (
                ActionQuality.PARTIAL
                if heuristic.quality != ActionQuality.INVALID
                else ActionQuality.INVALID
            )
            return ActionPrediction(
                ActionType.UNKNOWN,
                confidence=0.0,
                quality=quality,
                completion=heuristic.completion,
                source="posec3d_hmdb51+pose_heuristic",
                model_version=model_version,
            )
        return ActionPrediction(
            ActionType.WALK,
            confidence=min(heuristic.confidence, semantic.walk_probability),
            quality=ActionQuality.STRONG,
            completion=heuristic.completion,
            source="posec3d_hmdb51+pose_heuristic",
            model_version=model_version,
        )

    @staticmethod
    def _append_missing_slot(
        state: _TrackState,
        timestamp: float,
        frame_index: int,
    ) -> None:
        """在短暂漏检期间保留一个带掩码的时间槽。"""

        if state.last_seen_timestamp is not None and timestamp <= state.last_seen_timestamp:
            return
        state.poses.append(np.zeros((17, 3), dtype=np.float32))
        state.timestamps.append(float(timestamp))
        state.posec3d_prediction = None
        state.views.append(None)
        state.ground_points.append(None)
        state.camera_view_timeline.append(
            _CameraViewSample(
                frame_index=int(frame_index),
                timestamp=float(timestamp),
                view_angle=None,
                camera_zone=None,
                world_position=None,
                ground_anchor_source=None,
                missing=True,
            )
        )
        state.valid_pose_frames = sum(
            bool(np.any(item[:, 2] > 0.0)) for item in state.poses
        )
        state.valid_leg_frames = sum(
            bool(np.mean(item[[11, 12, 13, 14, 15, 16], 2] > 0.0) >= 0.45)
            for item in state.poses
        )

    def process_at(
        self,
        frame_bgr: np.ndarray,
        *,
        timestamp: float | None = None,
    ) -> tuple[VisionTrack, ...]:
        """按输入源时间戳处理一帧，保留 ``process`` 的兼容入口。"""
        previous = self._current_timestamp
        resolved_timestamp = time.time() if timestamp is None else float(timestamp)
        if not np.isfinite(resolved_timestamp):
            raise ValueError("timestamp must be finite")
        self._current_timestamp = resolved_timestamp
        try:
            return self.process(frame_bgr)
        finally:
            self._current_timestamp = previous

    def process(self, frame_bgr: np.ndarray) -> tuple[VisionTrack, ...]:
        """运行一帧 BGR 图像的生产处理链。

        处理顺序是检测/跟踪、批量姿态推理、外观刷新、时序步态更新、质量计算，
        最后输出 ``VisionTrack``。当前检测框异常时会把步态质量强制设为零，
        防止过期的缓存嵌入成为强证据。
        """
        if frame_bgr is None:
            return ()
        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError("frame_bgr must be a numpy.ndarray")
        if frame_bgr.size == 0:
            return ()
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must have shape (height, width, 3)")
        started = time.perf_counter()
        self._load()
        if any(
            component is None
            for component in (self.detector, self.pose, self.appearance, self.gait)
        ):
            raise ProductionVisionError("生产视觉模型未完成加载")
        assert self.detector is not None
        assert self.pose is not None
        assert self.appearance is not None
        assert self.gait is not None
        timestamp = self._current_timestamp
        if timestamp is None:
            timestamp = time.time()
        self.frame_index += 1
        inference_frame = self.enhancer.apply(frame_bgr)
        detections = self._detections_for_frame(inference_frame)
        active_detection_ids = {item.track_id for item in detections}
        for track_id, state in tuple(self.states.items()):
            if track_id not in active_detection_ids:
                self._append_missing_slot(state, timestamp, self.frame_index)
        if not detections:
            self._expire_states(set(), timestamp=timestamp)
            self._record_frame_latency(started)
            return ()

        boxes = [item.box for item in detections]
        poses = self.pose.extract(inference_frame, boxes)
        if len(poses) != len(boxes):
            raise ProductionVisionError(
                "RTMPose 输出数量与检测框数量不一致："
                f"poses={len(poses)}, boxes={len(boxes)}"
            )
        overlaps = occlusion_scores(np.asarray(boxes, dtype=np.float32))

        frame_metrics: list[
            tuple[
                np.ndarray | None,
                float,
                float,
                float,
                float,
                ActionPrediction,
                PoseC3DPrediction | None,
                bool,
                str,
            ]
        ] = []
        gait_indexes: list[int] = []
        for index, detection in enumerate(detections):
            state = self.states.get(detection.track_id)
            if state is None:
                state = _TrackState(
                    poses=deque(maxlen=self._pose_buffer_length),
                    timestamps=deque(maxlen=self._pose_buffer_length),
                    views=deque(maxlen=self._pose_buffer_length),
                    ground_points=deque(maxlen=self._pose_buffer_length),
                    camera_view_timeline=deque(maxlen=self._pose_buffer_length),
                    segment_start_frame=self.frame_index,
                )
                self.states[detection.track_id] = state
            hard_jump = False
            if (
                state.last_seen_timestamp is not None
                and timestamp - state.last_seen_timestamp
                > self.config.maximum_track_gap_seconds
            ):
                state.track_gap_frames += 1
                state.lifetime_track_gap_frames += 1
                self._start_new_subtracklet(
                    state,
                    self.frame_index,
                    ("track_gap",),
                )
            if state.real_boxes and _abrupt_track_jump(state.real_boxes[-1], detection.box):
                state.id_switches += 1
                state.lifetime_id_switches += 1
                hard_jump = True
                self._start_new_subtracklet(
                    state,
                    self.frame_index,
                    ("track_id_switch",),
                )
            state.frame_count += 1
            state.lifetime_frame_count += 1
            state.last_seen_frame = self.frame_index
            state.last_seen_timestamp = timestamp
            state.boxes.append(detection.box)
            if self._detector_refreshed_this_frame:
                state.real_boxes.append(detection.box)
                state.real_box_frames.append(self.frame_index)
                state.detector_gap_frames = 0
            canonical = _canonical_pose(
                None if hard_jump else poses[index],
                detection.box,
                self.config.keypoint_confidence,
            )
            visibility, leg_visibility = _pose_visibility(
                canonical,
                self.config.keypoint_confidence,
            )
            # Keep one time slot per observed track frame.  Invalid pose rows
            # remain masked zeros and are gap-filled by the gait preprocessor;
            # deleting them would compress the gait phase.
            state.poses.append(
                canonical.copy()
                if canonical is not None
                else np.zeros((17, 3), dtype=np.float32)
            )
            state.timestamps.append(float(timestamp))
            frame_view_angle = _view_angle(canonical)
            ground_point, ground_anchor_source = _ground_point(
                canonical,
                detection.box,
                self.config.ground_homography,
            )
            state.views.append(frame_view_angle)
            state.ground_points.append(ground_point)
            state.camera_view_timeline.append(
                _CameraViewSample(
                    frame_index=self.frame_index,
                    timestamp=float(timestamp),
                    view_angle=frame_view_angle,
                    camera_zone=_camera_zone(detection.box, frame_bgr.shape),
                    world_position=ground_point,
                    ground_anchor_source=ground_anchor_source,
                )
            )
            state.valid_pose_frames = sum(
                bool(np.any(item[:, 2] > 0.0)) for item in state.poses
            )
            state.valid_leg_frames = sum(
                bool(np.mean(item[[11, 12, 13, 14, 15, 16], 2] > 0.0) >= 0.45)
                for item in state.poses
            )
            walking_ratio, gait_cycles = _walking_metrics(
                state.poses,
                detection.box,
                state.timestamps,
            )
            heuristic_prediction = self.action_classifier.predict(
                tuple(state.poses),
                state.timestamps,
                walking_ratio=walking_ratio,
                gait_cycles=gait_cycles,
            )
            semantic_prediction = self._posec3d_prediction(
                state,
                frame_bgr.shape[:2],
            )
            action_prediction = self._fuse_action_predictions(
                heuristic_prediction,
                semantic_prediction,
                probability_threshold=self.config.posec3d_probability_threshold,
                margin_threshold=self.config.posec3d_margin_threshold,
                model_version=(
                    self.posec3d.model_version
                    if self.posec3d is not None
                    else heuristic_prediction.model_version
                ),
            )
            action_stable = self._advance_action_state(state, action_prediction)
            walk_authorized = bool(
                action_stable
                and state.action_type == ActionType.WALK
                and self._action_router.allows_walk(action_prediction)
            )
            if walk_authorized:
                gait_gate_reason = "walk_authorized"
            elif not action_stable:
                gait_gate_reason = "action_transition_pending"
            elif state.action_type != ActionType.WALK:
                gait_gate_reason = "stable_action_not_walk"
            elif not self._action_router.allows_walk(action_prediction):
                gait_gate_reason = self._action_router.quarantine_reason(
                    action_prediction
                )
            if not walk_authorized:
                # A candidate, degraded, partial or unknown action cannot
                # keep an older gait vector alive as if the current window
                # were still a complete WALK segment.
                state.gait = None
                state.last_gait_frame = None
            gait_due = (
                state.gait is None
                or state.last_gait_frame is None
                or state.frame_count - state.last_gait_frame
                >= self.config.gait_inference_stride
            )
            first_gait = state.gait is None or state.last_gait_frame is None
            gait_overdue = (
                not first_gait
                and state.frame_count - state.last_gait_frame
                >= self.config.gait_inference_stride
                + max(1, self.config.detector_inference_stride)
            )
            if (
                state.valid_pose_frames >= self.config.minimum_pose_frames
                # A complete cycle is required before GPGait++ sees the
                # current action segment; half-cycle fixed windows are too
                # easy to contaminate during entrance/exit transitions.
                and gait_cycles >= 1.0
                and walk_authorized
                and gait_due
                and (
                    first_gait
                    or not self._detector_refreshed_this_frame
                    or gait_overdue
                )
            ):
                gait_indexes.append(index)
            frame_metrics.append(
                (
                    canonical,
                    visibility,
                    leg_visibility,
                    walking_ratio,
                    gait_cycles,
                    action_prediction,
                    semantic_prediction,
                    walk_authorized,
                    gait_gate_reason,
                )
            )

        if gait_indexes:
            gait_values = self.gait.encode_batch(
                [
                    tuple(self.states[detections[index].track_id].poses)
                    for index in gait_indexes
                ]
            )
            for destination, value in zip(gait_indexes, gait_values):
                if value is None:
                    continue
                state = self.states[detections[destination].track_id]
                state.gait = value
                state.last_gait_frame = state.frame_count

        # 周期性外观刷新放到没有执行检测和步态编码的帧。首次外观仍立即提取，
        # 避免新 Track 因错峰策略延迟进入外观分支。
        appearance_indexes: list[int] = []
        gait_ran = bool(gait_indexes)
        for index, detection in enumerate(detections):
            state = self.states[detection.track_id]
            first_appearance = state.last_appearance_frame is None
            appearance_due = (
                first_appearance
                or state.frame_count - state.last_appearance_frame
                >= self.config.appearance_stride
            )
            appearance_overdue = (
                not first_appearance
                and state.frame_count - state.last_appearance_frame
                >= self.config.appearance_stride
                + max(2, self.config.detector_inference_stride)
            )
            periodic_slot_available = (
                not self._detector_refreshed_this_frame and not gait_ran
            )
            if appearance_due and (
                first_appearance or periodic_slot_available or appearance_overdue
            ):
                appearance_indexes.append(index)
        if appearance_indexes:
            appearance_boxes = [boxes[index] for index in appearance_indexes]
            # All learned branches see the same low-light transform.  Feeding
            # OSNet the raw frame while pose/detection see CLAHE creates a
            # branch-specific quality shift that the shared thresholds cannot
            # represent.
            appearance_values = self.appearance.extract(inference_frame, appearance_boxes)
            for destination, value in zip(appearance_indexes, appearance_values):
                state = self.states[detections[destination].track_id]
                state.last_appearance_frame = state.frame_count
                if value is not None:
                    previous = state.appearance
                    similarity: float | None = None
                    if previous is not None:
                        previous_vector = np.asarray(previous, dtype=np.float32).reshape(-1)
                        current_vector = np.asarray(value, dtype=np.float32).reshape(-1)
                        previous_norm = float(np.linalg.norm(previous_vector))
                        current_norm = float(np.linalg.norm(current_vector))
                        if (
                            previous_vector.shape == current_vector.shape
                            and previous_norm > 1e-8
                            and current_norm > 1e-8
                        ):
                            similarity = float(
                                np.dot(previous_vector, current_vector)
                                / (previous_norm * current_norm)
                            )
                    state.appearance_similarity_to_previous = similarity
                    state.appearance_change_suspected = bool(
                        similarity is not None
                        and similarity < self.config.appearance_change_suspect_similarity
                        and not {
                            "track_gap",
                            "track_id_switch",
                        }.intersection(state.segment_boundary_reasons)
                    )
                    state.appearance = value
                    state.appearance_generation += 1

        output: list[VisionTrack] = []
        active_ids: set[int] = set()
        frame_height, frame_width = frame_bgr.shape[:2]
        for index, detection in enumerate(detections):
            state = self.states[detection.track_id]
            (
                canonical,
                visibility,
                leg_visibility,
                walking_ratio,
                gait_cycles,
                action_prediction,
                semantic_prediction,
                walk_authorized,
                gait_gate_reason,
            ) = (
                frame_metrics[index]
            )
            x1, y1, x2, y2 = detection.box
            crop = frame_bgr[y1:y2, x1:x2]
            sharpness = 0.0
            if crop.size:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sharpness = float(np.clip(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 0, 1))
            box_geometry_valid = _box_valid(detection.box)
            truncated = x1 <= 1 or y1 <= 1 or x2 >= frame_width - 1 or y2 >= frame_height - 1
            box_valid = box_geometry_valid and not truncated
            reasons: list[str] = []
            if truncated:
                reasons.append("box_truncated")
            if leg_visibility < 0.45:
                reasons.append("low_leg_visibility")
            if state.valid_pose_frames < self.config.minimum_pose_frames:
                reasons.append("gait_sequence_immature")
            if state.detector_gap_frames:
                reasons.append("detector_gap")
            if state.track_gap_frames:
                reasons.append("track_gap")
            if bool(getattr(self.detector, "overflowed", False)):
                reasons.append("maximum_people_reached")
            sequence_maturity = min(
                1.0,
                len(state.poses) / max(self.config.minimum_pose_frames, 1),
            )
            pose_validity = min(
                1.0,
                state.valid_pose_frames / max(self.config.minimum_pose_frames, 1),
            )
            gait_quality = float(
                np.clip(
                    leg_visibility
                    * np.sqrt(pose_validity)
                    * np.sqrt(sequence_maturity)
                    * (0.25 + 0.75 * walking_ratio),
                    0.0,
                    1.0,
                )
            )
            # 之前成熟的时序状态可能在几个坏帧中继续保留。当前人物框异常或
            # 被图像边界裁剪时，不要让该缓存嵌入成为强证据。
            if not box_valid:
                gait_quality = 0.0
            lower_body_mean_score = _lower_body_mean_score(state.poses)
            longest_leg_gap_frames = _longest_leg_gap_frames(state.poses)
            track_gap_ratio = float(
                np.clip(
                    (state.detector_gap_frames + state.track_gap_frames)
                    / max(len(state.poses), 1),
                    0.0,
                    1.0,
                )
            )
            pose_jitter = _pose_jitter(state.poses)
            stable_view_angle, view_angle_confidence = _view_consensus(
                state.camera_view_timeline
            )
            quality = TrackQuality(
                detection_confidence=detection.confidence,
                box_height=float(y2 - y1),
                box_valid=box_valid,
                sharpness=sharpness,
                occlusion=float(overlaps[index]),
                keypoint_visibility=visibility,
                leg_visibility=leg_visibility,
                gait_branch_quality=gait_quality,
                contour_area=float((x2 - x1) * (y2 - y1)),
                contour_jitter=_box_jitter(state.boxes),
                id_switches=state.id_switches,
                frame_count=len(state.poses),
                valid_pose_frames=state.valid_pose_frames,
                valid_leg_frames=state.valid_leg_frames,
                detector_gap_frames=state.detector_gap_frames,
                track_gap_frames=state.track_gap_frames,
                timestamp_span_seconds=(
                    float(state.timestamps[-1] - state.timestamps[0])
                    if len(state.timestamps) >= 2
                    else 0.0
                ),
                lower_body_mean_score=lower_body_mean_score,
                longest_leg_gap_frames=longest_leg_gap_frames,
                track_gap_ratio=track_gap_ratio,
                pose_jitter=pose_jitter,
                gait_cycles=gait_cycles,
                walking_ratio=walking_ratio,
                view_angle=stable_view_angle,
                reasons=tuple(reasons),
            )
            window_frame_count = len(state.poses)
            world_trajectory = _world_trajectory(
                state.ground_points,
                state.timestamps,
                window_seconds=self.config.ground_trajectory_window_seconds,
                maximum_speed_mps=self.config.ground_max_speed_mps,
            )
            segment = CleanSubTracklet(
                source_track_id=detection.track_id,
                segment_id=state.segment_id,
                frame_start=max(
                    state.segment_start_frame,
                    self.frame_index - max(window_frame_count, 1) + 1,
                ),
                frame_end=self.frame_index,
                frame_count=window_frame_count,
                valid_pose_frames=state.valid_pose_frames,
                valid_leg_frames=state.valid_leg_frames,
                quality_score=gait_quality,
                boundary_reasons=state.segment_boundary_reasons,
                quality_reasons=tuple(
                    reason
                    for reason in reasons
                    if reason not in {"track_gap", "track_id_switch"}
                ),
                view_coverage=tuple(
                    sorted({view for view in state.views if view is not None})
                ),
            )
            track = VisionTrack(
                track_id=detection.track_id,
                box=detection.box,
                detection_confidence=detection.confidence,
                features=FeatureBundle(
                    appearance=state.appearance,
                    gait=state.gait,
                ),
                quality=quality,
                clean_subtracklet=segment,
                pose_keypoints=(canonical.copy() if canonical is not None else None),
                metadata={
                    "action_type": action_prediction.action_type.value,
                    "action_confidence": action_prediction.confidence,
                    "action_quality": action_prediction.quality.value,
                    "action_completion": action_prediction.completion,
                    "action_source": action_prediction.source,
                    "action_model_version": action_prediction.model_version,
                    "action_semantic_label": (
                        semantic_prediction.top_label
                        if semantic_prediction is not None
                        else None
                    ),
                    "action_semantic_index": (
                        semantic_prediction.top_index
                        if semantic_prediction is not None
                        else None
                    ),
                    "action_semantic_probability": (
                        semantic_prediction.walk_probability
                        if semantic_prediction is not None
                        else None
                    ),
                    "action_semantic_margin": (
                        semantic_prediction.margin
                        if semantic_prediction is not None
                        else None
                    ),
                    "action_semantic_reason": (
                        semantic_prediction.reason
                        if semantic_prediction is not None
                        else None
                    ),
                    "action_gait_authorized": walk_authorized,
                    "action_gait_gate_reason": gait_gate_reason,
                    "action_branch_id": (
                        f"gait_{action_prediction.action_type.value.lower()}"
                        if action_prediction.action_type != ActionType.UNKNOWN
                        else "unknown"
                    ),
                    "action_branch_generation": state.action_branch_generation,
                    "action_segment_id": state.action_segment_id,
                    "action_boundary_reason": state.action_boundary_reason,
                    "action_stable_type": state.action_type.value,
                    "action_transition_pending": state.action_candidate_type is not None,
                    "action_candidate_type": (
                        state.action_candidate_type.value
                        if state.action_candidate_type is not None
                        else None
                    ),
                    "action_candidate_frames": state.action_candidate_frames,
                    "action_unknown_frames": state.action_unknown_frames,
                    # These values are audit/conditioning metadata.  They do
                    # not change GPGait++'s image-coordinate input.
                    "camera_zone": _camera_zone(detection.box, frame_bgr.shape),
                    "frame_index": self.frame_index,
                    "frame_timestamp": float(timestamp),
                    "view_angle_raw": frame_view_angle,
                    "view_angle_confidence": view_angle_confidence,
                    "camera_view_timeline": tuple(
                        {
                            "frame_index": sample.frame_index,
                            "timestamp": sample.timestamp,
                            "view_angle": sample.view_angle,
                            "camera_zone": sample.camera_zone,
                            "world_position": sample.world_position,
                            "ground_anchor_source": sample.ground_anchor_source,
                            "missing": sample.missing,
                        }
                        for sample in state.camera_view_timeline
                    ),
                    "pose_jitter": pose_jitter,
                    "gait_lower_body_mean_score": lower_body_mean_score,
                    "gait_longest_leg_gap_frames": longest_leg_gap_frames,
                    "gait_track_gap_ratio": track_gap_ratio,
                    "world_position": world_trajectory.position,
                    "ground_speed": world_trajectory.speed,
                    "world_displacement": world_trajectory.displacement,
                    "world_direction_degrees": world_trajectory.direction_degrees,
                    "ground_trajectory_quality": world_trajectory.quality,
                    "ground_valid_samples": world_trajectory.valid_samples,
                    "ground_outlier_count": world_trajectory.outlier_count,
                    "ground_calibration_id": self.config.ground_calibration_id,
                    "ground_coordinate_frame": self.config.ground_coordinate_frame,
                    "ground_coordinate_aligned": bool(
                        self.config.ground_homography is not None
                        and self.config.ground_coordinate_frame is not None
                    ),
                    "ground_calibrated": self.config.ground_homography is not None,
                    "appearance_similarity_to_previous": (
                        state.appearance_similarity_to_previous
                    ),
                    "appearance_change_suspected": state.appearance_change_suspected,
                    "appearance_generation": state.appearance_generation,
                    "appearance_fresh": bool(
                        state.last_appearance_frame == state.frame_count
                        and index in appearance_indexes
                    ),
                    "lifetime_frame_count": state.lifetime_frame_count,
                    "lifetime_detector_gap_frames": state.lifetime_detector_gap_frames,
                    "lifetime_track_gap_frames": state.lifetime_track_gap_frames,
                    "lifetime_id_switches": state.lifetime_id_switches,
                },
            )
            output.append(track)
            self.latest[detection.track_id] = track
            active_ids.add(detection.track_id)
        self._expire_states(active_ids, timestamp=timestamp)
        self._record_frame_latency(started)
        return tuple(output)


__all__ = [
    "MODEL_DIRECTORY",
    "ProductionVisionAdapter",
    "ProductionVisionConfig",
    "ProductionVisionError",
    "production_readiness",
]
