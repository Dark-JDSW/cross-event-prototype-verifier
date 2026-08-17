"""生产视频前端：YOLO、ByteTrack、RTMPose、ReID 和步态。

整个重量级视觉栈都隐藏在既有的 ``VisionAdapter`` 接口之后。GUI 和身份策略
代码只接收 ``VisionTrack`` 值，不导入模型框架，也不需要知道时序状态如何维护。

适配器为每条轨迹维护检测、姿态、嵌入及时序质量状态。一帧图像会依次经过
检测/跟踪、批量姿态推理、外观刷新、步态窗口更新和质量计算。模型权重在首帧
到来时延迟加载，因此构造 GUI 时仍能保持响应。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from typing import Sequence

import cv2
import numpy as np

from .adapters import occlusion_scores
from .gait_graph import TemporalGaitEncoder
from .model_assets import tensor_state_fingerprint
from .osnet_ain import load_osnet_ain
from .types import FeatureBundle, TrackQuality
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
    gait_path: Path = MODEL_DIRECTORY / "gaitgraph2_grew_state.pt"
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
    gait_inference_stride: int = 3
    low_light_threshold: float = 100.0
    low_light_check_interval: int = 12
    state_retention_frames: int = 90
    state_retention_seconds: float = 3.0
    maximum_track_gap_seconds: float = 1.5
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
        if self.state_retention_seconds <= 0 or self.maximum_track_gap_seconds <= 0:
            raise ValueError("track retention/gap seconds must be positive")
        if self.maximum_track_gap_seconds >= self.state_retention_seconds:
            raise ValueError(
                "maximum_track_gap_seconds must be shorter than state_retention_seconds"
            )
        if self.require_cuda and self.torch_device.lower().startswith("cpu"):
            raise ValueError("生产视觉链路 require_cuda=True 时不能把 device 设为 CPU")

    @property
    def required_files(self) -> tuple[Path, ...]:
        """返回生产启动前所需的全部本地文件。"""
        return (
            self.detector_path,
            self.tracker_path,
            self.pose_path,
            self.appearance_path,
            self.gait_path,
        )

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
    def __init__(self, threshold: float, interval: int) -> None:
        """配置亮度采样间隔和可复用的 CLAHE 算子。"""
        self.threshold = float(threshold)
        self.interval = max(1, int(interval))
        self.frame_index = 0
        self.enabled = False
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def reset(self) -> None:
        """清除已采样亮度，使下一帧重新评估光照。"""
        self.frame_index = 0
        self.enabled = False

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """根据采样间隔返回原始帧或 CLAHE 增强帧。"""
        if self.frame_index % self.interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.enabled = float(gray.mean()) < self.threshold
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


class _RtmposeEstimator:
    """使用批量 CUDA ONNX 推理的自顶向下 RTMPose-s 适配器。"""
    _MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
    _STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
    _INPUT_WIDTH = 192
    _INPUT_HEIGHT = 256
    _PADDING = 1.25
    _SIMCC_SPLIT = 2.0
    _BUCKETS = (1, 2, 4, 8, 16, 32)

    def __init__(self, path: Path, *, require_cuda: bool = True) -> None:
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
            providers = [("CUDAExecutionProvider", {"device_id": 0})]
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
        for row, destination_index in enumerate(indexes):
            vector = features[row].astype(np.float32)
            norm = float(np.linalg.norm(vector))
            if norm > 1e-8:
                output[destination_index] = vector / norm
        return output


@dataclass
class _TrackState:
    """由一个 ByteTrack ID 持有的短期时序缓存。"""
    frame_count: int = 0
    last_seen_frame: int = 0
    poses: deque[np.ndarray] = field(default_factory=deque)
    timestamps: deque[float] = field(default_factory=deque)
    boxes: deque[Box] = field(default_factory=lambda: deque(maxlen=20))
    valid_pose_frames: int = 0
    valid_leg_frames: int = 0
    detector_gap_frames: int = 0
    track_gap_frames: int = 0
    last_seen_timestamp: float | None = None
    last_valid_timestamp: float | None = None
    appearance: np.ndarray | None = None
    last_appearance_frame: int | None = None
    gait: np.ndarray | None = None
    last_gait_frame: int | None = None
    id_switches: int = 0


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

    GaitGraph2 的 GREW 检查点是在原图坐标上训练的。这里如果再按检测框
    归一化，会把同一个人的全局位置和尺度改成一个与训练数据不兼容的分布，
    最终表现为不同人物的 gait embedding 几乎完全重合。``box`` 保留在签名
    中是为了让调用方继续显式传递检测上下文；它不应该参与模型输入变换。
    """
    if keypoints is None or np.asarray(keypoints).shape != (17, 3):
        return None
    points = np.asarray(keypoints, dtype=np.float32).copy()
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] >= confidence_floor)
    points[~valid] = 0.0
    if not bool(valid.any()):
        return None
    # Do not clip valid coordinates to the box. GREW stores raw image-space
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
    # v2 records the checkpoint-compatible GREW input contract: raw full-frame
    # RTMPose coordinates and OpenGait's partition-normalized COCO graph.  A
    # database produced by the old bbox-normalized path must not be reused as
    # if its gait prototypes were comparable to this stream.
    model_version = "yolo11l-bytetrack+rtmpose-s+osnet-ain+gaitgraph2-grew-opengait-input-v2"
    feature_schema = (
        "osnet-ain-512-imagenet-v1|"
        "gaitgraph2-grew-coco17-raw-full-frame-t60-tta3-v2"
    )
    preprocess_version = "rtmpose-s-256x192-simcc2|opengait-coco17-raw-t60-tta3-v2"
    joint_format = "COCO-17"
    tta_mode = "original+temporal-reverse+horizontal-flip"
    coordinate_contract = "raw-full-frame-image-coordinates"
    embedding_dimensions = {"appearance": 512, "gait": 384}

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
        """返回当前 GaitGraph2 窗口长度。"""
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

    def __init__(self, config: ProductionVisionConfig | None = None) -> None:
        """校验就绪状态，并准备延迟加载的模型槽位。"""
        self.config = config or ProductionVisionConfig()
        ready, issues = production_readiness(self.config, verify_hashes=True)
        if not ready:
            raise ProductionVisionError("；".join(issues))
        self.detector: _YoloByteTracker | None = None
        self.pose: _RtmposeEstimator | None = None
        self.appearance: _OsnetAppearanceExtractor | None = None
        self.gait: TemporalGaitEncoder | None = None
        self.enhancer = _LowLightEnhancer(
            self.config.low_light_threshold,
            self.config.low_light_check_interval,
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

    def _load(self) -> None:
        """只加载一次检测、姿态、外观和步态模型。"""
        if self.detector is not None:
            return
        try:
            self.detector = _YoloByteTracker(self.config)
            self.pose = _RtmposeEstimator(
                self.config.pose_path,
                require_cuda=self.config.require_cuda,
            )
            self.appearance = _OsnetAppearanceExtractor(
                self.config.appearance_path,
                self.config.torch_device,
            )
            self.gait = TemporalGaitEncoder(
                self.config.gait_path,
                device=self.config.torch_device,
                sequence_length=self.config.gait_sequence_length,
                use_tta=True,
            )
            gait_dimension = getattr(self.gait, "output_dimension", None)
            if gait_dimension is not None and gait_dimension != 384:
                raise ProductionVisionError(
                    "GaitGraph2 production contract requires a 384-dimensional TTA embedding"
                )
            appearance_dimension = getattr(self.appearance, "output_dimension", None)
            if appearance_dimension is not None and appearance_dimension != 512:
                raise ProductionVisionError(
                    "OSNet-AIN production contract requires a 512-dimensional embedding"
                )
        except Exception:
            self.detector = None
            self.pose = None
            self.appearance = None
            self.gait = None
            raise

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
        return f"YOLO/ByteTrack + RTMPose({pose_backend}{latency}) + OSNet-AIN + GaitGraph2"

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
                state.poses.clear()
                state.timestamps.clear()
                state.valid_pose_frames = 0
                state.valid_leg_frames = 0
                state.gait = None
                state.last_gait_frame = None

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
        # ByteTrack is stateful, but it only sees detector calls.  On skipped
        # frames predict a short constant-velocity box from the recent track
        # history instead of freezing the crop; this keeps pose/appearance
        # aligned with motion while the next persisted tracker update remains
        # authoritative.
        delta = max(1, self.frame_index - (self._last_detector_frame or self.frame_index))
        height, width = frame_bgr.shape[:2]
        predicted: list[_Detection] = []
        for detection in self._cached_detections:
            state = self.states.get(detection.track_id)
            box = detection.box
            if state is not None and len(state.boxes) >= 2:
                previous = np.asarray(state.boxes[-2], dtype=np.float32)
                current = np.asarray(state.boxes[-1], dtype=np.float32)
                velocity = current - previous
                raw = current + velocity * min(delta, 2)
                x1, y1, x2, y2 = (int(round(float(value))) for value in raw)
                x1 = max(0, min(x1, max(width - 1, 0)))
                y1 = max(0, min(y1, max(height - 1, 0)))
                x2 = max(x1 + 1, min(x2, width))
                y2 = max(y1 + 1, min(y2, height))
                box = (x1, y1, x2, y2)
                state.detector_gap_frames += 1
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
    def _append_missing_slot(state: _TrackState, timestamp: float) -> None:
        """在短暂漏检期间保留一个带掩码的时间槽。"""

        if state.last_seen_timestamp is not None and timestamp <= state.last_seen_timestamp:
            return
        state.poses.append(np.zeros((17, 3), dtype=np.float32))
        state.timestamps.append(float(timestamp))
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
        self._current_timestamp = time.time() if timestamp is None else float(timestamp)
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
        if frame_bgr is None or frame_bgr.size == 0:
            return ()
        started = time.perf_counter()
        self._load()
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
                self._append_missing_slot(state, timestamp)
        if not detections:
            self._expire_states(set(), timestamp=timestamp)
            self._record_frame_latency(started)
            return ()

        boxes = [item.box for item in detections]
        poses = self.pose.extract(inference_frame, boxes)
        overlaps = occlusion_scores(np.asarray(boxes, dtype=np.float32))

        frame_metrics: list[
            tuple[np.ndarray | None, float, float, float, float]
        ] = []
        gait_indexes: list[int] = []
        for index, detection in enumerate(detections):
            state = self.states.setdefault(
                detection.track_id,
                _TrackState(
                    poses=deque(maxlen=self.config.gait_sequence_length),
                    timestamps=deque(maxlen=self.config.gait_sequence_length),
                ),
            )
            if (
                state.last_seen_timestamp is not None
                and timestamp - state.last_seen_timestamp
                > self.config.maximum_track_gap_seconds
            ):
                state.track_gap_frames += 1
                state.poses.clear()
                state.timestamps.clear()
                state.gait = None
                state.last_gait_frame = None
                state.valid_pose_frames = 0
                state.valid_leg_frames = 0
            if state.boxes and _abrupt_track_jump(state.boxes[-1], detection.box):
                state.id_switches += 1
                state.poses.clear()
                state.timestamps.clear()
                state.gait = None
                state.last_gait_frame = None
                state.appearance = None
                state.last_appearance_frame = None
            state.frame_count += 1
            state.last_seen_frame = self.frame_index
            state.last_seen_timestamp = timestamp
            state.boxes.append(detection.box)
            canonical = _canonical_pose(
                poses[index],
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
            state.valid_pose_frames = sum(
                bool(np.any(item[:, 2] > 0.0)) for item in state.poses
            )
            state.valid_leg_frames = sum(
                bool(np.mean(item[[11, 12, 13, 14, 15, 16], 2] > 0.0) >= 0.45)
                for item in state.poses
            )
            if canonical is not None:
                state.last_valid_timestamp = timestamp
            walking_ratio, gait_cycles = _walking_metrics(
                state.poses,
                detection.box,
                state.timestamps,
            )
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
                and gait_cycles >= 0.5
                and gait_due
                and (
                    first_gait
                    or not self._detector_refreshed_this_frame
                    or gait_overdue
                )
            ):
                gait_indexes.append(index)
            frame_metrics.append(
                (canonical, visibility, leg_visibility, walking_ratio, gait_cycles)
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
                    state.appearance = value

        output: list[VisionTrack] = []
        active_ids: set[int] = set()
        frame_height, frame_width = frame_bgr.shape[:2]
        for index, detection in enumerate(detections):
            state = self.states[detection.track_id]
            canonical, visibility, leg_visibility, walking_ratio, gait_cycles = (
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
                frame_count=state.frame_count,
                valid_pose_frames=state.valid_pose_frames,
                valid_leg_frames=state.valid_leg_frames,
                detector_gap_frames=state.detector_gap_frames,
                track_gap_frames=state.track_gap_frames,
                timestamp_span_seconds=(
                    float(state.timestamps[-1] - state.timestamps[0])
                    if len(state.timestamps) >= 2
                    else 0.0
                ),
                gait_cycles=gait_cycles,
                walking_ratio=walking_ratio,
                view_angle=_view_angle(canonical),
                reasons=tuple(reasons),
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
