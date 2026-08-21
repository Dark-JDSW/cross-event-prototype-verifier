"""桌面应用使用的独立 OpenCV 视觉适配器。

验证器核心有意不负责模型推理。本模块是 GUI 的默认适配器：它使用 OpenCV
内置的 HOG 人体检测器以及轻量轮廓/运动描述符，使应用无需相邻的
``videotracker`` 项目或已下载的模型权重也能运行。

若要获得生产精度，可在此接口替换 :class:`OpenCvDemoAdapter`，接入检测器、
外观 ReID 和 OpenGait 特征；GUI 的其余部分及验证策略无需改变。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot
from typing import Protocol, Sequence

import cv2
import numpy as np

from .adapters import occlusion_scores
from ..types import FeatureBundle, TrackQuality


Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class VisionTrack:
    """一条跟踪检测及其模型特征和质量证据。

    这是每个视觉适配器的主要输出。管线无需知道它来自 HOG、YOLO/ByteTrack
    还是未来的检测器即可消费该值，从而保持视觉接口可替换。
    """

    track_id: int
    box: Box
    detection_confidence: float
    features: FeatureBundle
    quality: TrackQuality


class VisionAdapter(Protocol):
    """帧处理与具体视觉栈之间的小型接口。"""

    def process(self, frame_bgr: np.ndarray) -> tuple[VisionTrack, ...]:
        """返回一帧 BGR 图像坐标中的跟踪人物。"""

    def reset(self) -> None:
        """当输入源会话变化时清除短期跟踪状态。"""


@dataclass
class _TrackState:
    """诊断轨迹缓存，包含框、轮廓历史和运动能量。"""

    track_id: int
    box: Box
    missed: int = 0
    frame_count: int = 0
    contours: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=48))
    motion_energy: deque[float] = field(default_factory=lambda: deque(maxlen=48))


class _CentroidTracker:
    """用于诊断模式的无依赖最近质心跟踪器。"""

    def __init__(self, *, max_missing: int = 12) -> None:
        """创建具有有限丢帧宽限期的最近质心跟踪器。"""
        self.max_missing = max_missing
        self.next_id = 1
        self.states: dict[int, _TrackState] = {}

    @staticmethod
    def _center(box: Box) -> tuple[float, float]:
        """返回图像空间框的几何中心。"""
        return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)

    @staticmethod
    def _distance(left: Box, right: Box) -> float:
        """返回两个框中心之间的欧氏距离。"""
        lx, ly = _CentroidTracker._center(left)
        rx, ry = _CentroidTracker._center(right)
        return hypot(lx - rx, ly - ry)

    @staticmethod
    def _gate(box: Box) -> float:
        """返回用于匹配轨迹且考虑尺度的最大位移。"""
        width = max(1, box[2] - box[0])
        height = max(1, box[3] - box[1])
        return max(70.0, 1.25 * hypot(width, height))

    def update(self, boxes: Sequence[Box]) -> list[tuple[int, Box]]:
        """将当前框匹配到轨迹，并为未匹配框创建 ID。"""
        boxes = list(boxes)
        assigned: dict[int, int] = {}
        used_tracks: set[int] = set()
        used_boxes: set[int] = set()

        pairs = sorted(
            (
                self._distance(state.box, box),
                track_id,
                index,
            )
            for track_id, state in self.states.items()
            for index, box in enumerate(boxes)
        )
        for distance, track_id, index in pairs:
            if track_id in used_tracks or index in used_boxes:
                continue
            if distance > self._gate(self.states[track_id].box):
                continue
            assigned[index] = track_id
            used_tracks.add(track_id)
            used_boxes.add(index)

        for index, box in enumerate(boxes):
            if index not in assigned:
                track_id = self.next_id
                self.next_id += 1
                self.states[track_id] = _TrackState(track_id=track_id, box=box)
                assigned[index] = track_id

        for track_id, state in list(self.states.items()):
            if track_id in used_tracks or track_id in assigned.values():
                state.missed = 0
            else:
                state.missed += 1
                if state.missed > self.max_missing:
                    del self.states[track_id]

        result: list[tuple[int, Box]] = []
        for index, box in enumerate(boxes):
            track_id = assigned[index]
            state = self.states[track_id]
            state.box = box
            result.append((track_id, box))
        return result

    def reset(self) -> None:
        """清除所有诊断轨迹，并从 1 重新开始编号。"""
        self.states.clear()
        self.next_id = 1


class OpenCvDemoAdapter:
    """供独立 GUI 使用的纯 CPU 检测/跟踪/特征适配器。

    步态分支是轮廓与运动代理，并非 OpenGait 模型。它的设计目标是在保持
    重量级模型接口可替换的同时，用于运行端到端应用流程。
    """

    supports_automatic_registration = False
    model_version = "opencv-hog-diagnostic-v1"
    backend_status = "OpenCV HOG 诊断模式（禁止自动注册）"

    def __init__(
        self,
        *,
        max_processing_dimension: int = 960,
        max_detections: int = 8,
        detection_stride: int = 1,
    ) -> None:
        """准备 HOG/前景检测、质心跟踪和缓存。"""
        self.max_processing_dimension = max(320, int(max_processing_dimension))
        self.max_detections = max(1, int(max_detections))
        self.detection_stride = max(1, int(detection_stride))
        self.hog = None
        if hasattr(cv2, "HOGDescriptor") and hasattr(
            cv2,
            "HOGDescriptor_getDefaultPeopleDetector",
        ):
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        # OpenCV 5 不再通过 Python 绑定公开旧版 HOG 人体检测器。在用户安装
        # 固定版本的 4.x wheel 或提供生产检测器适配器之前，使用无依赖的
        # 前景检测器，确保 GUI 仍可启动。
        self.foreground = cv2.createBackgroundSubtractorMOG2(
            history=240,
            varThreshold=32,
            detectShadows=True,
        )
        self.tracker = _CentroidTracker()
        self.frame_index = 0
        self.latest: dict[int, VisionTrack] = {}

    def reset(self) -> None:
        """重置检测器、跟踪器和每条轨迹的缓存描述符。"""
        self.tracker.reset()
        self.frame_index = 0
        self.latest.clear()

    @staticmethod
    def _clip_box(box: Sequence[float], width: int, height: int) -> Box | None:
        """将候选框裁剪到图像边界内，并拒绝过小区域。"""
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        if x2 - x1 < 8 or y2 - y1 < 16:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _confidence(weight: float) -> float:
        """将未校准的 HOG margin 映射为保守的质量信号。"""
        # HOG 返回的 margin 不是校准概率。将其作为保守的检测质量信号，
        # 不要把它当作身份分数。
        return float(np.clip(0.55 + 0.12 * float(weight), 0.35, 0.98))

    @staticmethod
    def _iou(left: Box, right: Box) -> float:
        """计算用于诊断 NMS 的普通交并比。"""
        x1 = max(left[0], right[0])
        y1 = max(left[1], right[1])
        x2 = min(left[2], right[2])
        y2 = min(left[3], right[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection <= 0:
            return 0.0
        left_area = (left[2] - left[0]) * (left[3] - left[1])
        right_area = (right[2] - right[0]) * (right[3] - right[1])
        return intersection / max(left_area + right_area - intersection, 1)

    def _nms(self, boxes: list[Box], scores: list[float]) -> tuple[list[Box], list[float]]:
        """抑制重叠的诊断检测结果，并限制输出数量。"""
        keep: list[int] = []
        for index in sorted(range(len(boxes)), key=lambda item: scores[item], reverse=True):
            if all(self._iou(boxes[index], boxes[other]) < 0.45 for other in keep):
                keep.append(index)
            if len(keep) >= self.max_detections:
                break
        return [boxes[index] for index in keep], [scores[index] for index in keep]

    def _detect(self, frame_bgr: np.ndarray) -> tuple[list[Box], list[float]]:
        """优先运行 HOG；不可用时使用前景回退方案。"""
        height, width = frame_bgr.shape[:2]
        scale = min(1.0, self.max_processing_dimension / max(height, width))
        if scale < 0.999:
            small = cv2.resize(
                frame_bgr,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = frame_bgr
            scale = 1.0
        if self.hog is None:
            return self._detect_foreground(small, scale, width, height)
        try:
            locations, weights = self.hog.detectMultiScale(
                small,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05,
            )
        except cv2.error:
            return [], []
        raw_weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        boxes: list[Box] = []
        scores: list[float] = []
        for index, location in enumerate(locations):
            x, y, box_width, box_height = location
            box = self._clip_box(
                (x / scale, y / scale, (x + box_width) / scale, (y + box_height) / scale),
                width,
                height,
            )
            if box is None:
                continue
            weight = float(raw_weights[index]) if index < len(raw_weights) else 0.0
            boxes.append(box)
            scores.append(self._confidence(weight))
        return self._nms(boxes, scores)

    def _detect_foreground(
        self,
        frame_bgr: np.ndarray,
        scale: float,
        original_width: int,
        original_height: int,
    ) -> tuple[list[Box], list[float]]:
        """供不含旧版 HOG API 的 OpenCV 构建使用的回退检测器。"""

        mask = self.foreground.apply(frame_bgr)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = max(1, frame_bgr.shape[0] * frame_bgr.shape[1])
        min_area = max(400.0, frame_area * 0.004)
        boxes: list[Box] = []
        scores: list[float] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > frame_area * 0.85:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_height < 32 or box_height / max(box_width, 1) < 0.45:
                continue
            box = self._clip_box(
                (
                    x / scale,
                    y / scale,
                    (x + box_width) / scale,
                    (y + box_height) / scale,
                ),
                original_width,
                original_height,
            )
            if box is not None:
                boxes.append(box)
                scores.append(float(np.clip(0.55 + area / frame_area, 0.55, 0.85)))
        return self._nms(boxes, scores)

    @staticmethod
    def _appearance_descriptor(crop: np.ndarray) -> np.ndarray | None:
        """构造紧凑的 HSV 加灰度直方图诊断描述符。"""
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        histogram = cv2.normalize(histogram, histogram).reshape(-1)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray_hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        gray_hist = cv2.normalize(gray_hist, gray_hist).reshape(-1)
        descriptor = np.concatenate((histogram, gray_hist)).astype(np.float32)
        norm = float(np.linalg.norm(descriptor))
        return descriptor / norm if norm > 1e-8 else None

    @staticmethod
    def _contour_descriptor(crop: np.ndarray) -> np.ndarray | None:
        """构造用于运动历史的归一化低分辨率裁剪描述符。"""
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (16, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        low, high = float(gray.min()), float(gray.max())
        if high - low > 1e-5:
            gray = (gray - low) / (high - low)
        else:
            gray.fill(0.0)
        descriptor = gray.reshape(-1)
        norm = float(np.linalg.norm(descriptor))
        return descriptor / norm if norm > 1e-8 else None

    @staticmethod
    def _gait_descriptor(state: _TrackState) -> np.ndarray | None:
        """汇总轮廓均值、时间扩散和帧间变化。"""
        if not state.contours:
            return None
        contours = np.stack(tuple(state.contours), axis=0)
        mean = contours.mean(axis=0)
        spread = contours.std(axis=0)
        if len(contours) > 1:
            delta = np.abs(np.diff(contours, axis=0)).mean(axis=0)
        else:
            delta = np.zeros_like(mean)
        # 时间扩散和帧间差分可防止静止的衣物区域伪装成熟步态序列。
        vector = np.concatenate((mean, spread, delta))
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-8 else None

    @staticmethod
    def _motion_quality(state: _TrackState) -> tuple[float, float]:
        """根据轮廓能量估计运动质量和粗略周期数。"""
        energies = list(state.motion_energy)
        if len(energies) < 2:
            return 0.0, 0.0
        recent = float(np.mean(energies[-min(12, len(energies)) :]))
        quality = float(np.clip(recent / 0.08, 0.0, 1.0))
        peaks = 0
        for index in range(1, len(energies) - 1):
            if (
                energies[index] > energies[index - 1]
                and energies[index] >= energies[index + 1]
                and energies[index] > max(0.01, float(np.mean(energies)) * 1.15)
            ):
                peaks += 1
        cycles = float(np.clip(peaks / 2.0, 0.0, 3.0))
        return quality, cycles

    def process(self, frame_bgr: np.ndarray) -> tuple[VisionTrack, ...]:
        """检测、跟踪、描述并评估一帧 BGR 图像的质量。

        HOG 可用时优先使用，否则使用前景掩码。两条路径都明确只用于诊断，
        并将 ``supports_automatic_registration`` 设置为 ``False``。
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return ()
        self.frame_index += 1
        if self.frame_index % self.detection_stride == 0 or not self.tracker.states:
            boxes, scores = self._detect(frame_bgr)
        else:
            boxes = [state.box for state in self.tracker.states.values() if state.missed == 0]
            scores = [0.65] * len(boxes)
        tracked = self.tracker.update(boxes)
        if not tracked:
            self.latest.clear()
            return ()

        height, width = frame_bgr.shape[:2]
        overlap = occlusion_scores(np.asarray([box for _, box in tracked], dtype=np.float32))
        current_ids: set[int] = set()
        output: list[VisionTrack] = []
        score_by_box = {box: scores[index] for index, box in enumerate(boxes)}
        for index, (track_id, box) in enumerate(tracked):
            state = self.tracker.states[track_id]
            x1, y1, x2, y2 = box
            crop = frame_bgr[y1:y2, x1:x2]
            contour = self._contour_descriptor(crop)
            if contour is not None:
                if state.contours:
                    state.motion_energy.append(float(np.abs(contour - state.contours[-1]).mean()))
                state.contours.append(contour)
            state.frame_count += 1
            appearance = self._appearance_descriptor(crop)
            gait = self._gait_descriptor(state)
            motion_quality, gait_cycles = self._motion_quality(state)
            gait_quality = float(
                np.clip(
                    min(1.0, state.frame_count / 16.0)
                    * (0.35 + 0.65 * motion_quality),
                    0.0,
                    1.0,
                )
            )
            detection_confidence = float(score_by_box.get(box, 0.65))
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else None
            sharpness = 0.0
            if gray_crop is not None and gray_crop.size:
                variance = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
                sharpness = float(np.clip(variance / 500.0, 0.0, 1.0))
            quality = TrackQuality(
                detection_confidence=detection_confidence,
                box_height=float(y2 - y1),
                box_valid=True,
                sharpness=max(sharpness, 0.35),
                occlusion=float(overlap[index]) if index < len(overlap) else 0.0,
                contour_area=float((x2 - x1) * (y2 - y1)),
                frame_count=state.frame_count,
                gait_cycles=gait_cycles,
                walking_ratio=motion_quality,
                gait_branch_quality=gait_quality,
            )
            item = VisionTrack(
                track_id=track_id,
                box=box,
                detection_confidence=detection_confidence,
                features=FeatureBundle(appearance=appearance, gait=gait),
                quality=quality,
            )
            output.append(item)
            self.latest[track_id] = item
            current_ids.add(track_id)
        for track_id in tuple(self.latest):
            if track_id not in current_ids and track_id not in self.tracker.states:
                del self.latest[track_id]
        return tuple(output)


__all__ = ["Box", "OpenCvDemoAdapter", "VisionAdapter", "VisionTrack"]
