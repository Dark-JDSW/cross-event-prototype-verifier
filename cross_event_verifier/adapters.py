"""模型/跟踪器适配器和转换辅助函数。

本文件包含可直接从现有 ``videotracker`` 管线复用的轻量组件。重量级的
YOLO、FastReID 和 OpenGait 推理留在验证器核心之外，只需要产出一个
``FeatureBundle``。
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

import numpy as np

from .types import FeatureBundle, Observation, TrackQuality


class FeatureExtractor(Protocol):
    """由 ReID/步态提取器或测试替身实现的最小接口。"""

    def extract(self, frame: object, box: Sequence[float], **kwargs: object) -> FeatureBundle:
        """返回一个检测框对应的归一化外观/步态特征。"""
        ...


def occlusion_scores(boxes: Sequence[Sequence[float]]) -> np.ndarray:
    """计算其他检测框对每个框的覆盖率，而不是对称的 IoU。

    当一个大人物框包含一个小人物框时，小人物的证据应比大人物受到更大
    惩罚。用其他框的覆盖率可以比普通的两两 IoU 更好地表达这种不对称性。
    """

    values = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if len(values) <= 1:
        return np.zeros(len(values), dtype=np.float32)
    x1 = np.maximum(values[:, None, 0], values[None, :, 0])
    y1 = np.maximum(values[:, None, 1], values[None, :, 1])
    x2 = np.minimum(values[:, None, 2], values[None, :, 2])
    y2 = np.minimum(values[:, None, 3], values[None, :, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    np.fill_diagonal(intersection, 0.0)
    area = np.clip(values[:, 2] - values[:, 0], 0, None) * np.clip(
        values[:, 3] - values[:, 1], 0, None
    )
    return np.clip(intersection / (area[:, None] + 1e-8), 0.0, 1.0).max(axis=1)


def pose_feature(
    box: Sequence[float],
    keypoints: Sequence[Sequence[float]] | np.ndarray | None,
    *,
    keypoint_confidence: float = 0.35,
) -> np.ndarray | None:
    """创建独立的 41 维归一化姿态描述符。

    该描述符是适配器的回退方案，不是 OpenGait 时序嵌入的替代品。它保留
    归一化坐标、四个姿态角度和三个身体比例值，使现有 RTMPose 输出可以在
    分阶段迁移期间送入本项目的步态分支。
    """

    if keypoints is None:
        return None
    points = np.asarray(keypoints, dtype=np.float32)
    if points.shape != (17, 3):
        return None
    x1, y1, x2, y2 = map(float, box)
    width, height = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    coordinates: list[float] = []
    for x, y, confidence in points:
        if confidence < keypoint_confidence:
            coordinates.extend([0.0, 0.0])
        else:
            coordinates.extend([(x - x1) / width, (y - y1) / height])

    def angle(first: np.ndarray, second: np.ndarray) -> float:
        """把两个可见关键点之间的方向编码成相位值。"""
        if first[2] <= 0.30 or second[2] <= 0.30:
            return 0.0
        return float(np.arctan2(second[1] - first[1], second[0] - first[0]))

    angles = [
        angle(points[5], points[6]),
        angle(points[11], points[12]),
        angle(points[5], points[11]),
        angle(points[6], points[12]),
    ]
    angles = [value / (2.0 * np.pi) + 0.5 for value in angles]
    shoulder_left, shoulder_right = points[5], points[6]
    hip_left, hip_right = points[11], points[12]
    ankle_left, ankle_right = points[15], points[16]
    ratios = [0.0, 0.0, 0.0]
    if all(point[2] > 0.30 for point in (shoulder_left, shoulder_right, hip_left, hip_right)):
        shoulder_width = float(np.hypot(*(shoulder_right[:2] - shoulder_left[:2])))
        body_height = float(
            np.hypot(
                (hip_left[0] + hip_right[0] - shoulder_left[0] - shoulder_right[0]) / 2.0,
                (hip_left[1] + hip_right[1] - shoulder_left[1] - shoulder_right[1]) / 2.0,
            )
        )
        leg_height = float(
            (
                np.hypot(*(ankle_left[:2] - hip_left[:2]))
                + np.hypot(*(ankle_right[:2] - hip_right[:2]))
            )
            / 2.0
        )
        ratios = [
            shoulder_width / max(body_height, 1e-6),
            leg_height / max(body_height, 1e-6),
            body_height / max(height, 1e-6),
        ]
    vector = np.asarray(coordinates + angles + ratios, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else None


def observation_from_tracker(
    *,
    event_id: str,
    camera_id: str,
    capture_session_id: str,
    track_id: str,
    timestamp: float,
    box: Sequence[float],
    detection_confidence: float,
    appearance: Sequence[float] | np.ndarray | None,
    gait: Sequence[float] | np.ndarray | None,
    frame_count: int,
    gait_cycles: float,
    walking_ratio: float,
    keypoint_visibility: float = 0.0,
    leg_visibility: float | None = None,
    gait_branch_quality: float | None = None,
    occlusion: float = 0.0,
    sharpness: float = 1.0,
    contour_area: float = 0.0,
    contour_jitter: float = 0.0,
    id_switches: int = 0,
    valid_pose_frames: int | None = None,
    valid_leg_frames: int | None = None,
    detector_gap_frames: int = 0,
    track_gap_frames: int = 0,
    timestamp_span_seconds: float = 0.0,
    view_angle: str | None = None,
    appearance_request_id: str | None = None,
    model_version: str = "unconfigured",
    feature_schema: str = "unconfigured-v1",
    artifact_sha256: str = "unverified",
    preprocess_version: str = "unversioned-v1",
    joint_format: str = "unknown",
    sequence_length: int | None = None,
    tta_mode: str = "unknown",
    coordinate_contract: str = "unknown",
    embedding_dimensions: Mapping[str, int] | None = None,
    calibration_version: str = "heuristic-default-v1",
    **metadata: object,
) -> Observation:
    """从 YOLO/ByteTrack 轨迹片段构造一个 ``Observation``。

    这里负责把检测器坐标和模型输出转换为共享领域契约。适配器会有意地在
    嵌入旁边携带来源和质量元数据，使下游策略代码不必猜测向量的来源。
    """

    x1, y1, x2, y2 = map(float, box)
    quality = TrackQuality(
        detection_confidence=detection_confidence,
        box_height=max(0.0, y2 - y1),
        box_valid=x2 > x1 and y2 > y1,
        sharpness=sharpness,
        occlusion=occlusion,
        keypoint_visibility=keypoint_visibility,
        leg_visibility=leg_visibility,
        gait_branch_quality=gait_branch_quality,
        contour_area=contour_area,
        contour_jitter=contour_jitter,
        id_switches=id_switches,
        frame_count=frame_count,
        valid_pose_frames=valid_pose_frames,
        valid_leg_frames=valid_leg_frames,
        detector_gap_frames=detector_gap_frames,
        track_gap_frames=track_gap_frames,
        timestamp_span_seconds=timestamp_span_seconds,
        gait_cycles=gait_cycles,
        walking_ratio=walking_ratio,
        view_angle=view_angle,
    )
    return Observation(
        event_id=event_id,
        camera_id=camera_id,
        capture_session_id=capture_session_id,
        track_id=track_id,
        timestamp=timestamp,
        features=FeatureBundle(appearance=appearance, gait=gait),
        quality=quality,
        appearance_request_id=appearance_request_id,
        model_version=model_version,
        feature_schema=feature_schema,
        artifact_sha256=artifact_sha256,
        preprocess_version=preprocess_version,
        joint_format=joint_format,
        sequence_length=sequence_length,
        tta_mode=tta_mode,
        coordinate_contract=coordinate_contract,
        embedding_dimensions=dict(embedding_dimensions or {}),
        calibration_version=calibration_version,
        metadata=metadata,
    )
