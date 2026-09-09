"""地面控制点录入、校核和标定文件序列化。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class GroundControlPoint:
    """一个图像点与地面世界坐标的对应关系。"""

    point_id: str
    image_u: float
    image_v: float
    world_x: float
    world_y: float

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.point_id,
            "image": [self.image_u, self.image_v],
            "world": [self.world_x, self.world_y],
        }


def parse_ground_control_point(
    point_id: str,
    image_u: str,
    image_v: str,
    world_x: str,
    world_y: str,
) -> GroundControlPoint:
    """把界面文本字段转换为一个完整控制点。"""

    identifier = point_id.strip()
    if not identifier:
        raise ValueError("点 ID 不能为空")
    values: list[float] = []
    for label, raw in (
        ("像素 U", image_u),
        ("像素 V", image_v),
        ("世界 X", world_x),
        ("世界 Y", world_y),
    ):
        try:
            value = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label}必须是数字") from error
        if not np.isfinite(value):
            raise ValueError(f"{label}必须是有限数值")
        values.append(value)
    return GroundControlPoint(identifier, *values)


@dataclass(frozen=True)
class HomographyCheck:
    """地面 Homography 及其在控制点上的校核结果。"""

    matrix: tuple[float, ...]
    inliers: tuple[bool, ...]
    reprojection_errors: tuple[float, ...]
    rmse: float
    maximum_error: float

    def to_dict(self) -> dict[str, object]:
        return {
            "image_to_ground": list(self.matrix),
            "inliers": list(self.inliers),
            "reprojection_errors_m": list(self.reprojection_errors),
            "rmse_m": self.rmse,
            "maximum_error_m": self.maximum_error,
        }


def validate_ground_control_points(
    points: Sequence[GroundControlPoint],
) -> list[str]:
    """返回可操作的控制点错误；空列表表示可计算。"""

    errors: list[str] = []
    if len(points) < 4:
        errors.append("至少需要 4 个地面控制点")
        return errors
    identifiers = [point.point_id.strip() for point in points]
    if any(not identifier for identifier in identifiers):
        errors.append("控制点 ID 不能为空")
    if len(set(identifiers)) != len(identifiers):
        errors.append("控制点 ID 不能重复")
    values = np.asarray(
        [
            [point.image_u, point.image_v, point.world_x, point.world_y]
            for point in points
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        errors.append("控制点坐标必须是有限数值")
        return errors
    if len(np.unique(values[:, :2], axis=0)) != len(points):
        errors.append("图像控制点不能重复")
    if len(np.unique(values[:, 2:], axis=0)) != len(points):
        errors.append("世界控制点不能重复")
    if np.linalg.matrix_rank(values[:, :2] - values[:, :2].mean(axis=0)) < 2:
        errors.append("图像控制点不能共线")
    if np.linalg.matrix_rank(values[:, 2:] - values[:, 2:].mean(axis=0)) < 2:
        errors.append("世界控制点不能共线")
    return errors


def compute_ground_homography(
    points: Sequence[GroundControlPoint],
    *,
    ransac_threshold_m: float = 0.05,
) -> HomographyCheck:
    """从控制点计算 image→ground H，并返回可审计的误差。"""

    errors = validate_ground_control_points(points)
    if errors:
        raise ValueError("；".join(errors))
    if not np.isfinite(ransac_threshold_m) or ransac_threshold_m <= 0:
        raise ValueError("RANSAC 校核阈值必须为正数")
    source = np.asarray([[point.image_u, point.image_v] for point in points], dtype=np.float64)
    target = np.asarray([[point.world_x, point.world_y] for point in points], dtype=np.float64)
    matrix, mask = cv2.findHomography(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_threshold_m),
    )
    if (
        matrix is None
        or not np.isfinite(matrix).all()
        or np.linalg.matrix_rank(matrix) < 3
    ):
        raise ValueError("无法从当前控制点计算有效 Homography")
    projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    distances = np.linalg.norm(projected - target, axis=1)
    inliers = tuple(bool(value) for value in (mask.reshape(-1) if mask is not None else np.ones(len(points), dtype=bool)))
    inlier_distances = distances[np.asarray(inliers, dtype=bool)]
    if inlier_distances.size == 0:
        raise ValueError("控制点全部被判为外点，请检查标记或世界坐标")
    return HomographyCheck(
        matrix=tuple(float(value) for value in matrix.reshape(-1)),
        inliers=inliers,
        reprojection_errors=tuple(float(value) for value in distances),
        rmse=float(np.sqrt(np.mean(np.square(inlier_distances)))),
        maximum_error=float(np.max(distances)),
    )


def _point_from_dict(raw: Any, index: int) -> GroundControlPoint:
    if not isinstance(raw, dict):
        raise ValueError(f"第 {index} 个控制点不是对象")
    try:
        point_id = str(raw["id"]).strip()
        image = raw["image"]
        world = raw["world"]
        if len(image) != 2 or len(world) != 2:
            raise ValueError
        return GroundControlPoint(
            point_id=point_id,
            image_u=float(image[0]),
            image_v=float(image[1]),
            world_x=float(world[0]),
            world_y=float(world[1]),
        )
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ValueError(f"第 {index} 个控制点缺少 id/image/world 数值") from error


def load_ground_control_file(path: str | Path) -> dict[str, object]:
    """读取本页面导出的 ``gcp-v1`` 文件。"""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取标定文件：{error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != "gcp-v1":
        raise ValueError("标定文件必须使用 schema_version=gcp-v1")
    raw_points = payload.get("control_points")
    if not isinstance(raw_points, list):
        raise ValueError("标定文件缺少 control_points 列表")
    points = [_point_from_dict(raw, index + 1) for index, raw in enumerate(raw_points)]
    validation_errors = validate_ground_control_points(points)
    if validation_errors:
        raise ValueError("；".join(validation_errors))
    result: dict[str, object] = {
        "camera_id": str(payload.get("camera_id", "camera-1")),
        "coordinate_frame": str(payload.get("coordinate_frame", "ground")),
        "unit": str(payload.get("unit", "m")),
        "image_size": payload.get("image_size"),
        "points": points,
    }
    return result


def save_ground_control_file(
    path: str | Path,
    *,
    camera_id: str,
    coordinate_frame: str,
    unit: str,
    image_size: tuple[int, int] | None,
    points: Sequence[GroundControlPoint],
    check: HomographyCheck,
) -> None:
    """以稳定、可审计的 JSON 格式保存控制点和 H 校核结果。"""

    payload: dict[str, object] = {
        "schema_version": "gcp-v1",
        "camera_id": camera_id.strip(),
        "coordinate_frame": coordinate_frame.strip(),
        "unit": unit.strip() or "m",
        "image_size": list(image_size) if image_size is not None else None,
        "control_points": [point.to_dict() for point in points],
        "homography": check.to_dict(),
    }
    destination = Path(path)
    try:
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise ValueError(f"无法写入标定文件：{error}") from error
