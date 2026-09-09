"""Optional adapter for the official MMAction2 PoseC3D-HMDB51 checkpoint.

The adapter is intentionally lazy: the base project remains usable without
MMAction2, while an enabled PoseC3D gate fails closed when its runtime is not
installed or its input/output contract is invalid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


class PoseC3DError(RuntimeError):
    """Raised when the optional PoseC3D runtime or contract is invalid."""


@dataclass(frozen=True)
class PoseC3DPrediction:
    """Auditable semantic result for one track window."""

    top_index: int
    top_label: str
    walk_probability: float
    second_probability: float
    margin: float
    scores: tuple[float, ...] = ()
    reason: str = "ok"

    @property
    def semantic_walk(self) -> bool:
        return self.top_label == "walk"

    @classmethod
    def invalid(cls, reason: str) -> "PoseC3DPrediction":
        return cls(-1, "", 0.0, 0.0, 0.0, (), reason)


def _load_labels(path: Path) -> tuple[str, ...]:
    try:
        labels = tuple(
            line.strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except OSError as error:
        raise PoseC3DError(f"无法读取 HMDB51 标签表：{path}: {error}") from error
    if len(labels) != 51 or labels[49] != "walk":
        raise PoseC3DError(
            "HMDB51 标签契约无效：需要 51 个标签且 index 49 必须为 walk"
        )
    return labels


def _probabilities(raw_scores: Any, expected: int) -> np.ndarray:
    values = np.asarray(
        raw_scores.detach().cpu().numpy()
        if hasattr(raw_scores, "detach")
        else raw_scores,
        dtype=np.float64,
    ).reshape(-1)
    if values.size != expected or not np.isfinite(values).all():
        raise PoseC3DError(
            f"PoseC3D 输出必须是 {expected} 个有限分数，实际 shape={values.shape}"
        )
    total = float(values.sum())
    if np.all(values >= 0.0) and abs(total - 1.0) <= 1e-3:
        return values / max(total, 1e-12)
    shifted = values - float(values.max())
    exp_values = np.exp(np.clip(shifted, -80.0, 0.0))
    return exp_values / max(float(exp_values.sum()), 1e-12)


class PoseC3DActionClassifier:
    """Run official HMDB51 PoseC3D inference on one tracked person."""

    num_classes = 51
    walk_class_index = 49
    clip_length = 48

    def __init__(
        self,
        checkpoint_path: Path,
        config_path: Path,
        label_map_path: Path,
        *,
        device: str = "cuda:0",
        probability_threshold: float = 0.80,
        margin_threshold: float = 0.15,
        model: Any | None = None,
        inferencer: Callable[..., Any] | None = None,
    ) -> None:
        for name, value in {
            "probability_threshold": probability_threshold,
            "margin_threshold": margin_threshold,
        }.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.label_map_path = Path(label_map_path)
        self.labels = _load_labels(self.label_map_path)
        self.probability_threshold = float(probability_threshold)
        self.margin_threshold = float(margin_threshold)
        self.model_version = "posec3d-hmdb51-split1-walk-gate-v1"
        if model is not None:
            self.model = model
            self._inferencer = inferencer
            return
        try:
            # Older released MMAction2 transforms still reference ``np.Inf``;
            # NumPy 2 removed that alias.  Keep the compatibility shim local to
            # the optional runtime instead of constraining the base project.
            if not hasattr(np, "Inf"):
                setattr(np, "Inf", np.inf)
            from mmaction.apis import inference_skeleton, init_recognizer
        except (ImportError, ModuleNotFoundError) as error:
            raise PoseC3DError(
                "PoseC3D 需要安装官方 MMAction2/MMEngine/MMCV 运行时"
            ) from error
        try:
            self.model = init_recognizer(
                str(self.config_path),
                str(self.checkpoint_path),
                device=device,
            )
        except Exception as error:
            raise PoseC3DError(f"PoseC3D 模型加载失败：{error}") from error
        self._inferencer = inference_skeleton

    def _window(self, poses: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        values = np.asarray(poses, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (17, 3):
            raise PoseC3DError(f"PoseC3D 输入必须为 [T,17,3]，实际 {values.shape}")
        if values.shape[0] < self.clip_length:
            raise PoseC3DError(
                f"PoseC3D 需要至少 {self.clip_length} 帧，实际 {values.shape[0]}"
            )
        indices = np.linspace(0, values.shape[0] - 1, self.clip_length)
        window = values[np.rint(indices).astype(np.int64)]
        if not np.isfinite(window).all():
            raise PoseC3DError("PoseC3D 输入包含非有限坐标或分数")
        if not np.any(window[:, :, 2] > 0.0, axis=1).all():
            raise PoseC3DError("PoseC3D 输入包含无姿态帧")
        return window

    def predict(
        self,
        poses: Sequence[np.ndarray] | np.ndarray,
        *,
        image_shape: tuple[int, int],
    ) -> PoseC3DPrediction:
        try:
            window = self._window(poses)
            height, width = (int(image_shape[0]), int(image_shape[1]))
            if height <= 0 or width <= 0:
                raise PoseC3DError(f"图像尺寸无效：{image_shape}")
            pose_results = [
                {
                    "keypoints": frame[:, :2][None, ...].astype(np.float32),
                    "keypoint_scores": frame[:, 2][None, ...].astype(np.float32),
                }
                for frame in window
            ]
            if self._inferencer is None:
                raise PoseC3DError("PoseC3D inference_skeleton 未初始化")
            result = self._inferencer(self.model, pose_results, (height, width))
            raw_scores = getattr(result, "pred_score", None)
            if raw_scores is None and isinstance(result, dict):
                raw_scores = result.get("pred_score")
            if raw_scores is None:
                raise PoseC3DError("PoseC3D 结果缺少 pred_score")
            scores = _probabilities(raw_scores, self.num_classes)
            order = np.argsort(scores)[::-1]
            top_index = int(order[0])
            second_probability = float(scores[int(order[1])])
            walk_probability = float(scores[self.walk_class_index])
            return PoseC3DPrediction(
                top_index=top_index,
                top_label=self.labels[top_index],
                walk_probability=walk_probability,
                second_probability=second_probability,
                margin=float(scores[top_index] - second_probability),
                scores=tuple(float(item) for item in scores),
            )
        except PoseC3DError as error:
            return PoseC3DPrediction.invalid(str(error))
        except Exception as error:
            return PoseC3DPrediction.invalid(f"PoseC3D 推理失败：{error}")

    def warmup(self) -> None:
        """Run one valid-shape inference without affecting application state."""

        poses = np.zeros((self.clip_length, 17, 3), dtype=np.float32)
        poses[:, :, 2] = 1.0
        poses[:, :, 0] = np.linspace(20.0, 100.0, 17)
        poses[:, :, 1] = np.linspace(20.0, 220.0, 17)
        result = self.predict(poses, image_shape=(256, 192))
        if result.reason != "ok":
            raise PoseC3DError(f"PoseC3D warmup 失败：{result.reason}")


__all__ = [
    "PoseC3DActionClassifier",
    "PoseC3DPrediction",
    "PoseC3DError",
]
