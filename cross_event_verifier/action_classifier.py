"""轻量姿态动作分类器。

动作标签是步态证据的路由条件，而不是身份特征。这个模块只读取一条
COCO-17 姿态序列，输出带质量和版本信息的动作预测；它不写图库，也不
参与身份匹配。默认实现使用可解释的运动学规则，另外提供一个无第三方
机器学习依赖的原型模型，供人工标注后的离线训练和回放使用。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .actions import ActionPrediction, ActionQuality, ActionType


POSE_FEATURE_SCHEMA = "pose-action-kinematics-v1"


def _as_pose_array(poses: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """Validate and return a finite-shape ``[T,17,3]`` pose array."""

    values = np.asarray(poses, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (17, 3):
        raise ValueError(f"pose sequence must have shape [T,17,3], got {values.shape}")
    return values


def extract_pose_action_features(
    poses: Sequence[np.ndarray] | np.ndarray,
    timestamps: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Extract scale-normalized, action-only kinematic features.

    Missing joints are excluded from each statistic. No missing frame is
    interpolated here; the production pose protocol keeps the missing mask for
    the gait encoder and this classifier uses it only as a quality signal.
    """

    values = _as_pose_array(poses)
    frame_count = values.shape[0]
    if frame_count == 0:
        # Keep the feature contract identical for empty and non-empty
        # sequences.  ``ActionClassifierModel`` stores the feature width in
        # its mean/scale arrays, so an accidental extra slot here would turn
        # an empty annotation into a model-shape error rather than a clean
        # low-quality prediction.
        return np.zeros(11, dtype=np.float32)

    xy = values[:, :, :2]
    scores = values[:, :, 2]
    valid = np.isfinite(xy).all(axis=2) & np.isfinite(scores) & (scores > 0.0)
    pose_ratio = float(valid.any(axis=1).mean())
    leg_indexes = np.asarray([11, 12, 13, 14, 15, 16])
    leg_ratio = float(valid[:, leg_indexes].mean())

    hip_valid = valid[:, 11] & valid[:, 12]
    shoulder_valid = valid[:, 5] & valid[:, 6]
    centers = np.full((frame_count, 2), np.nan, dtype=np.float32)
    centers[hip_valid] = (xy[hip_valid, 11] + xy[hip_valid, 12]) * 0.5
    shoulders = np.full((frame_count, 2), np.nan, dtype=np.float32)
    shoulders[shoulder_valid] = (xy[shoulder_valid, 5] + xy[shoulder_valid, 6]) * 0.5

    body_sizes = np.linalg.norm(shoulders - centers, axis=1)
    body_sizes = body_sizes[np.isfinite(body_sizes) & (body_sizes > 1e-3)]
    if body_sizes.size == 0:
        body_sizes = np.linalg.norm(
            xy[:, 15] - xy[:, 12], axis=1
        )
        body_sizes = body_sizes[np.isfinite(body_sizes) & (body_sizes > 1e-3)]
    scale = float(np.median(body_sizes)) if body_sizes.size else 1.0

    center = np.zeros(2, dtype=np.float32)
    for axis in range(2):
        finite_axis = centers[np.isfinite(centers[:, axis]), axis]
        if finite_axis.size:
            center[axis] = float(np.median(finite_axis))
    normalized = (xy - center[None, None, :]) / max(scale, 1e-3)

    def mean_velocity(joints: Sequence[int]) -> float:
        if frame_count < 2:
            return 0.0
        delta = np.linalg.norm(np.diff(normalized[:, joints], axis=0), axis=2)
        pair_valid = valid[:-1, joints] & valid[1:, joints]
        if timestamps is not None and len(timestamps) == frame_count:
            time_delta = np.diff(np.asarray(timestamps, dtype=np.float64))
            positive = time_delta[time_delta > 1e-6]
            if positive.size:
                nominal = float(np.median(positive))
                # A real gap should reduce its contribution rather than look
                # like a high-speed action.
                delta = delta / np.maximum(
                    np.clip(time_delta / nominal, 1.0, 8.0)[:, None], 1.0
                )
                pair_valid &= (time_delta <= nominal * 2.5)[:, None]
        return float(delta[pair_valid].mean()) if pair_valid.any() else 0.0

    ankle_motion = mean_velocity((15, 16))
    leg_motion = mean_velocity((13, 14, 15, 16))

    hip_y = centers[:, 1] / max(scale, 1e-3)
    hip_y = hip_y[np.isfinite(hip_y)]
    hip_range = (
        float(np.percentile(hip_y, 95) - np.percentile(hip_y, 5))
        if hip_y.size >= 3
        else 0.0
    )
    hip_velocity = float(np.mean(np.abs(np.diff(hip_y)))) if hip_y.size >= 2 else 0.0

    bends: list[float] = []
    for hip_index, knee_index, ankle_index in ((11, 13, 15), (12, 14, 16)):
        leg_validity = valid[:, hip_index] & valid[:, knee_index] & valid[:, ankle_index]
        first = xy[:, hip_index] - xy[:, knee_index]
        second = xy[:, ankle_index] - xy[:, knee_index]
        denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        cosine = np.divide(
            np.sum(first * second, axis=1),
            denominator,
            out=np.zeros(frame_count, dtype=np.float32),
            where=denominator > 1e-6,
        )
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        bends.extend((1.0 - angle / np.pi)[leg_validity].tolist())
    knee_bend = float(np.median(bends)) if bends else 0.0
    knee_bend_range = (
        float(np.percentile(bends, 90) - np.percentile(bends, 10))
        if len(bends) >= 3
        else 0.0
    )

    phase = xy[:, 15, 1] - xy[:, 16, 1]
    phase_valid = valid[:, 15] & valid[:, 16]
    phase = phase[phase_valid]
    phase_amplitude = 0.0
    phase_crossing_rate = 0.0
    if phase.size >= 4:
        phase = (phase - np.median(phase)) / max(scale, 1e-3)
        phase_amplitude = float(np.percentile(phase, 90) - np.percentile(phase, 10))
        threshold = max(float(np.std(phase) * 0.15), 1e-3)
        signs = np.sign(phase[np.abs(phase) >= threshold])
        if signs.size >= 2:
            phase_crossing_rate = float(np.count_nonzero(signs[1:] != signs[:-1])) / frame_count

    return np.asarray(
        [
            pose_ratio,
            leg_ratio,
            ankle_motion,
            leg_motion,
            hip_range,
            hip_velocity,
            knee_bend,
            knee_bend_range,
            phase_amplitude,
            phase_crossing_rate,
            float(np.isfinite(values).all(axis=(1, 2)).mean()),
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class ActionClassifierModel:
    """A small standardized nearest-centroid action model."""

    classes: tuple[ActionType, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    centroids: np.ndarray
    minimum_frames: int = 12
    model_version: str = "pose-action-centroid-v1"

    def __post_init__(self) -> None:
        classes = tuple(ActionType.parse(value) for value in self.classes)
        if len(classes) < 2 or len(set(classes)) != len(classes):
            raise ValueError("action classifier needs at least two distinct classes")
        mean = np.asarray(self.feature_mean, dtype=np.float32).reshape(-1)
        scale = np.asarray(self.feature_scale, dtype=np.float32).reshape(-1)
        centroids = np.asarray(self.centroids, dtype=np.float32)
        if (
            mean.size == 0
            or scale.shape != mean.shape
            or centroids.shape != (len(classes), mean.size)
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or not np.isfinite(centroids).all()
        ):
            raise ValueError("invalid action classifier model arrays")
        if np.any(scale <= 0.0) or self.minimum_frames < 1:
            raise ValueError("action classifier scales must be positive")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "centroids", centroids)

    @classmethod
    def fit(
        cls,
        sequences: Iterable[Sequence[np.ndarray] | np.ndarray],
        labels: Iterable[ActionType | str],
        *,
        timestamps: Iterable[Sequence[float] | np.ndarray | None] | None = None,
        minimum_frames: int = 12,
        minimum_samples_per_class: int = 2,
    ) -> "ActionClassifierModel":
        """Fit class centroids from independently annotated pose sequences."""

        sequence_values = list(sequences)
        label_values = [ActionType.parse(value) for value in labels]
        if len(sequence_values) != len(label_values) or not sequence_values:
            raise ValueError("sequences and labels must be non-empty and have equal length")
        timestamp_values = list(timestamps) if timestamps is not None else [None] * len(sequence_values)
        if len(timestamp_values) != len(sequence_values):
            raise ValueError("timestamps must match sequence count")
        features = np.vstack(
            [extract_pose_action_features(sequence, stamp) for sequence, stamp in zip(sequence_values, timestamp_values)]
        )
        counts = {action: label_values.count(action) for action in set(label_values)}
        insufficient = [action.value for action, count in counts.items() if count < minimum_samples_per_class]
        if insufficient:
            raise ValueError(
                "each action class needs at least "
                f"{minimum_samples_per_class} samples: {', '.join(sorted(insufficient))}"
            )
        classes = tuple(sorted(counts, key=lambda value: value.value))
        label_array = np.asarray([label.value for label in label_values])
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
        standardized = (features - mean) / scale
        centroids = np.vstack(
            [standardized[label_array == action.value].mean(axis=0) for action in classes]
        )
        return cls(tuple(classes), mean, scale, centroids, minimum_frames=minimum_frames)

    def predict_features(
        self,
        features: np.ndarray,
        *,
        frame_count: int,
    ) -> ActionPrediction:
        """Predict from one feature vector with an auditable quality gate."""

        values = np.asarray(features, dtype=np.float32).reshape(-1)
        if values.shape != self.feature_mean.shape or not np.isfinite(values).all():
            return ActionPrediction(
                ActionType.UNKNOWN,
                0.0,
                ActionQuality.INVALID,
                source="action_classifier",
                model_version=self.model_version,
            )
        standardized = (values - self.feature_mean) / self.feature_scale
        distances = np.linalg.norm(self.centroids - standardized[None, :], axis=1)
        order = np.argsort(distances)
        nearest = int(order[0])
        nearest_distance = float(distances[nearest])
        second_distance = float(distances[order[1]]) if len(order) > 1 else nearest_distance
        margin = max(second_distance - nearest_distance, 0.0)
        confidence = float(np.clip(0.5 + margin / (2.0 + margin), 0.0, 0.99))
        pose_ratio = float(np.clip(values[0], 0.0, 1.0))
        leg_ratio = float(np.clip(values[1], 0.0, 1.0))
        completion = float(
            np.clip(
                min(frame_count / max(self.minimum_frames, 1), 1.0)
                * pose_ratio
                * np.sqrt(leg_ratio),
                0.0,
                1.0,
            )
        )
        quality = (
            ActionQuality.STRONG
            if frame_count >= self.minimum_frames and pose_ratio >= 0.70 and leg_ratio >= 0.60
            else ActionQuality.PARTIAL
            if frame_count > 0 and pose_ratio >= 0.30
            else ActionQuality.INVALID
        )
        action = self.classes[nearest] if confidence >= 0.55 else ActionType.UNKNOWN
        return ActionPrediction(
            action,
            confidence,
            quality,
            completion,
            source="action_classifier",
            model_version=self.model_version,
        )

    def predict(
        self,
        poses: Sequence[np.ndarray] | np.ndarray,
        timestamps: Sequence[float] | np.ndarray | None = None,
    ) -> ActionPrediction:
        values = _as_pose_array(poses)
        features = extract_pose_action_features(values, timestamps)
        return self.predict_features(features, frame_count=len(values))

    def save(self, path: Path) -> None:
        """Persist a model without pickle or executable code."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            classes=np.asarray([action.value for action in self.classes]),
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            centroids=self.centroids,
            minimum_frames=np.asarray(self.minimum_frames, dtype=np.int64),
            model_version=np.asarray(self.model_version),
            feature_schema=np.asarray(POSE_FEATURE_SCHEMA),
        )

    @classmethod
    def load(cls, path: Path) -> "ActionClassifierModel":
        with np.load(Path(path), allow_pickle=False) as data:
            schema = str(np.asarray(data["feature_schema"]).item())
            if schema != POSE_FEATURE_SCHEMA:
                raise ValueError(f"unsupported action feature schema: {schema!r}")
            return cls(
                tuple(ActionType.parse(value) for value in data["classes"].tolist()),
                data["feature_mean"],
                data["feature_scale"],
                data["centroids"],
                minimum_frames=int(np.asarray(data["minimum_frames"]).item()),
                model_version=str(np.asarray(data["model_version"]).item()),
            )


class PoseActionClassifier:
    """Runtime classifier with a conservative kinematic baseline."""

    def __init__(
        self,
        model: ActionClassifierModel | None = None,
        *,
        minimum_frames: int = 12,
    ) -> None:
        if minimum_frames < 1:
            raise ValueError("minimum_frames must be positive")
        self.model = model
        self.minimum_frames = int(minimum_frames)
        self.model_version = (
            model.model_version if model is not None else "pose-action-kinematic-v1"
        )

    @classmethod
    def from_path(cls, path: Path) -> "PoseActionClassifier":
        model = ActionClassifierModel.load(path)
        return cls(model, minimum_frames=model.minimum_frames)

    def predict(
        self,
        poses: Sequence[np.ndarray] | np.ndarray,
        timestamps: Sequence[float] | np.ndarray | None = None,
        *,
        walking_ratio: float | None = None,
        gait_cycles: float | None = None,
    ) -> ActionPrediction:
        values = _as_pose_array(poses)
        if self.model is not None:
            return self.model.predict(values, timestamps)
        features = extract_pose_action_features(values, timestamps)
        return self._predict_kinematic(
            features,
            frame_count=len(values),
            walking_ratio=walking_ratio,
            gait_cycles=gait_cycles,
        )

    def _predict_kinematic(
        self,
        features: np.ndarray,
        *,
        frame_count: int,
        walking_ratio: float | None,
        gait_cycles: float | None,
    ) -> ActionPrediction:
        pose_ratio = float(np.clip(features[0], 0.0, 1.0))
        leg_ratio = float(np.clip(features[1], 0.0, 1.0))
        ankle_motion, leg_motion = (float(max(value, 0.0)) for value in features[2:4])
        hip_range = float(max(features[4], 0.0))
        knee_bend, knee_bend_range = (float(np.clip(value, 0.0, 1.0)) for value in features[6:8])
        phase_crossing_rate = float(max(features[9], 0.0))
        completion = float(
            np.clip(
                min(frame_count / max(self.minimum_frames, 1), 1.0)
                * pose_ratio
                * np.sqrt(leg_ratio),
                0.0,
                1.0,
            )
        )
        if frame_count < self.minimum_frames or pose_ratio < 0.45 or leg_ratio < 0.45:
            quality = ActionQuality.PARTIAL if pose_ratio >= 0.30 else ActionQuality.INVALID
            return ActionPrediction(
                ActionType.UNKNOWN,
                float(np.clip(completion, 0.0, 0.6)),
                quality,
                completion,
                source="action_classifier",
                model_version=self.model_version,
            )

        # A large hip excursion with a substantial change in knee bend is the
        # most useful inexpensive cue for sit/stand. A sustained bent-knee
        # window without that excursion is routed to squat.
        if hip_range >= 0.85 and knee_bend >= 0.18 and knee_bend_range < 0.08:
            confidence = float(np.clip(0.60 + 0.15 * min(knee_bend, 1.0), 0.0, 0.90))
            return ActionPrediction(
                ActionType.SQUAT,
                confidence,
                ActionQuality.STRONG,
                completion,
                source="action_classifier",
                model_version=self.model_version,
            )
        if hip_range >= 0.65 and knee_bend_range >= 0.08:
            confidence = float(np.clip(0.60 + 0.15 * min(hip_range, 1.0), 0.0, 0.90))
            return ActionPrediction(
                ActionType.SIT_STAND,
                confidence,
                ActionQuality.STRONG,
                completion,
                source="action_classifier",
                model_version=self.model_version,
            )

        walking_score = float(np.clip(walking_ratio if walking_ratio is not None else 0.0, 0.0, 1.0))
        cycles = max(float(gait_cycles or 0.0), 0.0)
        if walking_score >= 0.55 and cycles >= 0.50:
            confidence = float(
                np.clip(
                    0.55
                    + 0.22 * walking_score
                    + 0.12 * min(cycles / 2.0, 1.0)
                    + 0.11 * leg_ratio,
                    0.0,
                    0.99,
                )
            )
            return ActionPrediction(
                ActionType.WALK,
                confidence,
                ActionQuality.STRONG,
                completion,
                source="action_classifier",
                model_version=self.model_version,
            )
        motion = max(ankle_motion, leg_motion, phase_crossing_rate * 0.1)
        if motion < 0.01 and hip_range < 0.25 and knee_bend < 0.20:
            return ActionPrediction(
                ActionType.STANDING,
                0.76,
                ActionQuality.STRONG,
                completion,
                source="action_classifier",
                model_version=self.model_version,
            )
        confidence = float(np.clip(0.60 + 0.15 * leg_ratio + 0.10 * min(motion / 0.1, 1.0), 0.0, 0.85))
        return ActionPrediction(
            ActionType.OTHER,
            confidence,
            ActionQuality.STRONG,
            completion,
            source="action_classifier",
            model_version=self.model_version,
        )


def write_training_manifest(path: Path, *, model: ActionClassifierModel, sample_count: int) -> None:
    """Write a small human-readable record next to a trained model."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "schema_version": POSE_FEATURE_SCHEMA,
                "model_version": model.model_version,
                "classes": [action.value for action in model.classes],
                "sample_count": int(sample_count),
                "minimum_frames": model.minimum_frames,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ActionClassifierModel",
    "POSE_FEATURE_SCHEMA",
    "PoseActionClassifier",
    "extract_pose_action_features",
    "write_training_manifest",
]
