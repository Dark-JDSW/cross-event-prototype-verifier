"""Quality-aware aggregation helpers for temporal biometric evidence.

The helpers in this module operate on already extracted feature vectors. They
do not alter the pose representation or the GaitGraph2 input contract; they
only control how valid observations are accumulated.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def _normalise_band(band: object) -> str:
    value = getattr(band, "value", band)
    return str(value).lower()


def gait_quality_weight(
    quality: float,
    band: object,
    *,
    strong_threshold: float,
) -> float:
    """Return a bounded contribution weight for one gait sample."""

    q = float(np.clip(quality, 0.0, 1.0))
    band_name = _normalise_band(band)
    if band_name == "invalid":
        return 0.0
    if band_name == "partial":
        threshold = max(float(strong_threshold), 1e-6)
        return float(min(0.5, 0.5 * q / threshold))
    return q


def weighted_unit_mean(
    vectors: Iterable[Sequence[float] | np.ndarray],
    weights: Sequence[float] | None = None,
) -> np.ndarray | None:
    """Compute a finite, L2-normalised weighted centroid."""

    rows = [np.asarray(vector, dtype=np.float32).reshape(-1) for vector in vectors]
    if not rows:
        return None
    dimension = rows[0].shape[0]
    if dimension == 0 or any(row.shape[0] != dimension for row in rows):
        return None
    matrix = np.vstack(rows)
    if not np.all(np.isfinite(matrix)):
        return None

    if weights is None:
        weight_array = np.ones((len(rows),), dtype=np.float32)
    else:
        weight_array = np.asarray(weights, dtype=np.float32).reshape(-1)
        if weight_array.shape[0] != len(rows):
            raise ValueError("weights must have the same length as vectors")
        if not np.all(np.isfinite(weight_array)):
            return None
        weight_array = np.clip(weight_array, 0.0, None)

    total = float(weight_array.sum())
    if total <= 1e-8:
        return None
    centroid = (matrix * weight_array[:, None]).sum(axis=0) / total
    norm = float(np.linalg.norm(centroid))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return (centroid / norm).astype(np.float32)


def weighted_cosine_stability(
    vectors: Iterable[Sequence[float] | np.ndarray],
    centroid: Sequence[float] | np.ndarray,
    weights: Sequence[float] | None = None,
) -> float:
    """Return the weighted mean cosine similarity to a centroid."""

    rows = [np.asarray(vector, dtype=np.float32).reshape(-1) for vector in vectors]
    centre = np.asarray(centroid, dtype=np.float32).reshape(-1)
    if not rows or centre.size == 0:
        return 0.0
    matrix = np.vstack(rows)
    if matrix.shape[1] != centre.shape[0]:
        return 0.0
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    centre_norm = float(np.linalg.norm(centre))
    if centre_norm <= 1e-8 or np.any(row_norms <= 1e-8):
        return 0.0
    cosines = (matrix @ centre) / (row_norms[:, 0] * centre_norm)
    cosines = np.clip(cosines, -1.0, 1.0)
    if weights is None:
        return float(np.mean(cosines))
    weight_array = np.asarray(weights, dtype=np.float32).reshape(-1)
    if weight_array.shape[0] != len(rows):
        raise ValueError("weights must have the same length as vectors")
    weight_array = np.clip(np.nan_to_num(weight_array, nan=0.0), 0.0, None)
    total = float(weight_array.sum())
    return float(np.dot(cosines, weight_array) / total) if total > 1e-8 else 0.0
