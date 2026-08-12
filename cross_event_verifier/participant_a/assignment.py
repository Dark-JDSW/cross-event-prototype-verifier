"""带前置可行性门控的全局一对一身份指派。

在人群较多的画面中，分别为每条轨迹选择最佳身份，可能把同一个人分配给
两条轨迹。本模块先标记不可能的边，再求解矩形线性指派问题，最后应用每
一行的间隔规则。SciPy 是可选依赖；内置的匈牙利算法回退实现让小型部署
只使用 NumPy 也能运行。
"""

from __future__ import annotations

import numpy as np


def _hungarian_min(cost: np.ndarray) -> list[tuple[int, int]]:
    """使用纯 Python 求解矩形代价矩阵的匈牙利算法。

    :func:`gated_global_assignment` 中可选的 SciPy 路径在大批量数据上更快，
    而本实现保证小型边缘部署只安装 NumPy 时仍可使用。
    """

    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("cost must be a 2-D matrix")
    rows, columns = matrix.shape
    if rows == 0 or columns == 0:
        return []
    transposed = False
    if rows > columns:
        matrix = matrix.T
        rows, columns = matrix.shape
        transposed = True

    # 使用最短增广路算法的 1 起始下标实现。
    u = np.zeros(rows + 1, dtype=np.float64)
    v = np.zeros(columns + 1, dtype=np.float64)
    p = np.zeros(columns + 1, dtype=np.int64)
    way = np.zeros(columns + 1, dtype=np.int64)
    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(columns + 1, np.inf, dtype=np.float64)
        used = np.zeros(columns + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = np.inf
            j1 = 0
            for j in range(1, columns + 1):
                if used[j]:
                    continue
                current = matrix[i0 - 1, j - 1] - u[i0] - v[j]
                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if not np.isfinite(delta):
                break
            for j in range(columns + 1):
                if used[j]:
                    u[int(p[j])] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs: list[tuple[int, int]] = []
    for j in range(1, columns + 1):
        if p[j] == 0:
            continue
        row, column = int(p[j] - 1), int(j - 1)
        pairs.append((column, row) if transposed else (row, column))
    return pairs


def _linear_sum_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    """优先使用已安装的 SciPy，否则使用本地匈牙利算法。"""
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        rows, columns = linear_sum_assignment(cost)
        return list(zip(rows.tolist(), columns.tolist()))
    except ImportError:
        return _hungarian_min(cost)


def gated_global_assignment(
    score_matrix: np.ndarray,
    appearance_matrix: np.ndarray | None = None,
    *,
    accept_threshold: float,
    appearance_floor: float = 0.0,
    margin_threshold: float = 0.0,
) -> dict[int, int]:
    """在应用可行性门后，将行指派到列。

    在匈牙利算法之前进行门控，可以防止低置信度配对占用另一行本可使用的
    列。间隔门被有意放在指派之后，并且只比较可行列，这与早期跟踪设计中
    的门控指派行为一致。
    """

    # 行表示当前轨迹，列表示正式身份。先计算可行性再优化，避免坏边占用
    # 更好行所需要的列。
    scores = np.asarray(score_matrix, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError("score_matrix must be 2-D")
    if appearance_matrix is None:
        appearance = np.ones_like(scores)
    else:
        appearance = np.asarray(appearance_matrix, dtype=np.float32)
        if appearance.shape != scores.shape:
            raise ValueError("appearance_matrix must have the same shape as score_matrix")
    if scores.size == 0:
        return {}

    feasible = np.isfinite(scores) & (scores >= accept_threshold) & (
        appearance >= appearance_floor
    )
    # 允许求解器暂时选择禁用边，之后再丢弃；对于不存在完整可行覆盖的
    # 矩形矩阵，这是必要的处理方式。
    cost = np.where(feasible, -scores, 1e6).astype(np.float64)
    result: dict[int, int] = {}
    for row, column in _linear_sum_assignment(cost):
        if row >= scores.shape[0] or column >= scores.shape[1] or not feasible[row, column]:
            continue
        if margin_threshold > 0:
            other_scores = scores[row, feasible[row] & (np.arange(scores.shape[1]) != column)]
            if other_scores.size and float(scores[row, column] - np.max(other_scores)) < margin_threshold:
                continue
        result[int(row)] = int(column)
    return result


__all__ = ["gated_global_assignment"]
