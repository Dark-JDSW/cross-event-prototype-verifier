"""面向时序 COCO-17 骨架的精简 GaitGraph2 推理模型。

本实现有意只支持推理。它复现官方 OpenGait GREW 检查点所使用的
GaitGraph2/ResGCN 网络结构，但运行时不导入 OpenGait 源码，这使桌面应用
保持独立，并让项目其他部分只依赖很小的 ``encode(sequence)`` 接口。

上游 OpenGait/GaitGraph2 代码和检查点属于学术用途资产；商业使用或再分发
前请查看 ``THIRD_PARTY_NOTICES.md``。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


_COCO_CONNECT = np.asarray(
    [5, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14],
    dtype=np.int64,
)
_COCO_FLIP = np.asarray(
    [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
    dtype=np.int64,
)


def _coco_adjacency(max_hop: int = 3) -> np.ndarray:
    """构建按跳数分区并经过度归一化的 COCO-17 图。"""
    nodes = 17
    neighbours = [
        (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    adjacency = np.zeros((nodes, nodes), dtype=np.float32)
    for index in range(nodes):
        adjacency[index, index] = 1.0
    for left, right in neighbours:
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0

    hop_distance = np.full((nodes, nodes), np.inf, dtype=np.float32)
    powers = [np.linalg.matrix_power(adjacency, distance) for distance in range(max_hop + 1)]
    reachable = np.stack(powers) > 0
    for distance in range(max_hop, -1, -1):
        hop_distance[reachable[distance]] = distance

    # OpenGait normalizes the *union* of all nodes reachable within
    # ``max_hop`` and only then partitions that normalized matrix by hop. It
    # is not the one-hop degree normalization used by a vanilla ST-GCN.
    reachable_within_hops = np.isfinite(hop_distance)
    degree = reachable_within_hops.sum(axis=0)
    inverse_degree = np.zeros((nodes, nodes), dtype=np.float32)
    for index, value in enumerate(degree):
        if value > 0:
            inverse_degree[index, index] = float(value) ** -1
    normalized = reachable_within_hops.astype(np.float32) @ inverse_degree
    result = np.zeros((max_hop + 1, nodes, nodes), dtype=np.float32)
    for hop in range(max_hop + 1):
        result[hop][hop_distance == hop] = normalized[hop_distance == hop]
    return result


class _SpatialGraphConv(nn.Module):
    """在 17 个 COCO 关节上执行按跳数分区的图卷积。"""

    def __init__(self, in_channels: int, out_channels: int, classes: int) -> None:
        """创建 1x1 投影，为每个跳数输出一个通道组。"""
        super().__init__()
        self.classes = classes
        self.gcn = nn.Conv2d(in_channels, out_channels * classes, 1)

    def forward(self, values: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """投影特征，并使用跳数矩阵聚合邻居。"""
        values = self.gcn(values)
        batch, channels, time_steps, joints = values.shape
        values = values.view(
            batch,
            self.classes,
            channels // self.classes,
            time_steps,
            joints,
        )
        return torch.einsum(
            "nkctv,kvw->nctw",
            values,
            adjacency[: self.classes],
        ).contiguous()


class _SpatialBasicBlock(nn.Module):
    """初始阶段使用的空间残差图卷积模块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        classes: int,
        residual: bool = False,
    ) -> None:
        """配置图卷积、归一化、激活和残差路径。"""
        super().__init__()
        if not residual:
            self.residual = None
        elif in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        self.conv = _SpatialGraphConv(in_channels, out_channels, classes)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, values: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """执行空间图聚合，再进行残差激活。"""
        residual = 0 if self.residual is None else self.residual(values)
        return self.relu(self.bn(self.conv(values, adjacency)) + residual)


class _SpatialBottleneckBlock(nn.Module):
    """与 GREW 检查点结构匹配的缩减宽度空间模块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        classes: int,
        residual: bool = False,
        reduction: int = 4,
    ) -> None:
        """构建降维投影、图卷积、升维投影和残差路径。"""
        super().__init__()
        if not residual:
            self.residual = None
        elif in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        inner = out_channels // reduction
        self.conv_down = nn.Conv2d(in_channels, inner, 1)
        self.bn_down = nn.BatchNorm2d(inner)
        self.conv = _SpatialGraphConv(inner, inner, classes)
        self.bn = nn.BatchNorm2d(inner)
        self.conv_up = nn.Conv2d(inner, out_channels, 1)
        self.bn_up = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, values: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """执行瓶颈空间变换并加上残差。"""
        residual = 0 if self.residual is None else self.residual(values)
        values = self.relu(self.bn_down(self.conv_down(values)))
        values = self.relu(self.bn(self.conv(values, adjacency)))
        values = self.bn_up(self.conv_up(values))
        return self.relu(values + residual)


class _TemporalBasicBlock(nn.Module):
    """对每个关节独立执行九帧时序卷积的模块。"""

    def __init__(self, channels: int, stride: int = 1, residual: bool = False) -> None:
        """配置时序步幅和可选的残差投影。"""
        super().__init__()
        if not residual:
            self.residual = None
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (stride, 1)),
                nn.BatchNorm2d(channels),
            )
        self.conv = nn.Conv2d(channels, channels, (9, 1), (stride, 1), (4, 0))
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, values: torch.Tensor, module_residual: torch.Tensor | int) -> torch.Tensor:
        """执行时序卷积并加上模块级残差。"""
        residual = 0 if self.residual is None else self.residual(values)
        return self.relu(self.bn(self.conv(values)) + residual + module_residual)


class _TemporalBottleneckBlock(nn.Module):
    """带可选时间下采样的瓶颈时序模块。"""

    def __init__(
        self,
        channels: int,
        stride: int = 1,
        residual: bool = False,
        reduction: int = 4,
        get_residual: bool = False,
    ) -> None:
        """构建与检查点兼容的时序瓶颈路径。"""
        super().__init__()
        temporal_stride = False
        if get_residual:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (2, 1)),
                nn.BatchNorm2d(channels),
            )
            temporal_stride = True
        elif not residual:
            self.residual = None
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (2, 1)),
                nn.BatchNorm2d(channels),
            )
            temporal_stride = True
        if temporal_stride:
            stride = 2
        inner = channels // reduction
        self.conv_down = nn.Conv2d(channels, inner, 1)
        self.bn_down = nn.BatchNorm2d(inner)
        self.conv = nn.Conv2d(inner, inner, (9, 1), (stride, 1), (4, 0))
        self.bn = nn.BatchNorm2d(inner)
        self.conv_up = nn.Conv2d(inner, channels, 1)
        self.bn_up = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, values: torch.Tensor, module_residual: torch.Tensor | int) -> torch.Tensor:
        """执行缩减时序卷积，并加入两种残差贡献。"""
        residual = 0 if self.residual is None else self.residual(values)
        values = self.relu(self.bn_down(self.conv_down(values)))
        values = self.relu(self.bn(self.conv(values)))
        values = self.bn_up(self.conv_up(values))
        return self.relu(values + residual + module_residual)


class _ResGcnModule(nn.Module):
    """一个先空间后时序、带可学习边权的 ResGCN 阶段。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block: str,
        adjacency: torch.Tensor,
        *,
        stride: int = 1,
        reduction: int = 4,
        get_residual: bool = False,
    ) -> None:
        """选择模块类型，并配置阶段级/模块级残差路径。"""
        super().__init__()
        # 官方 GREW GaitGraph2 图保存四个跳数矩阵，但 ResGCN 模块配置为
        # max_graph_distance=2，只使用前三个。保留这一差异对检查点兼容性至关重要。
        classes = 3
        if block == "initial":
            module_residual, block_residual = False, False
            spatial_type = _SpatialBasicBlock
            temporal_type = _TemporalBasicBlock
        elif block == "Basic":
            module_residual, block_residual = True, False
            spatial_type = _SpatialBasicBlock
            temporal_type = _TemporalBasicBlock
        else:
            module_residual, block_residual = False, True
            spatial_type = _SpatialBottleneckBlock
            temporal_type = _TemporalBottleneckBlock

        if not module_residual:
            self.residual = None
        elif stride == 1 and in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        if spatial_type is _SpatialBottleneckBlock:
            self.scn = spatial_type(
                in_channels,
                out_channels,
                classes,
                block_residual,
                reduction,
            )
            self.tcn = temporal_type(
                out_channels,
                stride,
                block_residual,
                reduction,
                get_residual,
            )
        else:
            self.scn = spatial_type(in_channels, out_channels, classes, block_residual)
            self.tcn = temporal_type(out_channels, stride, block_residual)
        self.edge = nn.Parameter(torch.ones_like(adjacency))

    def forward(self, values: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """应用学习到的边缩放、图卷积和时序模块。"""
        module_residual: torch.Tensor | int = (
            0 if self.residual is None else self.residual(values)
        )
        adjacency = adjacency.to(device=values.device, dtype=values.dtype)
        return self.tcn(self.scn(values, adjacency * self.edge), module_residual)


class _InputBranch(nn.Module):
    """GaitGraph2 三个输入模态之一的茎部（stem）分支。"""

    def __init__(self, adjacency: torch.Tensor) -> None:
        """创建批归一化和检查点中的三个茎部阶段。"""
        super().__init__()
        self.register_buffer("A", adjacency.clone())
        channels = (5, 64, 64, 32)
        self.bn = nn.BatchNorm2d(channels[0])
        self.layers = nn.ModuleList(
            [
                _ResGcnModule(channels[0], channels[1], "initial", adjacency),
                _ResGcnModule(channels[1], channels[2], "Bottleneck", adjacency),
                _ResGcnModule(channels[2], channels[3], "Bottleneck", adjacency),
            ]
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """归一化一个模态，并让它经过对应的 ResGCN 茎部。"""
        values = self.bn(values)
        for layer in self.layers:
            values = layer(values, self.A)
        return values


class _ResGcn(nn.Module):
    """完整的三分支 ResGCN 主干和 128 维投影头。"""

    def __init__(self, adjacency: torch.Tensor) -> None:
        """构建模态茎部、主干阶段、池化和投影层。"""
        super().__init__()
        self.graph = adjacency
        self.head = nn.ModuleList(_InputBranch(adjacency) for _ in range(3))
        channels = (32, 128, 128, 128, 256, 256, 256)
        layers: list[nn.Module] = []
        for index in range(len(channels) - 1):
            in_channels = channels[index] * 3 if index == 0 else channels[index]
            out_channels = channels[index + 1]
            stride = 1 if index == 0 or in_channels == out_channels else 2
            layers.append(
                _ResGcnModule(
                    in_channels,
                    out_channels,
                    "Bottleneck",
                    adjacency,
                    stride=stride,
                    get_residual=index == 0,
                )
            )
        self.backbone = nn.ModuleList(layers)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fcn = nn.Linear(256, 128)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """拼接分支特征，运行主干并返回嵌入。"""
        values = torch.cat(
            [branch(values[:, index]) for index, branch in enumerate(self.head)],
            dim=1,
        )
        for layer in self.backbone:
            values = layer(values, self.graph)
        values = self.global_pooling(values).squeeze(-1).squeeze(-1)
        return self.fcn(values)


class GaitGraph2Encoder(nn.Module):
    """仅用于推理、键名与检查点兼容的 GaitGraph2/ResGCN 编码器。

    内部会将便于表达多输入的 ``N,T,V,I,C`` 张量转换为图卷积所需的
    ``N,I,C,T,V``。
    """

    def __init__(self) -> None:
        """创建固定 COCO 图和与检查点兼容的 ResGCN。"""
        super().__init__()
        adjacency = torch.from_numpy(_coco_adjacency(3))
        self.ResGCN = _ResGcn(adjacency)

    def forward(self, transformed: torch.Tensor) -> torch.Tensor:
        """将多输入张量换轴为 ResGCN 布局并进行编码。"""
        # 输入为 N,T,V,I,C；ResGCN 需要 N,I,C,T,V。
        values = transformed.permute(0, 3, 4, 1, 2).contiguous()
        return self.ResGCN(values)


def _fill_missing(sequence: np.ndarray) -> np.ndarray:
    """按 OpenGait ``NormalizeEmpty`` 规则填补空关节。"""
    result = np.asarray(sequence, dtype=np.float32).copy()
    for frame in result:
        empty = frame[:, 0] == 0.0
        if not empty.any():
            continue
        # OpenGait computes the frame center over all 17 rows (including the
        # zero rows) and only then replaces the empty coordinates. Matching
        # that detail keeps missing-joint handling checkpoint-compatible.
        center = frame.mean(axis=0)
        frame[empty, 0] = center[0]
        frame[empty, 1] = center[1]
        frame[empty, 2] = 0.0
    return result


def _fixed_length(sequence: np.ndarray, length: int) -> np.ndarray:
    """将姿态窗口重采样为模型要求的固定时序长度。"""
    if len(sequence) == length:
        return sequence
    indexes = np.rint(np.linspace(0, len(sequence) - 1, length)).astype(np.int64)
    return sequence[indexes]


def gait_graph_multi_input(sequence: np.ndarray) -> np.ndarray:
    """将姿态转换为关节、速度和骨骼通道。

    输出的五个通道编码原始坐标、相对坐标、时序运动、骨骼方向和置信度，
    这些内容由转换后的 GaitGraph2 检查点使用。
    """

    values = _fill_missing(sequence)
    time_steps, joints, channels = values.shape
    output = np.zeros((time_steps, joints, 3, channels + 2), dtype=np.float32)
    output[:, :, 0, :channels] = values
    output[:, :, 0, channels:] = values[:, :, :2] - values[:, :1, :2]
    if time_steps > 2:
        output[:-2, :, 1, :2] = values[1:-1, :, :2] - values[:-2, :, :2]
        output[:-2, :, 1, 3:] = values[2:, :, :2] - values[:-2, :, :2]
    output[:, :, 1, 3] = values[:, :, 2]
    output[:, :, 2, :2] = values[:, :, :2] - values[:, _COCO_CONNECT, :2]
    bone_length = np.sqrt(np.square(output[:, :, 2, :2]).sum(axis=-1)) + 1e-4
    for channel in range(2):
        ratio = output[:, :, 2, channel] / bone_length
        output[:, :, 2, channels + channel] = np.arccos(np.clip(ratio, -1.0, 1.0))
    output[:, :, 2, 3] = values[:, :, 2]
    return output


class TemporalGaitEncoder:
    """加载 GREW 检查点并编码滚动姿态序列。

    ``encode`` 将时序重采样、缺失关键点处理、可选测试时增强、设备放置和
    L2 归一化隐藏在一个小接口后，该接口由生产视觉适配器使用。
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device | None = None,
        sequence_length: int = 60,
        use_tta: bool = True,
    ) -> None:
        """加载仅含张量的检查点，并准备推理。"""
        self.path = Path(checkpoint_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"GaitGraph2 checkpoint not found: {self.path}")
        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.sequence_length = max(25, int(sequence_length))
        self.use_tta = bool(use_tta)
        self.model = GaitGraph2Encoder().to(self.device)
        state = torch.load(self.path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @property
    def output_dimension(self) -> int:
        """返回暴露给 ``FeatureBundle`` 的嵌入维度。"""
        return 384 if self.use_tta else 128

    def encode(self, poses: Sequence[np.ndarray] | np.ndarray) -> np.ndarray | None:
        """编码一个姿态窗口；窗口不可用时返回 ``None``。"""
        return self.encode_batch([poses])[0]

    def encode_batch(
        self,
        pose_sequences: Sequence[Sequence[np.ndarray] | np.ndarray],
    ) -> list[np.ndarray | None]:
        """一次编码多个姿态窗口，并保持结果与输入顺序对齐。

        无效窗口在对应位置返回 ``None``，其余窗口的原始、时间反转和水平翻转
        变体会合并为一个模型批次。这样生产适配器无需了解 TTA 分组、设备传输
        和输出归一化细节，也不会为每个 Track 单独启动一次 GPU 前向。
        """

        output: list[np.ndarray | None] = [None] * len(pose_sequences)
        variants: list[np.ndarray] = []
        valid_indexes: list[int] = []
        for index, poses in enumerate(pose_sequences):
            try:
                sequence = np.asarray(poses, dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if (
                sequence.ndim != 3
                or sequence.shape[1:] != (17, 3)
                or len(sequence) < 25
            ):
                continue
            sequence = _fixed_length(sequence, self.sequence_length)
            variants.append(gait_graph_multi_input(sequence))
            if self.use_tta:
                variants.append(gait_graph_multi_input(sequence[::-1].copy()))
                variants.append(
                    gait_graph_multi_input(sequence[:, _COCO_FLIP].copy())
                )
            valid_indexes.append(index)
        if not variants:
            return output

        batch = torch.from_numpy(np.stack(variants)).to(self.device)
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    embeddings = self.model(batch)
            else:
                embeddings = self.model(batch)
        grouped = embeddings.float().cpu().numpy().reshape(len(valid_indexes), -1)
        for destination, vector in zip(valid_indexes, grouped):
            norm = float(np.linalg.norm(vector))
            if norm > 1e-8:
                output[destination] = (vector / norm).astype(np.float32)
        return output


__all__ = [
    "GaitGraph2Encoder",
    "TemporalGaitEncoder",
    "gait_graph_multi_input",
]
