"""仅用于推理的 OSNet-AIN x1.0 外观编码器。

这是 MIT 许可的 deep-person-reid OSNet-AIN 模型的精简适配。训练、数据集、
评测以及 Cython 排序扩展被有意排除；项目只需要与检查点兼容的特征推理。
第三方归属记录在 ``THIRD_PARTY_NOTICES.md`` 中。
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class _ConvLayer(nn.Module):
    """OSNet 主干使用的卷积、归一化和激活模块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        instance_norm: bool = False,
    ) -> None:
        """创建一个可配置的空间卷积模块。"""
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = (
            nn.InstanceNorm2d(out_channels, affine=True)
            if instance_norm
            else nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """执行卷积、归一化和 ReLU 激活。"""
        return self.relu(self.bn(self.conv(values)))


class _Conv1x1(nn.Module):
    """带批归一化和 ReLU 的逐点投影。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """创建与检查点兼容的 1x1 投影。"""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """投影通道宽度，但不改变空间分辨率。"""
        return self.relu(self.bn(self.conv(values)))


class _Conv1x1Linear(nn.Module):
    """激活有意保持线性的逐点投影。"""

    def __init__(self, in_channels: int, out_channels: int, *, batch_norm: bool = True) -> None:
        """创建带可选批归一化的投影。"""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """执行投影和可选归一化。"""
        values = self.conv(values)
        return self.bn(values) if self.bn is not None else values


class _LightConv3x3(nn.Module):
    """OSNet 多尺度分支使用的深度可分离 3x3 卷积。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """创建逐点扩展以及后续的深度空间滤波。"""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1,
            bias=False,
            groups=out_channels,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """执行可分离卷积、归一化和激活。"""
        return self.relu(self.bn(self.conv2(self.conv1(values))))


class _LightConvStream(nn.Module):
    """构成一个 OSNet 全尺度分支的多深度顺序堆叠。"""

    def __init__(self, in_channels: int, out_channels: int, depth: int) -> None:
        """创建包含指定数量轻量卷积层的分支。"""
        super().__init__()
        layers: list[nn.Module] = [_LightConv3x3(in_channels, out_channels)]
        layers.extend(_LightConv3x3(out_channels, out_channels) for _ in range(depth - 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """计算该分支中的全部层。"""
        return self.layers(values)


class _ChannelGate(nn.Module):
    """类似 Squeeze-and-Excitation 的门控，根据全局上下文重新加权通道。"""

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        """创建两层通道注意力瓶颈。"""
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=True)
        self.norm1 = None
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=True)
        self.gate_activation = nn.Sigmoid()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """计算通道门控值并乘到特征图上。"""
        gates = self.global_avgpool(values)
        gates = self.relu(self.fc1(gates))
        gates = self.gate_activation(self.fc2(gates))
        return values * gates


class _OsBlock(nn.Module):
    """包含四个感受野分支和残差路径的 OSNet 全尺度模块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        instance_inside: bool,
    ) -> None:
        """构建分支、通道门控、投影和可选实例归一化。"""
        super().__init__()
        middle = out_channels // 4
        self.conv1 = _Conv1x1(in_channels, middle)
        self.conv2 = nn.ModuleList(
            _LightConvStream(middle, middle, depth) for depth in range(1, 5)
        )
        self.gate = _ChannelGate(middle)
        self.conv3 = _Conv1x1Linear(
            middle,
            out_channels,
            batch_norm=not instance_inside,
        )
        self.downsample = (
            _Conv1x1Linear(in_channels, out_channels)
            if in_channels != out_channels
            else None
        )
        self.IN = (
            nn.InstanceNorm2d(out_channels, affine=True)
            if instance_inside
            else None
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """汇聚经过门控的分支，并加上投影后的残差。"""
        identity = values
        stem = self.conv1(values)
        mixed: torch.Tensor | int = 0
        for stream in self.conv2:
            mixed = mixed + self.gate(stream(stem))
        output = self.conv3(mixed)
        if self.IN is not None:
            output = self.IN(output)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(output + identity)


class OSBlock(_OsBlock):
    """使用批归一化输出特征的 OS 模块变体。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """创建标准 OSNet 模块变体。"""
        super().__init__(in_channels, out_channels, instance_inside=False)


class OSBlockINin(_OsBlock):
    """在模块内部使用实例归一化的 OS 模块变体。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """创建使用实例归一化的 OSNet-AIN 模块变体。"""
        super().__init__(in_channels, out_channels, instance_inside=True)


class OSNetAin(nn.Module):
    """与检查点兼容、输出 512 维嵌入的 OSNet-AIN x1.0。"""

    def __init__(self, num_classes: int = 2510) -> None:
        """构建 x1.0 主干以及与分类器检查点兼容的头部。"""
        super().__init__()
        self.feature_dim = 512
        self.conv1 = _ConvLayer(
            3,
            64,
            7,
            stride=2,
            padding=3,
            instance_norm=True,
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Sequential(OSBlockINin(64, 256), OSBlockINin(256, 256))
        self.pool2 = nn.Sequential(_Conv1x1(256, 256), nn.AvgPool2d(2, stride=2))
        self.conv3 = nn.Sequential(OSBlock(256, 384), OSBlockINin(384, 384))
        self.pool3 = nn.Sequential(_Conv1x1(384, 384), nn.AvgPool2d(2, stride=2))
        self.conv4 = nn.Sequential(OSBlockINin(384, 512), OSBlock(512, 512))
        self.conv5 = _Conv1x1(512, 512)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU())
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """让外观裁剪经过主干，并返回一个 512 维向量。"""
        values = self.maxpool(self.conv1(values))
        values = self.pool2(self.conv2(values))
        values = self.pool3(self.conv3(values))
        values = self.conv5(self.conv4(values))
        values = self.global_avgpool(values).flatten(1)
        return self.fc(values)


def load_osnet_ain(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> tuple[OSNetAin, torch.device]:
    """加载仅含张量的检查点，并返回评估模式下的模型/设备对。

    严格加载前会去掉 ``module.`` 等训练前缀，使不匹配的检查点在启动时直
    接失败，而不是静默地产生无效外观向量。
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"OSNet-AIN checkpoint not found: {path}")
    selected_device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    clean_state = {
        (name[7:] if name.startswith("module.") else name): tensor
        for name, tensor in state.items()
    }
    classifier = clean_state.get("classifier.weight")
    classes = int(classifier.shape[0]) if classifier is not None else 2510
    model = OSNetAin(num_classes=classes)
    model.load_state_dict(clean_state, strict=True)
    model.to(selected_device).eval()
    return model, selected_device


__all__ = ["OSNetAin", "load_osnet_ain"]
