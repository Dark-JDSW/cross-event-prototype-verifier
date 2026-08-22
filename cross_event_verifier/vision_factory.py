"""构造具体视觉适配器，同时不把初始化细节泄漏到 GUI。

``production`` 模式在条件不满足时直接失败，``demo`` 明确只用于诊断，
``auto`` 可以回退到 demo，但会关闭自动注册。这样的选择策略可避免低精度
检测器悄悄向正式身份图库写入内容。
"""

from __future__ import annotations

from typing import Callable

from .vision import OpenCvDemoAdapter, VisionAdapter


def build_vision_adapter(
    backend: str = "production",
    *,
    preload: bool = False,
    on_stage: Callable[[str, float], None] | None = None,
) -> VisionAdapter:
    """构造请求的后端，并可选执行分阶段模型加载与 Dummy warmup。"""

    def notify(text: str, progress: float) -> None:
        if on_stage is not None:
            on_stage(text, progress)

    selected = backend.strip().lower()
    if selected not in {"production", "demo", "auto"}:
        raise ValueError(f"unsupported vision backend: {backend}")
    if selected == "demo":
        notify("诊断后端已就绪", 1.0)
        return OpenCvDemoAdapter(detection_stride=1)

    from .production_vision import (
        ProductionVisionAdapter,
        ProductionVisionError,
        production_readiness,
    )

    notify("正在检查运行环境与模型资产…", 0.04)
    ready, issues = production_readiness()
    if ready:
        notify("正在校验模型清单与运行协议…", 0.08)
        adapter = ProductionVisionAdapter()
        if preload:
            adapter.preload(on_stage=on_stage)
            adapter.warmup(on_stage=on_stage)
            notify("生产视觉后端已就绪", 1.0)
        return adapter
    if selected == "auto":
        notify("生产后端不可用，切换诊断后端", 1.0)
        adapter = OpenCvDemoAdapter(detection_stride=1)
        adapter.startup_warning = (
            "生产视觉后端不可用，已进入仅供诊断的 HOG 模式；自动注册已关闭。"
            + "；".join(issues)
        )
        return adapter
    raise ProductionVisionError("；".join(issues))


__all__ = ["build_vision_adapter"]
