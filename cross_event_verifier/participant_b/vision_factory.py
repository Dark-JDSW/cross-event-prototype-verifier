"""构造具体视觉适配器，同时不把初始化细节泄漏到 GUI。

``production`` 模式在条件不满足时直接失败，``demo`` 明确只用于诊断，
``auto`` 可以回退到 demo，但会关闭自动注册。这样的选择策略可避免低精度
检测器悄悄向正式身份图库写入内容。
"""

from __future__ import annotations

from .vision import OpenCvDemoAdapter, VisionAdapter


def build_vision_adapter(backend: str = "production") -> VisionAdapter:
    """构造请求的生产、诊断或安全自动回退后端。"""

    selected = backend.strip().lower()
    if selected not in {"production", "demo", "auto"}:
        raise ValueError(f"unsupported vision backend: {backend}")
    if selected == "demo":
        return OpenCvDemoAdapter(detection_stride=1)

    from .production_vision import (
        ProductionVisionAdapter,
        ProductionVisionError,
        production_readiness,
    )

    ready, issues = production_readiness()
    if ready:
        return ProductionVisionAdapter()
    if selected == "auto":
        adapter = OpenCvDemoAdapter(detection_stride=1)
        adapter.startup_warning = (
            "生产视觉后端不可用，已进入仅供诊断的 HOG 模式；自动注册已关闭。"
            + "；".join(issues)
        )
        return adapter
    raise ProductionVisionError("；".join(issues))


__all__ = ["build_vision_adapter"]
