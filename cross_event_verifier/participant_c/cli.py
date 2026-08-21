"""独立验证器和桌面 GUI 的命令行入口。

CLI 有意保持命令分发简单：``demo`` 运行领域引擎，``doctor`` 检查生产就绪状态，
``download-models`` 引导外部权重，``gui`` 则委托给 Tk 应用。
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ..participant_a.engine import CrossEventVerifier
from ..types import FeatureBundle, Observation, TrackQuality


def _good_quality() -> TrackQuality:
    """为冒烟演示构造确定性的高质量证据。"""
    return TrackQuality(
        detection_confidence=0.92,
        box_height=180,
        sharpness=0.90,
        occlusion=0.05,
        keypoint_visibility=0.88,
        contour_area=2400,
        frame_count=24,
        gait_cycles=2,
        walking_ratio=0.92,
    )


def run_demo() -> int:
    """运行一次小型内存验证，并打印 JSON 决策。"""
    verifier = CrossEventVerifier()
    appearance = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    gait = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    verifier.register_identity("P1", FeatureBundle(appearance, gait), metadata={"label": "demo"})
    decision = verifier.verify(
        Observation(
            event_id="demo-event",
            camera_id="cam-a",
            capture_session_id="session-a",
            track_id="track-1",
            features=FeatureBundle(
                appearance=appearance + np.array([0.0, 0.05, 0.0, 0.0]),
                gait=gait + np.array([0.0, 0.0, 0.05, 0.0]),
            ),
            quality=_good_quality(),
        )
    )
    print(
        json.dumps(
            {
                "kind": decision.kind.value,
                "state": decision.state.value,
                "identity_id": decision.identity_id,
                "score": decision.score,
                "margin": decision.margin,
                "reasons": decision.reasons,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    verifier.close()
    return 0


def run_doctor() -> int:
    """不打开 GUI，打印生产模型、CUDA 和 ONNX 就绪状态。"""

    report: dict[str, object] = {}
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except ImportError as error:
        report["torch_error"] = str(error)
    try:
        import onnxruntime as ort

        report["onnxruntime"] = ort.__version__
        report["onnx_providers"] = ort.get_available_providers()
    except ImportError as error:
        report["onnxruntime_error"] = str(error)
    try:
        from ..participant_b.production_vision import (
            ProductionVisionConfig,
            production_readiness,
        )

        config = ProductionVisionConfig()
        ready, issues = production_readiness(config, verify_hashes=True)
        report["production_ready"] = ready
        report["issues"] = issues
        report["models"] = {
            path.name: path.stat().st_size if path.is_file() else None
            for path in config.required_files
        }
    except Exception as error:
        report["production_ready"] = False
        report["issues"] = [str(error)]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if bool(report.get("production_ready")) else 1


def run_download_models(names: list[str] | None, force: bool) -> int:
    """在安装 Python 依赖后引导模型二进制文件。"""

    from .model_assets import ModelAssetError, download_models

    try:
        paths = download_models(names=names, force=force)
    except ModelAssetError as error:
        print(f"模型下载失败：{error}")
        return 1
    except Exception as error:  # 网络/Torch 错误应转换为 CLI 消息
        print(f"模型下载失败：{error}")
        return 1
    print(f"已准备 {len(paths)} 个模型文件。")
    return 0


def main(argv: list[str] | None = None) -> int:
    """解析一条 CLI 命令，并返回进程风格的退出码。"""
    parser = argparse.ArgumentParser(description="Cross-event prototype verifier")
    parser.add_argument(
        "command",
        choices=["demo", "gui", "doctor", "download-models"],
        nargs="?",
        default="demo",
    )
    parser.add_argument(
        "--database",
        default="data/verifier-production-v1.sqlite3",
        help=(
            "SQLite file used by the GUI "
            "(default: data/verifier-production-v1.sqlite3)"
        ),
    )
    parser.add_argument(
        "--vision-backend",
        choices=["production", "auto", "demo"],
        default="production",
        help="GUI vision backend; production is the safe default",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="download only this manifest model; may be repeated",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="redownload model files after manifest verification fails",
    )
    args = parser.parse_args(argv)
    if args.command == "demo":
        return run_demo()
    if args.command == "doctor":
        return run_doctor()
    if args.command == "download-models":
        return run_download_models(args.models, args.force)
    if args.command == "gui":
        try:
            from .gui import launch_gui
        except ImportError as error:
            parser.error(
                "GUI requires opencv-python and the Python Tk runtime; "
                f"details: {error}"
            )
        try:
            return launch_gui(args.database, args.vision_backend)
        except RuntimeError as error:
            parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
