"""在安装 Python 包后下载并校验生产模型资产。

Python 包依赖有意不下载模型二进制文件。本模块是克隆项目后的显式模型引导
命令。每个来源都会先计算哈希，之后才允许可选的 checkpoint 转换读取它。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIRECTORY = PROJECT_ROOT / "models"
MODEL_MANIFEST = MODEL_DIRECTORY / "manifest.json"


class ModelAssetError(RuntimeError):
    """模型来源或转换后资产校验失败时抛出。"""


@dataclass(frozen=True)
class ModelAsset:
    """描述模型来源及后处理校验信息的清单条目。"""

    name: str
    bytes: int | None
    sha256: str | None
    tensor_sha256: str | None
    url: str
    source_bytes: int
    source_sha256: str
    postprocess: str | None = None


def _read_manifest(path: Path = MODEL_MANIFEST) -> dict[str, Any]:
    """加载并进行最低限度校验 JSON 模型清单。"""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        models = document["models"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ModelAssetError(f"无法读取模型清单：{path}: {error}") from error
    if not isinstance(models, dict):
        raise ModelAssetError(f"模型清单格式错误：{path}")
    return document


def _assets(document: dict[str, Any]) -> tuple[ModelAsset, ...]:
    """将清单字典转换为可信且路径安全的资产记录。"""
    values: list[ModelAsset] = []
    for raw_name, raw_value in document["models"].items():
        name = Path(str(raw_name)).name
        if name != raw_name:
            raise ModelAssetError(f"模型清单包含非法文件名：{raw_name}")
        if not isinstance(raw_value, dict):
            raise ModelAssetError(f"模型条目不是对象：{name}")
        download = raw_value.get("download")
        if not isinstance(download, dict):
            raise ModelAssetError(f"模型没有 download 配置：{name}")
        url = str(download.get("url", "")).strip()
        if not url.lower().startswith("https://"):
            raise ModelAssetError(f"模型下载地址必须使用 HTTPS：{name}")
        try:
            values.append(
                ModelAsset(
                    name=name,
                    bytes=(
                        int(raw_value["bytes"])
                        if raw_value.get("bytes") is not None
                        else None
                    ),
                    sha256=(
                        str(raw_value["sha256"]).lower()
                        if raw_value.get("sha256")
                        else None
                    ),
                    tensor_sha256=(
                        str(raw_value["tensor_sha256"]).lower()
                        if raw_value.get("tensor_sha256")
                        else None
                    ),
                    url=url,
                    source_bytes=int(download.get("source_bytes", raw_value["bytes"])),
                    source_sha256=str(
                        download.get("source_sha256", raw_value["sha256"])
                    ).lower(),
                    postprocess=(
                        str(download["postprocess"])
                        if download.get("postprocess")
                        else None
                    ),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ModelAssetError(f"模型条目字段错误：{name}: {error}") from error
    return tuple(values)


def _hash_file(path: Path) -> tuple[int, str]:
    """返回按有限大小分块读取文件的字节数和 SHA-256。"""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _verify_file(path: Path, expected_bytes: int, expected_sha256: str, label: str) -> None:
    """当文件大小或摘要与清单不符时抛出 ``ModelAssetError``。"""
    if not path.is_file():
        raise ModelAssetError(f"文件不存在：{label}")
    size, digest = _hash_file(path)
    if size != expected_bytes or digest.lower() != expected_sha256.lower():
        raise ModelAssetError(
            f"{label} 校验失败：实际 bytes={size}, sha256={digest}; "
            f"期望 bytes={expected_bytes}, sha256={expected_sha256}"
        )


def tensor_state_fingerprint(path: Path) -> str:
    """独立于 pickle 布局，对张量名称、类型、形状和字节内容计算哈希。

    该函数有意忽略 pickle/容器顺序，但保留影响推理的每个张量值。因此转换后
    的运行时加载可以使用更安全的 ``weights_only=True`` 路径。
    """

    try:
        import torch
    except ImportError as error:  # pragma: no cover - 取决于可选安装依赖
        raise ModelAssetError("校验 tensor-only 模型需要安装 torch") from error
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise ModelAssetError(f"tensor-only 文件格式错误：{path}")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            raise ModelAssetError(f"模型包含非 Tensor 项：{path}:{name}")
        tensor = value.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _verify_artifact(path: Path, asset: ModelAsset, label: str) -> None:
    """校验直接二进制资产或转换后的张量 checkpoint。"""
    if asset.postprocess:
        if not asset.tensor_sha256:
            raise ModelAssetError(f"转换模型缺少 tensor_sha256：{asset.name}")
        actual = tensor_state_fingerprint(path)
        if actual.lower() != asset.tensor_sha256.lower():
            raise ModelAssetError(
                f"{label} tensor 指纹不符：实际 {actual}，期望 {asset.tensor_sha256}"
            )
        return
    if asset.bytes is None or asset.sha256 is None:
        raise ModelAssetError(f"模型缺少最终文件校验字段：{asset.name}")
    _verify_file(path, asset.bytes, asset.sha256, label)


def _download(url: str, destination: Path) -> None:
    """将一个 HTTPS 来源以流式方式写入临时目标。"""
    request = Request(
        url,
        headers={
            "User-Agent": "cross-event-prototype-verifier/model-bootstrap",
        },
    )
    with urlopen(request, timeout=120) as response, destination.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)


def _convert_checkpoint(source: Path, destination: Path, operation: str) -> None:
    """从可信上游 checkpoint 中提取请求的状态字典。"""
    try:
        import torch
    except ImportError as error:  # pragma: no cover - 取决于可选安装依赖
        raise ModelAssetError(
            f"模型 {operation} 需要先安装 production/CUDA 依赖中的 torch"
        ) from error

    # 本次调用前已经根据固定哈希校验来源。因此显式的不安全 pickle 模式仅限于
    # manifest.json 列出的可信上游 checkpoint；运行时推理使用 weights_only=True
    # 加载转换后的纯张量文件。
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        state = checkpoint
    elif operation == "strip_state_dict":
        state = checkpoint.get("state_dict", checkpoint)
    elif operation == "extract_model":
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    else:
        raise ModelAssetError(f"未知 checkpoint 转换操作：{operation}")
    torch.save(state if operation == "extract_model" else {"state_dict": state}, destination)


def _select_assets(
    assets: Iterable[ModelAsset], names: Iterable[str] | None,
) -> tuple[ModelAsset, ...]:
    """按文件名过滤清单条目，并拒绝未知选择。"""
    values = tuple(assets)
    if names is None:
        return values
    wanted = {Path(name).name for name in names}
    unknown = wanted - {item.name for item in values}
    if unknown:
        raise ModelAssetError(f"模型清单中不存在：{', '.join(sorted(unknown))}")
    return tuple(item for item in values if item.name in wanted)


def download_models(
    *,
    model_directory: Path = MODEL_DIRECTORY,
    names: Iterable[str] | None = None,
    force: bool = False,
) -> tuple[Path, ...]:
    """下载缺失的生产资产，并返回其本地路径。

    每个资产先下载到同级临时文件，依据来源哈希校验，可选转换后再次校验，
    最后才复制到模型目录。下载失败绝不会替换已有的正确文件。
    """

    manifest_path = model_directory / "manifest.json"
    document = _read_manifest(manifest_path)
    assets = _select_assets(_assets(document), names)
    model_directory.mkdir(parents=True, exist_ok=True)
    completed: list[Path] = []
    for asset in assets:
        target = model_directory / asset.name
        if target.exists() and not force:
            try:
                _verify_artifact(target, asset, asset.name)
            except ModelAssetError:
                raise ModelAssetError(
                    f"本地模型已存在但校验失败：{target}；如需重新下载请使用 --force"
                )
            print(f"已存在且校验通过：{asset.name}")
            completed.append(target)
            continue

        with tempfile.TemporaryDirectory(
            prefix="cross-event-model-", dir=model_directory
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            source = temporary_root / f"{asset.name}.source"
            converted = temporary_root / f"{asset.name}.converted"
            print(f"下载 {asset.name} ...")
            _download(asset.url, source)
            _verify_file(
                source,
                asset.source_bytes,
                asset.source_sha256,
                f"{asset.name} 源文件",
            )
            if asset.postprocess:
                _convert_checkpoint(source, converted, asset.postprocess)
                candidate = converted
            else:
                candidate = source
            _verify_artifact(candidate, asset, asset.name)
            shutil.copyfile(candidate, target)
        print(f"完成并校验：{asset.name}")
        completed.append(target)
    return tuple(completed)


__all__ = ["ModelAssetError", "download_models", "tensor_state_fingerprint"]
