[CmdletBinding()]
# 只下载本地清单中列出的文件。每次下载都写入同级临时文件，校验后再原子移动
# 到 models/；这样传输失败不会替换已知正确的 checkpoint。
param(
    [string]$BaseUrl = "",
    [string]$ModelDirectory = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
# 基础 URL 有意通过外部方式提供，使凭据或签名 URL 永远不会进入源代码管理或
# SQLite 审计数据库。
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($BaseUrl)) {
    $BaseUrl = $env:CROSS_EVENT_MODEL_BASE_URL
}
if ([string]::IsNullOrWhiteSpace($BaseUrl)) {
    throw "请提供 -BaseUrl，或设置环境变量 CROSS_EVENT_MODEL_BASE_URL"
}
if ([string]::IsNullOrWhiteSpace($ModelDirectory)) {
    $ModelDirectory = Join-Path $projectRoot "models"
}
$ModelDirectory = (Resolve-Path $ModelDirectory).Path
$manifestPath = Join-Path $ModelDirectory "manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "找不到本地模型清单：$manifestPath"
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$modelEntries = @($manifest.models.PSObject.Properties)
$BaseUrl = $BaseUrl.TrimEnd('/')
$pythonCommand = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonCommand -PathType Leaf)) {
    $pythonCommand = (Get-Command python -ErrorAction Stop).Source
}
$tensorFingerprintProbe = @'
from pathlib import Path
import sys
from cross_event_verifier.model_assets import tensor_state_fingerprint
print(tensor_state_fingerprint(Path(sys.argv[1])))
'@

function Get-VerifiedModelHash {
    param(
        [string]$Path,
        [object]$Expected,
        [string]$Name
    )
    if ($null -ne $Expected.tensor_sha256) {
        # 转换后的 checkpoint 根据张量内容进行语义校验，而不是根据 pickle 容器
        # 字节校验，因此无害的序列化变化不会使部署契约失效。
        $actual = (& $pythonCommand -c $tensorFingerprintProbe $Path | Out-String).Trim()
        if ($LASTEXITCODE -ne 0 -or $actual.ToLowerInvariant() -ne ([string]$Expected.tensor_sha256).ToLowerInvariant()) {
            throw "模型 tensor 指纹不符：$Name"
        }
        return
    }
    $downloaded = Get-FileHash -LiteralPath $Path -Algorithm SHA256
    $downloadedSize = (Get-Item -LiteralPath $Path).Length
    if ($downloadedSize -ne [int64]$Expected.bytes -or $downloaded.Hash.ToLowerInvariant() -ne ([string]$Expected.sha256).ToLowerInvariant()) {
        throw "远端模型校验失败：$Name"
    }
}

foreach ($entry in $modelEntries) {
    # 构造目标路径前拒绝路径穿越。
    $name = [System.IO.Path]::GetFileName([string]$entry.Name)
    if ($name -ne [string]$entry.Name) {
        throw "模型清单包含非法文件名：$($entry.Name)"
    }
    $target = Join-Path $ModelDirectory $name
    $expected = $entry.Value
    if (Test-Path -LiteralPath $target -PathType Leaf) {
        try {
            Get-VerifiedModelHash -Path $target -Expected $expected -Name $name
            Write-Host "已存在且校验通过：$name"
            continue
        }
        catch {
            if (-not $Force) {
                throw "本地模型已存在但校验失败：$name；如需重新下载请显式添加 -Force"
            }
        }
    }

    $temporary = Join-Path $ModelDirectory ("." + $name + ".download")
    if (Test-Path -LiteralPath $temporary) {
        Remove-Item -LiteralPath $temporary -Force
    }
    try {
        $uri = $BaseUrl + "/" + [uri]::EscapeDataString($name)
        Write-Host "下载：$uri"
        Invoke-WebRequest -Uri $uri -OutFile $temporary -UseBasicParsing
        Get-VerifiedModelHash -Path $temporary -Expected $expected -Name $name
        Move-Item -LiteralPath $temporary -Destination $target -Force
        Write-Host "完成：$name"
    }
    finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
}

Write-Host "全部模型已同步并通过 manifest 校验。"
