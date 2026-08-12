[CmdletBinding()]
# 仅在 Python doctor 验证模型存在且哈希正确后打包清单列出的模型资产。归档先
# 暂存到系统临时目录，避免中断运行在仓库内留下半写入的模型包。
param(
    [string]$Output = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
# 一次性解析路径，并让后续文件操作始终限定在本项目和明确指定的输出归档内。
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modelDirectory = Join-Path $projectRoot "models"
$manifestPath = Join-Path $modelDirectory "manifest.json"

if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $projectRoot "dist\model-bundle-v1.zip"
}
$outputPath = [System.IO.Path]::GetFullPath($Output)

if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "找不到模型清单：$manifestPath"
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$modelEntries = @($manifest.models.PSObject.Properties)
if ($modelEntries.Count -eq 0) {
    throw "模型清单为空：$manifestPath"
}

# Python doctor 同时理解直接文件哈希和转换 checkpoint 使用的语义张量指纹。优先
# 使用项目虚拟环境，避免全局 Python 因缺包给出误导性报告。
$pythonCommand = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonCommand -PathType Leaf)) {
    $pythonCommand = (Get-Command python -ErrorAction Stop).Source
}
& $pythonCommand -m cross_event_verifier doctor
if ($LASTEXITCODE -ne 0) {
    throw "模型自检失败，停止打包"
}

foreach ($entry in $modelEntries) {
    # 清单名称按设计只能是文件名；拒绝路径组件可避免编辑清单时意外复制任意文件。
    $name = [System.IO.Path]::GetFileName([string]$entry.Name)
    if ($name -ne [string]$entry.Name) {
        throw "模型清单包含非法文件名：$($entry.Name)"
    }
    $path = Join-Path $modelDirectory $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "缺少模型文件：$path"
    }
    $expected = $entry.Value
    if ($null -ne $expected.tensor_sha256) {
        continue
    }
    $actual = Get-FileHash -LiteralPath $path -Algorithm SHA256
    $size = (Get-Item -LiteralPath $path).Length
    if ($size -ne [int64]$expected.bytes -or $actual.Hash.ToLowerInvariant() -ne ([string]$expected.sha256).ToLowerInvariant()) {
        throw "模型校验失败：$name"
    }
}

if ((Test-Path -LiteralPath $outputPath -PathType Leaf) -and -not $Force) {
    throw "输出已存在：$outputPath；如需覆盖请显式添加 -Force"
}
$parent = Split-Path -Parent $outputPath
New-Item -ItemType Directory -Path $parent -Force | Out-Null

$staging = Join-Path ([System.IO.Path]::GetTempPath()) ("cross-event-models-" + [guid]::NewGuid().ToString("N"))
try {
    # 复制到干净的 models/ 树，使生成的 zip 保持同步脚本和运行时 doctor 所需布局。
    $stagedModels = Join-Path $staging "models"
    New-Item -ItemType Directory -Path $stagedModels -Force | Out-Null
    Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $stagedModels "manifest.json")
    Copy-Item -LiteralPath (Join-Path $modelDirectory "README.md") -Destination (Join-Path $stagedModels "README.md")
    foreach ($entry in $modelEntries) {
        Copy-Item -LiteralPath (Join-Path $modelDirectory ([string]$entry.Name)) -Destination $stagedModels
    }
    Compress-Archive -Path (Join-Path $staging "models") -DestinationPath $outputPath -CompressionLevel Optimal -Force
    $archiveHash = Get-FileHash -LiteralPath $outputPath -Algorithm SHA256
    $archiveSize = (Get-Item -LiteralPath $outputPath).Length
    [pscustomobject]@{
        output = $outputPath
        bytes = $archiveSize
        sha256 = $archiveHash.Hash.ToLowerInvariant()
        files = $modelEntries.Count
    } | ConvertTo-Json
}
finally {
    if (Test-Path -LiteralPath $staging) {
        Remove-Item -LiteralPath $staging -Recurse -Force
    }
}
