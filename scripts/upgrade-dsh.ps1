<#
.SYNOPSIS
  dsh-prompt-optimizer 0.6 的前置：在 DSH 未运行时把宿主 dsh 升到指定版本。

.DESCRIPTION
  为什么必须"先卸载、再安装"，而不能直接 `npm install -g @deepseek-ai/dsh@<新版本>`：
  运行中的 dsh 已经加载了安装树里的原生模块（koffi.node、sharp 的 dll 等），
  Windows 不允许替换被映射的文件 → 就地升级会报 EBUSY/EPERM 并留下**混合版本**的安装树。
  因此顺序被钉死为：退出 DSH → 卸载 → 安装 → 校验。

  本脚本只做文件层的事：不启动服务、不改任何 profile、不碰 po06 插件本体。

.PARAMETER Version
  目标版本。默认 0.1.7-rc.1（插件 v0.6.10 对齐的宿主版本）。

.PARAMETER Preferred
  已存在目标版本时仍然强制重装（真机踩过：npm 缓存会让人以为装上了新版本）。

.EXAMPLE
  pwsh -File .\scripts\upgrade-dsh.ps1
  pwsh -File .\scripts\upgrade-dsh.ps1 -Version 0.1.7-rc.1 -Preferred
#>
[CmdletBinding()]
param(
  [string]$Version = '0.1.7-rc.1',
  [string]$Package = '@deepseek-ai/dsh',
  [switch]$Preferred
)

$ErrorActionPreference = 'Stop'

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "  OK   $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  WARN $msg" -ForegroundColor Yellow }
function Die($msg)  { Write-Host "  FAIL $msg" -ForegroundColor Red; exit 1 }

Step "0/5 前置检查：确认没有 dsh 正在运行"
# 判据必须是"真的在跑 dsh 宿主"，不能只匹配命令行里出现 "dsh" 字样：
# 文件路径里带 "dsh"（例如本脚本所在目录）会让任何无关进程变成误报，从而挡住升级。
$dshLaunch = '(@deepseek-ai[\\/]dsh[\\/]lib[\\/]bin\.js)|([\\/]dsh\.(cmd|ps1))|(@deepseek-ai[\\/]dsh[\\/]node_modules)'
$procs = Get-CimInstance Win32_Process -Filter "Name = 'node.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -and $_.CommandLine -match $dshLaunch }
if ($procs) {
  Write-Host "  仍在运行的 dsh 进程：" -ForegroundColor Yellow
  $procs | ForEach-Object { Write-Host ("    pid {0}  {1}" -f $_.ProcessId, $_.CommandLine) }
  Die "请先关闭 DSH（包括这个 Web GUI 和任何 dsh 终端），然后在一个**普通的 PowerShell 窗口**里重跑本脚本。"
}
Ok "没有 dsh 进程在跑"

Step "1/5 记录升级前状态"
$before = (& dsh --version 2>&1 | Out-String).Trim()
Ok "当前 dsh：$before"
$prefix = (& npm config get prefix 2>&1 | Out-String).Trim()
$installDir = Join-Path $prefix "node_modules\deepseek-ai"
Write-Host "  npm prefix : $prefix"
Write-Host "  安装目录   : $installDir"
if (-not (Test-Path $installDir)) { Warn "安装目录不存在，卸载步骤将被跳过（可能是首次安装）" }

Step "2/5 卸载旧版"
& npm uninstall -g $Package
$uninstallCode = $LASTEXITCODE
if ($uninstallCode -ne 0) { Warn "npm uninstall 退出码 $uninstallCode（若本来就只装了半份，这是正常的）" }

# 卸载后残留的目录会让新版本"看起来装上了、实际是旧的"——显式核一次。
if (Test-Path $installDir) {
  Warn "卸载后仍有残留：$installDir"
  Write-Host "  正在移除残留（这是就地升级留下混合版本的直接原因）" -ForegroundColor Yellow
  Remove-Item -Recurse -Force $installDir
}
Ok "旧版已清除"

Step "3/5 安装 $Package@$Version"
# --prefer-online：npm 本地缓存的 packument 过期会导致 "No matching version found"，
# 而该版本在 registry 上其实存在（作者真机踩过，见其 DSH-COMPAT 6.1）。
$installArgs = @('install', '-g', "$Package@$Version", '--prefer-online')
if ($Preferred) { $installArgs += '--force' }
& npm @installArgs
if ($LASTEXITCODE -ne 0) { Die "npm install 失败（退出码 $LASTEXITCODE）" }

Step "4/5 校验宿主版本"
$after = (& dsh --version 2>&1 | Out-String).Trim()
Write-Host "  升级后 dsh：$after"
if ($after -notlike "*$Version*") {
  Die "版本校验不通过：期望 $Version，实际 $after。不要继续，先查 npm 缓存与 registry。"
}
Ok "宿主已到 $Version（升级前是 $before）"

Step "5/5 校验 web profile 仍能装配（含 po06 插件）"
$dump = (& dsh --profile web --dump-config 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0) {
  Warn "web profile 组合失败，下面是原始输出："
  Write-Host $dump
  Die "profile 没组合起来 —— 插件装了但不生效，先解决这个再启动。"
}
if ($dump -notmatch [regex]::Escape('@dsh-external/dsh-po06')) {
  Warn "web profile 的组合结果里没有 @dsh-external/dsh-po06 —— 插件层掉了，检查 profile 的 package.json"
} else {
  Ok "web profile 组合结果里仍有 @dsh-external/dsh-po06（插件层在）"
}
if ($dump -match [regex]::Escape('@dsh-external/dsh-prompt-optimizer')) {
  Warn "同一 profile 里还有 0.5.x 旧插件 → 0.6 会以 DOUBLE_INTERCEPT 拒绝启用"
} else {
  Ok "没有 0.5.x 旧插件，不会撞双重拦截"
}

Write-Host @"

完成。现在启动：

    dsh --profile web

启动后把输入栏那排控件里的「档位」从 「关闭」 拨到 「标准」 或 「重度」 才会拦截（默认 off）。
另外，0.6 自己的启用配置在 $env:USERPROFILE\.dsh\po06.json —— 面板改档位会写它。
"@ -ForegroundColor Green
