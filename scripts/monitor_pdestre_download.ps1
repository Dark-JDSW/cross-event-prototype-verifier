param(
    [string]$JobId = "63d09ee1-a685-4e1e-a561-bed9bdca7156",
    [string]$LogPath = "F:\dataset\pdestre\reports\download_progress.log",
    [int]$IntervalSeconds = 60
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
$previousBytes = $null
$previousTime = $null

while ($true) {
    try {
        $job = Get-BitsTransfer -JobId $JobId
    } catch {
        Add-Content -LiteralPath $LogPath -Value "$(Get-Date -Format o) job-unavailable: $($_.Exception.Message)"
        break
    }
    $now = Get-Date
    $bytes = [int64]$job.BytesTransferred
    $total = [int64]$job.BytesTotal
    $rate = $null
    if ($null -ne $previousBytes -and $null -ne $previousTime) {
        $seconds = ($now - $previousTime).TotalSeconds
        if ($seconds -gt 0) {
            $rate = ($bytes - $previousBytes) / $seconds
        }
    }
    $percent = if ($total -gt 0) { 100.0 * $bytes / $total } else { 0.0 }
    $rateText = if ($null -ne $rate) {
        "rate_mib_s=$([Math]::Round($rate / 1MB, 3))"
    } else {
        "rate_mib_s=unknown"
    }
    Add-Content -LiteralPath $LogPath -Value "$(Get-Date -Format o) state=$($job.JobState) bytes=$bytes total=$total percent=$([Math]::Round($percent, 2)) $rateText"
    if ($job.JobState -in @("Transferred", "Error", "Cancelled")) {
        break
    }
    $previousBytes = $bytes
    $previousTime = $now
    Start-Sleep -Seconds $IntervalSeconds
}
