param(
    [string]$JobId = "63d09ee1-a685-4e1e-a561-bed9bdca7156",
    [string]$ArchivePath = "F:\dataset\pdestre\raw\dataset.tar",
    [string]$ExtractPath = "F:\dataset\pdestre\extracted",
    [string]$PythonExe = "C:\Users\wzige\PycharmProjects\cross-event-prototype-verifier\.venv\Scripts\python.exe",
    [string]$AuditScript = "C:\Users\wzige\PycharmProjects\cross-event-prototype-verifier\scripts\audit_external_walk_datasets.py"
)

$ErrorActionPreference = "Stop"
$reportDirectory = "F:\dataset\pdestre\reports"
New-Item -ItemType Directory -Force -Path $reportDirectory | Out-Null
$logPath = Join-Path $reportDirectory "finalize_download.log"

function Write-Log([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    $line | Tee-Object -FilePath $logPath -Append
}

try {
    while ($true) {
        $job = Get-BitsTransfer -JobId $JobId
        if ($job.JobState -eq "Transferred") {
            Write-Log "BITS transfer completed; finalizing destination."
            Complete-BitsTransfer -BitsJob $job
            break
        }
        if ($job.JobState -in @("Error", "Cancelled")) {
            throw "BITS job ended in $($job.JobState): $($job.ErrorDescription)"
        }
        Start-Sleep -Seconds 30
    }

    if (-not (Test-Path -LiteralPath $ArchivePath -PathType Leaf)) {
        throw "Completed BITS job did not produce archive: $ArchivePath"
    }
    $bsdtar = "D:\Anaconda\Library\bin\bsdtar.exe"
    if (-not (Test-Path -LiteralPath $bsdtar -PathType Leaf)) {
        throw "bsdtar is unavailable: $bsdtar"
    }
    Write-Log "Checking archive listing."
    & $bsdtar -tf $ArchivePath | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Archive listing failed with exit code $LASTEXITCODE"
    }

    if (Test-Path -LiteralPath $ExtractPath) {
        $existing = Get-ChildItem -LiteralPath $ExtractPath -Force | Select-Object -First 1
        if ($null -ne $existing) {
            throw "Extraction target is not empty: $ExtractPath"
        }
    } else {
        New-Item -ItemType Directory -Force -Path $ExtractPath | Out-Null
    }
    Write-Log "Extracting P-DESTRE archive to $ExtractPath."
    & $bsdtar -xf $ArchivePath -C $ExtractPath
    if ($LASTEXITCODE -ne 0) {
        throw "Archive extraction failed with exit code $LASTEXITCODE"
    }

    Write-Log "Building non-destructive external-dataset audit."
    & $PythonExe $AuditScript --root "F:\dataset" --require-pdestre
    if ($LASTEXITCODE -ne 0) {
        throw "Dataset audit failed with exit code $LASTEXITCODE"
    }
    Write-Log "Download, extraction, and audit completed."
} catch {
    Write-Log "FAILED: $($_ | Out-String)"
    throw
}
