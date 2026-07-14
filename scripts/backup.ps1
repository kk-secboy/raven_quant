param(
    [string]$BackupRoot = "E:\quantlab-backups",
    [ValidateRange(1, 365)]
    [int]$RetentionCount = 14,
    [string]$ProjectName = "quantlab-platform",
    [string]$EnvFile = "",
    [string]$ComposeFile = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repo ".venv\Scripts\python.exe"
$python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }
$arguments = @(
    (Join-Path $PSScriptRoot "backup.py"),
    "--backup-root", $BackupRoot,
    "--retention-count", [string]$RetentionCount,
    "--project-name", $ProjectName
)
if ($EnvFile) { $arguments += @("--env-file", $EnvFile) }
if ($ComposeFile) { $arguments += @("--compose-file", $ComposeFile) }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "QuantLab Python backup failed" }
