param(
    [Parameter(Mandatory = $true)]
    [string]$BackupDirectory,
    [switch]$ConfirmRestore,
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
    (Join-Path $PSScriptRoot "restore.py"),
    "--backup-directory", $BackupDirectory,
    "--project-name", $ProjectName
)
if ($ConfirmRestore) { $arguments += "--confirm-restore" }
if ($EnvFile) { $arguments += @("--env-file", $EnvFile) }
if ($ComposeFile) { $arguments += @("--compose-file", $ComposeFile) }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "QuantLab Python restore failed" }
