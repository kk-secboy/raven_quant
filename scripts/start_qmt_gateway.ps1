param(
    [string]$EnvFile = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
if (-not $EnvFile) { $EnvFile = Join-Path $repo "deploy\qmt-gateway.env" }
$EnvFile = (Resolve-Path -LiteralPath $EnvFile).Path
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "project virtual environment is missing: $python"
}
& $python -m quant_broker_gateway.cli --env-file $EnvFile
if ($LASTEXITCODE -ne 0) { throw "QMT gateway exited with code $LASTEXITCODE" }
