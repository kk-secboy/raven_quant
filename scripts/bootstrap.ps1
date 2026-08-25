param(
    [ValidateSet("core", "research", "full")]
    [string]$Profile = "full",
    [string]$Start = "2016-01-01",
    [string]$SnapshotStart = "2008-01-01",
    [string]$End = "latest"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    python -m venv (Join-Path $repo ".venv")
    & $python -m pip install -e "${repo}[dev]"
}

Push-Location $repo
try {
    & $python -m quant_data.cli probe
    & $python -m quant_data.cli bootstrap --profile $Profile --start $Start --snapshot-start $SnapshotStart --end $End
}
finally {
    Pop-Location
}
