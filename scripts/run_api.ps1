param([int]$Port = 8765)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment not found. Run scripts\setup.ps1 first."
}

Set-Location $ProjectRoot
& $Python -m uvicorn backend.app:app --host 127.0.0.1 --port $Port

