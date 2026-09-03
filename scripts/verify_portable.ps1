param([switch]$SkipTests)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ModelDirectory = Join-Path $ProjectRoot "models\distilbert-phishing-v2"
$ModelFile = Join-Path $ModelDirectory "model.safetensors"
$MetadataFile = Join-Path $ModelDirectory "research_metadata.json"

$RequiredFiles = @(
    $Python,
    $ModelFile,
    (Join-Path $ModelDirectory "config.json"),
    (Join-Path $ModelDirectory "tokenizer.json"),
    (Join-Path $ModelDirectory "decision_policy.json"),
    $MetadataFile,
    (Join-Path $ProjectRoot "extension\manifest.json")
)

foreach ($Path in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required file is missing: $Path"
    }
}

$Metadata = Get-Content -LiteralPath $MetadataFile -Raw | ConvertFrom-Json
$ActualHash = (Get-FileHash -LiteralPath $ModelFile -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualHash -ne $Metadata.model_sha256) {
    throw "The V2 model hash does not match the research metadata. Re-extract the package."
}

Push-Location $ProjectRoot
try {
    & $Python -c "import fastapi, numpy, sklearn, torch, transformers; print('Python dependencies: OK'); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { throw "Python dependency verification failed." }

    if (-not $SkipTests) {
        & $Python -m pytest
        if ($LASTEXITCODE -ne 0) { throw "Automated tests failed." }
    }
} finally {
    Pop-Location
}

Write-Host "Model SHA-256: $ActualHash"
Write-Host "Mail Risk Lab package verification: PASSED" -ForegroundColor Green
