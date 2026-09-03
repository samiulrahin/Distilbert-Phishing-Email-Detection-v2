param(
    [switch]$TrainDistilBert,
    [int]$BenchmarkRuns = 60
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment not found. Run scripts\setup.ps1 first."
}

Set-Location $ProjectRoot
$env:HF_HOME = Join-Path $ProjectRoot "tmp\huggingface"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
& $Python -m ml.download_dataset
& $Python -m ml.preprocess
& $Python -m ml.train_baseline

if ($TrainDistilBert) {
    & $Python -m ml.train_distilbert
}

& $Python -m evaluation.capture_environment
& $Python -m pytest
Push-Location (Join-Path $ProjectRoot "extension")
try {
    npm test
} finally {
    Pop-Location
}
& $Python -m evaluation.benchmark_runtime --runs $BenchmarkRuns
& $Python -m evaluation.benchmark_runtime --runs $BenchmarkRuns --force-baseline
& $Python -m evaluation.compile_results
