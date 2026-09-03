param(
    [string]$PythonPath = "",
    [switch]$Cuda
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"

if ($PythonPath) {
    $PythonCommand = $PythonPath
    $PythonArguments = @()
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonCommand = "py"
    $PythonArguments = @("-3.12")
} else {
    $PythonCommand = "python"
    $PythonArguments = @()
}

& $PythonCommand @PythonArguments -m venv $VenvPath
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip

if ($Cuda) {
    & $VenvPython -m pip install --force-reinstall torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
} else {
    & $VenvPython -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
}

& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements-ml.txt")
& $VenvPython -c "import sys, torch; print(sys.version); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
