param([switch]$Cuda)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

function Resolve-CompatiblePython {
    # Use the installed Python 3.11 first.
    # Python 3.11.3 has been confirmed to be installed and working.
    $Candidates = @(
        @{ Command = "py"; Arguments = @("-3.11") },
        @{ Command = "python"; Arguments = @() }
    )

    foreach ($Candidate in $Candidates) {
        if (-not (Get-Command $Candidate.Command -ErrorAction SilentlyContinue)) {
            continue
        }

        try {
            $Probe = & $Candidate.Command @($Candidate.Arguments) -c "import sys; print('OK' if (3,11) <= sys.version_info[:2] < (3,13) else 'NO')" 2>$null
        }
        catch {
            continue
        }

        if ($LASTEXITCODE -eq 0 -and $Probe -eq "OK") {
            return [pscustomobject]@{
                Command = $Candidate.Command
                Arguments = $Candidate.Arguments
            }
        }
    }

    throw "64-bit Python 3.11 or 3.12 was not found. Install it from python.org, then run this installer again."
}

Write-Host "Mail Risk Lab portable installation" -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"

# Create the virtual environment if it does not already exist
if (-not (Test-Path -LiteralPath $VenvPython)) {
    $Python = Resolve-CompatiblePython

    Write-Host "Using Python: $($Python.Command) $($Python.Arguments -join ' ')" -ForegroundColor Green
    Write-Host "Creating the local Python environment..."

    & $Python.Command @($Python.Arguments) -m venv $VenvPath

    if ($LASTEXITCODE -ne 0) {
        throw "Python could not create the virtual environment."
    }
}
else {
    Write-Host "Reusing the existing local Python environment."
}

# Upgrade pip
Write-Host "Upgrading pip..."
& $VenvPython -m pip install --upgrade pip

if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed."
}

# Install PyTorch
if ($Cuda) {
    Write-Host "Installing NVIDIA CUDA PyTorch..."
    & $VenvPython -m pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
}
else {
    Write-Host "Installing CPU PyTorch for maximum compatibility..."
    & $VenvPython -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
}

if ($LASTEXITCODE -ne 0) {
    throw "PyTorch installation failed. Check the internet connection and free disk space."
}

# Install project dependencies
Write-Host "Installing project dependencies..."

$RequirementsFile = Join-Path $ProjectRoot "requirements-portable.txt"

if (-not (Test-Path -LiteralPath $RequirementsFile)) {
    throw "requirements-portable.txt was not found at $RequirementsFile"
}

& $VenvPython -m pip install -r $RequirementsFile

if ($LASTEXITCODE -ne 0) {
    throw "Project dependency installation failed."
}

# Verify the portable installation
Write-Host "Verifying portable installation..."

$VerifyScript = Join-Path $PSScriptRoot "verify_portable.ps1"

if (-not (Test-Path -LiteralPath $VerifyScript)) {
    throw "verify_portable.ps1 was not found at $VerifyScript"
}

& $VerifyScript -SkipTests

if ($LASTEXITCODE -ne 0) {
    throw "Post-installation verification failed."
}

Write-Host ""
Write-Host "Mail Risk Lab portable installation completed successfully." -ForegroundColor Green
Write-Host "Virtual environment: $VenvPath" -ForegroundColor Green