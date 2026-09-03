$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$DemoUrl = "http://127.0.0.1:8765/demo"
$HealthUrl = "http://127.0.0.1:8765/health"

if (-not (Test-Path -LiteralPath $Python)) {
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show(
        "The project environment is not installed. Run scripts\setup.ps1 first.",
        "Mail Risk Lab",
        "OK",
        "Warning"
    ) | Out-Null
    exit 1
}

function Test-MailRiskService {
    try {
        $Health = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 2
        return $Health.status -eq "ok"
    }
    catch {
        return $false
    }
}

if (-not (Test-MailRiskService)) {
    $RunScript = Join-Path $PSScriptRoot "run_api.ps1"
    Start-Process powershell.exe -WindowStyle Minimized -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", ('"{0}"' -f $RunScript)
    )

    $Ready = $false
    for ($Attempt = 0; $Attempt -lt 45; $Attempt++) {
        Start-Sleep -Seconds 1
        if (Test-MailRiskService) {
            $Ready = $true
            break
        }
    }
    if (-not $Ready) {
        Add-Type -AssemblyName PresentationFramework
        [System.Windows.MessageBox]::Show(
            "The model did not become ready within 45 seconds. Open the minimized service window for details.",
            "Mail Risk Lab",
            "OK",
            "Error"
        ) | Out-Null
        exit 1
    }
}

Start-Process $DemoUrl
