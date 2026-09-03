param(
    [string]$PackageName = "Mail_Risk_Lab_V2_Transfer_Package_2026-08-04",
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$WorkspaceRoot = Split-Path -Parent $ProjectRoot
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $WorkspaceRoot "deliverables"
}
$PackageDirectory = Join-Path $OutputRoot $PackageName
$ZipPath = Join-Path $OutputRoot ($PackageName + ".zip")
$HashPath = $ZipPath + ".sha256.txt"

if (Test-Path -LiteralPath $PackageDirectory) {
    throw "Package directory already exists: $PackageDirectory. Use a different PackageName."
}
if (Test-Path -LiteralPath $ZipPath) {
    throw "Package ZIP already exists: $ZipPath. Use a different PackageName."
}

New-Item -ItemType Directory -Path $PackageDirectory -Force | Out-Null

function Copy-RelativeFile {
    param([string]$RelativePath)
    $Source = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path -LiteralPath $Source)) {
        throw "Required project file is missing: $RelativePath"
    }
    $Destination = Join-Path $PackageDirectory $RelativePath
    New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Destination
}

function Copy-ProjectTree {
    param([string]$RelativeDirectory)
    $SourceRoot = Join-Path $ProjectRoot $RelativeDirectory
    foreach ($File in Get-ChildItem -LiteralPath $SourceRoot -Recurse -File) {
        $Relative = $File.FullName.Substring($ProjectRoot.Length + 1)
        if ($Relative -match '(^|\\)(__pycache__|\.pytest_cache|node_modules)(\\|$)') { continue }
        if ($File.Extension -in @('.pyc', '.pyo')) { continue }
        $Destination = Join-Path $PackageDirectory $Relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force | Out-Null
        Copy-Item -LiteralPath $File.FullName -Destination $Destination
    }
}

$RootFiles = @(
    ".gitignore",
    "README.md",
    "USER_GUIDE.md",
    "PORTABLE_PACKAGE_README.md",
    "FACULTY_DEMONSTRATION_GUIDE.md",
    "PROJECT_PROPOSAL_TO_IMPLEMENTATION_MAP.md",
    "Install_Mail_Risk_Lab.bat",
    "Start_Mail_Risk_Lab.bat",
    "Verify_Mail_Risk_Lab.bat",
    "requirements-portable.txt",
    "requirements.txt",
    "requirements-ml.txt",
    "pyproject.toml"
)
foreach ($File in $RootFiles) { Copy-RelativeFile $File }

foreach ($Directory in @("backend", "demo", "docs", "extension", "ml", "report_assets", "scripts")) {
    Copy-ProjectTree $Directory
}

$DataFiles = @(
    "data\augmentation\hard_cases.csv",
    "data\processed\README.md",
    "data\raw\README.md",
    "data\raw\download_metadata.json",
    "data\splits\README.md",
    "data\splits\data_quality_report.json"
)
foreach ($File in $DataFiles) { Copy-RelativeFile $File }

foreach ($File in Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "evaluation") -Recurse -File) {
    $Relative = $File.FullName.Substring($ProjectRoot.Length + 1)
    if ($Relative -match '(^|\\)__pycache__(\\|$)' -or $File.Extension -eq ".pyc") { continue }
    if ($File.Name -match '^(api_server|setup).*\.log$') { continue }
    if ($File.Name -eq "distilbert_training_error.log") { continue }
    if ($File.Name -eq "runtime_log.csv") { continue }
    $Destination = Join-Path $PackageDirectory $Relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force | Out-Null
    Copy-Item -LiteralPath $File.FullName -Destination $Destination
}

$ModelFiles = @(
    "models\README.md",
    "models\tfidf-logistic-regression.joblib",
    "models\baseline_threshold.json",
    "models\distilbert-phishing-v2\config.json",
    "models\distilbert-phishing-v2\decision_policy.json",
    "models\distilbert-phishing-v2\decision_threshold.json",
    "models\distilbert-phishing-v2\model.safetensors",
    "models\distilbert-phishing-v2\research_metadata.json",
    "models\distilbert-phishing-v2\special_tokens_map.json",
    "models\distilbert-phishing-v2\tokenizer.json",
    "models\distilbert-phishing-v2\tokenizer_config.json",
    "models\distilbert-phishing-v2\training_args.bin",
    "models\distilbert-phishing-v2\vocab.txt"
)
foreach ($File in $ModelFiles) { Copy-RelativeFile $File }

$AcademicDirectory = Join-Path $PackageDirectory "Academic_Project_Documents"
New-Item -ItemType Directory -Path $AcademicDirectory -Force | Out-Null
$AcademicSources = @(
    @{ Pattern = "25048264+Proposal.pdf"; Destination = "25048264_Proposal.pdf" },
    @{ Pattern = "UFCF9Y-60-M*assessment specification*.pdf"; Destination = "Assessment_Specification.pdf" },
    @{ Pattern = "Project report structure(1).pdf"; Destination = "Project_Report_Structure.pdf" },
    @{ Pattern = "Project_Map.md"; Destination = "Project_Map.md" },
    @{ Pattern = "25048264_Project_Architecture_and_High_Mark_Implementation_Plan.md"; Destination = "Architecture_and_Implementation_Plan.md" },
    @{ Pattern = "25048264 Faculty Update Draft.docx"; Destination = "25048264 Faculty Update Draft.docx" },
    @{ Pattern = "Ethics Review Checklist .docx"; Destination = "Ethics_Review_Checklist.docx" }
)
foreach ($Entry in $AcademicSources) {
    $Matches = @(Get-ChildItem -LiteralPath $WorkspaceRoot -File | Where-Object { $_.Name -like $Entry.Pattern })
    if ($Matches.Count -ne 1) {
        throw "Expected one academic source matching '$($Entry.Pattern)' but found $($Matches.Count)."
    }
    Copy-Item -LiteralPath $Matches[0].FullName -Destination (Join-Path $AcademicDirectory $Entry.Destination)
}

$Contents = @"
# Package Contents and Boundaries

This archive is the runnable Mail Risk Lab V2 faculty-transfer package.

Included:

- Final DistilBERT V2 weights, tokenizer, calibration, and selective decision policy.
- TF-IDF logistic-regression baseline.
- Chrome Manifest V3 extension, local FastAPI service, full demonstration interface, and tests.
- Controlled examples, challenge sets, curated metrics, predictions, plots, screenshots, and Version 1 evidence snapshot.
- Installation, verification, faculty-demonstration, user, architecture, ethics, threat-model, model-card, dataset-card, limitation, and evidence documentation.
- Proposal, assessment specification, report-structure guidance, ethics checklist, implementation plan, project map, and faculty progress brief.

Intentionally excluded because they are not required to run or demonstrate the artefact:

- `.venv`, `.git`, caches, temporary renders, local server logs, and machine-specific setup logs.
- Intermediate training checkpoints and the obsolete Version 1 weight file. Version 1 results and its recorded model hash remain included.
- Raw and split public-dataset CSV files. Their download URL, source SHA-256, quality report, split description, training code, and saved evaluation evidence remain included. The final V2 model is already trained.
- Any Gmail credentials, personal-inbox content, or participant data.

Model SHA-256: 1087417f895a8e064a71456a7d3da9a968da6d8fc2edcb15f671a875180f4daa
"@
Set-Content -LiteralPath (Join-Path $PackageDirectory "PACKAGE_CONTENTS.md") -Value $Contents -Encoding UTF8

$ManifestPath = Join-Path $PackageDirectory "PACKAGE_MANIFEST_SHA256.txt"
$ManifestLines = foreach ($File in Get-ChildItem -LiteralPath $PackageDirectory -Recurse -File | Sort-Object FullName) {
    if ($File.FullName -eq $ManifestPath) { continue }
    $Relative = $File.FullName.Substring($PackageDirectory.Length + 1).Replace("\", "/")
    $Hash = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$Hash  $Relative"
}
Set-Content -LiteralPath $ManifestPath -Value $ManifestLines -Encoding ASCII

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
Compress-Archive -LiteralPath $PackageDirectory -DestinationPath $ZipPath -CompressionLevel Optimal

Add-Type -AssemblyName System.IO.Compression.FileSystem
$Archive = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
try {
    $Buffer = New-Object byte[] 1048576
    foreach ($Entry in $Archive.Entries) {
        if (-not $Entry.Name) { continue }
        $Stream = $Entry.Open()
        try {
            while ($Stream.Read($Buffer, 0, $Buffer.Length) -gt 0) { }
        } finally {
            $Stream.Dispose()
        }
    }
    $ArchiveCount = $Archive.Entries.Count
} finally {
    $Archive.Dispose()
}

$ZipHash = (Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content -LiteralPath $HashPath -Value "$ZipHash  $([IO.Path]::GetFileName($ZipPath))" -Encoding ASCII

$PackageFiles = @(Get-ChildItem -LiteralPath $PackageDirectory -Recurse -File)
[pscustomobject]@{
    PackageDirectory = $PackageDirectory
    ZipPath = $ZipPath
    ZipSHA256 = $ZipHash
    Files = $PackageFiles.Count
    UncompressedMB = [math]::Round((($PackageFiles | Measure-Object Length -Sum).Sum / 1MB), 2)
    ZipMB = [math]::Round(((Get-Item -LiteralPath $ZipPath).Length / 1MB), 2)
    ArchiveEntries = $ArchiveCount
} | ConvertTo-Json
