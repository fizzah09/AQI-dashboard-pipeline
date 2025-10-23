# Runs end-to-end Hopsworks demo: verify -> fetch training data -> train -> list models
# Usage: from repo root or any folder: .\scripts\run_hopsworks_pipeline.ps1

$ErrorActionPreference = "Stop"

Write-Host "=============================================="
Write-Host " AQI Pipeline Demo (Hopsworks)" -ForegroundColor Cyan
Write-Host "=============================================="

# Resolve project root (parent of scripts folder)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

# Ensure PYTHONPATH so imports resolve
$env:PYTHONPATH = $ProjectRoot

# Load .env into current PowerShell process environment (so Python os.getenv can see it)
$envFile = Join-Path $ProjectRoot ".env"
if (-not (Test-Path $envFile)) {
    throw ".env not found at $envFile"
}

Get-Content $envFile | ForEach-Object {
    if ($_ -match '^[\s]*#') { return }
    if ($_ -match '^[\s]*$') { return }
    if ($_ -match '^[\s]*([A-Za-z0-9_]+)[\s]*=[\s]*"?(.*)"?[\s]*$') {
        $name = $matches[1]
        $value = $matches[2]
        # Trim surrounding quotes if present
        if ($value.StartsWith('"') -and $value.EndsWith('"')) { $value = $value.Trim('"') }
        [System.Environment]::SetEnvironmentVariable($name, $value, 'Process')
    }
}

# 0) Show Python and env summary
python --version
$projName = $env:HOPSWORKS_PROJECT_NAME
if (-not $projName) {
    $projName = (Select-String -Path $envFile -Pattern '^HOPSWORKS_PROJECT_NAME=').Line.Split('=')[1]
}
Write-Host ("Project: {0}" -f $projName)

function Invoke-Step {
    param(
        [Parameter(Mandatory=$true)][string]$Title,
        [Parameter(Mandatory=$true)][scriptblock]$Action
    )
    Write-Host "`n[STEP] $Title" -ForegroundColor Green
    try {
        & $Action
        Write-Host "[OK] $Title" -ForegroundColor Green
    } catch {
        Write-Host "[FAIL] $Title" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        exit 1
    }
}

# 1) Verify Hopsworks connectivity and Feature Store
Invoke-Step -Title "Verify Hopsworks" -Action {
    python scripts/verify_hopsworks.py
}

# 2) Fetch training data from Hopsworks Feature Store
Invoke-Step -Title "Fetch training data from Feature Store" -Action {
    $pyFetch = @"
import pathlib
from modeling.data_loader_hopswork import load_training_data_from_hopsworks
pathlib.Path('data').mkdir(exist_ok=True)
df = load_training_data_from_hopsworks(days_back=30)
out = 'data/ml_training_data_latest.csv'
df.to_csv(out, index=False)
print(f'Saved training data -> {out} | rows={len(df)} cols={len(df.columns)}')
"@
    python -c $pyFetch
}

# 3) Train the model
Invoke-Step -Title "Train model" -Action {
    if (-not (Test-Path 'data/ml_training_data_latest.csv')) {
        throw "data/ml_training_data_latest.csv not found"
    }
    python run_training.py --data data/ml_training_data_latest.csv --target pollutant_aqi
}

# 4) Summarize Model Registry
Invoke-Step -Title "List registered models" -Action {
    $pyList = @"
from modeling.model_registry import list_registered_models
list_registered_models()
"@
    python -c $pyList
}

# 5) Show artifacts (optional)
Write-Host "`nArtifacts:" -ForegroundColor Yellow
if (Test-Path 'modeling/models') { Get-ChildItem -Recurse modeling/models -File | Select-Object FullName,Length,LastWriteTime }
if (Test-Path 'modeling/evaluation') { Get-ChildItem -Recurse modeling/evaluation -File | Select-Object FullName,Length,LastWriteTime }

# 6) Show verification JSON (optional)
if (Test-Path './hopsworks_verification_report.json') {
    Write-Host "`nHopsworks Verification Report:" -ForegroundColor Yellow
    Get-Content './hopsworks_verification_report.json'
}

Write-Host "`nDone." -ForegroundColor Cyan
