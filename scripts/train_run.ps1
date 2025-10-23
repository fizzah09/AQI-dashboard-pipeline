<#
PowerShell helper to run training: either with local CSV or fetch from Hopsworks.
Usage:
  .\scripts\train_run.ps1 -UseHopsworks -DaysBack 30 -Target pollutant_aqi
  .\scripts\train_run.ps1 -DataPath data\ml_training_data_1year.csv -Target pollutant_aqi
#>
param(
    [switch]$UseHopsworks,
    [int]$DaysBack = 30,
    [string]$DataPath = "data\ml_training_data_1year.csv",
    [string]$Target = "pollutant_aqi"
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot
$env:PYTHONPATH = $ProjectRoot

# Load .env into process
$envFile = Join-Path $ProjectRoot '.env'
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^[\s]*#') { return }
        if ($_ -match '^[\s]*$') { return }
        if ($_ -match '^[\s]*([A-Za-z0-9_]+)[\s]*=[\s]*"?(.*)"?[\s]*$') {
            $name = $matches[1]; $value = $matches[2];
            if ($value.StartsWith('"') -and $value.EndsWith('"')) { $value = $value.Trim('"') }
            [System.Environment]::SetEnvironmentVariable($name, $value, 'Process')
        }
    }
}

Write-Host "Running training with settings:`n  UseHopsworks=$UseHopsworks`n  DaysBack=$DaysBack`n  DataPath=$DataPath`n  Target=$Target`n" -ForegroundColor Cyan

if ($UseHopsworks) {
    $cmd = "python run_training.py --use-hopsworks --days-back $DaysBack --target $Target"
} else {
    $cmd = "python run_training.py --data $DataPath --target $Target"
}

Write-Host "Executing: $cmd" -ForegroundColor Yellow
Invoke-Expression $cmd
