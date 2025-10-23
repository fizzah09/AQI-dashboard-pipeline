<#
CLI wrapper for run_training.py that accepts arbitrary args and ensures .env is loaded.
Usage:
  .\scripts\run_training_cli.ps1 -- --use-hopsworks --days-back 30 --target pollutant_aqi
The `--` separates PS args from the python args.
#>
param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [String[]]$Args
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

if ($Args -and $Args[0] -eq '--') { $Args = $Args[1..($Args.Length-1)] }

$pyArgs = $Args -join ' '
if (-not $pyArgs) { $pyArgs = '--data data\ml_training_data_1year.csv --target pollutant_aqi' }

$cmd = "python run_training.py $pyArgs"
Write-Host "Executing: $cmd" -ForegroundColor Yellow
Invoke-Expression $cmd
