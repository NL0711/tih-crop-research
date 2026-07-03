Param(
    [Parameter(Mandatory=$true)][string]$Config,
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [Parameter(Mandatory=$true)][int]$Gpus,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$ExtraArgs
)

$Port = $env:PORT
if (-not $Port) { $Port = 29500 }
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$rootDir = Join-Path $scriptDir ".."
$env:PYTHONPATH = "$rootDir;$env:PYTHONPATH"
$testScript = Join-Path $scriptDir "test.py"

Write-Host "Running distributed test on $Gpus GPUs with port $Port"
& torchrun --nproc_per_node=$Gpus --master_port=$Port $testScript $Config $Checkpoint --launcher pytorch @ExtraArgs
