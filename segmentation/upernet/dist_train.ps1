Param(
    [Parameter(Mandatory=$true)][string]$Config,
    [Parameter(Mandatory=$true)][int]$Gpus,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$ExtraArgs
)

$Port = $env:PORT
if (-not $Port) { $Port = 29600 }
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$rootDir = Join-Path $scriptDir ".."
$env:PYTHONPATH = "$rootDir;$env:PYTHONPATH"
$trainScript = Join-Path $scriptDir "train.py"

Write-Host "Running distributed training on $Gpus GPUs with port $Port"
& torchrun --nproc_per_node=$Gpus --master_port=$Port $trainScript $Config --launcher pytorch @ExtraArgs
