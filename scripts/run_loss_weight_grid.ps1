param(
    [string]$DataDir = "data",
    [string]$OutputRoot = "outputs/loss_weight_grid",
    [string]$Device = "auto",
    [string]$SplitMode = "standard",
    [string]$Seeds = "42",
    [string]$EncoderBackend = "sbert",
    [string]$EncoderModel = "all-MiniLM-L6-v2",
    [string]$LambdaExpValues = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    [string]$LambdaImpValues = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    [string]$JointLossVariant = "full",
    [string]$ContextLossType = "bpr",
    [string]$SelectMetric = "Recall@10",
    [switch]$SummarizeOnly,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"

function Parse-DoubleList {
    param([string]$Text)
    return $Text.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" } | ForEach-Object { [double]$_ }
}

function Parse-IntList {
    param([string]$Text)
    return $Text.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" } | ForEach-Object { [int]$_ }
}

function Get-Mean {
    param([double[]]$Values)
    if ($null -eq $Values -or $Values.Count -eq 0) {
        return $null
    }
    return ($Values | Measure-Object -Average).Average
}

function Get-Std {
    param([double[]]$Values)
    if ($null -eq $Values -or $Values.Count -eq 0) {
        return $null
    }
    if ($Values.Count -eq 1) {
        return 0.0
    }
    $mean = Get-Mean $Values
    $sum = 0.0
    foreach ($v in $Values) {
        $sum += [math]::Pow(($v - $mean), 2)
    }
    return [math]::Sqrt($sum / $Values.Count)
}

function Format-WeightTag {
    param([double]$Value)
    return ("{0:0.0}" -f $Value).Replace(".", "p")
}

$seedList = Parse-IntList $Seeds
$lambdaExpList = Parse-DoubleList $LambdaExpValues
$lambdaImpList = Parse-DoubleList $LambdaImpValues

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

if (-not $SummarizeOnly) {
    foreach ($seed in $seedList) {
        foreach ($lambdaExp in $lambdaExpList) {
            foreach ($lambdaImp in $lambdaImpList) {
                $expTag = Format-WeightTag $lambdaExp
                $impTag = Format-WeightTag $lambdaImp
                $runName = "exp_${expTag}_imp_${impTag}_seed${seed}"
                $saveDir = Join-Path $OutputRoot $runName

                Write-Host ""
                Write-Host "=== Running $runName ==="

                $argsList = @(
                    "train.py",
                    "--data-dir", $DataDir,
                    "--save-dir", $saveDir,
                    "--device", $Device,
                    "--split-mode", $SplitMode,
                    "--seed", "$seed",
                    "--encoder-backend", $EncoderBackend,
                    "--encoder-model", $EncoderModel,
                    "--joint-loss-variant", $JointLossVariant,
                    "--context-loss-type", $ContextLossType,
                    "--lambda-exp", ("{0:0.0}" -f $lambdaExp),
                    "--lambda-imp", ("{0:0.0}" -f $lambdaImp)
                ) + $ExtraArgs

                & python @argsList
                if ($LASTEXITCODE -ne 0) {
                    throw "Experiment $runName failed with exit code $LASTEXITCODE"
                }
            }
        }
    }
}

$summaryFiles = Get-ChildItem -Path $OutputRoot -Recurse -Filter "summary.json" | Sort-Object FullName
if ($summaryFiles.Count -eq 0) {
    throw "No summary.json files found under $OutputRoot"
}

$firstSummary = Get-Content $summaryFiles[0].FullName -Raw | ConvertFrom-Json
$metricNames = @($firstSummary.test_metrics.PSObject.Properties.Name)

$detailRows = foreach ($file in $summaryFiles) {
    $saveDir = Split-Path $file.FullName -Parent
    $runName = Split-Path $saveDir -Leaf
    $summary = Get-Content $file.FullName -Raw | ConvertFrom-Json
    $match = [regex]::Match($runName, '^exp_(\d+p\d+)_imp_(\d+p\d+)_seed(\d+)$')
    if (-not $match.Success) {
        throw "Failed to parse run name: $runName"
    }

    $lambdaExp = [double]($match.Groups[1].Value.Replace("p", "."))
    $lambdaImp = [double]($match.Groups[2].Value.Replace("p", "."))
    $seed = [int]$match.Groups[3].Value

    $row = [ordered]@{
        run_name = $runName
        seed = $seed
        lambda_exp = $lambdaExp
        lambda_imp = $lambdaImp
        joint_loss = $JointLossVariant
        context_loss = $ContextLossType
        best_val_recall10 = [double]$summary.best_val_recall10
    }

    foreach ($metricName in $metricNames) {
        $row["test_$metricName"] = [double]$summary.test_metrics.$metricName
    }

    [PSCustomObject]$row
}

Write-Host ""
Write-Host "=== Per-Run Summary ==="
$detailRows | Sort-Object lambda_exp, lambda_imp, seed | Format-Table -AutoSize

$groupedRows = $detailRows | Group-Object -Property lambda_exp, lambda_imp
$aggregateRows = foreach ($group in $groupedRows) {
    $rows = @($group.Group)
    $first = $rows[0]

    $row = [ordered]@{
        lambda_exp = [double]$first.lambda_exp
        lambda_imp = [double]$first.lambda_imp
        seeds = ($rows.seed | Sort-Object | ForEach-Object { [string]$_ }) -join ","
        joint_loss = [string]$first.joint_loss
        context_loss = [string]$first.context_loss
        best_val_recall10_mean = Get-Mean @($rows | ForEach-Object { [double]$_.best_val_recall10 })
        best_val_recall10_std = Get-Std @($rows | ForEach-Object { [double]$_.best_val_recall10 })
    }

    foreach ($metricName in $metricNames) {
        $columnName = "test_$metricName"
        $values = @($rows | ForEach-Object { [double]$_.$columnName })
        $row["${columnName}_mean"] = Get-Mean $values
        $row["${columnName}_std"] = Get-Std $values
    }

    [PSCustomObject]$row
}

$selectColumn = "test_$SelectMetric" + "_mean"
if (-not ($aggregateRows[0].PSObject.Properties.Name -contains $selectColumn)) {
    throw "SelectMetric '$SelectMetric' is not present in aggregate rows"
}

$bestRow = $aggregateRows | Sort-Object -Property @{Expression = $selectColumn; Descending = $true}, @{Expression = "test_MRR_mean"; Descending = $true} | Select-Object -First 1

Write-Host ""
Write-Host "=== Aggregate Summary ==="
$aggregateRows | Sort-Object lambda_exp, lambda_imp | Format-Table -AutoSize

Write-Host ""
Write-Host "=== Best Combo By $SelectMetric ==="
$bestRow | Format-List

$detailCsvPath = Join-Path $OutputRoot "loss_weight_grid_detail.csv"
$aggregateCsvPath = Join-Path $OutputRoot "loss_weight_grid_aggregate.csv"
$bestCsvPath = Join-Path $OutputRoot "loss_weight_grid_best.csv"

$detailRows | Export-Csv -Path $detailCsvPath -NoTypeInformation -Encoding UTF8
$aggregateRows | Export-Csv -Path $aggregateCsvPath -NoTypeInformation -Encoding UTF8
@($bestRow) | Export-Csv -Path $bestCsvPath -NoTypeInformation -Encoding UTF8

Write-Host "Saved per-run summary to $detailCsvPath"
Write-Host "Saved aggregate summary to $aggregateCsvPath"
Write-Host "Saved best combo to $bestCsvPath"
