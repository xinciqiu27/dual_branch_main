param(
    [string]$DataDir = "data",
    [string]$OutputRoot = "outputs/candidate_config_compare",
    [string]$Device = "auto",
    [string]$SplitMode = "standard",
    [string]$Seeds = "42,52,62",
    [string]$EncoderBackend = "sbert",
    [string]$EncoderModel = "all-MiniLM-L6-v2",
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"

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

$seedList = Parse-IntList $Seeds

$configs = @(
    [PSCustomObject]@{
        Name = "no_imp_aux_bpr"
        JointLossVariant = "no_imp_aux"
        ContextLossType = "bpr"
        LambdaExp = $null
        LambdaImp = $null
    },
    [PSCustomObject]@{
        Name = "full_exp0p3_imp0p3"
        JointLossVariant = "full"
        ContextLossType = "bpr"
        LambdaExp = 0.3
        LambdaImp = 0.3
    },
    [PSCustomObject]@{
        Name = "full_exp0p5_imp0p3"
        JointLossVariant = "full"
        ContextLossType = "bpr"
        LambdaExp = 0.5
        LambdaImp = 0.3
    }
)

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

foreach ($seed in $seedList) {
    foreach ($cfg in $configs) {
        $runName = "{0}_seed{1}" -f $cfg.Name, $seed
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
            "--joint-loss-variant", $cfg.JointLossVariant,
            "--context-loss-type", $cfg.ContextLossType
        )

        if ($null -ne $cfg.LambdaExp) {
            $argsList += @("--lambda-exp", ("{0:0.0}" -f $cfg.LambdaExp))
        }
        if ($null -ne $cfg.LambdaImp) {
            $argsList += @("--lambda-imp", ("{0:0.0}" -f $cfg.LambdaImp))
        }

        $argsList += $ExtraArgs

        & python @argsList
        if ($LASTEXITCODE -ne 0) {
            throw "Experiment $runName failed with exit code $LASTEXITCODE"
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

    $cfg = $configs | Where-Object { $runName.StartsWith($_.Name + "_seed") } | Select-Object -First 1
    if ($null -eq $cfg) {
        throw "Failed to match config for run name: $runName"
    }

    $seedText = $runName.Substring($runName.LastIndexOf("seed") + 4)
    $seed = [int]$seedText

    $row = [ordered]@{
        run_name = $runName
        base_run = $cfg.Name
        seed = $seed
        joint_loss = $cfg.JointLossVariant
        context_loss = $cfg.ContextLossType
        lambda_exp = if ($null -eq $cfg.LambdaExp) { "" } else { "{0:0.0}" -f $cfg.LambdaExp }
        lambda_imp = if ($null -eq $cfg.LambdaImp) { "" } else { "{0:0.0}" -f $cfg.LambdaImp }
        best_val_recall10 = [double]$summary.best_val_recall10
    }

    foreach ($metricName in $metricNames) {
        $row["test_$metricName"] = [double]$summary.test_metrics.$metricName
    }

    [PSCustomObject]$row
}

Write-Host ""
Write-Host "=== Per-Run Summary ==="
$detailRows | Sort-Object base_run, seed | Format-Table -AutoSize

$aggregateRows = foreach ($cfg in $configs) {
    $rows = @($detailRows | Where-Object { $_.base_run -eq $cfg.Name })
    if ($rows.Count -eq 0) {
        continue
    }

    $row = [ordered]@{
        run_name = $cfg.Name
        seeds = ($rows.seed | Sort-Object | ForEach-Object { [string]$_ }) -join ","
        joint_loss = $cfg.JointLossVariant
        context_loss = $cfg.ContextLossType
        lambda_exp = if ($null -eq $cfg.LambdaExp) { "" } else { "{0:0.0}" -f $cfg.LambdaExp }
        lambda_imp = if ($null -eq $cfg.LambdaImp) { "" } else { "{0:0.0}" -f $cfg.LambdaImp }
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

Write-Host ""
Write-Host "=== Aggregate Summary ==="
$aggregateRows | Format-Table -AutoSize

$detailCsvPath = Join-Path $OutputRoot "candidate_compare_detail.csv"
$aggregateCsvPath = Join-Path $OutputRoot "candidate_compare_aggregate.csv"

$detailRows | Export-Csv -Path $detailCsvPath -NoTypeInformation -Encoding UTF8
$aggregateRows | Export-Csv -Path $aggregateCsvPath -NoTypeInformation -Encoding UTF8

Write-Host "Saved per-run summary to $detailCsvPath"
Write-Host "Saved aggregate summary to $aggregateCsvPath"
