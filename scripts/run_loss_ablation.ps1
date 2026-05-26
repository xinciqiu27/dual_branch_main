param(
    [string]$DataDir = "data",
    [string]$OutputRoot = "outputs/loss_ablation_multi_seed",
    [string]$Device = "auto",
    [string]$SplitMode = "standard",
    [string]$Seeds = "42,52,62",
    [string]$EncoderBackend = "sbert",
    [string]$EncoderModel = "all-MiniLM-L6-v2",
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"

$recallKs = @(2, 4, 5, 6, 8, 10)
$seedList = $Seeds.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" } | ForEach-Object { [int]$_ }

$configs = @(
    [PSCustomObject]@{
        Name = "full_bpr"
        JointLossVariant = "full"
        ContextLossType = "bpr"
    },
    [PSCustomObject]@{
        Name = "no_imp_aux_bpr"
        JointLossVariant = "no_imp_aux"
        ContextLossType = "bpr"
    }
)

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
        ) + $ExtraArgs

        & python @argsList
        if ($LASTEXITCODE -ne 0) {
            throw "Experiment $runName failed with exit code $LASTEXITCODE"
        }
    }
}

$detailRows = foreach ($seed in $seedList) {
    foreach ($cfg in $configs) {
        $runName = "{0}_seed{1}" -f $cfg.Name, $seed
        $saveDir = Join-Path $OutputRoot $runName
        $summaryPath = Join-Path $saveDir "summary.json"

        if (-not (Test-Path $summaryPath)) {
            continue
        }

        $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
        [PSCustomObject]@{
            run_name = $runName
            base_run = $cfg.Name
            seed = $seed
            joint_loss = $cfg.JointLossVariant
            context_loss = $cfg.ContextLossType
            best_val_recall10 = [double]$summary.best_val_recall10
            test_recall2 = [double]$summary.test_metrics.'Recall@2'
            test_recall4 = [double]$summary.test_metrics.'Recall@4'
            test_recall5 = [double]$summary.test_metrics.'Recall@5'
            test_recall6 = [double]$summary.test_metrics.'Recall@6'
            test_recall8 = [double]$summary.test_metrics.'Recall@8'
            test_recall10 = [double]$summary.test_metrics.'Recall@10'
            test_ndcg10 = [double]$summary.test_metrics.'NDCG@10'
            test_mrr = [double]$summary.test_metrics.MRR
        }
    }
}

Write-Host ""
Write-Host "=== Per-Run Summary ==="
$detailRows | Sort-Object base_run, seed | Format-Table -AutoSize

$aggregateRows = foreach ($cfg in $configs) {
    $rows = $detailRows | Where-Object { $_.base_run -eq $cfg.Name }
    if ($rows.Count -eq 0) {
        continue
    }

    $valRecallValues = @($rows | ForEach-Object { [double]$_.best_val_recall10 })
    $testRecall2Values = @($rows | ForEach-Object { [double]$_.test_recall2 })
    $testRecall4Values = @($rows | ForEach-Object { [double]$_.test_recall4 })
    $testRecall5Values = @($rows | ForEach-Object { [double]$_.test_recall5 })
    $testRecall6Values = @($rows | ForEach-Object { [double]$_.test_recall6 })
    $testRecall8Values = @($rows | ForEach-Object { [double]$_.test_recall8 })
    $testRecall10Values = @($rows | ForEach-Object { [double]$_.test_recall10 })
    $testNdcgValues = @($rows | ForEach-Object { [double]$_.test_ndcg10 })
    $testMrrValues = @($rows | ForEach-Object { [double]$_.test_mrr })

    [PSCustomObject]@{
        run_name = $cfg.Name
        seeds = ($seedList -join ",")
        best_val_recall10_mean = Get-Mean $valRecallValues
        best_val_recall10_std = Get-Std $valRecallValues
        test_recall2_mean = Get-Mean $testRecall2Values
        test_recall2_std = Get-Std $testRecall2Values
        test_recall4_mean = Get-Mean $testRecall4Values
        test_recall4_std = Get-Std $testRecall4Values
        test_recall5_mean = Get-Mean $testRecall5Values
        test_recall5_std = Get-Std $testRecall5Values
        test_recall6_mean = Get-Mean $testRecall6Values
        test_recall6_std = Get-Std $testRecall6Values
        test_recall8_mean = Get-Mean $testRecall8Values
        test_recall8_std = Get-Std $testRecall8Values
        test_recall10_mean = Get-Mean $testRecall10Values
        test_recall10_std = Get-Std $testRecall10Values
        test_ndcg10_mean = Get-Mean $testNdcgValues
        test_ndcg10_std = Get-Std $testNdcgValues
        test_mrr_mean = Get-Mean $testMrrValues
        test_mrr_std = Get-Std $testMrrValues
    }
}

Write-Host ""
Write-Host "=== Aggregate Summary ==="
$aggregateRows | Format-Table -AutoSize

$detailCsvPath = Join-Path $OutputRoot "loss_ablation_detail.csv"
$aggregateCsvPath = Join-Path $OutputRoot "loss_ablation_aggregate.csv"

$detailRows | Export-Csv -Path $detailCsvPath -NoTypeInformation -Encoding UTF8
$aggregateRows | Export-Csv -Path $aggregateCsvPath -NoTypeInformation -Encoding UTF8

Write-Host "Saved per-run summary to $detailCsvPath"
Write-Host "Saved aggregate summary to $aggregateCsvPath"
