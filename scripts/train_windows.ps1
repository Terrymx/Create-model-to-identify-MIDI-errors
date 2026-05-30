param(
    [string]$DataRoot = "E:\downloads\桌面\dku\CS309\project\maestro-v3.0.0-midi\maestro-v3.0.0",
    [string]$Python = "E:\downloads\桌面\dku\CS309\project\code\venv\Scripts\python.exe",
    [string]$Output = "checkpoints\bigru_wrong_note.pt",
    [string]$EvalSplit = "test",
    [int]$CleanEpochs = 1,
    [int]$Epochs = 20,
    [int]$BatchSize = 16,
    [int]$WindowSize = 256,
    [double]$TrainErrorRate = 0.15,
    [double]$ErrorRate = 0.08,
    [double]$DetThreshold = 0.3,
    [double]$DetPosWeight = 3.0,
    [double[]]$KindClassWeights = @(1.0, 6.0, 4.0),
    [double[]]$ThresholdSweep = @(0.2, 0.25, 0.3, 0.35, 0.4, 0.5),
    [string]$SaveMetric = "task_score",
    [double]$LrFactor = 0.5,
    [int]$LrPatience = 4,
    [double]$LrThreshold = 0.002,
    [double]$MinLr = 0.00001,
    [int]$EarlyStopPatience = 0,
    [int]$NumWorkers = 0,
    [Nullable[int]]$MaxFiles = $null
)

if (-not (Test-Path $Python)) {
    $LocalVenvPython = Join-Path $PSScriptRoot "..\venv\Scripts\python.exe"
    if (Test-Path $LocalVenvPython) {
        $Python = (Resolve-Path $LocalVenvPython).Path
    }
    else {
        throw "Cannot find Python executable: $Python. If your venv is inside code_new, create/use code_new\venv or pass -Python <path>."
    }
}

if (-not (Test-Path $DataRoot)) {
    throw "Cannot find MAESTRO data root: $DataRoot"
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $RepoRoot
try {
    & $Python -m pip install -e .

    $arguments = @(
        "-m", "midi_error_detector.train",
        "--data-root", $DataRoot,
        "--eval-split", $EvalSplit,
        "--clean-epochs", $CleanEpochs,
        "--epochs", $Epochs,
        "--batch-size", $BatchSize,
        "--window-size", $WindowSize,
        "--train-error-rate", $TrainErrorRate,
        "--error-rate", $ErrorRate,
        "--det-threshold", $DetThreshold,
        "--det-pos-weight", $DetPosWeight,
        "--kind-class-weights"
    )

    foreach ($weight in $KindClassWeights) {
        $arguments += $weight
    }

    $arguments += @(
        "--threshold-sweep"
    )

    foreach ($threshold in $ThresholdSweep) {
        $arguments += $threshold
    }

    $arguments += @(
        "--save-metric", $SaveMetric,
        "--lr-factor", $LrFactor,
        "--lr-patience", $LrPatience,
        "--lr-threshold", $LrThreshold,
        "--min-lr", $MinLr,
        "--early-stop-patience", $EarlyStopPatience,
        "--num-workers", $NumWorkers,
        "--output", $Output
    )

    if ($null -ne $MaxFiles) {
        $arguments += @("--max-files", $MaxFiles)
    }

    & $Python @arguments
}
finally {
    Pop-Location
}
