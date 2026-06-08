param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,

    [string]$Python = "python"
)

$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "src"
$env:PYTHONDONTWRITEBYTECODE = "1"

New-Item -ItemType Directory -Force -Path "checkpoints", "training_logs" | Out-Null

$teacherCheckpoint = "checkpoints\transformer_clean_likelihood.pt"
$teacherLog = "training_logs\external_likelihood_teacher.log"
$teacherErr = "training_logs\external_likelihood_teacher.err.log"

& $Python -B -u -m midi_error_detector.train `
    --model transformer `
    --data-root $DataRoot `
    --clean-only `
    --clean-epochs 8 `
    --epochs 0 `
    --batch-size 8 `
    --window-size 256 `
    --num-layers 4 `
    --transformer-d-model 192 `
    --transformer-heads 4 `
    --transformer-ffn-dim 512 `
    --det-loss-weight 0 `
    --pitch-loss-weight 0 `
    --kind-loss-weight 0 `
    --masked-pitch-loss-weight 1.0 `
    --masked-pitch-rate 0.18 `
    --lr 0.0003 `
    --lr-patience 0 `
    --num-workers 0 `
    --output $teacherCheckpoint 1> $teacherLog 2> $teacherErr

if ($LASTEXITCODE -ne 0) {
    throw "Clean likelihood training failed with exit code $LASTEXITCODE"
}

function Train-Branch {
    param(
        [string]$Name,
        [string]$InitCheckpoint,
        [string]$Output
    )

    $log = "training_logs\$Name.log"
    $err = "training_logs\$Name.err.log"

    & $Python -B -u -m midi_error_detector.train `
        --model transformer `
        --explicit-correction-evidence `
        --evidence-checkpoint $teacherCheckpoint `
        --freeze-detector-backbone `
        --init-checkpoint $InitCheckpoint `
        --data-root $DataRoot `
        --clean-epochs 0 `
        --epochs 18 `
        --early-stop-patience 6 `
        --batch-size 8 `
        --window-size 256 `
        --num-layers 4 `
        --transformer-d-model 192 `
        --transformer-heads 4 `
        --transformer-ffn-dim 512 `
        --curriculum-error-rate-stages "0.08,0.12;0.02,0.05,0.08;0.005,0.01,0.02" `
        --error-rate 0.01 `
        --det-threshold 0.70 `
        --det-pos-weight 2.3 `
        --clean-theory-weight 1.5 `
        --error-theory-weight 1.5 `
        --pitch-loss-weight 0 `
        --kind-loss-weight 0 `
        --ranking-loss-weight 0.06 `
        --ranking-margin 0.65 `
        --ranking-top-k 64 `
        --hard-replay-size 512 `
        --hard-replay-epochs 1 `
        --asymmetric-hard-replay `
        --fn-replay-fraction 0.75 `
        --fn-replay-weight 1.5 `
        --fp-replay-weight 0.4 `
        --target-precision 0.8 `
        --kind-class-weights 1 6 4 `
        --threshold-sweep 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.93 0.95 `
        --save-metric precision_recall_score `
        --lr 0.001 `
        --lr-patience 3 `
        --lr-factor 0.5 `
        --lr-threshold 0.001 `
        --num-workers 0 `
        --output $Output 1> $log 2> $err

    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Train-Branch `
    -Name "external_likelihood_branch_a_step2" `
    -InitCheckpoint "checkpoints\transformer_explicit_surprise_step2.pt" `
    -Output "checkpoints\external_likelihood_branch_a_step2.pt"

Train-Branch `
    -Name "external_likelihood_branch_b_stage1" `
    -InitCheckpoint "checkpoints\transformer_explicit_correction_detector.pt" `
    -Output "checkpoints\external_likelihood_branch_b_stage1.pt"
