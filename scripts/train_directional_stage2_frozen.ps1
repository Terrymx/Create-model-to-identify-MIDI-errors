param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,

    [string]$Python = "python",

    [string]$BaseCheckpoint = "checkpoints\transformer_keyboard_aware_unified_detector.pt"
)

$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "src"
$env:PYTHONDONTWRITEBYTECODE = "1"

& $Python -B -u -m midi_error_detector.train `
    --model transformer `
    --unified-correction `
    --explicit-correction-evidence `
    --directional-forward-checkpoint "checkpoints\transformer_forward_likelihood_leakage_safe.pt" `
    --directional-backward-checkpoint "checkpoints\transformer_backward_likelihood_leakage_safe.pt" `
    --freeze-detector-backbone `
    --init-checkpoint $BaseCheckpoint `
    --data-root $DataRoot `
    --eval-split validation `
    --clean-epochs 0 `
    --epochs 28 `
    --early-stop-patience 8 `
    --batch-size 8 `
    --window-size 256 `
    --num-layers 4 `
    --transformer-d-model 192 `
    --transformer-heads 4 `
    --transformer-ffn-dim 512 `
    --correction-embedding-dim 32 `
    --correction-evidence-groups 4 `
    --curriculum-error-rate-stages "0.08,0.12;0.02,0.05,0.08;0.005,0.01,0.02" `
    --error-rate 0.01 `
    --det-threshold 0.70 `
    --det-pos-weight 2.3 `
    --clean-theory-weight 1.5 `
    --error-theory-weight 1.5 `
    --pitch-loss-weight 0 `
    --kind-loss-weight 0 `
    --masked-pitch-loss-weight 0 `
    --clean-mask-batches-per-epoch 0 `
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
    --threshold-sweep 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.93 0.95 `
    --save-metric precision_recall_score `
    --lr 0.001 `
    --lr-patience 3 `
    --lr-factor 0.5 `
    --lr-threshold 0.001 `
    --num-workers 0 `
    --output "checkpoints\transformer_keyboard_aware_unified_directional_frozen.pt" `
    1> "training_logs\keyboard_aware_unified_directional_frozen.log" `
    2> "training_logs\keyboard_aware_unified_directional_frozen.err.log"

if ($LASTEXITCODE -ne 0) {
    throw "Frozen Directional Stage 2 failed with exit code $LASTEXITCODE"
}
