param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,

    [string]$Python = "python"
)

$env:PYTHONPATH = "src"
$env:PYTHONDONTWRITEBYTECODE = "1"

foreach ($direction in @("forward", "backward")) {
    & $Python -B -u scripts\train_directional_likelihood.py `
        --data-root $DataRoot `
        --direction $direction `
        --epochs 8 `
        --early-stop-patience 2 `
        --batch-size 8 `
        --window-size 256 `
        --num-layers 4 `
        --d-model 192 `
        --heads 4 `
        --ffn-dim 512 `
        --lr 0.0003 `
        --num-workers 0 `
        --output "checkpoints\transformer_${direction}_likelihood_leakage_safe.pt" `
        1> "training_logs\directional_${direction}_likelihood_leakage_safe.log" `
        2> "training_logs\directional_${direction}_likelihood_leakage_safe.err.log"

    if ($LASTEXITCODE -ne 0) {
        throw "$direction likelihood training failed with exit code $LASTEXITCODE"
    }
}
