param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,

    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

& "$PSScriptRoot\train_keyboard_aware_explicit_detector.ps1" `
    -DataRoot $DataRoot `
    -Python $Python

& "$PSScriptRoot\train_directional_stage2_frozen.ps1" `
    -DataRoot $DataRoot `
    -Python $Python `
    -BaseCheckpoint "checkpoints\transformer_keyboard_aware_unified_detector.pt"

& "$PSScriptRoot\train_directional_stage2.ps1" `
    -DataRoot $DataRoot `
    -Python $Python `
    -BaseCheckpoint "checkpoints\transformer_keyboard_aware_unified_detector.pt"
