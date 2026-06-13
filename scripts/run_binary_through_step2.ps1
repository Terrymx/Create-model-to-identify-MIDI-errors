param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,

    [string]$Python = "python",

    [int]$WaitForProcessId = 0
)

$ErrorActionPreference = "Stop"

if ($WaitForProcessId -gt 0) {
    $process = Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue
    if ($null -ne $process) {
        Write-Output "waiting for existing Step 2 process $WaitForProcessId"
        Wait-Process -Id $WaitForProcessId
    }
}

& "$PSScriptRoot\train_keyboard_aware_binary_step1a.ps1" `
    -DataRoot $DataRoot `
    -Python $Python

& "$PSScriptRoot\train_keyboard_aware_binary_step2.ps1" `
    -DataRoot $DataRoot `
    -Python $Python `
    -InitCheckpoint "checkpoints\transformer_keyboard_aware_binary_step1a.pt"
