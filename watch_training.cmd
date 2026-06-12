@echo off
title MIDI Training Progress
cd /d "E:\downloads\桌面\dku\CS309\project\code_new"
echo Real-time MIDI training progress. Closing this window will not stop training.
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -LiteralPath 'training_logs\keyboard_aware_unified_step1a.err.log' -Tail 20 -Wait"
pause
