#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/media/wwh/7382E9627565AA99/Create-model-to-identify-MIDI-errors}"
DATA_ROOT="${DATA_ROOT:-/media/wwh/7382E9627565AA99/maestro-v3.0.0-midi/maestro-v3.0.0}"
PYTHON="${PYTHON:-/home/wwh/anaconda3/envs/midi-error-detector/bin/python}"

cd "$PROJECT_DIR"
mkdir -p checkpoints training_logs
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1

common_args=(
    --model transformer
    --data-root "$DATA_ROOT"
    --eval-split test
    --clean-epochs 0
    --epochs 36
    --early-stop-patience 10
    --batch-size 8
    --window-size 256
    --num-layers 4
    --transformer-d-model 192
    --transformer-heads 4
    --transformer-ffn-dim 512
    --curriculum-error-rate-stages "0.08,0.12;0.02,0.05,0.08;0.005,0.01,0.02"
    --error-rate 0.01
    --det-threshold 0.70
    --det-pos-weight 2.3
    --clean-theory-weight 1.5
    --error-theory-weight 1.5
    --pitch-loss-weight 0
    --kind-loss-weight 0
    --masked-pitch-loss-weight 0.35
    --masked-pitch-rate 0.18
    --clean-mask-batches-per-epoch 900
    --ranking-loss-weight 0.06
    --ranking-margin 0.65
    --ranking-top-k 64
    --hard-replay-size 512
    --hard-replay-epochs 1
    --asymmetric-hard-replay
    --fn-replay-fraction 0.75
    --fn-replay-weight 1.5
    --fp-replay-weight 0.4
    --target-precision 0.8
    --threshold-sweep 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.93 0.95
    --save-metric precision_recall_score
    --lr 0.0003
    --lr-patience 4
    --lr-factor 0.5
    --lr-threshold 0.001
    --num-workers 0
)

echo "starting binary Step 1A"
"$PYTHON" -B -u -m midi_error_detector.train \
    "${common_args[@]}" \
    --output checkpoints/transformer_keyboard_aware_binary_step1a.pt \
    > training_logs/keyboard_aware_binary_step1a.log \
    2> training_logs/keyboard_aware_binary_step1a.err.log

echo "starting binary explicit-surprise Step 2"
"$PYTHON" -B -u -m midi_error_detector.train \
    "${common_args[@]}" \
    --init-checkpoint checkpoints/transformer_keyboard_aware_binary_step1a.pt \
    --explicit-surprise \
    --surprise-train-mask-rate 0.25 \
    --surprise-eval-groups 4 \
    --surprise-embedding-dim 16 \
    --output checkpoints/transformer_keyboard_aware_binary_step2.pt \
    > training_logs/keyboard_aware_binary_step2.log \
    2> training_logs/keyboard_aware_binary_step2.err.log

echo "binary Step 1A and Step 2 complete"
