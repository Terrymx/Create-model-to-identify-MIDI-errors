#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: $0 binary-unified|three-class}"
PROJECT_DIR="${PROJECT_DIR:-/media/wwh/7382E9627565AA99/Create-model-to-identify-MIDI-errors}"
DATA_ROOT="${DATA_ROOT:-/media/wwh/7382E9627565AA99/maestro-v3.0.0-midi/maestro-v3.0.0}"
PYTHON="${PYTHON:-/home/wwh/anaconda3/envs/midi-error-detector/bin/python}"

case "$MODE" in
    binary-unified)
        BASE_CHECKPOINT="checkpoints/transformer_keyboard_aware_binary_unified_step2.pt"
        OUTPUT="checkpoints/transformer_binary_unified_directional_frozen.pt"
        LOG="training_logs/binary_unified_directional_frozen.log"
        ERR_LOG="training_logs/binary_unified_directional_frozen.err.log"
        ARCH_ARGS=(--unified-correction)
        ;;
    three-class)
        BASE_CHECKPOINT="checkpoints/transformer_keyboard_aware_step2.pt"
        OUTPUT="checkpoints/transformer_threeclass_directional_frozen.pt"
        LOG="training_logs/threeclass_directional_frozen.log"
        ERR_LOG="training_logs/threeclass_directional_frozen.err.log"
        ARCH_ARGS=()
        ;;
    *)
        echo "unknown mode: $MODE" >&2
        exit 2
        ;;
esac

cd "$PROJECT_DIR"
mkdir -p checkpoints training_logs
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" -B -u -m midi_error_detector.train \
    --model transformer \
    "${ARCH_ARGS[@]}" \
    --explicit-correction-evidence \
    --directional-forward-checkpoint checkpoints/transformer_forward_likelihood_leakage_safe.pt \
    --directional-backward-checkpoint checkpoints/transformer_backward_likelihood_leakage_safe.pt \
    --freeze-detector-backbone \
    --init-checkpoint "$BASE_CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    --eval-split validation \
    --clean-epochs 0 \
    --epochs 28 \
    --early-stop-patience 8 \
    --batch-size 8 \
    --window-size 256 \
    --num-layers 4 \
    --transformer-d-model 192 \
    --transformer-heads 4 \
    --transformer-ffn-dim 512 \
    --correction-embedding-dim 32 \
    --correction-evidence-groups 4 \
    --curriculum-error-rate-stages "0.08,0.12;0.02,0.05,0.08;0.005,0.01,0.02" \
    --error-rate 0.01 \
    --det-threshold 0.70 \
    --det-pos-weight 2.3 \
    --clean-theory-weight 1.5 \
    --error-theory-weight 1.5 \
    --pitch-loss-weight 0 \
    --kind-loss-weight 0 \
    --masked-pitch-loss-weight 0 \
    --clean-mask-batches-per-epoch 0 \
    --ranking-loss-weight 0.06 \
    --ranking-margin 0.65 \
    --ranking-top-k 64 \
    --hard-replay-size 512 \
    --hard-replay-epochs 1 \
    --asymmetric-hard-replay \
    --fn-replay-fraction 0.75 \
    --fn-replay-weight 1.5 \
    --fp-replay-weight 0.4 \
    --target-precision 0.8 \
    --threshold-sweep 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.93 0.95 \
    --save-metric precision_recall_score \
    --lr 0.001 \
    --lr-patience 3 \
    --lr-factor 0.5 \
    --lr-threshold 0.001 \
    --num-workers 0 \
    --output "$OUTPUT" \
    > "$LOG" \
    2> "$ERR_LOG"
