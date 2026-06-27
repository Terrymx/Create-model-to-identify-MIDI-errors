#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/wwh/7382E9627565AA99/Create-model-to-identify-MIDI-errors"
PYTHON="/home/wwh/anaconda3/envs/midi-error-detector/bin/python"
DATA_ROOT="/media/wwh/7382E9627565AA99/maestro-v3.0.0-midi/maestro-v3.0.0"

cd "$ROOT"
export PYTHONPATH="src:scripts"

echo "[$(date --iso-8601=seconds)] E2 learned patch energy verifier started"
"$PYTHON" -u scripts/run_e2_learned_patch_verifier.py \
  --cache-dir training_logs/counterfactual_c_030_025_piece_cache \
  --data-root "$DATA_ROOT" \
  --output-json training_logs/e2_learned_patch_verifier.json \
  --output-md training_logs/e2_learned_patch_verifier.md \
  --checkpoint-dir checkpoints/e2_learned_patch_verifier \
  --target-precision 0.80 \
  --seed 43 \
  --motif-radius 4 \
  --motif-min-similarity 0.84 \
  --motif-exclude-radius 16 \
  --patch-radius 16 \
  --epochs 60 \
  --batch-size 384 \
  --hidden-dim 192 \
  --lr 0.0004 \
  --correction-weight 0.35
echo "[$(date --iso-8601=seconds)] E2 learned patch energy verifier completed"
