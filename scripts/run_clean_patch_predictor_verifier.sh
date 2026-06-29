#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/wwh/7382E9627565AA99/Create-model-to-identify-MIDI-errors"
PYTHON="/home/wwh/anaconda3/envs/midi-error-detector/bin/python"
DATA_ROOT="/media/wwh/7382E9627565AA99/maestro-v3.0.0-midi/maestro-v3.0.0"

cd "$ROOT"
export PYTHONPATH="src:scripts"
export PYTHONDONTWRITEBYTECODE=1

echo "[$(date --iso-8601=seconds)] clean-MIDI patch predictor verifier started"
"$PYTHON" -u scripts/run_clean_patch_predictor_verifier.py \
  --cache-dir training_logs/counterfactual_c_030_025_piece_cache \
  --data-root "$DATA_ROOT" \
  --output-json training_logs/clean_patch_predictor_verifier.json \
  --output-md training_logs/clean_patch_predictor_verifier.md \
  --checkpoint-dir checkpoints/clean_patch_predictor_verifier \
  --target-precision 0.80 \
  --seed 41 \
  --motif-radius 4 \
  --motif-min-similarity 0.84 \
  --motif-exclude-radius 16 \
  --patch-radius 16 \
  --epochs 24 \
  --batch-size 512 \
  --hidden-dim 160 \
  --lr 0.0006 \
  --max-clean-patches 240000
echo "[$(date --iso-8601=seconds)] clean-MIDI patch predictor verifier completed"
