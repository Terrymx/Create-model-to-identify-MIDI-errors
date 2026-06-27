#!/usr/bin/env bash
set -euo pipefail

repo="/media/wwh/7382E9627565AA99/Create-model-to-identify-MIDI-errors"
python="/home/wwh/anaconda3/envs/midi-error-detector/bin/python"
data_root="/media/wwh/7382E9627565AA99/maestro-v3.0.0-midi/maestro-v3.0.0"
cache_dir="training_logs/counterfactual_c_030_025_piece_cache"

cd "$repo"
export PYTHONPATH="src:scripts"
export PYTHONDONTWRITEBYTECODE=1

echo "[$(date --iso-8601=seconds)] P1 patch structural verifier started"
"$python" -u scripts/run_patch_structural_verifier.py \
  --cache-dir "$cache_dir" \
  --data-root "$data_root" \
  --output-json training_logs/patch_structural_verifier.json \
  --output-md training_logs/patch_structural_verifier.md \
  --checkpoint-dir checkpoints/patch_structural_verifier \
  --target-precision 0.80 \
  --seed 41 \
  --motif-radius 4 \
  --motif-min-similarity 0.84 \
  --motif-exclude-radius 16 \
  --patch-radii 4 8 16 \
  --epochs 45 \
  --batch-size 512 \
  --hidden-dim 192 \
  --lr 0.0005 \
  --correction-weight 0.25

echo "[$(date --iso-8601=seconds)] P1 patch structural verifier completed"
