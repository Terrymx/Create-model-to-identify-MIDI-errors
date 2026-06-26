#!/usr/bin/env bash
set -euo pipefail

repo="/media/wwh/7382E9627565AA99/Create-model-to-identify-MIDI-errors"
python="/home/wwh/anaconda3/envs/midi-error-detector/bin/python"
data_root="/media/wwh/7382E9627565AA99/maestro-v3.0.0-midi/maestro-v3.0.0"
cache_dir="training_logs/counterfactual_c_030_025_piece_cache"

cd "$repo"
export PYTHONPATH="src:scripts"
export PYTHONDONTWRITEBYTECODE=1

echo "[$(date --iso-8601=seconds)] evaluating motif/repetition verifier"
"$python" -u scripts/run_motif_repetition_verifier.py \
  --cache-dir "$cache_dir" \
  --data-root "$data_root" \
  --output-json training_logs/motif_repetition_fdr_verifier.json \
  --output-md training_logs/motif_repetition_fdr_verifier.md \
  --checkpoint-dir checkpoints/motif_repetition_fdr_verifier \
  --target-precision 0.80 \
  --seed 41 \
  --motif-radius 4 \
  --motif-min-similarity 0.84 \
  --motif-exclude-radius 16

echo "[$(date --iso-8601=seconds)] motif/repetition verifier completed"
