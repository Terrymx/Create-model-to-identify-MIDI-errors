#!/usr/bin/env bash
set -euo pipefail

repo="/media/wwh/7382E9627565AA99/Create-model-to-identify-MIDI-errors"
python="/home/wwh/anaconda3/envs/midi-error-detector/bin/python"
cache_dir="training_logs/counterfactual_b_030_025_piece_cache"

cd "$repo"
export PYTHONPATH="src:scripts"
export PYTHONDONTWRITEBYTECODE=1

while [[ ! -f "$cache_dir/metadata.json" ]]; do
  sleep 20
done

"$python" -u scripts/run_counterfactual_piece_b.py \
  --cache-dir "$cache_dir" \
  --output-json training_logs/counterfactual_b_030_025_piece.json \
  --output-md training_logs/counterfactual_b_030_025_piece.md \
  --checkpoint-dir checkpoints/counterfactual_b_030_025_piece \
  --target-precision 0.80 \
  --seed 41
