#!/usr/bin/env bash
set -euo pipefail

repo="/media/wwh/7382E9627565AA99/Create-model-to-identify-MIDI-errors"
python="/home/wwh/anaconda3/envs/midi-error-detector/bin/python"
data_root="/media/wwh/7382E9627565AA99/maestro-v3.0.0-midi/maestro-v3.0.0"
b_cache="training_logs/counterfactual_b_030_025_piece_cache"
c_cache="training_logs/counterfactual_c_030_025_piece_cache"

cd "$repo"
export PYTHONPATH="src:scripts"
export PYTHONDONTWRITEBYTECODE=1

echo "[$(date --iso-8601=seconds)] building piece-consistent C cache"
"$python" -u scripts/build_counterfactual_local_cache.py \
  --cache-dir "$b_cache" \
  --output-dir "$c_cache" \
  --data-root "$data_root" \
  --forward-checkpoint checkpoints/transformer_forward_likelihood_leakage_safe.pt \
  --backward-checkpoint checkpoints/transformer_backward_likelihood_leakage_safe.pt \
  --batch-size 32 \
  --seed 41 \
  --piece-consistent

echo "[$(date --iso-8601=seconds)] evaluating B2/B3 plus C combinations"
"$python" -u scripts/run_counterfactual_piece_c.py \
  --cache-dir "$c_cache" \
  --output-json training_logs/counterfactual_c_030_025_piece.json \
  --output-md training_logs/counterfactual_c_030_025_piece.md \
  --checkpoint-dir checkpoints/counterfactual_c_030_025_piece \
  --target-precision 0.80 \
  --seed 41

echo "[$(date --iso-8601=seconds)] C comparison completed; D not started"
