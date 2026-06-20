#!/usr/bin/env bash
set -euo pipefail

repo="/media/wwh/7382E9627565AA99/Create-model-to-identify-MIDI-errors"
python="/home/wwh/anaconda3/envs/midi-error-detector/bin/python"
data_root="/media/wwh/7382E9627565AA99/maestro-v3.0.0-midi/maestro-v3.0.0"
b_cache="training_logs/counterfactual_d_piece_cache"
c_cache="training_logs/counterfactual_d_local_cache"

cd "$repo"
export PYTHONPATH="src:scripts"
export PYTHONDONTWRITEBYTECODE=1

echo "[$(date --iso-8601=seconds)] building piece-consistent B cache"
"$python" -u scripts/build_counterfactual_candidate_cache.py \
  --threeclass-checkpoint checkpoints/transformer_threeclass_validation_directional_frozen.pt \
  --binary-checkpoint checkpoints/transformer_binary_deleteaux_validation_directional_frozen.pt \
  --forward-checkpoint checkpoints/transformer_forward_likelihood_leakage_safe.pt \
  --backward-checkpoint checkpoints/transformer_backward_likelihood_leakage_safe.pt \
  --data-root "$data_root" \
  --output-dir "$b_cache" \
  --batch-size 8 \
  --seed 41 \
  --piece-consistent

echo "[$(date --iso-8601=seconds)] building piece-consistent local C cache"
"$python" -u scripts/build_counterfactual_local_cache.py \
  --cache-dir "$b_cache" \
  --output-dir "$c_cache" \
  --data-root "$data_root" \
  --forward-checkpoint checkpoints/transformer_forward_likelihood_leakage_safe.pt \
  --backward-checkpoint checkpoints/transformer_backward_likelihood_leakage_safe.pt \
  --batch-size 32 \
  --seed 41 \
  --piece-consistent

echo "[$(date --iso-8601=seconds)] fitting matched C1 and evaluating D1/D2/D3"
"$python" -u scripts/run_counterfactual_global_search.py \
  --cache-dir "$c_cache" \
  --output-json training_logs/counterfactual_global_d.json \
  --output-md training_logs/counterfactual_global_d.md \
  --checkpoint checkpoints/counterfactual_global_d/c1_piece_consistent.joblib \
  --target-precision 0.80 \
  --seed 41

echo "[$(date --iso-8601=seconds)] completed counterfactual D"
