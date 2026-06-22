#!/usr/bin/env bash
set -euo pipefail

repo="/media/wwh/7382E9627565AA99/Create-model-to-identify-MIDI-errors"
python="/home/wwh/anaconda3/envs/midi-error-detector/bin/python"
data_root="/media/wwh/7382E9627565AA99/maestro-v3.0.0-midi/maestro-v3.0.0"

cd "$repo"
export PYTHONPATH="src:scripts"
export PYTHONDONTWRITEBYTECODE=1

echo "[$(date --iso-8601=seconds)] building piece-consistent B cache at 0.30/0.25"
"$python" -u scripts/build_counterfactual_candidate_cache.py \
  --threeclass-checkpoint checkpoints/transformer_threeclass_validation_directional_frozen.pt \
  --binary-checkpoint checkpoints/transformer_binary_deleteaux_validation_directional_frozen.pt \
  --forward-checkpoint checkpoints/transformer_forward_likelihood_leakage_safe.pt \
  --backward-checkpoint checkpoints/transformer_backward_likelihood_leakage_safe.pt \
  --data-root "$data_root" \
  --output-dir training_logs/counterfactual_b_030_025_piece_cache \
  --batch-size 8 \
  --seed 41 \
  --piece-consistent \
  --threeclass-candidate-threshold 0.30 \
  --binary-candidate-threshold 0.25

echo "[$(date --iso-8601=seconds)] B cache completed"
