#!/usr/bin/env bash
set -euo pipefail

repo="/media/wwh/7382E9627565AA99/Create-model-to-identify-MIDI-errors"
python="/home/wwh/anaconda3/envs/midi-error-detector/bin/python"
data_root="/media/wwh/7382E9627565AA99/maestro-v3.0.0-midi/maestro-v3.0.0"
cache_dir="training_logs/counterfactual_b_cache"
local_cache_dir="training_logs/counterfactual_bc_cache"

cd "$repo"
export PYTHONPATH="src:scripts"
export PYTHONDONTWRITEBYTECODE=1

echo "[$(date --iso-8601=seconds)] building shared B cache"
"$python" -u scripts/build_counterfactual_candidate_cache.py \
  --threeclass-checkpoint checkpoints/transformer_threeclass_validation_directional_frozen.pt \
  --binary-checkpoint checkpoints/transformer_binary_deleteaux_validation_directional_frozen.pt \
  --forward-checkpoint checkpoints/transformer_forward_likelihood_leakage_safe.pt \
  --backward-checkpoint checkpoints/transformer_backward_likelihood_leakage_safe.pt \
  --data-root "$data_root" \
  --output-dir "$cache_dir" \
  --batch-size 8

echo "[$(date --iso-8601=seconds)] evaluating B1/B2/B3"
"$python" -u scripts/run_counterfactual_edit_verifier.py \
  --cache-dir "$cache_dir" \
  --old-context-json training_logs/frozen_union_context_verifier_expanded.json \
  --old-model-dir checkpoints/frozen_union_context_verifier_expanded \
  --output-json training_logs/counterfactual_b_verifier.json \
  --output-md training_logs/counterfactual_b_verifier.md \
  --checkpoint-dir checkpoints/counterfactual_b_verifier

echo "[$(date --iso-8601=seconds)] building local C cache"
"$python" -u scripts/build_counterfactual_local_cache.py \
  --cache-dir "$cache_dir" \
  --output-dir "$local_cache_dir" \
  --data-root "$data_root" \
  --forward-checkpoint checkpoints/transformer_forward_likelihood_leakage_safe.pt \
  --backward-checkpoint checkpoints/transformer_backward_likelihood_leakage_safe.pt \
  --batch-size 32

echo "[$(date --iso-8601=seconds)] evaluating B+C"
"$python" -u scripts/run_counterfactual_edit_verifier.py \
  --cache-dir "$local_cache_dir" \
  --old-context-json training_logs/frozen_union_context_verifier_expanded.json \
  --old-model-dir checkpoints/frozen_union_context_verifier_expanded \
  --output-json training_logs/counterfactual_bc_verifier.json \
  --output-md training_logs/counterfactual_bc_verifier.md \
  --checkpoint-dir checkpoints/counterfactual_bc_verifier

echo "[$(date --iso-8601=seconds)] completed counterfactual B+C"
