#!/usr/bin/env bash
set -euo pipefail

repo="/media/wwh/7382E9627565AA99/Create-model-to-identify-MIDI-errors"
python="/home/wwh/anaconda3/envs/midi-error-detector/bin/python"
data_root="/media/wwh/7382E9627565AA99/maestro-v3.0.0-midi/maestro-v3.0.0"

cd "$repo"
export PYTHONPATH="src:scripts"
export PYTHONDONTWRITEBYTECODE=1

echo "[$(date --iso-8601=seconds)] starting verifier improvement suite"
"$python" -u scripts/run_verifier_improvement_suite.py \
  --threeclass-checkpoint checkpoints/transformer_threeclass_validation_directional_frozen.pt \
  --binary-checkpoint checkpoints/transformer_binary_deleteaux_validation_directional_frozen.pt \
  --forward-checkpoint checkpoints/transformer_forward_likelihood_leakage_safe.pt \
  --backward-checkpoint checkpoints/transformer_backward_likelihood_leakage_safe.pt \
  --data-root "$data_root" \
  --old-context-json training_logs/frozen_union_context_verifier_expanded.json \
  --old-model-dir checkpoints/frozen_union_context_verifier_expanded \
  --theory-context-json training_logs/frozen_union_theory_verifier.json \
  --theory-model-dir checkpoints/frozen_union_theory_verifier \
  --output-json training_logs/verifier_improvement_suite.json \
  --output-md training_logs/verifier_improvement_suite.md \
  --ranker-output checkpoints/verifier_pairwise_ranker.pt \
  --target-precision 0.80 \
  --batch-size 8 \
  --window-size 256 \
  --stride 128 \
  --error-rate 0.01 \
  --ranker-epochs 30 \
  --negatives-per-positive 4
echo "[$(date --iso-8601=seconds)] completed verifier improvement suite"
