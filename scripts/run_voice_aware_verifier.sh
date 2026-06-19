#!/usr/bin/env bash
set -euo pipefail

method="${1:?usage: run_voice_aware_verifier.sh onset_matching|global_beam}"
beam_width="${2:-8}"

case "$method" in
  onset_matching)
    tag="onset"
    ;;
  global_beam)
    tag="beam"
    ;;
  *)
    echo "unknown voice method: $method" >&2
    exit 2
    ;;
esac

repo="/media/wwh/7382E9627565AA99/Create-model-to-identify-MIDI-errors"
python="/home/wwh/anaconda3/envs/midi-error-detector/bin/python"
data_root="/media/wwh/7382E9627565AA99/maestro-v3.0.0-midi/maestro-v3.0.0"
three_checkpoint="checkpoints/transformer_threeclass_validation_directional_frozen.pt"
binary_checkpoint="checkpoints/transformer_binary_deleteaux_validation_directional_frozen.pt"
forward_checkpoint="checkpoints/transformer_forward_likelihood_leakage_safe.pt"
backward_checkpoint="checkpoints/transformer_backward_likelihood_leakage_safe.pt"
verifier_prefix="frozen_union_voice_${tag}_verifier"
calibration_prefix="frozen_union_voice_${tag}_calibration"

cd "$repo"
export PYTHONPATH="src:scripts"
export PYTHONDONTWRITEBYTECODE=1

echo "[$(date --iso-8601=seconds)] starting verifier method=$method beam_width=$beam_width"
"$python" -u scripts/run_frozen_union_candidate_context_verifier.py \
  --threeclass-checkpoint "$three_checkpoint" \
  --binary-checkpoint "$binary_checkpoint" \
  --forward-checkpoint "$forward_checkpoint" \
  --backward-checkpoint "$backward_checkpoint" \
  --data-root "$data_root" \
  --threeclass-candidate-thresholds 0.60 \
  --binary-candidate-thresholds 0.50 \
  --target-precision 0.80 \
  --error-rate 0.01 \
  --batch-size 8 \
  --window-size 256 \
  --stride 128 \
  --voice-method "$method" \
  --voice-beam-width "$beam_width" \
  --hgb-only \
  --output-json "training_logs/${verifier_prefix}.json" \
  --output-md "training_logs/${verifier_prefix}.md" \
  --checkpoint-dir "checkpoints/${verifier_prefix}" \
  > "training_logs/${verifier_prefix}.log" \
  2> "training_logs/${verifier_prefix}.err.log"

echo "[$(date --iso-8601=seconds)] starting calibration method=$method"
"$python" -u scripts/calibrate_frozen_context_verifier.py \
  --threeclass-checkpoint "$three_checkpoint" \
  --binary-checkpoint "$binary_checkpoint" \
  --forward-checkpoint "$forward_checkpoint" \
  --backward-checkpoint "$backward_checkpoint" \
  --data-root "$data_root" \
  --context-json "training_logs/${verifier_prefix}.json" \
  --model-dir "checkpoints/${verifier_prefix}" \
  --output-json "training_logs/${calibration_prefix}.json" \
  --output-md "training_logs/${calibration_prefix}.md" \
  --target-precision 0.80 \
  --voice-method "$method" \
  --voice-beam-width "$beam_width" \
  --max-runs 4 \
  > "training_logs/${calibration_prefix}.log" \
  2> "training_logs/${calibration_prefix}.err.log"

echo "[$(date --iso-8601=seconds)] completed method=$method"
