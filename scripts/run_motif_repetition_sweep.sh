#!/usr/bin/env bash
set -euo pipefail

repo="/media/wwh/7382E9627565AA99/Create-model-to-identify-MIDI-errors"
python="/home/wwh/anaconda3/envs/midi-error-detector/bin/python"
data_root="/media/wwh/7382E9627565AA99/maestro-v3.0.0-midi/maestro-v3.0.0"
cache_dir="training_logs/counterfactual_c_030_025_piece_cache"

cd "$repo"
export PYTHONPATH="src:scripts"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p training_logs/motif_repetition_sweep
mkdir -p checkpoints/motif_repetition_sweep

configs=(
  "4 0.84 16"
  "4 0.82 16"
  "4 0.80 16"
  "4 0.86 16"
  "4 0.88 16"
  "3 0.84 16"
  "3 0.82 16"
  "5 0.84 16"
  "5 0.82 16"
  "4 0.84 8"
  "4 0.84 32"
)

echo "[$(date --iso-8601=seconds)] motif/repetition focused sweep started"
for config in "${configs[@]}"; do
  read -r radius similarity exclude_radius <<< "$config"
  similarity_tag="${similarity/./}"
  tag="r${radius}_s${similarity_tag}_e${exclude_radius}"
  echo "[$(date --iso-8601=seconds)] running $tag"
  "$python" -u scripts/run_motif_repetition_verifier.py \
    --cache-dir "$cache_dir" \
    --data-root "$data_root" \
    --output-json "training_logs/motif_repetition_sweep/${tag}.json" \
    --output-md "training_logs/motif_repetition_sweep/${tag}.md" \
    --checkpoint-dir "checkpoints/motif_repetition_sweep/${tag}" \
    --target-precision 0.80 \
    --seed 41 \
    --motif-radius "$radius" \
    --motif-min-similarity "$similarity" \
    --motif-exclude-radius "$exclude_radius"
done

"$python" -u scripts/summarize_motif_sweep.py \
  training_logs/motif_repetition_sweep/*.json \
  --output-md training_logs/motif_repetition_sweep_summary.md \
  --min-precision 0.80 \
  --limit 40

echo "[$(date --iso-8601=seconds)] motif/repetition focused sweep completed"
