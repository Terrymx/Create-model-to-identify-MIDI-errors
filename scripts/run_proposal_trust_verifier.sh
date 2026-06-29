#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src:scripts

/home/wwh/anaconda3/envs/midi-error-detector/bin/python -u scripts/run_proposal_trust_verifier.py \
  --cache-dir training_logs/counterfactual_c_030_025_piece_cache \
  --data-root /media/wwh/7382E9627565AA99/maestro-v3.0.0-midi/maestro-v3.0.0 \
  --c2-checkpoint-dir checkpoints/clean_patch_predictor_verifier \
  --e1-checkpoint checkpoints/e1_edit_energy_verifier/e1_edit_energy.pt \
  --output-json training_logs/proposal_trust_verifier.json \
  --output-md training_logs/proposal_trust_verifier.md \
  --checkpoint-dir checkpoints/proposal_trust_verifier \
  --target-precision 0.80 \
  --seed 41 \
  --motif-radius 4 \
  --motif-min-similarity 0.84 \
  --motif-exclude-radius 16 \
  --batch-size 512
