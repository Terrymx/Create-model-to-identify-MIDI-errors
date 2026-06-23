#!/usr/bin/env bash
set -euo pipefail

WORKERS="${WORKERS:-4}"
PYTHON_BIN="${PYTHON_BIN:-/home/wwh/anaconda3/envs/midi-error-detector/bin/python}"
DATA_ROOT="${DATA_ROOT:-/media/wwh/7382E9627565AA99/maestro-v3.0.0-midi/maestro-v3.0.0}"
PROGRESS_DIR="${PROGRESS_DIR:-training_logs/incremental_global_context_progress}"
OUTPUT_JSON="${OUTPUT_JSON:-training_logs/incremental_global_context_formal.json}"
LOG_DIR="${LOG_DIR:-training_logs/incremental_global_context_parallel}"

mkdir -p "${LOG_DIR}"
export PYTHONPATH="src:scripts${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

COMMON_ARGS=(
  --cache-dir training_logs/counterfactual_c_030_025_piece_cache
  --data-root "${DATA_ROOT}"
  --threeclass-checkpoint checkpoints/transformer_threeclass_validation_directional_frozen.pt
  --binary-checkpoint checkpoints/transformer_binary_deleteaux_validation_directional_frozen.pt
  --forward-checkpoint checkpoints/transformer_forward_likelihood_leakage_safe.pt
  --backward-checkpoint checkpoints/transformer_backward_likelihood_leakage_safe.pt
  --verifier-checkpoint checkpoints/counterfactual_c_030_025_piece_fair/b2_c_radius4_8_16_small_leaf.joblib
  --progress-dir "${PROGRESS_DIR}"
  --output-json "${OUTPUT_JSON}"
  --seed 41
  --beam-width 4
  --calibration-precision 0.81
)

run_workers() {
  local phase="$1"
  local pids=()
  local worker
  for ((worker = 0; worker < WORKERS; worker++)); do
    "${PYTHON_BIN}" -u scripts/run_incremental_global_context_formal.py \
      "${COMMON_ARGS[@]}" \
      --phase "${phase}" \
      --worker-index "${worker}" \
      --worker-count "${WORKERS}" \
      >"${LOG_DIR}/${phase}.worker${worker}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "${phase} worker failed; inspect ${LOG_DIR}" >&2
    return 1
  fi
}

run_finalize() {
  local label="$1"
  "${PYTHON_BIN}" -u scripts/run_incremental_global_context_formal.py \
    "${COMMON_ARGS[@]}" \
    --phase finalize \
    >"${LOG_DIR}/finalize.${label}.log" 2>&1
}

run_workers calibration
run_finalize calibration
run_workers test
run_finalize test
