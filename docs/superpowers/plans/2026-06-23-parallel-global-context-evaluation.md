# Parallel Global-Context Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe four-worker calibration/test execution that reuses the existing per-piece checkpoints and preserves serial evaluation results.

**Architecture:** Extract pure partitioning and checkpoint-aggregation helpers into the formal runner. Add phase and worker CLI controls, atomic writes, and a remote shell launcher that sequences calibration, selection, test, and final aggregation.

**Tech Stack:** Python 3, unittest, NumPy, PyTorch, Bash, JSON checkpoints.

---

### Task 1: Deterministic worker partitioning

**Files:**
- Modify: `tests/test_incremental_global_context_formal.py`
- Modify: `scripts/run_incremental_global_context_formal.py`

- [ ] Add failing tests proving four worker slices are disjoint, complete, and preserve sorted order.
- [ ] Run `python -m unittest tests.test_incremental_global_context_formal -v` and confirm the missing helper failure.
- [ ] Add `partition_piece_ids(piece_ids, worker_index, worker_count)` with argument validation.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Safe checkpoint aggregation

**Files:**
- Modify: `tests/test_incremental_global_context_formal.py`
- Modify: `scripts/run_incremental_global_context_formal.py`

- [ ] Add failing tests for atomic JSON writing and aggregation that rejects a missing expected piece.
- [ ] Run the focused tests and confirm the missing helper failures.
- [ ] Add atomic write/read helpers and an aggregation helper that reads every expected piece for one split/config.
- [ ] Change `_run_config` to compute only the assigned slice while preserving the existing checkpoint path.
- [ ] Run the focused tests and the full incremental-global test set.

### Task 3: Phase-separated orchestration

**Files:**
- Modify: `tests/test_incremental_global_context_formal.py`
- Modify: `scripts/run_incremental_global_context_formal.py`

- [ ] Add failing tests for selected-calibration persistence and incomplete calibration detection.
- [ ] Add `--phase`, `--worker-index`, and `--worker-count`.
- [ ] Implement calibration-only workers, test-only workers, and single-process finalization.
- [ ] Ensure finalization selects only after all 12×34 calibration checkpoints exist and writes the final result only after all test checkpoints exist.
- [ ] Run tests and `py_compile`.

### Task 4: Remote launcher and migration

**Files:**
- Create: `scripts/run_incremental_global_context_parallel.sh`

- [ ] Add a launcher that starts four workers with separate logs and waits for each phase.
- [ ] Sync changed files to the remote host.
- [ ] Run remote unit tests and a checkpoint-only four-worker smoke.
- [ ] Record the current checkpoint count, stop the old PID, launch the parallel runner, and verify four live Python workers with no duplicate or corrupt checkpoints.
- [ ] Report the preserved checkpoint count, new PIDs, and revised ETA.
