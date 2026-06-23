# Cross-Configuration Piece Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorder calibration around pieces so all missing configurations reuse one piece scorer and its immutable edit-state caches.

**Architecture:** Extract one-piece preparation and one-configuration search into focused helpers. Add a piece-major calibration function that constructs one scorer, runs every missing configuration independently, and preserves the existing checkpoint format.

**Tech Stack:** Python 3, unittest, NumPy, PyTorch, JSON checkpoints.

---

### Task 1: Piece-major scheduling

**Files:**
- Modify: `tests/test_incremental_global_context_formal.py`
- Modify: `scripts/run_incremental_global_context_formal.py`

- [ ] Write a failing test proving a piece-major scheduler creates one scorer for multiple missing configurations.
- [ ] Run the focused test and confirm the helper is missing.
- [ ] Implement the minimal scheduler with injected scorer/search functions.
- [ ] Verify each configuration receives an independent search call while the scorer object is reused.

### Task 2: Formal runner integration

**Files:**
- Modify: `scripts/run_incremental_global_context_formal.py`
- Modify: `scripts/run_incremental_global_context_parallel.sh`

- [ ] Extract piece preparation and result serialization from `_run_config`.
- [ ] Add `_run_piece_configs` using one real `IncrementalPieceScorer`.
- [ ] Change calibration phase to piece-major execution and retain test phase behavior.
- [ ] Replace the four-worker launcher with a single shared-cache calibration process.
- [ ] Run all incremental-global tests and Python/Bash syntax checks.

### Task 3: Remote equivalence benchmark

**Files:**
- Create: `scripts/benchmark_cross_config_piece_cache.py`

- [ ] Evaluate at least two configurations on one remote piece with independent scorers.
- [ ] Evaluate the same configurations with one shared scorer.
- [ ] Assert identical edits, TP, FP, and candidate selections.
- [ ] Report elapsed time and window-cache misses for both modes.
- [ ] Continue the formal run only if equality holds and work decreases.
