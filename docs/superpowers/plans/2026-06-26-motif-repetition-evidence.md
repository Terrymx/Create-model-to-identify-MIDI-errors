# Motif Repetition Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add lightweight motif/repetition structural evidence to the existing counterfactual B/C verifier without retraining detector or teacher models.

**Architecture:** Create a pure NumPy feature module that scores each candidate and proposal against repeated interval-pattern neighborhoods in the same piece. Add a small experiment runner that appends these features to B2/C2 verifier features and evaluates the same precision-constrained recall protocol.

**Tech Stack:** Python, NumPy, scikit-learn HistGradientBoosting, existing counterfactual candidate cache `.npz` files.

---

### Task 1: Motif feature module

**Files:**
- Create: `scripts/motif_repetition_features.py`
- Test: `tests/test_motif_repetition_features.py`

- [ ] Write failing unittest coverage for transposition-invariant matching, self-neighborhood exclusion, proposal consensus gain, and empty-match fallback.
- [ ] Implement pure NumPy helpers:
  - `motif_feature_size() -> int`
  - `compute_motif_repetition_features(...) -> np.ndarray`
- [ ] Run `python -m unittest tests.test_motif_repetition_features -v`.

### Task 2: Verifier runner

**Files:**
- Create: `scripts/run_motif_repetition_verifier.py`
- Test: `tests/test_motif_repetition_verifier.py`

- [ ] Write failing tests that verify motif features append to B2 and C2 matrices without changing their prefixes.
- [ ] Implement the runner using existing cache loaders and `evaluate_score_rows`.
- [ ] Run the focused unittest files and `python -m py_compile` on new scripts.

### Task 3: Remote experiment handoff

**Files:**
- Create: `scripts/run_motif_repetition_verifier.sh`

- [ ] Add a shell wrapper pointing at the current B2+C cache and output paths.
- [ ] Sync only the new scripts/tests to the remote machine.
- [ ] Launch the motif verifier run and report expected duration.
