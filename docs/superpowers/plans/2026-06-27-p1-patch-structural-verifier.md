# P1 Patch Structural Verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add patch-level structural evidence to the current `C2_motif_hgb` verifier and test whether recall improves at `Precision >= 0.80`.

**Architecture:** Keep the current best detector/verifier path intact. Use E1 only as a proposal selector to choose the edited pitch for each candidate, then compute observed-vs-edited local patch consistency features from piece-consistent note context. Train an HGB verifier on `C2_motif + patch` features and compare against `C2_motif_hgb`.

**Tech Stack:** Python, NumPy, PyTorch for E1 proposal logits, scikit-learn HGB, existing piece-consistent B/C cache, `PieceConsistentVoiceDataset`.

---

### Task 1: Patch Feature Module

**Files:**
- Create: `scripts/patch_structural_features.py`
- Test: `tests/test_patch_structural_features.py`

- [ ] Write failing tests for:
  - feature dimension is stable;
  - edited pitch closer to local pitch-class/pitch consensus gets positive gain;
  - edge candidates produce finite zero-padded features.

- [ ] Implement `compute_patch_structural_features()` for radii `(4, 8, 16)`.

- [ ] Run unit tests.

### Task 2: Patch Verifier Runner

**Files:**
- Create: `scripts/run_patch_structural_verifier.py`
- Create: `scripts/run_patch_structural_verifier.sh`

- [ ] Load the existing `counterfactual_c_030_025_piece_cache`.
- [ ] Append accepted motif features: `radius=4`, `min_similarity=0.84`, `exclude_radius=16`.
- [ ] Train E1 selector and use proposal logits to choose edited pitch.
- [ ] Append P1 patch features to train/calibration/test.
- [ ] Train and evaluate:
  - `C2_motif_hgb`;
  - `C2_motif_patch_hgb`;
  - `C2_motif_patch_e1_correction`.

- [ ] Save JSON, Markdown, and checkpoints.

### Task 3: Remote Run and Decision

**Files:**
- Modify: `experiments.md` after result is known.

- [ ] Run local tests and `py_compile`.
- [ ] Commit and push branch.
- [ ] Sync scripts/tests to remote.
- [ ] Run remote tests.
- [ ] Start `scripts/run_patch_structural_verifier.sh`.
- [ ] Accept P1 only if recall exceeds `0.6123` at precision `>=0.80`.
