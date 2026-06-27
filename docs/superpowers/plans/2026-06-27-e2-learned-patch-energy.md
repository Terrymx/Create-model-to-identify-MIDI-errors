# E2 Learned Patch Energy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a learned patch-level edit-energy model that compares observed MIDI context against candidate-edited context, then test whether it improves recall at `Precision >= 0.80`.

**Architecture:** Keep `C2_motif_hgb` as the accepted detection baseline. Build note-window patch tensors around each candidate, create one edited patch per B proposal, train a shared patch encoder to score candidate detection and proposal correction, then evaluate both direct E2 scores and HGB with appended E2 evidence.

**Tech Stack:** Python, NumPy, PyTorch, scikit-learn HGB, existing piece-consistent B/C cache, `PieceConsistentVoiceDataset`.

---

### Task 1: E2 Patch Tensor and Model Module

**Files:**
- Create: `scripts/e2_learned_patch_energy.py`
- Test: `tests/test_e2_learned_patch_energy.py`

- [ ] Write failing tests for:
  - patch pair tensors have shape `(N, L, F)` for observed and `(N, K, L, F)` for edited proposals;
  - edited patch updates the center pitch and pitch-class columns while leaving observed patch unchanged;
  - model forward returns one candidate logit per row and one proposal logit per proposal.

- [ ] Implement tensor construction, normalizer, shared patch encoder, and training/prediction helpers.

### Task 2: E2 Verifier Runner

**Files:**
- Create: `scripts/run_e2_learned_patch_verifier.py`
- Create: `scripts/run_e2_learned_patch_verifier.sh`

- [ ] Load `counterfactual_c_030_025_piece_cache`.
- [ ] Append accepted motif evidence using `radius=4`, `min_similarity=0.84`, `exclude_radius=16`.
- [ ] Build observed/proposal patch tensors from piece-consistent validation/test datasets.
- [ ] Train E2 on train cache with BCE detection loss plus replacement proposal CE.
- [ ] Evaluate:
  - `C2_motif_hgb`;
  - `C2_motif_hgb_e2_correction`;
  - `E2_patch_energy`;
  - `C2_motif_e2_hgb`.
- [ ] Save JSON, Markdown, and checkpoints.

### Task 3: Verification and Remote Run

**Files:**
- Modify: `experiments.md` after result is known.

- [ ] Run unit tests and `py_compile`.
- [ ] Commit and push branch.
- [ ] Sync E2 files to remote.
- [ ] Run remote tests.
- [ ] Start `scripts/run_e2_learned_patch_verifier.sh`.
- [ ] Accept E2 for detection only if recall exceeds `0.6123` at precision `>=0.80`; otherwise keep it only as correction evidence if Top1/Top3 improves.
