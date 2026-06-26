# E1 Edit-Energy Verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an E1 learned counterfactual edit-energy verifier, then compare it against the current `C2_motif` frontier under `Precision >= 0.80`.

**Architecture:** Reuse the existing piece-consistent B/C candidate cache. Add a PyTorch model with a candidate detection head and a proposal correction head. The model consumes candidate-level context features and per-proposal edit features; detection is trained with weighted BCE and replacement proposal ranking is trained with cross-entropy when the clean target pitch appears in the proposal set.

**Tech Stack:** Python, NumPy, PyTorch, scikit-learn HGB baseline, existing `run_counterfactual_edit_verifier.py`, `run_motif_repetition_verifier.py`, and `evaluate_score_rows`.

---

### Task 1: Feature Builder and Proposal Targets

**Files:**
- Create: `scripts/e1_edit_energy_verifier.py`
- Test: `tests/test_e1_edit_energy_verifier.py`

- [ ] Write failing tests for:
  - `proposal_target_indices()` returns target proposal indices only for true replacement candidates.
  - `build_e1_feature_tensors()` returns aligned candidate and proposal tensors.

- [ ] Implement the minimal feature builder:
  - candidate features: existing `C2_motif` candidate matrix.
  - proposal features: `b_features`, `c_features`, `b_ranking`, `c_ranking`, normalized observed/proposed pitch, and normalized pitch delta.

- [ ] Run unit tests and confirm they pass.

### Task 2: E1 Model and Training Loop

**Files:**
- Modify: `scripts/e1_edit_energy_verifier.py`
- Test: `tests/test_e1_edit_energy_verifier.py`

- [ ] Add `E1EditEnergyNet` with:
  - proposal encoder;
  - candidate encoder;
  - candidate detection head;
  - per-proposal correction head.

- [ ] Add a smoke test proving a forward pass returns one detection logit per candidate and one proposal logit per proposal.

- [ ] Implement train/evaluate helpers using weighted BCE plus optional correction CE.

- [ ] Run unit tests and `py_compile`.

### Task 3: Runner and Remote Script

**Files:**
- Modify: `scripts/e1_edit_energy_verifier.py`
- Create: `scripts/run_e1_edit_energy_verifier.sh`

- [ ] Implement CLI:
  - load train/calibration/test caches;
  - append motif features with the accepted `r4_s084_e16` setting;
  - train `C2_motif_hgb` baseline;
  - train `E1_edit_energy`;
  - evaluate both with existing threshold-selection protocol.

- [ ] Save:
  - `training_logs/e1_edit_energy_verifier.json`;
  - `training_logs/e1_edit_energy_verifier.md`;
  - `checkpoints/e1_edit_energy_verifier/e1_edit_energy.pt`.

- [ ] Sync to remote, run unit tests remotely, then start the shell runner.

### Task 4: Result Decision

**Files:**
- Modify: `experiments.md`

- [ ] Pull remote result.

- [ ] If `E1_edit_energy` exceeds `C2_motif` recall `0.6122` at precision `>=0.80`, accept E1 and design E2.

- [ ] If E1 does not improve recall, record the negative result and design E2 only if correction-head diagnostics show useful but underused signal.
