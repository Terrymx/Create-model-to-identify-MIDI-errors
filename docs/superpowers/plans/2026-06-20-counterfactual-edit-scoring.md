# Counterfactual Edit Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add target-only bidirectional edit gain, local edited-window rescoring, and whole-piece beam selection for replacement-error candidates.

**Architecture:** A shared collector produces a cached proposal table from the frozen candidate generators and directional teachers. A local route enriches shortlisted proposals with edited-neighborhood likelihood deltas and trains a proposal verifier. A global route consumes the same shortlist in a whole-piece beam search with cached recomputation. All proposals and features use only post-corruption observed MIDI.

**Tech Stack:** Python, PyTorch, NumPy, scikit-learn HGB, unittest, MAESTRO MIDI dataset.

---

### Task 1: Target-only counterfactual probability interface

**Files:**
- Create: `scripts/counterfactual_edit_features.py`
- Test: `tests/test_counterfactual_edit_features.py`

- [ ] **Step 1: Write failing tests**

Test that one directional distribution produces:

- positive gain when a proposed pitch is more probable than observed;
- forward/backward agreement and disagreement values;
- finite values for probabilities near zero;
- no dependence on clean target pitch.

- [ ] **Step 2: Run RED**

Run remotely:

```bash
PYTHONPATH=src:scripts /home/wwh/anaconda3/envs/midi-error-detector/bin/python \
  -m unittest tests.test_counterfactual_edit_features -v
```

Expected: import failure because `counterfactual_edit_features` does not exist.

- [ ] **Step 3: Implement minimal B features**

Implement:

```python
def counterfactual_target_features(
    forward_probability,
    backward_probability,
    observed_pitch,
    proposed_pitch,
) -> torch.Tensor
```

Return forward gain, backward gain, mean/min/max gain, absolute disagreement,
proposal probabilities, observed probabilities, agreement flags, pitch
distance, direction, octave relation, and keyboard-neighbor relation.

- [ ] **Step 4: Run GREEN**

Run the Task 1 test and the existing directional leakage tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/counterfactual_edit_features.py tests/test_counterfactual_edit_features.py
git commit -m "Add target-only counterfactual edit features"
```

### Task 2: Inference-only proposal generation

**Files:**
- Modify: `scripts/counterfactual_edit_features.py`
- Test: `tests/test_counterfactual_edit_features.py`

- [ ] **Step 1: Write failing tests**

Test proposal union and deduplication from detector, forward, and backward Top-K:

- observed pitch is excluded;
- pitches are clipped to `0..127`;
- no clean target or corruption kind is accepted by the API;
- at most the configured proposal count remains;
- deterministic tie-breaking.

- [ ] **Step 2: Run RED**

Expected: missing `build_replacement_proposals`.

- [ ] **Step 3: Implement proposal generation**

Use only frozen model probability tensors. Default source Top-K is `3` per
source; retain at most `4` union proposals before B scoring and at most `2`
after B scoring.

- [ ] **Step 4: Run GREEN**

Run the focused tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/counterfactual_edit_features.py tests/test_counterfactual_edit_features.py
git commit -m "Add inference-only replacement proposals"
```

### Task 3: Shared candidate/proposal cache

**Files:**
- Create: `scripts/build_counterfactual_candidate_cache.py`
- Modify: `scripts/run_frozen_union_candidate_context_verifier.py`
- Test: `tests/test_counterfactual_candidate_cache.py`

- [ ] **Step 1: Write failing tests**

Test that cache rows preserve candidate file ID, absolute note position,
observed pitch, proposal pitch, label, replacement target for metrics only, base
verifier features, and B features. Verify proposal generation cannot read the
replacement target field.

- [ ] **Step 2: Run RED**

Expected: missing cache builder.

- [ ] **Step 3: Implement cache collection**

Expose full directional pitch distributions at candidate positions, collect
detector correction/pitch distributions, create proposal rows, and write
compressed `.npz` files for train/calibration/test plus JSON metadata.

- [ ] **Step 4: Verify**

Run unit tests and a two-file remote smoke. Confirm cache reload reproduces row
counts and B scores.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_counterfactual_candidate_cache.py scripts/run_frozen_union_candidate_context_verifier.py tests/test_counterfactual_candidate_cache.py
git commit -m "Cache counterfactual replacement proposals"
```

### Task 4: B-only proposal verifier and ablations

**Files:**
- Create: `scripts/run_counterfactual_edit_verifier.py`
- Test: `tests/test_counterfactual_edit_verifier.py`

- [ ] **Step 1: Write failing tests**

Test candidate aggregation by maximum proposal score, proposal Top-1 selection,
precision-constrained calibration, and B1/B2/B3 feature slicing.

- [ ] **Step 2: Run RED**

Expected: missing verifier helpers.

- [ ] **Step 3: Implement B verifier**

Train HGB models for B1/B2/B3, append B features to the historical best verifier
features, select thresholds on calibration files, and report detection and
replacement Top-1/Top-3 metrics.

- [ ] **Step 4: Remote formal B run**

Use correct directional-frozen checkpoints and historical candidate protocol.
Store outputs under `training_logs/counterfactual_b_*`.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_counterfactual_edit_verifier.py tests/test_counterfactual_edit_verifier.py experiments.md
git commit -m "Add B-only counterfactual verifier"
```

### Task 5: Safe edited-window construction

**Files:**
- Modify: `scripts/counterfactual_edit_features.py`
- Test: `tests/test_counterfactual_edit_features.py`

- [ ] **Step 1: Write failing invariance tests**

Verify editing pitch `p -> q` changes only safe pitch-derived columns `0`, `6`,
and `7`; all timing, velocity, theory, corruption, and target fields remain
unchanged.

- [ ] **Step 2: Run RED**

Expected: missing `apply_observed_pitch_edit`.

- [ ] **Step 3: Implement safe edit**

Update normalized pitch and pitch-class sine/cosine using the same formulas as
dataset feature extraction.

- [ ] **Step 4: Run GREEN**

Run focused tests plus directional target-leakage audit.

- [ ] **Step 5: Commit**

```bash
git add scripts/counterfactual_edit_features.py tests/test_counterfactual_edit_features.py
git commit -m "Add leakage-safe observed pitch edits"
```

### Task 6: Local C likelihood deltas

**Files:**
- Modify: `scripts/build_counterfactual_candidate_cache.py`
- Modify: `scripts/counterfactual_edit_features.py`
- Test: `tests/test_counterfactual_edit_features.py`

- [ ] **Step 1: Write failing tests**

Use deterministic fake probability tensors to verify radius `4/8/16` sums,
improved/degraded neighbor counts, directional sign agreement, and padding
isolation.

- [ ] **Step 2: Run RED**

Expected: missing `local_edit_impact_features`.

- [ ] **Step 3: Implement C features**

For the two best B proposals, run edited forward/backward windows and append
local likelihood deltas. Preserve unedited B cache so B and C are comparable.

- [ ] **Step 4: Run C smoke and formal run**

Store outputs under `training_logs/counterfactual_c_*`.

- [ ] **Step 5: Commit**

```bash
git add scripts/counterfactual_edit_features.py scripts/build_counterfactual_candidate_cache.py tests/test_counterfactual_edit_features.py experiments.md
git commit -m "Add local counterfactual edit impact"
```

### Task 7: Whole-piece beam state and conflicts

**Files:**
- Create: `scripts/counterfactual_beam_search.py`
- Test: `tests/test_counterfactual_beam_search.py`

- [ ] **Step 1: Write failing tests**

Test keep/replace transitions, one edit per note, deterministic pruning, edit
budget, conflict penalties, and cache keys derived from sorted edit sets.

- [ ] **Step 2: Run RED**

Expected: missing beam search module.

- [ ] **Step 3: Implement beam core**

Implement beam widths `4/8/16`, proposal floors, edit-count penalties, and
batched whole-piece recomputation callback. Keep scoring callback-injected so
the beam logic is unit-testable without neural models.

- [ ] **Step 4: Run GREEN**

Run beam tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/counterfactual_beam_search.py tests/test_counterfactual_beam_search.py
git commit -m "Add whole-piece counterfactual beam search"
```

### Task 8: Global D1/D2/D3 experiments

**Files:**
- Create: `scripts/run_counterfactual_global_search.py`
- Modify: `experiments.md`
- Test: `tests/test_counterfactual_global_search.py`

- [ ] **Step 1: Write failing tests**

Test conversion from beam-selected edits to note-level predictions and
precision/recall metrics, including unchanged pieces and overlapping windows.

- [ ] **Step 2: Run RED**

Expected: missing global-search evaluator.

- [ ] **Step 3: Implement global runner**

Load shared proposal caches, rebuild whole observed pieces, evaluate D1/D2/D3,
and select beam settings only on calibration pieces.

- [ ] **Step 4: Run smoke and formal global search**

Run beam widths `4/8/16` and calibrated edit budgets. Save outputs under
`training_logs/counterfactual_global_*`.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_counterfactual_global_search.py tests/test_counterfactual_global_search.py experiments.md
git commit -m "Evaluate global counterfactual edit search"
```

### Task 9: Final verification and comparison

**Files:**
- Modify: `experiments.md`
- Modify: `OPTIMIZATION_PLAN.md`

- [ ] **Step 1: Run all relevant tests**

```bash
PYTHONPATH=src:scripts /home/wwh/anaconda3/envs/midi-error-detector/bin/python \
  -m unittest \
  tests.test_counterfactual_edit_features \
  tests.test_counterfactual_candidate_cache \
  tests.test_counterfactual_edit_verifier \
  tests.test_counterfactual_beam_search \
  tests.test_counterfactual_global_search \
  tests.test_verifier_improvement_suite \
  tests.test_verifier_theory_features \
  tests.test_voice_assignment -v
```

- [ ] **Step 2: Compare A/B/C/D**

Report calibration-selected test precision, recall, F1, replacement recall,
correction Top-1/Top-3, runtime, and candidate/edit counts.

- [ ] **Step 3: Apply decision rule**

Retain only variants that improve the matched baseline at precision `>=0.80`.
Proceed to conformal/FDR calibration if ranking improves.

- [ ] **Step 4: Commit results**

```bash
git add experiments.md OPTIMIZATION_PLAN.md
git commit -m "Record counterfactual edit scoring results"
```
