# Verifier Theory Interaction Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add candidate-level music-theory and piano-error interaction features, then rerun the strongest frozen cascade verifier and piece-level calibration.

**Architecture:** Put deterministic feature construction in a focused `scripts/verifier_theory_features.py` module. The frozen verifier imports that module and appends its output to existing raw, detector, directional, local-score and piece-relative features. Existing detector checkpoints, candidate generation, dataset splits and calibration protocol remain unchanged.

**Tech Stack:** Python 3.10+, PyTorch tensors, NumPy, pytest, scikit-learn HistGradientBoosting.

---

### Task 1: Theory Interaction Feature Module

**Files:**
- Create: `scripts/verifier_theory_features.py`
- Create: `tests/test_verifier_theory_features.py`

- [ ] **Step 1: Write failing tests for output shape and finite values**

```python
import torch

from verifier_theory_features import build_theory_interaction_features


def base_inputs():
    raw = torch.zeros(1, 5, 36)
    valid = torch.tensor([[True, True, True, True, False]])
    three = torch.full((1, 5), 0.6)
    binary = torch.full((1, 5), 0.6)
    delete = torch.zeros(1, 5)
    forward = torch.zeros(1, 5)
    backward = torch.zeros(1, 5)
    return raw, valid, three, binary, delete, forward, backward


def test_theory_interactions_have_stable_finite_shape():
    values = build_theory_interaction_features(*base_inputs())
    assert values.shape == (1, 5, 57)
    assert torch.isfinite(values).all()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
$env:PYTHONPATH="src;scripts"
python -m pytest tests/test_verifier_theory_features.py -v
```

Expected: FAIL because `verifier_theory_features` does not exist.

- [ ] **Step 3: Write failing semantic tests**

```python
def test_ornament_protection_increases_for_resolving_passing_note():
    raw, valid, three, binary, delete, forward, backward = base_inputs()
    raw[0, 2, 26:33] = torch.tensor([1, 1, 1, 1, 0, 1, 1])
    values = build_theory_interaction_features(
        raw, valid, three, binary, delete, forward, backward
    )
    assert values[0, 2, -8] > values[0, 1, -8]


def test_accidental_touch_suspicion_uses_short_neighbor_delete_evidence():
    raw, valid, three, binary, delete, forward, backward = base_inputs()
    raw[0, 2, 4] = 1.0 / 24.0
    raw[0, 2, 33] = 0.05
    delete[0, 2] = 0.95
    values = build_theory_interaction_features(
        raw, valid, three, binary, delete, forward, backward
    )
    assert values[0, 2, -6] > values[0, 1, -6]


def test_padding_does_not_change_valid_local_statistics():
    raw, valid, three, binary, delete, forward, backward = base_inputs()
    first = build_theory_interaction_features(
        raw, valid, three, binary, delete, forward, backward
    )
    raw[0, 4] = 999.0
    second = build_theory_interaction_features(
        raw, valid, three, binary, delete, forward, backward
    )
    assert torch.allclose(first[:, :4], second[:, :4])
```

- [ ] **Step 4: Implement the minimal feature module**

Implement:

```python
THEORY_COLUMNS = {
    "in_major_scale": 8,
    "in_minor_scale": 9,
    "harmonic_density": 12,
    "harmonic_consonance": 14,
    "major_scale_distance": 18,
    "minor_scale_distance": 19,
    "chord_tone": 20,
    "step_in": 26,
    "step_out": 27,
    "passing_tone": 29,
    "neighbor_tone": 30,
    "resolves_by_step": 31,
    "non_chord_resolution": 32,
    "local_duration": 33,
    "downbeat_strength": 34,
    "subdivision_strength": 35,
}
```

The returned 57 dimensions are:

- 45 local statistics: mean/max at radii 4 and 8 plus center-minus-mean8 for nine theory sources;
- 12 interactions: ornament protection, unresolved non-chord suspicion, accidental-touch suspicion, strong-beat harmonic conflict, model consensus, model disagreement, consensus-theory support, disagreement-theory conflict, directional confirmation, directional disagreement, short-note surprise, delete-theory agreement.

All local operations must mask invalid notes and all outputs must use `torch.nan_to_num`.

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```powershell
$env:PYTHONPATH="src;scripts"
python -m pytest tests/test_verifier_theory_features.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```powershell
git add scripts/verifier_theory_features.py tests/test_verifier_theory_features.py
git commit -m "Add verifier theory interaction features"
```

### Task 2: Integrate Features Into Frozen Verifier

**Files:**
- Modify: `scripts/run_frozen_union_candidate_context_verifier.py`
- Modify: `tests/test_verifier_theory_features.py`

- [ ] **Step 1: Write a failing integration test**

Add a test that calls a small exported helper:

```python
from run_frozen_union_candidate_context_verifier import append_theory_features


def test_append_theory_features_preserves_prefix_and_adds_57_columns():
    base = torch.randn(1, 5, 11)
    raw, valid, three, binary, delete, forward, backward = base_inputs()
    result = append_theory_features(
        base, raw, valid, three, binary, delete, forward, backward
    )
    assert torch.equal(result[..., :11], base)
    assert result.shape[-1] == 68
```

- [ ] **Step 2: Run the integration test and verify RED**

Run:

```powershell
$env:PYTHONPATH="src;scripts"
python -m pytest tests/test_verifier_theory_features.py -v
```

Expected: FAIL because `append_theory_features` is missing.

- [ ] **Step 3: Implement the integration**

Import `build_theory_interaction_features`, add an `append_theory_features` helper, and call it after the existing verifier feature concatenation. Pass:

- raw 36-dimensional features;
- valid mask;
- three-frozen probability;
- binary-frozen probability;
- binary delete probability from the final encoded detector signal column;
- forward and backward normalized surprise.

Keep all old feature columns in their current order and append the 57 new columns.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
$env:PYTHONPATH="src;scripts"
python -m pytest tests/test_verifier_theory_features.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Run syntax validation**

Run:

```powershell
python -m py_compile scripts/verifier_theory_features.py scripts/run_frozen_union_candidate_context_verifier.py
```

Expected: exit code 0.

- [ ] **Step 6: Commit**

```powershell
git add scripts/run_frozen_union_candidate_context_verifier.py tests/test_verifier_theory_features.py
git commit -m "Use theory interactions in frozen cascade verifier"
```

### Task 3: Run Focused Verifier Experiment

**Files:**
- Generate: `training_logs/frozen_union_theory_verifier.log`
- Generate: `training_logs/frozen_union_theory_verifier.err.log`
- Generate: `training_logs/frozen_union_theory_verifier.json`
- Generate: `training_logs/frozen_union_theory_verifier.md`
- Generate: `checkpoints/frozen_union_theory_verifier/`

- [ ] **Step 1: Sync scripts to the training server**

Upload:

```text
scripts/verifier_theory_features.py
scripts/run_frozen_union_candidate_context_verifier.py
```

- [ ] **Step 2: Run the focused configuration**

Run only:

```text
three threshold = 0.60
binary threshold = 0.50
models = HistGradientBoosting and HistGradientBoosting small-leaf
```

The script receives a new `--hgb-only` option so RandomForest and Logistic Regression are skipped.

- [ ] **Step 3: Verify experiment completion**

Confirm:

- no traceback or runtime error;
- both HGB variants produced joblib checkpoints;
- JSON and Markdown reports exist;
- test frontier precision and recall are reported.

### Task 4: Run Piece-Level Calibration

**Files:**
- Generate: `training_logs/frozen_union_theory_calibration.log`
- Generate: `training_logs/frozen_union_theory_calibration.err.log`
- Generate: `training_logs/frozen_union_theory_calibration.json`
- Generate: `training_logs/frozen_union_theory_calibration.md`

- [ ] **Step 1: Run frozen-aware calibration**

Use the focused verifier output and evaluate:

- raw score;
- candidate-density adjustment;
- piece-rank blend;
- piece-z blend;
- precision margins from 0.00 through 0.05.

- [ ] **Step 2: Select the formal result**

Select the maximum test recall among strategies whose test precision is at least 0.80. Report:

- verifier candidate threshold pair;
- verifier model;
- calibration strategy;
- selected score threshold;
- precision;
- recall;
- F1;
- improvement over `P=0.8008, R=0.5963`.

- [ ] **Step 3: Record the experiment**

Append the result and whether recall crossed 0.60 to `experiments.md`. Do not claim success if the constrained recall does not improve.

