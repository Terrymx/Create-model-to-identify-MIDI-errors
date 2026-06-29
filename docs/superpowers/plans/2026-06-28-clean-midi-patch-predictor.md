# Clean MIDI Patch Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a patch predictor and use observed-vs-edited patch likelihood gains as new verifier evidence for `C2_motif_hgb`.

**Architecture:** Keep the accepted candidate cache, motif features, and HGB verifier. The initial clean-only masked predictor was negative, so the active version trains a candidate-denoising contrastive objective: clean target pitch is the positive class and corrupted observed pitch is the negative class when they differ. Cached candidates are then scored by comparing observed patches with proposal-edited patches.

**Tech Stack:** Python, NumPy, PyTorch, scikit-learn HGB, existing `PieceConsistentVoiceDataset`, existing counterfactual candidate cache, existing motif verifier runner patterns.

---

### Task 1: Clean Patch Dataset and Tensor Builder

**Files:**
- Create: `scripts/clean_patch_predictor.py`
- Test: `tests/test_clean_patch_predictor.py`

- [x] **Step 1: Write the failing dataset test**

Add this test to `tests/test_clean_patch_predictor.py`:

```python
from __future__ import annotations

import unittest

import numpy as np

from clean_patch_predictor import (
    CleanPatchBatch,
    build_clean_patch_batch,
    build_candidate_patch_batch,
)


def _features_from_pitches(pitches: list[int]) -> np.ndarray:
    features = np.zeros((len(pitches), 36), dtype=np.float32)
    pitches_array = np.asarray(pitches, dtype=np.float32)
    features[:, 0] = pitches_array / 127.0
    features[:, 2] = 0.25
    features[:, 3] = 0.125
    features[:, 4] = np.diff(pitches_array, prepend=pitches_array[0]) / 24.0
    phase = 2.0 * np.pi * (pitches_array % 12.0) / 12.0
    features[:, 6] = np.sin(phase)
    features[:, 7] = np.cos(phase)
    return features


class CleanPatchPredictorTests(unittest.TestCase):
    def test_build_clean_patch_batch_masks_center_pitch_columns(self) -> None:
        features = _features_from_pitches([60, 62, 64, 65, 67])

        batch = build_clean_patch_batch(
            piece_features=features,
            center_positions=np.asarray([2], dtype=np.int64),
            radius=1,
        )

        self.assertIsInstance(batch, CleanPatchBatch)
        self.assertEqual(batch.context.shape, (1, 3, 39))
        self.assertEqual(batch.mask.shape, (1, 3))
        self.assertEqual(batch.target_pitch.shape, (1,))
        self.assertEqual(int(batch.target_pitch[0]), 64)
        self.assertAlmostEqual(float(batch.context[0, 1, 0]), 0.0)
        self.assertAlmostEqual(float(batch.context[0, 1, 6]), 0.0)
        self.assertAlmostEqual(float(batch.context[0, 1, 7]), 0.0)
        self.assertAlmostEqual(float(batch.context[0, 0, 0]), 62 / 127.0)

    def test_build_candidate_patch_batch_returns_observed_and_edited_contexts(self) -> None:
        features = _features_from_pitches([60, 62, 64, 65, 67])

        observed, edited, mask = build_candidate_patch_batch(
            piece_features=features,
            candidate_positions=np.asarray([2], dtype=np.int64),
            observed_pitch=np.asarray([64], dtype=np.float32),
            proposals=np.asarray([[63, 67]], dtype=np.float32),
            radius=1,
        )

        self.assertEqual(observed.context.shape, (1, 3, 39))
        self.assertEqual(edited.context.shape, (2, 3, 39))
        self.assertEqual(mask.shape, (1, 2))
        self.assertEqual(observed.target_pitch.tolist(), [64])
        self.assertEqual(edited.target_pitch.tolist(), [63, 67])
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_clean_patch_predictor.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'clean_patch_predictor'`.

- [x] **Step 3: Implement the minimal tensor builder**

Create `scripts/clean_patch_predictor.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PITCH_COLUMNS = (0, 6, 7)


@dataclass(frozen=True)
class CleanPatchBatch:
    context: np.ndarray
    mask: np.ndarray
    target_pitch: np.ndarray


def _pitch_columns_for(pitch: float) -> tuple[float, float, float]:
    phase = 2.0 * np.pi * ((float(pitch) % 12.0) / 12.0)
    return float(pitch) / 127.0, float(np.sin(phase)), float(np.cos(phase))


def _copy_patch(piece_features: np.ndarray, center: int, radius: int) -> tuple[np.ndarray, np.ndarray]:
    length = radius * 2 + 1
    base_dim = int(piece_features.shape[1])
    patch = np.zeros((length, base_dim + 3), dtype=np.float32)
    mask = np.zeros(length, dtype=np.float32)
    denominator = float(max(radius, 1))
    for offset in range(-radius, radius + 1):
        source = int(center + offset)
        target = offset + radius
        if source < 0 or source >= len(piece_features):
            continue
        patch[target, :base_dim] = piece_features[source]
        patch[target, base_dim] = float(offset) / denominator
        patch[target, base_dim + 1] = 1.0 if offset == 0 else 0.0
        patch[target, base_dim + 2] = 1.0
        mask[target] = 1.0
    return patch, mask


def build_clean_patch_batch(
    *,
    piece_features: np.ndarray,
    center_positions: np.ndarray,
    radius: int,
) -> CleanPatchBatch:
    rows = len(center_positions)
    length = radius * 2 + 1
    feature_dim = int(piece_features.shape[1]) + 3
    context = np.zeros((rows, length, feature_dim), dtype=np.float32)
    mask = np.zeros((rows, length), dtype=np.float32)
    target_pitch = np.zeros(rows, dtype=np.int64)
    for row, center in enumerate(np.asarray(center_positions, dtype=np.int64)):
        patch, patch_mask = _copy_patch(piece_features, int(center), radius)
        if 0 <= center < len(piece_features):
            target_pitch[row] = int(round(float(piece_features[int(center), 0]) * 127.0))
        patch[radius, list(PITCH_COLUMNS)] = 0.0
        context[row] = patch
        mask[row] = patch_mask
    return CleanPatchBatch(context=context, mask=mask, target_pitch=target_pitch)


def build_candidate_patch_batch(
    *,
    piece_features: np.ndarray,
    candidate_positions: np.ndarray,
    observed_pitch: np.ndarray,
    proposals: np.ndarray,
    radius: int,
) -> tuple[CleanPatchBatch, CleanPatchBatch, np.ndarray]:
    observed = build_clean_patch_batch(
        piece_features=piece_features,
        center_positions=candidate_positions,
        radius=radius,
    )
    observed_targets = np.asarray(np.rint(observed_pitch), dtype=np.int64)
    observed = CleanPatchBatch(
        context=observed.context,
        mask=observed.mask,
        target_pitch=observed_targets,
    )
    proposal_rows = []
    proposal_targets = []
    proposal_mask = np.isfinite(proposals).astype(np.float32)
    for row, center in enumerate(np.asarray(candidate_positions, dtype=np.int64)):
        for pitch in proposals[row]:
            batch = build_clean_patch_batch(
                piece_features=piece_features,
                center_positions=np.asarray([center], dtype=np.int64),
                radius=radius,
            )
            proposal_rows.append(batch.context[0])
            proposal_targets.append(int(round(float(pitch))))
    edited = CleanPatchBatch(
        context=np.asarray(proposal_rows, dtype=np.float32),
        mask=np.repeat(observed.mask, proposals.shape[1], axis=0),
        target_pitch=np.asarray(proposal_targets, dtype=np.int64),
    )
    return observed, edited, proposal_mask
```

- [x] **Step 4: Run the test to verify it passes**

Run:

```bash
$env:PYTHONPATH='scripts'; python -m pytest tests/test_clean_patch_predictor.py -v
```

Expected: PASS.

### Task 2: Masked Patch Predictor Model

**Files:**
- Modify: `scripts/clean_patch_predictor.py`
- Test: `tests/test_clean_patch_predictor.py`

- [x] **Step 1: Add failing tests for model shape and energy scoring**

Append tests that instantiate `CleanPatchPredictor`, pass a `CleanPatchBatch`,
and assert:

```python
self.assertEqual(tuple(logits.shape), (2, 128))
self.assertEqual(tuple(energy.shape), (2,))
self.assertLessEqual(float(energy.min()), float(energy.max()))
```

- [x] **Step 2: Run the model tests and verify they fail**

Run:

```bash
$env:PYTHONPATH='scripts'; python -m pytest tests/test_clean_patch_predictor.py -v
```

Expected: FAIL because `CleanPatchPredictor` and `patch_negative_log_likelihood` are not defined.

- [x] **Step 3: Implement model and scoring helpers**

Add a compact bidirectional GRU encoder that maps masked patch context to a 128-way center-pitch distribution. Add `patch_negative_log_likelihood(logits, target_pitch)` returning per-row cross entropy.

- [x] **Step 4: Run tests**

Run:

```bash
$env:PYTHONPATH='scripts'; python -m pytest tests/test_clean_patch_predictor.py -v
```

Expected: PASS.

### Task 3: Candidate Patch Feature Extraction

**Files:**
- Modify: `scripts/clean_patch_predictor.py`
- Test: `tests/test_clean_patch_predictor.py`

- [x] **Step 1: Add failing tests for verifier feature shape**

Add a test for `build_patch_predictor_features(observed_energy, proposal_energy, proposal_mask)` asserting a `(N, 8)` feature matrix containing observed energy, best edited energy, mean edited energy, best gain, mean gain, margin, valid proposal count, and any-improved flag.

- [x] **Step 2: Run and verify failure**

Run:

```bash
$env:PYTHONPATH='scripts'; python -m pytest tests/test_clean_patch_predictor.py -v
```

Expected: FAIL because `build_patch_predictor_features` is missing.

- [x] **Step 3: Implement feature aggregation**

Use lower energy as better. Define gain as `observed_energy - edited_energy`, so positive gain means the edit is more plausible than the observed note.

- [x] **Step 4: Run tests**

Run:

```bash
$env:PYTHONPATH='scripts'; python -m pytest tests/test_clean_patch_predictor.py -v
```

Expected: PASS.

### Task 4: Formal Runner

**Files:**
- Create: `scripts/run_clean_patch_predictor_verifier.py`
- Create: `scripts/run_clean_patch_predictor_verifier.sh`
- Create: `scripts/run_clean_patch_denoising_verifier.sh`
- Test: `tests/test_clean_patch_predictor.py`

- [x] Load `train`, `calibration`, and `test` arrays from the existing counterfactual C cache.
- [x] Run the clean-only version and reject it for recall improvement.
- [x] Add denoising candidate pretraining examples from train cache rows.
- [x] Train with target-vs-observed contrastive margin on corrupted train candidates.
- [ ] Append accepted motif features with `radius=4`, `min_similarity=0.84`, `exclude_radius=16`.
- [ ] Score observed and proposal-edited candidate patches for train/calibration/test.
- [ ] Evaluate:
  - `C2_motif_hgb`;
  - `PatchPredictor_direct`;
  - `C2_motif_patchpredictor_hgb`;
  - `C2_motif_patchpredictor_correction`.
- [ ] Save JSON, Markdown, and model checkpoint.

### Task 5: Verification and Experiment Logging

**Files:**
- Modify: `experiments.md`

- [ ] Run focused unit tests:

```bash
$env:PYTHONPATH='scripts'; python -m pytest tests/test_clean_patch_predictor.py tests/test_e2_learned_patch_energy.py -v
```

- [ ] Run syntax check:

```bash
python -m py_compile scripts/clean_patch_predictor.py scripts/run_clean_patch_predictor_verifier.py
```

- [ ] Run a small local smoke if data/checkpoints are locally available.
- [ ] Run the formal experiment remotely using the existing SSH/run-script pattern.
- [ ] Record accepted or rejected results in `experiments.md`.
- [ ] Accept the new detector only if it improves recall over `0.6123` at precision `>=0.80` under a matched protocol.
