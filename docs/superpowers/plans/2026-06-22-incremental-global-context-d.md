# Incremental Global-Context D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a genuine piece-level sequential editor whose accepted pitch edits update the neural context and verifier scores used by later decisions.

**Architecture:** A pure state module maps absolute-note edits to overlapping windows and deterministic cache keys. A neural rescoring module rebuilds only affected windows from a coherent piece state, reruns the four frozen models, and refreshes B2+C(4/8/16) candidate features. A runner drives beam width 4, calibrates score floor/edit budget on validation pieces, measures runtime on a smoke subset, and applies the selected configuration once to test.

**Tech Stack:** Python, NumPy, PyTorch, scikit-learn/joblib, unittest, existing MAESTRO piece-consistent datasets and frozen checkpoints.

---

### Task 1: Piece edit state and affected windows

**Files:**
- Create: `scripts/incremental_global_context.py`
- Test: `tests/test_incremental_global_context.py`

- [ ] **Step 1: Write failing tests**

Test `state_key(piece_id, edits)`, `apply_piece_edits(features, edits)`, and
`affected_window_starts(position, note_count, window_size, stride)`. Require
sorted deterministic keys, pitch columns `0/6/7` updates only, and inclusion of
the non-stride-aligned final window.

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=src:scripts python -m unittest tests.test_incremental_global_context -v
```

Expected: import failure for `incremental_global_context`.

- [ ] **Step 3: Implement the pure state functions**

Add immutable `PieceEdit`, deterministic state/window keys, feature editing, and
window-start calculation without importing neural models.

- [ ] **Step 4: Run GREEN**

Run the Task 1 test module and expect all tests to pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/incremental_global_context.py tests/test_incremental_global_context.py
git commit -m "Add incremental piece edit state"
```

### Task 2: Candidate dependency and cache planning

**Files:**
- Modify: `scripts/incremental_global_context.py`
- Modify: `tests/test_incremental_global_context.py`

- [ ] **Step 1: Write failing tests**

Test candidate-to-window dependency mapping for C radius 16 and cache keys that
include only edits intersecting a requested window.

- [ ] **Step 2: Run RED**

Expected: missing `candidate_dependency_windows` and `window_state_key`.

- [ ] **Step 3: Implement dependency functions**

Map absolute candidate positions to every inference window needed by detector,
B, and C evidence. Ensure an edit outside a window does not invalidate its
cache key.

- [ ] **Step 4: Run GREEN**

Run the Task 1/2 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/incremental_global_context.py tests/test_incremental_global_context.py
git commit -m "Map global edits to affected windows"
```

### Task 3: Frozen-window neural rescoring

**Files:**
- Create: `scripts/incremental_window_rescorer.py`
- Test: `tests/test_incremental_window_rescorer.py`

- [ ] **Step 1: Write failing tests**

Use small fake directional/detector models to verify that a state edit rebuilds
the window, calls each model once on a cache miss, reuses cached outputs on a
hit, and changes candidate scores only when dependencies intersect.

- [ ] **Step 2: Run RED**

Expected: missing rescorer module.

- [ ] **Step 3: Implement `IncrementalWindowRescorer`**

Load the fitted B2+C2 HGB and frozen models through dependency injection.
Produce refreshed detector signals, proposals, B2 features, C2 features, and
verifier probabilities for requested candidate rows.

- [ ] **Step 4: Run GREEN**

Run the rescorer tests and syntax checks.

- [ ] **Step 5: Commit**

```bash
git add scripts/incremental_window_rescorer.py tests/test_incremental_window_rescorer.py
git commit -m "Rescore edited context windows"
```

### Task 4: Incremental beam driver

**Files:**
- Create: `scripts/run_incremental_global_context_d.py`
- Test: `tests/test_incremental_global_context_d.py`

- [ ] **Step 1: Write failing tests**

Test keep/edit branching, two proposals per note, one edit per absolute note,
state-dependent candidate ordering, edit budget, deterministic beam pruning,
and conversion of final states to note-level predictions.

- [ ] **Step 2: Run RED**

Expected: missing runner functions.

- [ ] **Step 3: Implement the beam driver**

Process candidates by current verifier score. After every accepted edit, ask the
rescorer for refreshed scores of all dependent remaining candidates. Use beam
width 4 and state/window caches. Expose counters for neural evaluations, cache
hits, and elapsed time.

- [ ] **Step 4: Run GREEN**

Run beam-driver and prior tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_incremental_global_context_d.py tests/test_incremental_global_context_d.py
git commit -m "Add genuine incremental global context search"
```

### Task 5: Calibration, smoke timing, and formal run

**Files:**
- Create: `scripts/run_incremental_global_context_d.sh`
- Modify: `experiments.md`

- [ ] **Step 1: Run remote tests**

Run all incremental tests plus existing counterfactual B/C/D tests in the
complete remote environment.

- [ ] **Step 2: Run smoke timing**

Run 4 validation and 2 test pieces. Record seconds per piece, neural window
evaluations, cache hit rate, peak GPU memory, and projected full runtime.

- [ ] **Step 3: Choose practical calibration grid**

If projected runtime is under 8 hours, use beam width 4 and edit budgets
`1.0%, 1.25%, 1.5%`. If it exceeds 8 hours, keep beam width 4 but rescore only
the top 32 currently affected candidates. This choice uses timing only, not
test labels.

- [ ] **Step 4: Run calibration and test**

Select score floor and edit budget on calibration at precision `>=0.81`, then
apply once to test. Save JSON/Markdown outputs and checkpoint identifiers.

- [ ] **Step 5: Record and commit**

Append matched C versus genuine incremental D metrics and runtime diagnostics
to `experiments.md`, then commit.
