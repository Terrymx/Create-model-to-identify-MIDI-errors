# Cross-Configuration Piece Cache Design

## Goal

Reduce genuine incremental global-context calibration time by reusing identical
neural and verifier computations across the 12 calibration configurations,
without changing any configuration's beam-search result.

## Root cause

The current formal runner iterates over configurations first and pieces second.
It constructs a new `IncrementalPieceScorer` for every `(configuration, piece)`
pair. The scorer already caches by immutable edit state, but the cache is
discarded before the next configuration processes the same piece.

## New execution order

Calibration will iterate over pieces first:

1. load one piece and construct one `IncrementalPieceScorer`;
2. run each missing configuration with an independent beam search;
3. reuse scorer results keyed only by the sorted pitch-edit state;
4. write the existing per-configuration, per-piece JSON checkpoint;
5. release the scorer and its caches before moving to the next piece.

The score floor, edit budget, beam state, considered positions, cumulative
utility, and final edits remain private to each configuration. Only deterministic
`edit state -> candidate scores` and neural window outputs are shared.

## Compatibility and safety

- Existing 166 checkpoints remain valid and are skipped.
- Checkpoint paths and JSON schemas do not change.
- Calibration selection and test evaluation remain unchanged.
- Test evaluation uses one selected configuration, so it does not need
  cross-configuration sharing.
- Atomic checkpoint writes remain enabled.
- Memory is bounded to one piece's shared state cache and released between
  pieces.

## Acceptance

Before formal continuation, one remote piece is evaluated both with independent
scorers and with a shared scorer. For every tested configuration, the selected
edit list and TP/FP values must match exactly. The shared run must perform fewer
window-cache misses or finish faster. If equality fails, the existing runner is
retained.
