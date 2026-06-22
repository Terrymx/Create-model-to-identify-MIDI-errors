# Incremental Global-Context D Design

## Goal

Evaluate whether sequential MIDI edits become more reliable when every accepted
edit updates the context used to score later edits. The experiment uses the
current piece-consistent, unique-note protocol and the retained
`B2+C(4/8/16)` verifier.

## State and data flow

Each beam state belongs to one complete MIDI piece and contains a sorted set of
absolute-note pitch edits. Applying the state produces one coherent piece-level
feature matrix. A candidate transition either keeps the current state or adds
one of the candidate's two retained correction proposals.

For every added edit, the runner finds every overlapping length-256 window that
contains the edited absolute note. It rebuilds those windows from the current
piece state and reruns the frozen three-class detector, binary detector,
forward teacher, and backward teacher. Candidate detector evidence, B2
proposal evidence, radius-4/8/16 local edit impact, and verifier probabilities
are then refreshed for candidates whose required context intersects a changed
window.

Later transitions therefore observe all edits already present in the beam
state. Scores are not reused from the original unedited cache when their
context has changed.

## Search

The first implementation uses beam width 4 and the top two correction proposals
already retained by C. Candidate order is descending matched-verifier score,
with absolute note position as a deterministic tie-breaker. The edit budget and
minimum acceptance score are selected only on calibration pieces.

Beam states are cached by `(piece_id, sorted absolute-position/pitch edits)`.
Window inference is cached by `(piece_id, window_start, relevant state edits)`
so states sharing unaffected windows reuse neural outputs.

## Evaluation

The matched `B2+C(4/8/16)` baseline and incremental D use the same HGB seed,
candidate set, error denominator, and calibration precision target `0.81`.
Test precision must remain at least `0.80`. Report precision, recall, F1,
runtime, neural window evaluations, cache hit rate, and edits per piece.

The initial smoke uses a few validation and test pieces to measure wall time and
memory. The full run starts only if projected runtime is practical; otherwise
beam width or the number of rescored candidates is reduced using a
calibration-only decision.

## Non-goals

- No whole-piece Transformer call beyond its trained window length.
- No clean target pitch or corruption provenance is used for proposals or
  scoring.
- No D2/D3 conflict penalties until the genuine incremental D1 baseline is
  measured.
