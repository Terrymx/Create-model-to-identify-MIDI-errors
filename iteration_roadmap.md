# MIDI Wrong-Note Detection Iteration Roadmap

## Product Goal

Detect sparse performance mistakes after automatic MIDI transcription. The primary
metric is recall under a practical precision constraint:

- target operating point: precision >= 0.80
- near-term target: recall >= 0.60
- long-term target: precision and recall both >= 0.80

Pitch correction is an optional recommendation. Reliable error localization remains
the main task.

## Current Evidence

The Step 2 explicit-surprise checkpoint is the current baseline:

- checkpoint: `checkpoints/transformer_explicit_surprise_step2.pt`
- precision `0.8012`, recall `0.5307` at threshold `0.85`
- precision `0.7188`, recall `0.6123` at threshold `0.70`
- mean clean surprise `2.5835`
- mean error surprise `6.7222`

Explicit contextual surprise is useful, but the score ranking still mixes genuine
errors with plausible ornamentation and other clean notes.

## Planned Route

### Step A - Multi-Window Surprise Validation

Freeze the Step 2 model and measure masked-context surprise with note windows of
`64`, `128`, and `256` on the same validation examples. Compare true-positive and
false-positive candidates at the high-recall Stage 1 operating point.

The hypothesis is:

- genuine wrong notes remain surprising across multiple context scales
- passing, neighbor, and ornamental tones may be locally surprising but become
  plausible in phrase-level context

This first experiment is diagnostic only. It reports the separation/AUC of each
scale and simple combinations such as the mean and minimum surprise. It does not
select a final threshold on the test set.

### Step B - Multi-Window Surprise Training

Proceed only if Step A shows complementary separation beyond the existing
single-window surprise. Add local, mid-range, and wide surprise channels to the
error head, then retrain with the Step 2 curriculum and asymmetric hard replay.

The formal ablation is:

- Step 1A without explicit surprise
- Step 2 with one `256`-note surprise channel
- Step 2B with `64`, `128`, and `256` surprise channels

### Step C - Production Cascade

Consider this only after the multi-window experiment. Freeze the strongest main
model, generate high-recall candidates, and train a lightweight verifier:

1. Stage 1 uses the strongest Transformer at a high-recall candidate threshold.
2. Stage 2 verifies only those candidates with richer local and theory context.
3. A held-out calibration split selects the final operating threshold.

Stage 2 cannot recover a note missed by Stage 1, so Stage 1 candidate recall must
stay comfortably above the final recall target.

### Step D - Ensemble or Larger Verifier

Use only after the single-model cascade is understood:

- average several independently trained Step 2 models, or
- replace the small verifier with a larger candidate-context model

This is a later stability and ceiling experiment, not the first diagnostic.

## Decision Rule

- If multi-window combinations separate TP from FP better than the `256`-note
  surprise alone, implement Step B.
- If the scales are nearly redundant, skip Step B and move to the cascade probe.
- Preserve a new baseline only if it improves the held-out PR frontier, especially
  recall at precision >= `0.80`.

## Experimental Guardrails

- Never select thresholds on the test split.
- Split verifier training and calibration by MIDI file, not by individual note.
- Always report overall recall against every error note, not recall only within the
  Stage 1 candidate set.
- Record Stage 1 candidate coverage, final TP/FP/FN, and the full threshold sweep.
- Keep the sparse evaluation corruption rate at `0.01`.
