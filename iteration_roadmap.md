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

## New Main Pipeline: Explicit Correction-to-Detection

This is the primary direction after the Step 2 and verifier experiments.

### Stage 1 - From-Scratch Explicit Detection

Train the detector from scratch. Detection remains the main task; correction is an
explicit evidence branch inside the same pipeline.

For each note:

1. mask pitch-derived and leakage-prone features at the target position;
2. predict a clean pitch distribution from bidirectional context;
3. explicitly compute:
   - observed-pitch surprise;
   - observed-pitch probability;
   - top predicted pitch probability;
   - top-vs-observed probability gap;
   - pitch-distribution entropy;
   - whether the top pitch differs from the observed pitch;
   - evidence availability;
4. project these seven interpretable values and concatenate them with the
   Transformer state before the detection head;
5. optimize detection as the main objective, with pitch correction and clean
   masked reconstruction as auxiliary supervision.

The correction evidence is detached by default before entering the detection head.
This prevents the detector loss from manipulating the pitch distribution into an
easy but musically meaningless classification signal.

Training data use:

- clean MIDI: masked pitch reconstruction, teaching what a plausible note should be;
- corrupted MIDI: wrong-note detection, error kind, and clean replacement pitch;
- early epochs: higher and more diverse corruption rates;
- late epochs: sparse corruption rates aligned with deployment;
- each epoch: FN-heavy asymmetric hard replay from the previous epoch.

This stage does not initialize from any previous detector checkpoint.

### Stage 2 - Add New Structural Evidence

If Stage 1 improves the PR frontier, add evidence that is not already present:

- phrase/motif repetition consistency;
- cross-occurrence comparison of repeated musical material;
- longer-range form and cadence context;
- separately trained forward/backward token likelihoods.

### Stage 3 - Ensemble

Train several Stage 1/2 models with different seeds and context choices. Prefer an
heterogeneous ensemble over copies of the exact same model.

### Stage 4 - Precision-Constrained Cascade

Use a high-recall Stage 1 candidate generator and a verifier only after the main
model receives stronger structural evidence. Select the operating threshold on a
file-held-out calibration split with a precision safety margin.

### Stage 5 - Product Calibration

Calibrate to a target precision above the deployment requirement, and report a
confidence interval or risk-control bound so that validation-to-test drift is less
likely to cross below precision `0.80`.

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

The first lightweight probe produced only about `+0.0106` recall at precision
`>=0.80` on the exploratory test frontier. This supports retaining a verifier as a
post-processing ablation, but not scaling it before adding genuinely new contextual
signals.

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

## 2026-06-11 Finalized Post-Step-2 Roadmap

Primary objective:

`maximize recall subject to precision >= 0.80`

### Gate 1 - Select the Directional Stage 2 Training Strategy

Compare two variants that use the same Step 2 initialization, directional teachers,
MAESTRO split, corruption curriculum, replay policy, and validation metric:

1. Joint fine-tuning:
   - freeze the forward and backward likelihood teachers
   - train the detector encoder, fusion projection, and error head
   - completed best validation result: precision `0.8064`, recall `0.4936`
2. Frozen detector:
   - freeze the Step 2 detector encoder and both directional teachers
   - train only the fusion projection and error head
   - validation selects the epoch and threshold; test remains untouched until selection

The frozen variant initializes from `transformer_explicit_surprise_step2.pt`. The
17-value correction-evidence branch is new, but the detector base is the established
Step 2 model rather than the weaker from-scratch explicit-correction branch.

Select the variant with the highest validation recall at precision `>= 0.80`. Then
evaluate both selected checkpoints once on the same test split.

### Stage 2 - Stronger Probabilistic Music-Theory Evidence

Represent theory as confidence-bearing continuous evidence rather than hard rules:

- local key and modulation confidence
- chord identity, inversion, and chord-confidence estimates
- passing, neighbor, suspension, and appoggiatura likelihood
- resolution plausibility
- beat strength combined with harmonic role
- voice, bass, and melody role
- phrase, cadence, and local structural context

The main target is reducing false positives on rare but valid ornamental notes.

### Stage 3 - Unified Candidate Dataset

Run the selected detector at a lower, high-recall threshold. Store each candidate
with:

- detector probability and logit
- internal, forward, and backward surprise
- top pitch alternative, probability gap, and entropy
- theory and local-structure evidence
- cross-model agreement
- ground-truth error label

Split train, calibration, and test by MIDI piece. No piece may contribute candidates
to more than one split.

### Stage 4 - Candidate Verifier Baselines

Train and compare the following models on exactly the same candidate dataset:

- logistic regression
- linear and RBF SVM
- XGBoost
- small MLP

These models serve both as traditional machine-learning paper baselines and as
Stage 2 cascade verifiers. Select by overall recall at precision `>= 0.80`, not by
candidate-only recall.

### Stage 5 - Heterogeneous Ensemble

Combine models with meaningfully different error patterns:

- original Step 2
- frozen Directional Stage 2
- different random seeds
- different corruption curricula
- Transformer and BiGRU
- the strongest traditional verifier when useful

Use validation-trained stacking or calibrated score fusion rather than blindly
averaging highly correlated models.

### Stage 6 - High-Recall Cascade

Use the strongest detector or ensemble at a deliberately low candidate threshold,
then apply the best verifier to remove false positives. Measure and report:

- Stage 1 candidate recall ceiling
- final precision, recall, F1, TP, FP, and FN
- recall loss introduced by the verifier

This is the main route toward recall `0.60+` while retaining precision `>= 0.80`.

### Stage 7 - Piece-Level Calibration

Calibrate on complete held-out pieces rather than treating notes as exchangeable:

- adapt to each piece's score distribution and candidate density
- consider a high-confidence threshold plus a per-piece top-K policy
- reserve a precision safety margin above `0.80`
- report piece-level variation and validation-to-test drift

Do not start Stages 2-7 until Gate 1 selects the main detector.
