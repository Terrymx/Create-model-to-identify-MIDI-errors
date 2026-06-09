# Directional Likelihood Training Analysis

## Training Result

Both one-sided clean-music likelihood models completed four epochs:

| Model | Best validation loss | Perplexity | Pitch accuracy |
| --- | ---: | ---: | ---: |
| forward | 0.6705 | 1.9553 | 0.7717 |
| backward | 0.6980 | 2.0097 | 0.7522 |

The forward model improved through epoch 4 by validation loss. The backward model
also continued improving at epoch 4, so four epochs would not be a convincing
capacity limit.

## Validity Finding

These perplexity and accuracy values must not be used as evidence that Stage 2 has
succeeded.

The input sequence was shifted by one note and causal attention was enabled, but
the 36-dimensional feature vector contains derived fields that cross note
boundaries. For example:

- the previous note's `next_interval` and `step_out` depend on the current target;
- passing, neighbor, and resolution flags use both neighboring notes;
- backward prediction can receive the same information through `prev_interval`
  and `step_in`;
- harmonic, chord, and key features may be computed using notes that are not
  available in a strictly one-sided context.

The target pitch can therefore leak through a neighboring note's derived feature
vector. This plausibly explains the unusually low perplexity near `2.0`.

The checkpoints remain useful as an implementation diagnostic, but they are not
valid Stage 2 evidence and should not be fused into the detector.

## Position in the Roadmap

The project remains at the beginning of Stage 2: add genuinely new structural
evidence to the strongest detector platform.

- strongest main detector: Step 2 explicit-surprise checkpoint;
- current operating point: precision `0.8012`, recall `0.5307`;
- high-recall point: precision `0.7188`, recall `0.6123`;
- current hard-candidate internal-surprise AUC: `0.5426`;
- earlier external bidirectional teacher hard-candidate AUC: `0.5302`;
- earlier external-teacher fusion recall gain: only about `0.0012`.

The directional models are intended to provide non-redundant one-sided evidence,
but that hypothesis has not yet been tested without leakage.

## Next Experiment

### 1. Rebuild leakage-safe directional inputs

Use target-local non-pitch information, but exclude the target pitch and every
derived feature that can encode it. Context notes may expose their own pitch, but
their features must be computable from the permitted direction only.

The safest first version should use raw/local note attributes and recompute any
directional deltas after ordering the sequence. It should not reuse bidirectional
passing-tone, neighbor-tone, resolution, chord, or window-key fields.

### 2. Retrain forward and backward teachers

Train from scratch with validation early stopping. Allow up to eight epochs because
the backward validation curve was still improving at epoch 4.

### 3. Run the signal gate before detector training

On the same sparse validation corruption set, report:

- clean validation perplexity;
- all-note clean/error AUC and JS divergence;
- hard-candidate TP/FP AUC at Step 2 threshold `0.55`;
- Pearson and Spearman correlation with Step 2 internal surprise;
- forward, backward, mean, minimum, maximum, and disagreement signals.

Proceed only if the combined directional signal improves hard-candidate ranking
meaningfully beyond AUC `0.5426`. A practical gate is at least `+0.02` AUC, or a
clear file-held-out PR gain from a lightweight frozen-backbone probe.

### 4. Low-cost fusion probe

If the signal gate passes, freeze the Step 2 backbone and train only a small
evidence projection plus the detection head. Feed explicit forward/backward
probabilities, surprises, entropies, top alternatives, and directional
disagreement.

Continue to full Stage 2 training only if this probe improves recall by at least
`0.03` at precision `>=0.80`.

### 5. Later stages

Only after a successful Stage 2 single model:

1. train heterogeneous seeds/corruption curricula for an ensemble;
2. add a precision-constrained verifier;
3. perform piece-level calibration and risk control.

