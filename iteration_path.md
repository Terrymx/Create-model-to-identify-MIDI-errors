# MIDI Wrong-Note Detector Iteration Path

This document records the main iteration path for the MAESTRO MIDI wrong-note detector and the reasoning behind the next optimization steps.

## Goal

The target use case is realistic sparse wrong-note detection in classical piano MIDI:

- a whole piece may contain only a few to a few dozen wrong notes
- false positives are costly because users need to inspect every reported note
- the long-term target is both precision and recall above `0.80`

Current best single-stage model is close on precision but not recall:

- high-precision threshold around `0.97`
- precision around `0.87`
- recall around `0.51`

This means threshold tuning alone cannot reach `80/80`; the model needs either better candidate generation, better calibration, or a second-stage verifier.

## Iteration 1: Baseline BiGRU

Initial model used a BiGRU over note windows with synthetic corruptions.

Problem:

- after many epochs, task score stayed limited
- detection recall/precision tradeoff was not good enough
- the model had limited capacity for long-range note context

Conclusion:

- move to a Transformer encoder for stronger contextual modeling

## Iteration 2: Transformer

The Transformer model improved sequence modeling and became the main architecture.

Key changes:

- `--model transformer`
- compact 4-layer Transformer encoder
- same heads for detection, action kind, and pitch correction

Result:

- previous Transformer baseline reached about `task_score=0.7743`
- better than the BiGRU direction, but sparse-error precision was still poor before recalibration

Conclusion:

- model capacity helped, but training/eval distribution still did not match real sparse-error use

## Iteration 3: Chord And Scale-Degree Features

Added music-theory features:

- key/scale membership
- major/minor profile strength
- scale degree
- local chord tone
- local chord root/role
- same-onset harmonic density and consonance

Purpose:

- give the model explicit harmonic context instead of requiring it to infer all tonal structure from raw pitch/time features

Result:

- helped enough to keep as part of the main feature set
- did not fully solve sparse-error precision/recall tradeoff

Conclusion:

- theory features are useful, but the model still needs to be calibrated for realistic low error rates

## Iteration 4: Low-Error-Rate Evaluation

Ran evaluation at sparse synthetic error rates such as `0.005`, `0.01`, and `0.02`.

Finding:

- under realistic low error rates, precision dropped sharply because false positives dominate when true errors are rare
- this confirmed that high-recall settings are not automatically useful in real usage

Conclusion:

- optimize explicitly for sparse-error precision, not only general task score

## Iteration 5: Low-Error-Rate Precision Fine-Tune

Fine-tuned from the Transformer theory checkpoint using sparse train/eval error rates:

- train error rates: `0.005`, `0.01`, `0.02`, `0.05`
- eval error rate: `0.01`
- save metric: `precision_task_score`
- threshold sweep: `0.8` to `0.99`

Best checkpoint:

- `checkpoints\transformer_precision_finetune.pt`
- best epoch `28`
- `precision_task_score=0.8106`
- `best_det_f0_5_precision=0.8753`
- `best_det_f0_5_recall=0.5006`
- `best_det_f0_5_threshold=0.97`

Conclusion:

- precision improved substantially
- recall became the main bottleneck

## Iteration 6: Precision-First Inference Post-Processing

Updated `predict.py` to better match real usage:

- default to checkpoint F0.5-selected threshold
- report only non-keep action candidates by default
- allow `--max-candidates`
- allow `--min-separation`
- support long MIDI through overlapping note windows

Result:

- real long MIDI smoke test ran successfully
- a 7894-note piece produced only 2 high-confidence candidates at threshold `0.97`

Conclusion:

- output became more usable, but underlying recall remained limited

## Iteration 7: Melodic-Theory Features

Added more melodic-context features:

- passing tone
- neighbor tone
- step in/out
- same-direction motion
- stepwise resolution
- non-chord-tone resolution
- local duration ratio
- approximate metrical strength

Best checkpoint:

- `checkpoints\transformer_melodic_theory_precision.pt`
- best epoch `24`
- `precision_task_score=0.8108`
- `best_det_f0_5_precision=0.8693`
- `best_det_f0_5_recall=0.5102`
- `best_det_f0_5_threshold=0.97`

Interpretation:

- tiny score improvement, mostly from recall
- precision did not improve
- features may be noisy because piano MIDI is multi-voice and adjacent notes are not always the same melodic line

Conclusion:

- melodic features alone are not enough

## Iteration 8: False-Positive Theory Analysis

Analyzed false positives at sparse error rate `0.01`.

Important finding:

- false positives were not mainly passing/neighbor/resolution tones
- passing/neighbor flags were only around `1%` of false positives
- many false positives were also chord tones

Conclusion:

- hard-coded passing/neighbor protection cannot solve the main false-positive source
- current false positives likely come from more complex patterns such as texture, repeated notes, chord voicing, arpeggiation, local density, or imperfect synthetic labels

## Iteration 9: Harmony-Aware Post-Processing Probe

Added lightweight harmony scoring:

- score current pitch harmony
- score predicted replacement pitch harmony
- compute harmony gain
- test filters like minimum harmony gain and current-pitch protection

Full test result at threshold `0.97`:

| Strategy | Precision | Recall | F0.5 |
| --- | ---: | ---: | ---: |
| baseline | `0.8693` | `0.5102` | `0.7620` |
| `min_gain_0.00` | `0.8690` | `0.4381` | `0.7262` |
| `min_gain_0.05` | `0.8804` | `0.3533` | `0.6781` |
| `protect_current_0.65_no_gain` | `0.8699` | `0.4164` | `0.7143` |

Conclusion:

- simple harmony filters remove false positives, but remove too many true positives
- classical harmony is likely useful, but not as a hard rule
- it should be used in a learned calibration/reranking stage

## Current Bottleneck

The current single-stage model has this tradeoff:

- high precision: about `P=0.87`, `R=0.51`
- lower threshold: recall can rise toward `0.65`, but precision falls too much

This indicates that the first-stage model can generate more true candidates, but cannot rank/filter them well enough by itself.

## Next Plan: Two-Stage Candidate Reranker

The most promising next direction is a two-stage system:

1. First-stage Transformer uses a lower threshold such as `0.85` or `0.90` to improve candidate recall.
2. Second-stage verifier/reranker decides which candidates should actually be reported.

Reranker inputs:

- first-stage error probability
- keep/replace/delete probabilities
- top pitch probability and pitch margin
- predicted pitch shift
- local note/theory features
- chord/scale features
- melodic features
- duration, density, beat strength
- optional harmony gain in later versions

Expected benefit:

- keep the extra true positives recovered by the lower threshold
- learn to reject false positives from hard-negative examples
- avoid brittle hand-written harmony rules

Initial target:

- move from about `P=0.87/R=0.51`
- toward `P>=0.82/R>=0.60`
- then iterate toward the long-term `80/80+` goal

## Next Experiments

1. Collect first-stage candidates at threshold `0.85`.
2. Label candidates as TP/FP using synthetic sparse-error labels.
3. Train a small PyTorch MLP reranker on candidate-level features.
4. Evaluate reranker thresholds on test split.
5. If promising, add hard-negative mining and per-piece top-K evaluation.
