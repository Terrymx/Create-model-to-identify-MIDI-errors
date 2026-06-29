# MIDI Error Correction Optimization Plan

Last updated: 2026-06-28

## Objective

Maximize wrong-note recall subject to precision `>= 0.80` on sparse-error MIDI.
The accepted current detection mainline is `C2_motif_hgb`, with `P=0.8004`,
`R=0.6123`, and `F1=0.6938` under the unified piece-consistent protocol. The
expanded candidate pool has recall ceiling about `0.8328`, so the immediate
bottleneck is no longer candidate discovery. The active problem is separating
true errors from locally plausible candidates that are already in the candidate
pool.

## 2026-06-28 Active Target: Proposal-Selector Trust Verifier

Direction A result is now the main decision point. The proposal-selector gap
analysis was run remotely and saved as:

- `training_logs/proposal_selector_gap_analysis.md`
- `training_logs/proposal_selector_gap_analysis.json`

Key results:

- total test errors: `15639`;
- current `C2_motif_hgb` true positives: `9575`, recall `0.6123`;
- candidate false negatives: `3326`;
- candidate false negatives where the clean target is already in proposals:
  `2199`;
- ideal recall if only target-present candidate FNs were rescued: `0.7529`;
- ideal recall if all candidate FNs were rescued: `0.8249`.

For the `2199` target-present replacement FNs, existing proposal selectors rank
the target as follows:

| Selector | Top-1 | Top-2 | Top-3 | Mean rank |
| --- | ---: | ---: | ---: | ---: |
| B ranking | `0.5880` | `0.7808` | `0.9068` | `1.7244` |
| C aligned ranking | `0.5880` | `0.7808` | `0.9218` | `1.7094` |
| E1 proposal scores | `0.7904` | `0.9472` | `0.9914` | `1.2710` |

Conclusion: the bottleneck for this FN region is not proposal discovery and is
probably not proposal ranking either. E1 already ranks the correct pitch near
the top for almost every target-present candidate FN. The active optimization
should therefore be a proposal-selector trust verifier / candidate boost:

1. keep the accepted `C2_motif_hgb` detector and the current candidate cache;
2. focus only on replacement candidates whose proposal set has strong E1/B/C
   agreement and musical compatibility;
3. add candidate-level features that summarize proposal evidence, especially
   E1 best score, E1 margin, B/C/E1 agreement, target-like pitch relation,
   motif absence/weakness, and current C2 score;
4. train/calibrate a small rescue verifier or score boost on calibration data;
5. accept only if recall improves over `0.6123` while precision remains
   `>=0.80`.

This is intentionally not a new proposal generator. It answers: when a strong
proposal already exists, why does `C2_motif_hgb` still reject the candidate?

## 2026-06-28 Paused Direction: Clean-MIDI Patch Predictor

The next optimization target is a real clean-MIDI patch predictor / masked
denoising model. The previous E2 learned patch-energy experiment is rejected as
a detection mainline because it trained directly on sparse candidate rows and
did not improve recall. The replacement direction is to learn musical patch
structure from clean MIDI first, then use the learned patch likelihood as new
evidence for the existing verifier.

Core hypothesis:

```text
If a candidate is a short, in-key, or locally plausible wrong note, local
single-note and counterfactual evidence may be ambiguous. A patch model trained
on clean MIDI can still prefer the B/E1 proposed edit over the observed patch
when the observed note disrupts beat-level, onset-group, voice-contour, or
phrase-local structure.
```

The first clean-only implementation completed as a negative result: the patch
model learned center-pitch prediction, but its likelihood gain mostly acted as a
conservative precision filter. It removed `404` false positives but also lost
`310` true positives, reducing recall from `0.6123` to `0.6057`.

The denoising/contrastive implementation was also negative: it reduced recall
to `0.5947` at precision `0.8039`. The patch-predictor route is therefore
paused unless a later error slice shows a narrower patch-specific use case.

The attempted implementation kept the accepted detector and candidate cache,
but changed the patch model's training target from clean-only prediction to
candidate-denoising contrastive ranking:

1. train a patch predictor on train-split candidate rows, using the clean target
   pitch as the positive and the corrupted observed pitch as the negative when
   they differ;
2. at verifier time, build observed and proposal-edited patches around each
   cached candidate;
3. compute patch likelihood / reconstruction-energy features such as
   `edited_energy - observed_energy`, best proposal energy, and proposal margin;
4. append those features to `C2_motif_hgb` and evaluate under the same
   calibration/test protocol;
5. use E1-style proposal selection only as correction evidence, not as a
   replacement detector.

Original success gates:

- minimal success: improve over `C2_motif_hgb` at matched seed/protocol;
- target success: recall `> 0.6123` at precision `>= 0.80`;
- strong success: recall `>= 0.63` at precision `>= 0.80`, with measurable
  recovery of short-note or in-key false negatives.

Rejected near-term directions:

- more shallow HGB feature stacking without new information;
- the current D global sequential selector;
- the current supervised E2 candidate-row model;
- FDR as the primary recall-improvement mechanism;
- detector retraining before verifier evidence improves.

## Current Evidence

- traditional end-to-end classifiers have high precision but very low recall;
- counterfactual B/C evidence improves candidate ranking and remains part of the
  accepted verifier;
- motif/repetition evidence is the latest accepted recall improvement and is
  the current detection mainline;
- E1 and shallow patch features improve correction ranking but do not replace
  `C2_motif_hgb` for detection;
- FDR/risk-control improves some high-precision operating points but reduces
  recall at the active `P >= 0.80` target;
- the current E2 patch-energy model failed because it learned from candidate
  labels instead of learning normal clean-MIDI patch structure first.

## Updated Priority

### Priority 1: Proposal-selector trust verifier

Use E1/B/C proposal evidence to rescue target-present replacement FNs that the
current detector rejects. The highest-value slice is `replacement +
target-present + strong E1 rank + low C2 score`, especially where motif evidence
is weak or absent.

### Priority 2: Error-family specialization

Split the rescue analysis by error family, especially short in-key
replacements, out-of-key replacements, repeated-note/ornament contexts,
register/voice-crossing cases, and deletion touches.

### Priority 3: Clean or denoising patch evidence, only if narrowed

The broad clean/denoising patch predictor is negative and should not remain the
primary route. Revisit it only for a specific slice where proposal-selector
evidence is insufficient.

### Priority 4: Real AMT residual evaluation

Before making strong deployment claims, transcribe a small MAESTRO audio subset,
align predicted MIDI to reference MIDI, and compare real residual errors with
the synthetic benchmark. This supports paper validity and future calibration,
but it is not the fastest path to improving the current synthetic recall
frontier.

### Priority 5: Detector retraining or ensembling

Only revisit detector retraining or larger ensembles after verifier evidence
improves. The current candidate ceiling leaves enough room that better
candidate ranking should be tested first.

## Historical B+C Design

The following sections preserve the older counterfactual B/C/D plan because it
defines the accepted candidate cache, proposal, and verifier interfaces used by
the new patch-predictor experiment.

### Priority 1: Counterfactual edit scoring

For every replacement candidate, compare the observed pitch with several
proposed pitches. The main signal is not merely whether the observed note looks
unlikely, but whether a concrete replacement makes the sequence more likely.

The correction objective follows a noisy-channel interpretation:

`score(edit) = musical_plausibility(edited MIDI) + edit_prior - edit_cost`

Candidate proposals come only from information available at inference:

- the existing correction/pitch head Top-K;
- forward teacher Top-K;
- backward teacher Top-K;
- optionally the union of these proposals, deduplicated and clipped to the
  piano range.

The clean target pitch and synthetic corruption provenance must never be used to
create proposals or features.

### Priority 2: Global edit selection

If counterfactual scoring improves ranking, select a compatible set of edits
over the whole MIDI rather than accepting notes independently. The selector
will penalize conflicting edits, excessive edit count, voice discontinuity, and
harmony degradation.

The local B+C experiment and global selection will now both be implemented.
They share one cached candidate table and one set of B features:

- local route: rerun edited windows and learn candidate-level C features;
- global route: use the same shortlisted proposals in a whole-piece beam search,
  periodically recomputing directional likelihood after accepted edits.

The global route is not a brute-force search over every pitch. B retains at most
two replacement proposals per candidate, and the beam explores only compatible
edits above a calibration-selected proposal floor.

### Priority 3: Risk-controlled calibration

After ranking improves, use piece-level conformal or false-discovery-rate
control to preserve precision under calibration/test drift. This is a
calibration layer, not a substitute for better evidence.

### Priority 4: Human feedback and active learning

Use confirmed corrections and high-disagreement candidates to update the
replacement verifier and synthetic corruption distribution.

## B+C Design: Bidirectional Counterfactual Edit Gain

### Scope

The first implementation targets replacement candidates only. Delete edits
change sequence length and timing, while delete recall is already strong; delete
counterfactuals are deferred until replacement scoring is validated.

### B: Target-only counterfactual gain

Directional teachers predict each target pitch from one-sided context without
seeing that target pitch. Therefore one teacher pass gives a complete
distribution over all 128 pitches. For a proposed pitch `q` and observed pitch
`p`, compute:

- `forward_gain = log P_f(q | left context) - log P_f(p | left context)`;
- `backward_gain = log P_b(q | right context) - log P_b(p | right context)`;
- mean, minimum, maximum, and absolute disagreement of the two gains;
- proposal rank and probability under each teacher;
- whether both teachers prefer the same proposed pitch;
- gain over the best non-observed alternative;
- distance and keyboard relation between `p` and `q`.

This stage requires no edited-window rerun and can score every proposed pitch
for every candidate cheaply.

### C: Local edit-impact consistency

Run only for the best few proposals retained by B. Copy the observed feature
window and update only inference-visible pitch fields:

- normalized MIDI pitch;
- pitch-class sine;
- pitch-class cosine.

Do not recompute clean-key, clean-chord, corruption-type, or target-derived
fields. Re-run both directional teachers and measure the edited pitch's impact
on a local radius around the candidate:

- left/right neighborhood log-likelihood delta;
- number of neighboring notes improved or degraded;
- minimum directional neighborhood gain;
- forward/backward sign agreement;
- voice-local likelihood delta when voice assignment is confident;
- edit benefit normalized by pitch distance.

The default local radii are `4`, `8`, and `16` notes. B scores all proposals;
C reruns at most the best two proposals per candidate.

### D: Whole-piece counterfactual beam search

The whole-piece route consumes B-shortlisted proposals and processes them in
temporal order. A beam state contains:

- the accepted edit set;
- the current observed MIDI feature sequence;
- cumulative B and C gains;
- cumulative whole-piece directional-likelihood change;
- edit-count and edit-conflict penalties.

At each candidate, the beam can keep the note unchanged or accept one proposed
replacement. States are pruned by a calibration-selected objective:

`state_score = edit_gain + neighborhood_gain + global_gain - edit_cost - conflict_cost`

To control cost, whole-piece likelihood is not recomputed for every rejected
branch. It is recomputed after accepted edits in batches and cached by the edit
set hash. Initial experiments use beam widths `4`, `8`, and `16`, with a maximum
edit budget tied to piece note count and candidate density.

The global selector reports both independent-edit and beam-selected detection
metrics. It proceeds regardless of whether local C exceeds the current frontier,
because joint edits may expose collective sequence effects that independent
ranking cannot measure.

### Proposal and decision structure

Each candidate produces rows of:

`(candidate note, action=replace, proposed pitch, B features, C features)`

A lightweight HGB verifier ranks proposal rows. Candidate error probability is
the maximum valid replacement score. The selected correction pitch is the
highest-scoring proposal.

Evaluation reports both:

- detection: recall subject to precision `>=0.80`;
- correction: Top-1/Top-3 replacement pitch accuracy among detected replacement
  errors.

### Required ablations

| Run | Added evidence |
| --- | --- |
| A | historical best verifier |
| B1 | target-only forward gain |
| B2 | target-only forward + backward gain |
| B3 | B2 + agreement/disagreement features |
| C1 | B3 + radius-4 edited-neighborhood delta |
| C2 | C1 + radii 8/16 |
| C3 | C2 + confident same-voice delta |
| D1 | B3 proposals + whole-piece beam, no C features |
| D2 | C2 proposals + whole-piece beam |
| D3 | C3 proposals + whole-piece beam and voice-conflict penalty |

The first success gate is any B/C/D variant improving recall over `0.5963` at
precision `>=0.80`. Local and global routes are both executed; a route is
retained only if its calibration-selected operating point improves the matched
baseline on test.

## Leakage and Protocol Requirements

- corrupt complete pieces before slicing windows when whole-piece state is
  needed;
- all proposal generation uses observed MIDI and frozen model outputs only;
- calibration files select model, thresholds, and proposal limits;
- the test split is evaluated once with selected settings;
- report replace and delete recall separately;
- retain the historical candidate protocol for the primary comparison;
- any piece-consistent protocol result must include a matched baseline.

## Implementation Order

1. expose full forward/backward pitch distributions for candidate positions;
2. implement and unit-test B feature computation;
3. run a cached B-only experiment;
4. implement safe edited-window construction and invariance tests;
5. implement C local likelihood deltas;
6. train B+C verifier and run ablations;
7. add global edit selection only after the signal gate passes;
8. add conformal/FDR calibration after ranking is stable.

## Research Basis

- noisy-channel and language-model rescoring for transcription/correction;
- second-pass deliberation models for correcting an initial sequence;
- edit-based sequence correction rather than full sequence regeneration;
- mixture-of-experts anomaly ranking for heterogeneous contexts;
- conformal risk control and online false-discovery-rate control for precision
  guarantees under distribution drift.

Key references:

- https://arxiv.org/abs/1411.1623
- https://arxiv.org/abs/2003.07962
- https://arxiv.org/abs/2005.12592
- https://arxiv.org/abs/2309.06520
- https://arxiv.org/abs/1810.01403
- https://arxiv.org/abs/2208.02814
- https://arxiv.org/abs/1802.09098
