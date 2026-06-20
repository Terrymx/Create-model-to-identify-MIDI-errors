# MIDI Error Correction Optimization Plan

Last updated: 2026-06-20

## Objective

Maximize wrong-note recall subject to precision `>= 0.80` on sparse-error MIDI.
The current best result is `P=0.8008`, `R=0.5963`. The frozen candidate union
has recall ceiling `0.7284`, so the immediate bottleneck is candidate ranking,
especially replacement errors.

## Current Evidence

- traditional end-to-end classifiers have high precision but very low recall;
- theory interactions, pairwise ranking, score fusion, short/scale calibration,
  and post-corruption register calibration do not beat the current frontier;
- delete candidate recall is already above `0.91`;
- replace candidate recall is about `0.68`;
- false negatives are concentrated among short notes and scale tones;
- manually partitioning these contexts produces only small score shifts and is
  now considered auxiliary evidence rather than the main optimization route.

## Updated Priority

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

The first success gate is B2/B3 improving recall over `0.5963` at precision
`>=0.80`. Neighborhood reruns proceed even if the gain is small, but global
edit decoding proceeds only if B+C produces a repeatable improvement.

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
