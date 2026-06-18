# Verifier Theory Interaction Enhancement

## Goal

Improve recall beyond the current calibrated result:

- precision: 0.8008
- recall: 0.5963
- candidate sources: three-class frozen at 0.60 and binary frozen at 0.50
- verifier: small-leaf histogram gradient boosting

The success target is recall above 0.5963 while preserving precision at or above
0.80. Crossing recall 0.60 is the immediate experiment goal.

## Scope

This experiment changes only candidate-level verifier features.

It does not change:

- MAESTRO corruption generation;
- detector architecture or weights;
- frozen directional teachers;
- candidate labels;
- train/calibration/test piece splits;
- the target precision constraint.

## Existing Signals

The 36-dimensional note representation already contains:

- scale membership and scale distance;
- harmonic density and consonance;
- chord-tone and chord-role evidence;
- previous and next melodic intervals;
- step-in, step-out and direction;
- passing-tone and neighbor-tone indicators;
- stepwise and non-chord resolution;
- relative duration;
- downbeat and subdivision strength.

The verifier also receives detector probabilities, correction confidence,
delete-auxiliary probability, internal/directional surprise, local score
statistics and piece-relative features.

## New Features

### Local theory context

For radii 4 and 8, compute masked local mean and maximum values for:

- chord-tone confidence;
- passing-tone indicator;
- neighbor-tone indicator;
- non-chord resolution;
- relative duration;
- downbeat strength;
- subdivision strength;
- scale distance;
- harmonic consonance.

These distinguish an isolated suspicious note from a locally consistent
ornament or harmonic pattern.

### Continuous interaction features

Add continuous features rather than hard filtering:

- ornament protection:
  passing/neighbor/resolution evidence combined with stepwise motion;
- unresolved non-chord suspicion:
  low chord support, low scale support and weak resolution;
- accidental-touch suspicion:
  short duration, small pitch displacement, binary delete score and onset
  density;
- strong-beat harmonic conflict:
  strong beat combined with weak chord/scale support and high surprise;
- model-theory agreement:
  detector consensus or disagreement combined with the theory scores;
- directional confirmation:
  high forward and backward surprise combined with weak ornament protection.

No feature directly suppresses or forces a prediction. The traditional
verifier learns their weights from candidate labels.

## Experiment

Use the strongest existing candidate configuration:

- three-class frozen threshold: 0.60;
- binary delete-aux frozen threshold: 0.50.

Train and compare:

- HistGradientBoosting;
- HistGradientBoosting with smaller leaves.

Use validation pieces for verifier training and calibration selection. Evaluate
test only after model and threshold selection. Then run the existing
piece-density, piece-rank and piece-z calibration strategies.

## Verification

Unit tests will verify:

- interaction features have stable dimensions;
- ornament protection increases for passing/neighbor/resolving examples;
- accidental-touch suspicion increases for short neighboring delete candidates;
- invalid/padded notes cannot affect local statistics;
- generated features contain no NaN or infinite values.

The experiment is accepted if it improves recall over 0.5963 at precision at
or above 0.80. If it does not, retain the current calibrated verifier and use
the feature ablation as a negative experimental result.
