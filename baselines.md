# Paper Baselines and Evaluation Plan

This document collects the baseline plan for a paper-style version of the project.

## Problem Setting

The target application is post-processing for AI music transcription.

Most audio-to-MIDI or audio-to-score systems transcribe what was played. This is appropriate for studio recordings or platform releases, where the released performance is usually treated as the canonical music. For live performance, rehearsal, student recordings, or imperfect covers, the performance itself may contain wrong notes, extra touches, or accidental slips. A transcription system will often preserve those mistakes in the MIDI output.

This project studies a reference-free post-transcription step:

```text
transcribed MIDI -> suspicious note detection -> correction/delete suggestions
```

The key constraint is that no reference score is available. The model must judge note plausibility from symbolic musical context.

## Relation to Existing Work

| Direction | Typical Setup | Difference From This Project |
| --- | --- | --- |
| Automatic music transcription | Audio is converted to MIDI or score | Transcribes what was played; does not decide whether the performer played an unintended note. |
| Score-informed performance error detection | Performance is compared with a reference score | Requires the correct score; our setting is reference-free. |
| MIDI degradation / AMT post-processing | Clean MIDI is synthetically degraded and corrected | Very close technically; our framing focuses on live-performance-derived transcription where mistakes may be real performance errors. |
| Symbolic music language modeling | Predict masked or future notes from context | Useful as a stronger baseline or auxiliary task; can score whether the observed note is plausible. |
| Symbolic denoising / token correction | Corrupted tokens are restored to clean tokens | Closest methodological direction for future work; wrong-note detection can be derived from low observed-token likelihood or high replacement-token confidence. |

Useful references to cite later:

- Nakamura et al., performance error detection with symbolic alignment, score/reference-based.
- MIDI Degradation Toolkit, synthetic symbolic degradation and correction for AMT post-processing.
- MidiBERT-piano / MusicBERT, BERT-style symbolic music pre-training.
- Symbolic token denoising / note-token correction work, which motivates replacing pure binary detection with contextual token reconstruction.

## Evaluation Protocol

Use MAESTRO MIDI as clean symbolic data. Generate sparse synthetic errors online or with a fixed seed for reproducible evaluation.

Recommended evaluation splits:

- train: MAESTRO train split
- validation: optional model selection if we want to keep test clean for final reporting
- test: MAESTRO test split for headline numbers

Recommended sparse test error rates:

- `0.005`: very sparse realistic setting
- `0.01`: main setting used so far
- `0.02`: moderately noisy performance/transcription setting

Metrics:

- note-level precision, recall, F1
- F0.5, because false positives are costly in score editing
- precision-constrained recall at `precision >= 0.80`
- false positives per piece
- top-1 and top-3 correction accuracy for replace errors
- delete accuracy for extra-touch errors
- per-error-type recall
- per-feature false-positive analysis: chord tone, scale tone, passing tone, neighbor tone, resolution

Primary reporting metric:

```text
precision-constrained recall at precision >= 0.80, eval error_rate=0.01
```

Secondary metric:

```text
F0.5 at best threshold
```

## Baseline Families

### B0: Rule Baselines

Purpose: show how far simple music-theory heuristics get without learning.

Candidate rules:

- flag non-scale tones under estimated local key
- flag non-chord tones under local vertical context
- flag large melodic jumps followed by no stepwise recovery
- flag short non-chord tones without passing/neighbor/resolution behavior
- top-K per piece by heuristic suspiciousness

Expected role:

- likely high precision in obvious cases
- likely low recall because many wrong notes are musically plausible

Status: not implemented yet.

### B1: BiGRU Detector

Purpose: simple contextual sequence classifier baseline.

Existing checkpoint/result:

| Checkpoint | Eval Setting | Best Threshold | Precision | Recall | F1 | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `bigru_wrong_note_theory_scheduler_taskscore.pt` | error_rate `0.08` | `0.5` | `0.5994` | `0.8343` | `0.6976` | High recall, too many false positives. |

Need:

- re-evaluate under the common paper setting `error_rate=0.01`
- report precision-constrained recall and F0.5

### B2: Transformer Detector

Purpose: stronger contextual detector using self-attention.

Existing checkpoints:

- `transformer_wrong_note_taskscore.pt`
- `transformer_chord_degree_taskscore.pt`

Need:

- run unified eval at `error_rate=0.005/0.01/0.02`
- include with and without theory features if possible

### B3: Precision-First Transformer

Purpose: show what sparse-error calibration alone can do.

Existing result:

| Checkpoint | Threshold | Precision | Recall | F0.5 | Precision Task Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `transformer_precision_finetune.pt` | `0.97` | `0.8753` | `0.5006` | not separately recorded here | `0.8106` |

Interpretation:

- good precision
- recall too low for the desired editing assistant

### B4: Melodic-Theory Transformer

Purpose: test whether explicit musical-context features reduce false positives.

Existing result:

| Checkpoint | Threshold | Precision | Recall | F0.5 | Precision Task Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `transformer_melodic_theory_precision.pt` | `0.97` | `0.8693` | `0.5102` | `0.7620` | `0.8108` |

Interpretation:

- slightly better recall than precision-first
- still not enough recall under high precision

### B5: Theory-Weighted Recall Fine-Tune

Purpose: recover recall by emphasizing theory-suspicious errors.

Existing result:

| Checkpoint | Best-F1 Threshold | Precision | Recall | F1 | F0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `transformer_theory_weighted_recall.pt` | `0.8` | `0.7545` | `0.6308` | `0.6871` | `0.7539` |

Interpretation:

- recall improved
- precision fell below the desired `0.80`
- mostly moves along the threshold trade-off curve

### B6: Coverage + Sparse Calibration

Purpose: combine diverse error coverage with sparse final calibration.

Existing result:

| Checkpoint | Precision-Constrained Threshold | Precision | Recall | F1 | F0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `transformer_coverage_calibrated.pt` | `0.85` | `0.8037` | `0.5751` | `0.6705` | `0.7540` |

Interpretation:

- restored precision above `0.80`
- recall dropped again
- does not improve the precision/recall frontier

### B7: Hard Ranking

Purpose: improve score ordering between hard positives and hard negatives.

Existing result:

| Checkpoint | Precision-Constrained Threshold | Precision | Recall | F1 | F0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `transformer_hard_ranking.pt` | `0.70` | `0.8063` | `0.5548` | `0.6573` | `0.7494` |

Interpretation:

- kept precision
- recall decreased
- simple in-batch ranking was too conservative

### B8: From-Scratch Hard Replay Curriculum

Purpose: test whether previous checkpoints impose a bad score-ordering prior.

Current run:

- `checkpoints\transformer_from_scratch_hard_replay.pt`
- clean warm-up
- coverage training
- sparse calibration
- online hard ranking
- epoch-to-epoch hard replay

Status:

- currently running
- expected to show whether starting from scratch changes the precision/recall frontier

### B9: Masked Context + Curriculum + Asymmetric Replay

Purpose: test whether recall collapse is mainly caused by an overly conservative training prior.

Design:

- keep the current Transformer, note features, detection/pitch/kind heads, masked pitch auxiliary objective, optimizer, scheduler, and eval protocol unchanged
- change only the training schedule and replay composition
- use a three-stage corruption curriculum:
  - early stage: `0.08, 0.12`, to expose diverse error types
  - middle stage: `0.02, 0.05, 0.08`, to bridge coverage and sparse deployment
  - late stage: `0.005, 0.01, 0.02`, to calibrate to the realistic sparse-error prior
- split hard replay into false-negative and false-positive buckets:
  - FN windows: `75%` of replay capacity, detection weight `1.5`
  - FP windows: `25%` of replay capacity, detection weight `0.4`

Hypothesis:

- previous hard replay was dominated by "do not report this" pressure, making the detector over-conservative
- curriculum keeps early error diversity while aligning the final prior with `eval_error_rate=0.01`
- FN-heavy replay should improve `recall @ precision >= 0.80`

Decision rule:

- if `recall @ P>=0.80 > 0.58`, proceed to an architecture method that exposes masked-context surprise to the detector head
- if recall lands in `0.45-0.58`, tune curriculum timing and FN replay weight
- if recall remains `<0.40`, move to loss/calibration redesign

## Proposed Main Method Candidate

The most paper-worthy next method is not another pure detector. It should combine:

1. contextual detector probability;
2. masked-pitch or denoising-based observed-token likelihood;
3. top predicted replacement pitch confidence;
4. sparse top-K post-processing per piece.

Potential wrong-note score:

```text
score(note) =
  alpha * detector_probability
  + beta * (1 - P(observed_pitch | context))
  + gamma * (P(top_pitch | context) - P(observed_pitch | context))
  + delta * correction_head_confidence
```

This would align better with symbolic music language modeling literature: instead of only asking whether a note is wrong, the model learns what note should be plausible at that position.

## Immediate To-Do List

1. Implement a unified eval script that takes a checkpoint and outputs one row of the baseline table.
2. Add rule baselines.
3. Re-evaluate all saved checkpoints under the same `error_rate=0.005/0.01/0.02` protocol.
4. Add per-error-type recall and false positives per piece.
5. Add false-negative / false-positive export for qualitative analysis.
6. Start a denoising or masked-pitch auxiliary objective experiment.
