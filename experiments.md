# Training Experiments

This file records model settings and high-level results so runs can be compared
without digging through terminal logs.

## Current Dataset And Evaluation

- Dataset: MAESTRO v3.0.0 MIDI-only
- Data root: `E:\downloads\桌面\dku\CS309\project\maestro-v3.0.0-midi\maestro-v3.0.0`
- Eval split: `test`
- Online corruption:
  - train error rate: `0.15`
  - eval error rate: `0.08`
- Main checkpoint metric: `task_score`
- Threshold sweep: `0.2 0.25 0.3 0.35 0.4 0.5`

## Run Log

| Date | Checkpoint | Model | Feature Size | Key Parameters | Best Test Metrics | Notes |
| --- | --- | --- | ---: | --- | --- | --- |
| 2026-05-30 | `checkpoints\bigru_wrong_note_theory_scheduler_taskscore.pt` | BiGRU | 16 | `clean_epochs=1`, `epochs=40`, `batch_size=16`, `window_size=256`, `det_pos_weight=3.0`, `kind_class_weights=1 6 4`, scheduler on `task_score` | `task_score=0.7426`, `best_det_f1=0.6451`, `best_det_precision=0.5287`, `best_det_recall=0.8271`, `replace_top3=0.8229`, `replace_kind_acc=0.8572`, `delete_kind_acc=0.9391` | Best BiGRU/theory baseline from earlier README notes. |
| 2026-05-31 | `checkpoints\transformer_wrong_note_taskscore.pt` | Transformer | 16 | `clean_epochs=1`, `epochs=40`, `batch_size=8`, `window_size=256`, `num_layers=4`, `d_model=192`, `heads=4`, `ffn_dim=512`, `det_pos_weight=3.0`, `kind_class_weights=1 6 4` | epoch 20: `task_score=0.7462`, `best_det_f1=0.6732`, `best_det_precision=0.5826`, `best_det_recall=0.7971`, `replace_top3=0.8141`; epoch 40: `task_score=0.7743`, `best_det_f1=0.6976`, `best_det_precision=0.5994`, `best_det_recall=0.8343`, `replace_top3=0.8500`, `replace_kind_acc=0.8520`, `delete_kind_acc=0.9383` | First Transformer run; clearly beats the BiGRU baseline. Best threshold stayed at `0.5`. |
| 2026-06-01 | `checkpoints\transformer_chord_degree_taskscore.pt` | Transformer | 24 | `clean_epochs=1`, `epochs=80`, `early_stop_patience=10`, `batch_size=8`, `window_size=256`, `num_layers=4`, `d_model=192`, `heads=4`, `ffn_dim=512`, `train_error_rate=0.15`, `error_rate=0.08`, `det_pos_weight=3.0`, `kind_class_weights=1 6 4`, threshold sweep `0.15 0.2 0.25 0.3 0.35 0.4 0.5`, scheduler on `task_score` | best epoch 79: `task_score=0.8174`, `best_det_f1=0.7486`, `best_det_precision=0.6618`, `best_det_recall=0.8615`, `best_det_threshold=0.5`, `replace_top1=0.5946`, `replace_top3=0.8970`, `replace_kind_acc=0.8754`, `delete_kind_acc=0.9648`, final epoch 80 `task_score=0.8163` | Chord/degree features improved over the 16-feature Transformer baseline by `+0.0431` task_score. Best checkpoint saved at epoch 79. |

## Completed Run: Transformer With Chord/Degree Features

Code change: `FEATURE_SIZE=24`.

New feature groups added on top of the previous 16 features:

- estimated major scale degree, `-1` for chromatic notes
- estimated minor scale degree, `-1` for chromatic notes
- distance to nearest major-scale pitch class
- distance to nearest minor-scale pitch class
- local triad chord-tone flag
- local chord-root sine
- local chord-root cosine
- local chord role: root, third, fifth, or non-chord tone

Suggested command:

```powershell
cd E:\downloads\桌面\dku\CS309\project\code_new

E:\downloads\桌面\dku\CS309\project\code\venv\Scripts\python.exe -u -m midi_error_detector.train `
  --model transformer `
  --data-root "E:\downloads\桌面\dku\CS309\project\maestro-v3.0.0-midi\maestro-v3.0.0" `
  --clean-epochs 1 `
  --epochs 80 `
  --early-stop-patience 10 `
  --batch-size 8 `
  --window-size 256 `
  --num-layers 4 `
  --transformer-d-model 192 `
  --transformer-heads 4 `
  --transformer-ffn-dim 512 `
  --train-error-rate 0.15 `
  --error-rate 0.08 `
  --det-threshold 0.3 `
  --det-pos-weight 3.0 `
  --kind-class-weights 1 6 4 `
  --threshold-sweep 0.15 0.2 0.25 0.3 0.35 0.4 0.5 `
  --save-metric task_score `
  --lr-patience 4 `
  --lr-factor 0.5 `
  --lr-threshold 0.002 `
  --num-workers 0 `
  --output checkpoints\transformer_chord_degree_taskscore.pt
```

Final best checkpoint: epoch 79, `task_score=0.8174`.

## Precision-First Roadmap

Real use is expected to have sparse errors: usually only a few to a few dozen wrong notes per piece.
The next experiments should therefore optimize for high precision and acceptable recall, not only F1.

Planned sequence:

1. Low-error-rate evaluation with the current best checkpoint.
2. Low-error-rate fine-tuning, preferably with mixed or staged error rates such as `0.005`, `0.01`, `0.02`, `0.05`, `0.10`, `0.15`.
3. Precision-first inference post-processing, such as high thresholds, top-K per piece, or local peak filtering.
4. More music-theory features aimed at reducing false positives: passing tones, neighbor tones, resolution, meter/downbeat strength, melodic interval context, and key-confidence gating.
5. Dataset expansion after the precision problem is quantified, with ASAP first and larger piano MIDI corpora only after quality checks.

## Low Error Rate Precision Eval

Checkpoint: `checkpoints\transformer_chord_degree_taskscore.pt`.

Command output files:

- `training_logs\low_error_precision_eval.json`
- `training_logs\low_error_precision_eval.md`

Summary:

| Target Error Rate | Observed | Best Threshold By F0.5 | Precision | Recall | F0.5 | FP | FN | Note |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `0.005` | `0.0058` | `0.95` | `0.3621` | `0.7038` | `0.4011` | `10400` | `2485` | Too many false positives for real use. |
| `0.010` | `0.0115` | `0.95` | `0.5335` | `0.7040` | `0.5607` | `10328` | `4967` | Better, but still too many false positives. |
| `0.020` | `0.0229` | `0.95` | `0.6928` | `0.6932` | `0.6929` | `10254` | `10232` | Precision becomes usable only when the piece has many errors. |

Interpretation:

- The best threshold among `0.5` to `0.95` is always `0.95` when optimizing F0.5.
- The model produces roughly `10k` false positives across the full test split even at `threshold=0.95`.
- This confirms that the next step should not be more recall; it should be calibration/fine-tuning for sparse-error precision.
- For realistic use, we should test stricter thresholds beyond `0.95` and add per-piece candidate limits before changing the model architecture again.

## Planned Run: Precision-First Low-Error Fine-Tune

Goal: reduce false positives under sparse real-world error rates while keeping useful recall.

Starting point: `checkpoints\transformer_chord_degree_taskscore.pt` at epoch 79.

Key changes from the previous run:

- initialize from the best chord/degree Transformer checkpoint
- train on sparse mixed error rates: `0.005 0.01 0.02 0.05`
- evaluate at `error_rate=0.01`
- reduce detection positive weight from `3.0` to `1.0`
- use high detection thresholds: `0.8 0.85 0.9 0.95 0.97 0.98 0.99`
- save by `precision_task_score`, which emphasizes detection F0.5

Suggested command:

```powershell
cd E:\downloads\妗岄潰\dku\CS309\project\code_new

E:\downloads\妗岄潰\dku\CS309\project\code\venv\Scripts\python.exe -u -m midi_error_detector.train `
  --model transformer `
  --init-checkpoint checkpoints\transformer_chord_degree_taskscore.pt `
  --data-root "E:\downloads\妗岄潰\dku\CS309\project\maestro-v3.0.0-midi\maestro-v3.0.0" `
  --clean-epochs 0 `
  --epochs 30 `
  --early-stop-patience 8 `
  --batch-size 8 `
  --window-size 256 `
  --num-layers 4 `
  --transformer-d-model 192 `
  --transformer-heads 4 `
  --transformer-ffn-dim 512 `
  --train-error-rates 0.005 0.01 0.02 0.05 `
  --error-rate 0.01 `
  --det-threshold 0.95 `
  --det-pos-weight 1.0 `
  --kind-class-weights 1 6 4 `
  --threshold-sweep 0.8 0.85 0.9 0.95 0.97 0.98 0.99 `
  --save-metric precision_task_score `
  --lr 0.0001 `
  --lr-patience 3 `
  --lr-factor 0.5 `
  --lr-threshold 0.001 `
  --num-workers 0 `
  --output checkpoints\transformer_precision_finetune.pt
```

## Result: Precision-First Low-Error Fine-Tune

Checkpoint: `checkpoints\transformer_precision_finetune.pt`.

Status: completed 30 corrupt-stage epochs. Best checkpoint saved at epoch 28 by `precision_task_score`.

Key configuration:

- initialized from `checkpoints\transformer_chord_degree_taskscore.pt`
- train error rates cycled through `0.005`, `0.01`, `0.02`, `0.05`
- eval error rate `0.01`, observed `0.0115`
- default detection threshold `0.95`
- threshold sweep `0.8 0.85 0.9 0.95 0.97 0.98 0.99`
- `det_pos_weight=1.0`
- saved by `precision_task_score`

Best checkpoint metrics:

| Metric | Value |
| --- | ---: |
| `epoch` | `28` |
| `precision_task_score` | `0.8106` |
| `task_score` | `0.7800` |
| `best_det_f0_5` | `0.7613` |
| `best_det_f0_5_precision` | `0.8753` |
| `best_det_f0_5_recall` | `0.5006` |
| `best_det_f0_5_threshold` | `0.97` |
| `best_det_f1` | `0.6852` |
| `best_det_precision` | `0.7338` |
| `best_det_recall` | `0.6426` |
| `best_det_threshold` | `0.85` |
| `det_precision` at threshold `0.95` | `0.8410` |
| `det_recall` at threshold `0.95` | `0.5452` |
| `det_f1` at threshold `0.95` | `0.6616` |
| `det_f0_5` at threshold `0.95` | `0.7587` |
| `replace_pitch_top3` | `0.9130` |
| `replace_kind_acc` | `0.8367` |
| `delete_kind_acc` | `0.9667` |

Interpretation:

- The precision-first fine-tune improved sparse-error precision substantially versus the low-error eval baseline.
- At observed error rate `0.0115`, the previous checkpoint reached precision `0.5335` at threshold `0.95`; this run reaches `0.8753` with the F0.5-selected threshold `0.97`.
- Overall task score also improved slightly over the prior Transformer baseline: `0.7800` vs `0.7743`.
- Recall drops as intended, but remains usable enough for a precision-first workflow where false positives are costly.
- Next useful step is post-processing: high-confidence reporting, per-piece candidate limits, and optional top-K / local peak filtering before expanding datasets.

## Result: Precision-First Inference Post-Processing

Status: implemented in `src\midi_error_detector\predict.py` after the precision fine-tune.

Purpose:

- prioritize precision for realistic sparse-error use cases where a whole piece may only contain a few wrong notes
- default to the checkpoint's F0.5-selected threshold before falling back to F1/default training threshold
- suppress low-confidence and crowded candidates before presenting results to a user
- support long MIDI files by running inference in overlapping note windows

New inference controls:

| Option | Default | Use |
| --- | ---: | --- |
| `--threshold` | checkpoint `best_det_f0_5_threshold` | high precision detection cutoff |
| `--action-filter` | `non-keep` | remove high-score candidates whose predicted edit action is keep |
| `--min-action-prob` | `0.0` | optional stricter action-confidence filter |
| `--max-candidates` | none | per-piece candidate cap |
| `--min-separation` | `0.0` | suppress nearby lower-confidence candidates |
| `--sort-by` | `time` | chronological or confidence-ranked output |
| `--include-summary` | off | include note/window/candidate metadata |
| `--window-size` | checkpoint `window_size` | notes per inference window |
| `--window-overlap` | `32` | overlap between windows |

Smoke test:

```powershell
E:\downloads\桌面\dku\CS309\project\code\venv\Scripts\python.exe -B -m midi_error_detector.predict `
  --checkpoint checkpoints\transformer_precision_finetune.pt `
  --midi "E:\downloads\桌面\dku\CS309\project\maestro-v3.0.0-midi\maestro-v3.0.0\2004\MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi" `
  --top-k 3 `
  --max-candidates 10 `
  --min-separation 0.08 `
  --include-summary
```

Smoke-test result:

| Metric | Value |
| --- | ---: |
| notes | `7894` |
| windows | `36` |
| threshold | `0.97` |
| action filter | `non-keep` |
| candidate cap | `10` |
| returned candidates | `2` |

The two returned candidates were both high-confidence replacement suggestions from pitch `77` to pitch `76`, with detection probabilities `0.9838` and `0.9780`.

## Run: Melodic-Theory Feature Fine-Tune

Status: started after precision-first post-processing.

Code changes:

- expanded `FEATURE_SIZE` from `24` to `36`
- added local melodic-context features for passing tone, neighbor tone, stepwise resolution, motion direction, local duration ratio, and approximate metrical strength
- added compatible checkpoint loading so the old Transformer input projection can copy its first 24 feature columns while new feature columns stay randomly initialized

Planned command:

```powershell
E:\downloads\桌面\dku\CS309\project\code\venv\Scripts\python.exe -B -u -m midi_error_detector.train `
  --model transformer `
  --init-checkpoint checkpoints\transformer_precision_finetune.pt `
  --data-root "E:\downloads\桌面\dku\CS309\project\maestro-v3.0.0-midi\maestro-v3.0.0" `
  --clean-epochs 0 `
  --epochs 24 `
  --early-stop-patience 7 `
  --batch-size 8 `
  --window-size 256 `
  --num-layers 4 `
  --transformer-d-model 192 `
  --transformer-heads 4 `
  --transformer-ffn-dim 512 `
  --train-error-rates 0.005 0.01 0.02 0.05 `
  --error-rate 0.01 `
  --det-threshold 0.95 `
  --det-pos-weight 1.0 `
  --kind-class-weights 1 6 4 `
  --threshold-sweep 0.8 0.85 0.9 0.95 0.97 0.98 0.99 `
  --save-metric precision_task_score `
  --lr 0.00005 `
  --lr-patience 3 `
  --lr-factor 0.5 `
  --lr-threshold 0.001 `
  --num-workers 0 `
  --output checkpoints\transformer_melodic_theory_precision.pt
```

Final result:

Status: completed 24 corrupt-stage epochs. Best checkpoint saved at epoch 24 by `precision_task_score`.

Checkpoint: `checkpoints\transformer_melodic_theory_precision.pt`.

Best checkpoint metrics:

| Metric | Value |
| --- | ---: |
| `epoch` | `24` |
| `precision_task_score` | `0.8108` |
| `task_score` | `0.7790` |
| `best_det_f0_5` | `0.7620` |
| `best_det_f0_5_precision` | `0.8693` |
| `best_det_f0_5_recall` | `0.5102` |
| `best_det_f0_5_threshold` | `0.97` |
| `best_det_f1` | `0.6832` |
| `best_det_precision` | `0.7227` |
| `best_det_recall` | `0.6478` |
| `best_det_threshold` | `0.85` |
| `det_precision` at threshold `0.95` | `0.8321` |
| `det_recall` at threshold `0.95` | `0.5533` |
| `det_f1` at threshold `0.95` | `0.6646` |
| `det_f0_5` at threshold `0.95` | `0.7559` |
| `replace_pitch_top3` | `0.9116` |
| `replace_kind_acc` | `0.8382` |
| `delete_kind_acc` | `0.9685` |

Interpretation:

- The melodic-theory features produced a tiny `precision_task_score` improvement over the previous precision fine-tune: `0.8108` vs `0.8106`.
- Precision did not improve: F0.5-selected precision moved from `0.8753` to `0.8693`, while recall improved from `0.5006` to `0.5102`.
- This means the new features are not a clear precision win yet. They may be helping borderline recall/action behavior, but the next precision-focused step should be calibration/post-processing on per-piece false positives rather than simply training longer.

## Result: Harmony-Aware Post-Processing Probe

Status: tested without retraining.

Code changes:

- added lightweight harmony scoring in `src\midi_error_detector\harmony.py`
- added optional harmony metadata and adjusted ranking to `predict.py`
- added `scripts\evaluate_harmony_postprocess.py` to sweep harmony filters on sparse-error evaluation

Evaluation setup:

- checkpoint: `checkpoints\transformer_melodic_theory_precision.pt`
- test split, synthetic `error_rate=0.01`
- detection threshold `0.97`
- baseline means `probability >= 0.97` and predicted action is not keep

| Strategy | TP | FP | FN | Precision | Recall | F0.5 | FP Drop | TP Drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | `8560` | `1287` | `8219` | `0.8693` | `0.5102` | `0.7620` | `0` | `0` |
| `min_gain_-0.10` | `7533` | `1149` | `9246` | `0.8677` | `0.4490` | `0.7313` | `138` | `1027` |
| `min_gain_0.00` | `7351` | `1108` | `9428` | `0.8690` | `0.4381` | `0.7262` | `179` | `1209` |
| `min_gain_0.05` | `5928` | `805` | `10851` | `0.8804` | `0.3533` | `0.6781` | `482` | `2632` |
| `protect_current_0.65_no_gain` | `6987` | `1045` | `9792` | `0.8699` | `0.4164` | `0.7143` | `242` | `1573` |
| `protect_current_0.75_no_gain` | `7018` | `1048` | `9761` | `0.8701` | `0.4183` | `0.7155` | `239` | `1542` |

Interpretation:

- Simple harmony-gain filtering is too blunt: it removes false positives, but it removes many more true positives.
- Strict `min_gain_0.05` gets precision to `0.8804`, but recall collapses from `0.5102` to `0.3533`, so F0.5 drops sharply.
- Classical harmony is still likely useful, but not as a hard filter based on local triad membership. The next version should use per-piece top-K reranking or a learned calibration head instead of dropping all candidates with weak harmony gain.

## Result: Candidate Threshold Sweep

Status: completed first-stage threshold sweep for the two-stage reranker plan.

Purpose:

- check whether a lower first-stage threshold can provide enough candidate recall for a second-stage reranker to reach the long-term `80/80+` precision/recall target
- use the same sparse synthetic setting as the previous evaluations

Evaluation setup:

- checkpoint: `checkpoints\transformer_melodic_theory_precision.pt`
- test split, synthetic `error_rate=0.01`
- candidate definition: `error_probability >= threshold` and predicted action is not keep
- total error notes: `16779`

| Candidate Threshold | Precision | Recall | F0.5 | TP | FP | FN | Candidates |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.50` | `0.5273` | `0.7766` | `0.5635` | `13031` | `11681` | `3748` | `24712` |
| `0.60` | `0.5750` | `0.7481` | `0.6029` | `12552` | `9276` | `4227` | `21828` |
| `0.70` | `0.6270` | `0.7140` | `0.6427` | `11981` | `7126` | `4798` | `19107` |
| `0.75` | `0.6557` | `0.6971` | `0.6636` | `11697` | `6141` | `5082` | `17838` |
| `0.80` | `0.6873` | `0.6752` | `0.6849` | `11330` | `5154` | `5449` | `16484` |
| `0.85` | `0.7227` | `0.6478` | `0.7063` | `10869` | `4171` | `5910` | `15040` |

Interpretation:

- Even at candidate threshold `0.50`, first-stage recall is only `0.7766`, below the desired `0.80+` target.
- Lowering the threshold further may cross `0.80`, but it will produce a very large false-positive pool and may be hard for a small reranker to recover precision from.
- This means the current 80/80 bottleneck is mainly first-stage candidate generation, not only second-stage filtering.
- Next optimization should focus on changing the synthetic error generation and training objective so the base Transformer assigns higher probabilities to currently missed true errors.

## Run: Theory-Weighted Recall Fine-Tune

Status: started after candidate threshold sweep showed first-stage recall is the main bottleneck.

Purpose:

- keep the overall task as wrong-note detection
- improve first-stage candidate recall so a later reranker has enough true errors to work with
- use music theory more directly in training instead of only as input features

Code changes:

- added more classical-piano-oriented synthetic error types:
  - `scale_slip`
  - `chord_slip`
  - `octave_displacement`
- kept existing nearby, neighbor, and extra-touch corruptions
- added per-note detection loss weights:
  - theory-plausible clean notes can receive extra negative-class protection
  - theory-suspicious corrupted notes can receive extra positive-class weight
- exposed training args:
  - `--clean-theory-weight`
  - `--error-theory-weight`

Planned command:

```powershell
E:\downloads\桌面\dku\CS309\project\code\venv\Scripts\python.exe -B -u -m midi_error_detector.train `
  --model transformer `
  --init-checkpoint checkpoints\transformer_melodic_theory_precision.pt `
  --data-root "E:\downloads\桌面\dku\CS309\project\maestro-v3.0.0-midi\maestro-v3.0.0" `
  --clean-epochs 0 `
  --epochs 24 `
  --early-stop-patience 7 `
  --batch-size 8 `
  --window-size 256 `
  --num-layers 4 `
  --transformer-d-model 192 `
  --transformer-heads 4 `
  --transformer-ffn-dim 512 `
  --train-error-rates 0.01 0.02 0.05 0.10 `
  --error-rate 0.01 `
  --det-threshold 0.7 `
  --det-pos-weight 2.0 `
  --clean-theory-weight 1.5 `
  --error-theory-weight 2.0 `
  --kind-class-weights 1 6 4 `
  --threshold-sweep 0.4 0.5 0.6 0.7 0.8 0.85 0.9 `
  --save-metric best_det_f1 `
  --lr 0.00005 `
  --lr-patience 3 `
  --lr-factor 0.5 `
  --lr-threshold 0.001 `
  --num-workers 0 `
  --output checkpoints\transformer_theory_weighted_recall.pt
```

Success criteria:

- first-stage candidate recall should exceed the previous `0.7766` at threshold `0.50`
- ideally threshold `0.50` candidate recall should cross `0.80`
- precision can be lower at this stage, but should not collapse so badly that reranking becomes unrealistic

Observed result:

- stopped at epoch 16/24 by early stopping
- best checkpoint came from epoch 9
- best test F1 threshold `0.8`: precision `0.7545`, recall `0.6308`, F1 `0.6871`
- best F0.5 threshold `0.9`: precision `0.8237`, recall `0.5632`, F0.5 `0.7539`
- recall improved compared with the precision-first baseline, but precision dropped and F0.5 did not beat the previous `0.7620`

Conclusion:

- theory-weighted positive learning helped recover recall
- the model became too willing to report errors under sparse-error evaluation
- next step should keep high/mid error-rate coverage for learning error types, but finish with sparse calibration and save checkpoints using precision-constrained recall

## Run: Coverage + Sparse Calibration Fine-Tune

Status: started after theory-weighted recall run stopped at epoch 16.

Purpose:

- keep diverse synthetic errors so the model learns multiple wrong-note types
- avoid teaching the model that real pieces have dense error priors
- calibrate the final decision boundary on sparse wrong-note distributions
- select checkpoints by recall only when precision stays above the target floor

Code changes:

- added a final calibration phase:
  - `--calibration-epochs`
  - `--calibration-error-rates`
- added precision-constrained metrics:
  - `precision_constrained_threshold`
  - `precision_constrained_precision`
  - `precision_constrained_recall`
  - `precision_constrained_f1`
  - `precision_recall_score`
- added `--target-precision`, default `0.8`
- added `--save-metric precision_recall_score`

Planned command:

```powershell
E:\downloads\妗岄潰\dku\CS309\project\code\venv\Scripts\python.exe -B -u -m midi_error_detector.train `
  --model transformer `
  --init-checkpoint checkpoints\transformer_theory_weighted_recall.pt `
  --data-root "E:\downloads\妗岄潰\dku\CS309\project\maestro-v3.0.0-midi\maestro-v3.0.0" `
  --clean-epochs 0 `
  --epochs 24 `
  --calibration-epochs 10 `
  --early-stop-patience 8 `
  --batch-size 8 `
  --window-size 256 `
  --num-layers 4 `
  --transformer-d-model 192 `
  --transformer-heads 4 `
  --transformer-ffn-dim 512 `
  --train-error-rates 0.02 0.05 0.08 `
  --calibration-error-rates 0.005 0.01 0.02 `
  --error-rate 0.01 `
  --det-threshold 0.75 `
  --det-pos-weight 1.6 `
  --clean-theory-weight 2.2 `
  --error-theory-weight 1.2 `
  --target-precision 0.8 `
  --kind-class-weights 1 6 4 `
  --threshold-sweep 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.93 0.95 `
  --save-metric precision_recall_score `
  --lr 0.00003 `
  --lr-patience 3 `
  --lr-factor 0.5 `
  --lr-threshold 0.001 `
  --num-workers 0 `
  --output checkpoints\transformer_coverage_calibrated.pt
```

Success criteria:

- maintain precision `>=0.80` at the selected threshold
- improve precision-constrained recall beyond the previous high-precision recall around `0.56`
- ideally approach precision `>=0.80` and recall `>=0.65` before considering hard-negative mining

Observed result:

- run stopped at epoch 9/24 by early stopping
- best checkpoint was saved at epoch 1 by `precision_recall_score`
- checkpoint: `checkpoints\transformer_coverage_calibrated.pt`

Best saved test metrics:

- `precision_recall_score=0.8334`
- `precision_constrained_threshold=0.85`
- `precision_constrained_precision=0.8037`
- `precision_constrained_recall=0.5751`
- `precision_constrained_f1=0.6705`
- `best_det_threshold=0.70`
- `best_det_precision=0.7222`
- `best_det_recall=0.6451`
- `best_det_f1=0.6815`
- `best_det_f0_5_threshold=0.95`
- `best_det_f0_5_precision=0.8827`
- `best_det_f0_5_recall=0.4763`
- `best_det_f0_5=0.7540`
- `replace_pitch_top3=0.9007`
- `replace_kind_acc=0.7808`
- `task_score=0.7611`
- `precision_task_score=0.7947`

Comparison:

- previous theory-weighted recall checkpoint: precision `0.7545`, recall `0.6308`, F1 `0.6871`, F0.5 `0.7539`
- coverage + calibration at precision-constrained threshold: precision improved to `0.8037`, but recall fell to `0.5751`
- coverage + calibration best F1 threshold: recall `0.6451`, but precision only `0.7222`
- F0.5 is nearly unchanged: `0.7540` vs previous `0.7539`

Conclusion:

- sparse calibration successfully restored precision above `0.80`
- it did not recover enough recall; the model still cannot reach precision `>=0.80` and recall `>=0.65` simultaneously
- the next useful step should be hard-negative/hard-positive mining or a threshold-aware training objective, not another ordinary fine-tune

## Run: Hard Pairwise Ranking Fine-Tune

Status: started after coverage + sparse calibration showed that threshold calibration alone does not improve the precision/recall frontier.

Purpose:

- improve score ordering rather than only calibrating the decision threshold
- push hard true errors above hard clean notes within each batch
- reduce the overlap between missed true errors and false-positive clean notes

Code changes:

- added hard pairwise ranking loss for detection logits
- for each batch, the loss compares:
  - lowest-scoring true errors as hard positives
  - highest-scoring clean notes as hard negatives
- new args:
  - `--ranking-loss-weight`
  - `--ranking-margin`
  - `--ranking-top-k`

Planned command:

```powershell
E:\downloads\妗岄潰\dku\CS309\project\code\venv\Scripts\python.exe -B -u -m midi_error_detector.train `
  --model transformer `
  --init-checkpoint checkpoints\transformer_coverage_calibrated.pt `
  --data-root "E:\downloads\妗岄潰\dku\CS309\project\maestro-v3.0.0-midi\maestro-v3.0.0" `
  --clean-epochs 0 `
  --epochs 18 `
  --calibration-epochs 6 `
  --early-stop-patience 6 `
  --batch-size 8 `
  --window-size 256 `
  --num-layers 4 `
  --transformer-d-model 192 `
  --transformer-heads 4 `
  --transformer-ffn-dim 512 `
  --train-error-rates 0.01 0.02 0.05 `
  --calibration-error-rates 0.005 0.01 `
  --error-rate 0.01 `
  --det-threshold 0.75 `
  --det-pos-weight 1.4 `
  --clean-theory-weight 2.0 `
  --error-theory-weight 1.4 `
  --ranking-loss-weight 0.25 `
  --ranking-margin 1.0 `
  --ranking-top-k 64 `
  --target-precision 0.8 `
  --kind-class-weights 1 6 4 `
  --threshold-sweep 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.93 0.95 `
  --save-metric precision_recall_score `
  --lr 0.00002 `
  --lr-patience 3 `
  --lr-factor 0.5 `
  --lr-threshold 0.001 `
  --num-workers 0 `
  --output checkpoints\transformer_hard_ranking.pt
```

Success criteria:

- beat coverage calibration precision-constrained recall `0.5751` while keeping precision `>=0.80`
- ideally move toward precision `>=0.80`, recall `>=0.65`
- avoid dropping F0.5 below the previous range around `0.754`

Observed result:

- run stopped at epoch 15/18 by early stopping
- best checkpoint was saved at epoch 9 by `precision_recall_score`
- checkpoint: `checkpoints\transformer_hard_ranking.pt`

Best saved test metrics:

- `precision_recall_score=0.8169`
- `precision_constrained_threshold=0.70`
- `precision_constrained_precision=0.8063`
- `precision_constrained_recall=0.5548`
- `precision_constrained_f1=0.6573`
- `best_det_threshold=0.60`
- `best_det_precision=0.7621`
- `best_det_recall=0.5973`
- `best_det_f1=0.6697`
- `best_det_f0_5_threshold=0.80`
- `best_det_f0_5_precision=0.8521`
- `best_det_f0_5_recall=0.5056`
- `best_det_f0_5=0.7494`
- `replace_pitch_top3=0.9028`
- `replace_kind_acc=0.8162`
- `task_score=0.7646`
- `precision_task_score=0.7978`
- `ranking_loss=0.5757`

Comparison:

- coverage + calibration checkpoint: precision-constrained precision `0.8037`, recall `0.5751`, F1 `0.6705`, F0.5 `0.7540`
- hard ranking checkpoint: precision-constrained precision `0.8063`, recall `0.5548`, F1 `0.6573`, F0.5 `0.7494`
- ranking loss improved `replace_kind_acc` and kept high precision, but did not improve recall at the precision floor

Conclusion:

- the simple in-batch hard ranking loss was too conservative for sparse wrong-note recall
- it likely pushed both hard positives and hard negatives apart in a way that preserved precision but did not raise enough missed true errors above threshold
- next step should inspect false negatives/false positives directly and build a mined evaluation/training set, instead of relying on batch-local ranking alone

## Run: From-Scratch Transformer with Coverage, Calibration, and Epoch Hard Replay

Status: started after repeated fine-tuning failed to improve the precision/recall frontier.

Reasoning:

- previous runs may be limited by the checkpoint's learned error prior and score ordering
- further fine-tuning can preserve an early bias instead of relearning the task
- starting from scratch lets the model learn with the corrected training curriculum from the beginning

Training strategy:

- clean warm-up for basic MIDI reconstruction and keep/replace/delete heads
- coverage phase with mixed error rates to learn diverse synthetic error types
- sparse calibration phase to match realistic low-error evaluation
- online hard ranking in every training epoch:
  - hard positives: low-scoring true errors in each batch
  - hard negatives: high-scoring clean notes in each batch
- epoch-to-epoch hard replay:
  - each training epoch stores the hardest windows by false-negative and false-positive pressure
  - before the next epoch, the model replays those stored windows once
  - this gives the next epoch direct access to the previous epoch's hard positives and hard negatives

Planned command:

```powershell
E:\downloads\妗岄潰\dku\CS309\project\code\venv\Scripts\python.exe -B -u -m midi_error_detector.train `
  --model transformer `
  --data-root "E:\downloads\妗岄潰\dku\CS309\project\maestro-v3.0.0-midi\maestro-v3.0.0" `
  --clean-epochs 1 `
  --epochs 36 `
  --calibration-epochs 10 `
  --early-stop-patience 10 `
  --batch-size 8 `
  --window-size 256 `
  --num-layers 4 `
  --transformer-d-model 192 `
  --transformer-heads 4 `
  --transformer-ffn-dim 512 `
  --train-error-rates 0.01 0.02 0.05 0.08 `
  --calibration-error-rates 0.005 0.01 0.02 `
  --error-rate 0.01 `
  --det-threshold 0.75 `
  --det-pos-weight 1.8 `
  --clean-theory-weight 2.0 `
  --error-theory-weight 1.5 `
  --ranking-loss-weight 0.12 `
  --ranking-margin 0.75 `
  --ranking-top-k 64 `
  --hard-replay-size 768 `
  --hard-replay-epochs 1 `
  --target-precision 0.8 `
  --kind-class-weights 1 6 4 `
  --threshold-sweep 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.93 0.95 `
  --save-metric precision_recall_score `
  --lr 0.0003 `
  --lr-patience 4 `
  --lr-factor 0.5 `
  --lr-threshold 0.001 `
  --num-workers 0 `
  --output checkpoints\transformer_from_scratch_hard_replay.pt
```

Success criteria:

- beat the best fine-tuned precision-constrained recall `0.5751` at precision `>=0.80`
- preferably recover recall near `0.65` without precision falling below `0.80`
- if it still fails, the bottleneck is likely synthetic data design / missing offline hard-case mining rather than checkpoint initialization

Observed result:

- completed all 36 corrupt epochs
- best checkpoint was saved at epoch 33 by `precision_recall_score`
- checkpoint: `checkpoints\transformer_from_scratch_hard_replay.pt`

Best saved test metrics:

- `precision_recall_score=0.5238`
- `precision_constrained_threshold=0.50`
- `precision_constrained_precision=0.8217`
- `precision_constrained_recall=0.3226`
- `precision_constrained_f1=0.4633`
- `best_det_threshold=0.50`
- `best_det_precision=0.8217`
- `best_det_recall=0.3226`
- `best_det_f1=0.4633`
- `best_det_f0_5_threshold=0.50`
- `best_det_f0_5_precision=0.8217`
- `best_det_f0_5_recall=0.3226`
- `best_det_f0_5=0.6276`
- `replace_pitch_top1=0.2390`
- `replace_pitch_top3=0.7435`
- `replace_kind_acc=0.5244`
- `delete_kind_acc=0.7172`
- `task_score=0.5487`
- `precision_task_score=0.6411`
- `ranking_loss=0.6307`

Comparison:

- best previous fine-tuned precision-constrained recall: `0.5751` at precision `0.8037`
- from-scratch hard replay precision-constrained recall: `0.3226` at precision `0.8217`
- best previous F0.5 range: around `0.754`
- from-scratch hard replay F0.5: `0.6276`

Conclusion:

- from-scratch training did not solve the recall bottleneck
- epoch hard replay made the detector much too conservative for sparse-error recall
- the clean warm-up plus low-error calibration appears to teach a strong "mostly clean" prior before the model learns robust wrong-note recall
- this rules out checkpoint initialization as the main bottleneck; the larger issue is the training target/data design
- next step should not be more replay/fine-tuning; it should be a direct error-analysis/mining pipeline that identifies which error types and musical contexts are missed, then changes synthetic corruption and loss around those concrete failures
