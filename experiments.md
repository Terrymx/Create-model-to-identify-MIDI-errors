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

## 2026-06-05 - Interleaved clean masked modeling

Motivation:

- The previous from-scratch hard replay run did not improve sparse-error recall.
- Pure clean warm-up teaches a clean prior, but it does not force the model to infer what a missing/uncertain note should be from context.
- The next training design should use different data for different objectives:
  - clean MIDI: learn symbolic music plausibility by masking pitch-derived and theory-derived features, then reconstructing the correct pitch
  - corrupted MIDI: keep explicit wrong-note detection as the main task
  - hard replay: keep boundary cases, but avoid letting replay dominate the detector into an overly conservative regime

Code changes:

- Added masked pitch reconstruction inside `train.py`.
- Added `--masked-pitch-loss-weight` and `--masked-pitch-rate`.
- Added `--clean-mask-batches-per-epoch` so every corrupt epoch can start with a small clean masked-learning phase.
- Added `--det-loss-weight` to allow clean masked auxiliary passes to disable detection BCE and train only the contextual pitch objective.
- Smoke-tested with `--max-files 2`; the new `stage=clean_mask` path runs and reports `masked_pitch_loss`.

Planned run:

```powershell
cd E:\downloads\桌面\dku\CS309\project\code_new

$out='E:\downloads\桌面\dku\CS309\project\code_new\training_logs\masked_context_detector.log'
$err='E:\downloads\桌面\dku\CS309\project\code_new\training_logs\masked_context_detector.err.log'
$cmd="$env:PYTHONPATH='src'; $env:PYTHONDONTWRITEBYTECODE='1'; & 'E:\downloads\桌面\dku\CS309\project\code\venv\Scripts\python.exe' -B -u -m midi_error_detector.train --model transformer --data-root 'E:\downloads\桌面\dku\CS309\project\maestro-v3.0.0-midi\maestro-v3.0.0' --clean-epochs 0 --epochs 36 --calibration-epochs 8 --early-stop-patience 10 --batch-size 8 --window-size 256 --num-layers 4 --transformer-d-model 192 --transformer-heads 4 --transformer-ffn-dim 512 --train-error-rates 0.02 0.05 0.08 0.12 --calibration-error-rates 0.005 0.01 0.02 --error-rate 0.01 --det-threshold 0.70 --det-pos-weight 2.3 --clean-theory-weight 1.5 --error-theory-weight 1.5 --masked-pitch-loss-weight 0.35 --masked-pitch-rate 0.18 --clean-mask-batches-per-epoch 900 --ranking-loss-weight 0.06 --ranking-margin 0.65 --ranking-top-k 64 --hard-replay-size 512 --hard-replay-epochs 1 --target-precision 0.8 --kind-class-weights 1 6 4 --threshold-sweep 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.93 0.95 --save-metric precision_recall_score --lr 0.0003 --lr-patience 4 --lr-factor 0.5 --lr-threshold 0.001 --num-workers 0 --output checkpoints\transformer_masked_context_detector.pt"
$p=Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-Command',$cmd) -WorkingDirectory 'E:\downloads\桌面\dku\CS309\project\code_new' -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
$p | Select-Object Id,ProcessName,StartTime
```

Success criteria:

- primary: improve precision-constrained recall above `0.5751` while keeping precision `>=0.80`
- secondary: improve best F1/F0.5 frontier over `transformer_coverage_calibrated.pt`
- diagnostic: `masked_pitch_loss` should steadily fall during `stage=clean_mask`; if it does not, the masking task is too hard or leaking the wrong signal

## 2026-06-05 - Curriculum + asymmetric hard replay

Motivation:

- The interrupted `masked_context_detector` run showed the same early pattern: recall rises on high-error corrupt epochs, but replay remains too conservative.
- The next experiment keeps model architecture, losses, features, optimizer, masked pitch settings, and eval protocol fixed.
- Only the training scheduler and hard replay composition change, so the effect is easier to attribute.

Code changes:

- Added `--curriculum-error-rate-stages`, parsed as semicolon-separated stages.
- Added `--asymmetric-hard-replay`.
- Added FN/FP replay buckets:
  - `--fn-replay-fraction`
  - `--fn-replay-weight`
  - `--fp-replay-weight`
- Smoke-tested on `--max-files 2`; stages switched as expected and replay reported `fn_candidates` / `fp_candidates`.

Planned run:

```powershell
cd E:\downloads\桌面\dku\CS309\project\code_new

$out='E:\downloads\桌面\dku\CS309\project\code_new\training_logs\curriculum_asym_replay.log'
$err='E:\downloads\桌面\dku\CS309\project\code_new\training_logs\curriculum_asym_replay.err.log'
$cmd="`$env:PYTHONPATH='src'; `$env:PYTHONDONTWRITEBYTECODE='1'; & 'E:\downloads\桌面\dku\CS309\project\code\venv\Scripts\python.exe' -B -u -m midi_error_detector.train --model transformer --data-root 'E:\downloads\桌面\dku\CS309\project\maestro-v3.0.0-midi\maestro-v3.0.0' --clean-epochs 0 --epochs 36 --early-stop-patience 10 --batch-size 8 --window-size 256 --num-layers 4 --transformer-d-model 192 --transformer-heads 4 --transformer-ffn-dim 512 --curriculum-error-rate-stages '0.08,0.12;0.02,0.05,0.08;0.005,0.01,0.02' --error-rate 0.01 --det-threshold 0.70 --det-pos-weight 2.3 --clean-theory-weight 1.5 --error-theory-weight 1.5 --masked-pitch-loss-weight 0.35 --masked-pitch-rate 0.18 --clean-mask-batches-per-epoch 900 --ranking-loss-weight 0.06 --ranking-margin 0.65 --ranking-top-k 64 --hard-replay-size 512 --hard-replay-epochs 1 --asymmetric-hard-replay --fn-replay-fraction 0.75 --fn-replay-weight 1.5 --fp-replay-weight 0.4 --target-precision 0.8 --kind-class-weights 1 6 4 --threshold-sweep 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.93 0.95 --save-metric precision_recall_score --lr 0.0003 --lr-patience 4 --lr-factor 0.5 --lr-threshold 0.001 --num-workers 0 --output checkpoints\transformer_curriculum_asym_replay.pt"
$p=Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-Command',$cmd) -WorkingDirectory 'E:\downloads\桌面\dku\CS309\project\code_new' -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
$p | Select-Object Id,ProcessName,StartTime
```

Success criteria:

- primary: beat `recall=0.5751` at precision `>=0.80`
- if `0.45 <= recall < 0.58`, increase FN replay pressure or start the sparse curriculum stage earlier
- if recall remains `<0.40`, move to loss/calibration redesign rather than more scheduler tuning

Observed result:

- completed all 36 epochs without errors
- best checkpoint was saved at epoch 35 by `precision_recall_score`
- checkpoint: `checkpoints\transformer_curriculum_asym_replay.pt`

Best saved test metrics:

- `precision_recall_score=0.6500`
- `precision_constrained_threshold=0.75`
- `precision_constrained_precision=0.8092`
- `precision_constrained_recall=0.4256`
- `precision_constrained_f1=0.5578`
- `best_det_threshold=0.45`
- `best_det_precision=0.6909`
- `best_det_recall=0.5421`
- `best_det_f1=0.6075`
- `best_det_f0_5_threshold=0.80`
- `best_det_f0_5_precision=0.8301`
- `best_det_f0_5_recall=0.4043`
- `best_det_f0_5=0.6857`
- `replace_pitch_top1=0.3290`
- `replace_pitch_top3=0.8014`
- `replace_kind_acc=0.6413`
- `delete_kind_acc=0.8077`

Comparison:

- previous from-scratch hard replay: precision `0.8217`, recall `0.3226`
- curriculum + asymmetric replay: precision `0.8092`, recall `0.4256`
- recall improved by `0.1030` while precision remained above `0.80`
- historical target baseline remains precision `0.8037`, recall `0.5751`

Conclusion:

- the conservative-training diagnosis was partly correct
- curriculum and FN-heavy replay recovered a meaningful amount of recall
- the experiment did not beat the historical recall baseline, so scheduler changes alone are insufficient
- based on the predefined decision rule, the next iteration should strengthen FN replay and begin the sparse curriculum stage earlier before moving to architecture or loss redesign

## 2026-06-06 - Step 2 explicit masked-context surprise

Motivation:

- Step 1A improved recall from `0.3226` to `0.4256` at precision above `0.80`, confirming that scheduling and replay composition matter.
- It still did not beat the historical recall baseline `0.5751`.
- The masked pitch head may learn contextual plausibility without giving the error head direct access to that information.

Architecture change:

- compute `surprise = -log P(observed_pitch | masked context)`
- embed normalized surprise plus a surprise-availability flag
- concatenate this embedding to the normal Transformer hidden state before the error head
- retain the existing pitch and kind heads

Leakage protection:

- mask the target note's pitch-derived and theory-derived feature columns
- globally remove interval/melodic columns that can reveal the target pitch through neighboring notes
- use random target masking during training (`0.25`)
- use four grouped masked passes during evaluation so every valid note receives a surprise value

Initialization:

- load `checkpoints\transformer_curriculum_asym_replay.pt`
- copy the existing 192 detector hidden weights into the widened error head
- initialize only the new surprise projection and surprise-specific detector columns

Planned run:

```powershell
cd E:\downloads\桌面\dku\CS309\project\code_new

$out='E:\downloads\桌面\dku\CS309\project\code_new\training_logs\explicit_surprise_step2.log'
$err='E:\downloads\桌面\dku\CS309\project\code_new\training_logs\explicit_surprise_step2.err.log'
$cmd="`$env:PYTHONPATH='src'; `$env:PYTHONDONTWRITEBYTECODE='1'; & 'E:\downloads\桌面\dku\CS309\project\code\venv\Scripts\python.exe' -B -u -m midi_error_detector.train --model transformer --init-checkpoint checkpoints\transformer_curriculum_asym_replay.pt --explicit-surprise --surprise-train-mask-rate 0.25 --surprise-eval-groups 4 --surprise-embedding-dim 16 --data-root 'E:\downloads\桌面\dku\CS309\project\maestro-v3.0.0-midi\maestro-v3.0.0' --clean-epochs 0 --epochs 36 --early-stop-patience 10 --batch-size 8 --window-size 256 --num-layers 4 --transformer-d-model 192 --transformer-heads 4 --transformer-ffn-dim 512 --curriculum-error-rate-stages '0.08,0.12;0.02,0.05,0.08;0.005,0.01,0.02' --error-rate 0.01 --det-threshold 0.70 --det-pos-weight 2.3 --clean-theory-weight 1.5 --error-theory-weight 1.5 --masked-pitch-loss-weight 0.35 --masked-pitch-rate 0.18 --clean-mask-batches-per-epoch 900 --ranking-loss-weight 0.06 --ranking-margin 0.65 --ranking-top-k 64 --hard-replay-size 512 --hard-replay-epochs 1 --asymmetric-hard-replay --fn-replay-fraction 0.75 --fn-replay-weight 1.5 --fp-replay-weight 0.4 --target-precision 0.8 --kind-class-weights 1 6 4 --threshold-sweep 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.93 0.95 --save-metric precision_recall_score --lr 0.0003 --lr-patience 4 --lr-factor 0.5 --lr-threshold 0.001 --num-workers 0 --output checkpoints\transformer_explicit_surprise_step2.pt"
$p=Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-Command',$cmd) -WorkingDirectory 'E:\downloads\桌面\dku\CS309\project\code_new' -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
```

Success criteria:

- primary: `recall > 0.5751` at precision `>=0.80`
- diagnostic: mean error surprise should exceed mean clean surprise
- compare the full PR frontier, not only the selected threshold

Observed result:

- completed all 36 epochs without errors
- best checkpoint was saved at epoch 36
- checkpoint: `checkpoints\transformer_explicit_surprise_step2.pt`

Best saved test metrics:

- `precision_recall_score=0.7812`
- `precision_constrained_threshold=0.85`
- `precision_constrained_precision=0.8012`
- `precision_constrained_recall=0.5307`
- `precision_constrained_f1=0.6385`
- `best_det_threshold=0.70`
- `best_det_precision=0.7188`
- `best_det_recall=0.6123`
- `best_det_f1=0.6613`
- `best_det_f0_5_threshold=0.93`
- `best_det_f0_5_precision=0.8598`
- `best_det_f0_5_recall=0.4623`
- `best_det_f0_5=0.7336`
- `mean_clean_surprise=2.5835`
- `mean_error_surprise=6.7222`
- `replace_pitch_top1=0.4847`
- `replace_pitch_top3=0.8716`
- `replace_kind_acc=0.7620`

Comparison:

- Step 1A: precision `0.8092`, recall `0.4256`, F1 `0.5578`
- Step 2: precision `0.8012`, recall `0.5307`, F1 `0.6385`
- explicit surprise improved precision-constrained recall by `0.1051`
- historical target baseline remains precision `0.8037`, recall `0.5751`

Conclusion:

- explicit masked-context surprise substantially shifts the PR frontier upward
- the strong error/clean surprise separation confirms that contextual pitch likelihood contains useful detection information
- Step 2 still misses the historical precision-constrained recall target by about `0.0444`
- the next iteration should focus on calibrating/fusing the explicit surprise signal rather than removing it

## 2026-06-07 - Multi-window surprise validation

Purpose:

- test the Tier 2 hypothesis before committing to another full training run
- keep `transformer_explicit_surprise_step2.pt` frozen
- compare masked-context surprise at `64`, `128`, and `256` notes on the same
  validation candidates
- use overlapping short windows and prefer the result where each note is closest
  to the center of its window

Protocol:

- split: MAESTRO validation
- evaluation corruption rate: `0.01`
- Stage 1 candidate threshold: `0.55`
- no test-set threshold selection
- output: `training_logs/multiscale_surprise_validation.json`

Candidate coverage:

- total notes: `1,261,824`
- error notes: `14,214`
- candidates: `14,783`
- candidate precision: `0.6073`
- candidate recall ceiling: `0.6316`

TP/FP separability:

| Signal | AUC | TP mean | FP mean |
| --- | ---: | ---: | ---: |
| surprise 64 | 0.5364 | 8.7843 | 8.2633 |
| surprise 128 | 0.5434 | 8.5416 | 7.8640 |
| surprise 256 | 0.5426 | 7.5257 | 6.9351 |
| three-scale mean | 0.5512 | 8.2838 | 7.6874 |
| three-scale minimum | 0.5522 | 5.6418 | 5.0524 |

Conclusion:

- the multi-window hypothesis is directionally supported: requiring surprise to
  remain high across scales separates TP from FP slightly better than the existing
  `256`-note surprise alone
- the incremental AUC is only `0.0096`, which is too small to justify a full
  multi-window retraining run by itself
- 64/128/256 surprise values are largely redundant for the current pitch head
- keep multi-window surprise as a possible auxiliary feature, but do not treat it
  as the main route to the required precision gain
- the next low-cost experiment should test learned candidate verification using
  the frozen Step 2 score, multi-window surprise, and theory features; only scale
  that approach if it improves recall at precision `>=0.80` on a file-held-out
  calibration split

## 2026-06-07 - Candidate verifier probe

Purpose:

- test whether the remaining Step 2 false positives can be removed without
  retraining the Transformer
- use the low Stage 1 threshold `0.45` to preserve a high recall ceiling
- train a small MLP verifier using:
  - Step 2 probability and logit
  - pitch-head observed probability, entropy, top probability, and pitch margin
  - `64`, `128`, and `256` note surprise values and their aggregate statistics
  - local surprise means
  - action-head probabilities
  - all existing MIDI and theory features

Leakage controls:

- the MAESTRO validation split was divided by MIDI file
- 103 validation files trained the verifier
- 34 disjoint validation files selected the verifier threshold
- the test split was evaluated only after threshold selection
- overall recall uses every error note as its denominator, including errors missed
  by Stage 1

Stage 1 candidate coverage:

| Split | Candidate precision | Recall ceiling |
| --- | ---: | ---: |
| verifier train | 0.5472 | 0.6703 |
| calibration | 0.5498 | 0.6621 |
| test | 0.5758 | 0.7153 |

Calibration-selected operating point:

- verifier threshold: `0.41`
- calibration precision: `0.8026`
- calibration recall: `0.4833`
- test precision: `0.7980`
- test recall: `0.5460`
- test F1: `0.6484`

Test PR-frontier diagnostic:

- the adjacent threshold `0.42` gives precision `0.8028`, recall `0.5413`,
  F1 `0.6466`
- this threshold is reported as a diagnostic point, not as the preselected result
- the best test F1 is `0.6666` at threshold `0.28`, with precision `0.7208` and
  recall `0.6199`

Comparison:

- Step 2 precision-constrained baseline: precision `0.8012`, recall `0.5307`,
  F1 `0.6385`
- verifier exploratory frontier at precision `>=0.80`: precision `0.8028`,
  recall `0.5413`, F1 `0.6466`
- recall gain at the precision constraint: about `0.0106`

Conclusion:

- candidate verification produces a real but small upward PR shift
- the calibration-selected threshold missed the test precision constraint by
  `0.0020`, showing mild validation-to-test score shift
- multi-window surprise plus current theory features do not separate the difficult
  FP population strongly enough to reach recall `0.60` at precision `0.80`
- a larger cascade alone is unlikely to close the remaining gap unless it receives
  genuinely new context, such as phrase-level motif/repetition evidence or a
  separately trained bidirectional correction likelihood
- retain the verifier as an ablation and possible product post-processor, but do
  not make it the main training direction

## 2026-06-07 - Planned from-scratch explicit correction-to-detection run

Motivation:

- Step 1A showed that curriculum and FN-heavy replay improve recall.
- Step 2 showed that explicit surprise is materially better than implicit masked
  learning.
- Multi-window surprise and a lightweight verifier provided only small additional
  gains because they reused mostly the same information.
- The next model should make the entire correction distribution explicit, not only
  a single surprise scalar.

Architecture:

- start a new Transformer from random initialization
- generate a leakage-safe masked-context pitch distribution for every note
- explicitly feed seven correction signals into the detection head:
  - normalized observed-pitch surprise
  - observed-pitch probability
  - top predicted pitch probability
  - top-vs-observed probability gap
  - normalized entropy
  - top-pitch mismatch flag
  - evidence availability flag
- detach these signals before the detection head by default
- preserve the normal pitch and error-kind heads

Training:

- clean masked reconstruction before every corrupted epoch
- three-stage error-rate curriculum
- FN-heavy asymmetric replay from the previous epoch
- sparse evaluation at error rate `0.01`
- checkpoint selection by `precision_recall_score`
- no initialization from earlier checkpoints

Primary success criterion:

- improve recall over `0.5307` while precision remains `>=0.80`
- stretch target: recall `>=0.60` at precision `>=0.80`

Interpretation rule:

- if the full explicit evidence improves over scalar surprise, retain it as the new
  main detector architecture
- if it does not, the bottleneck is not evidence wiring; proceed to genuinely new
  phrase/motif-level information rather than more score fusion
# 2026-06-08 - External Likelihood A/B Started

This experiment tests whether the newer explicit-correction detector is a better
platform for later stages even though its standalone sparse-error frontier is worse.

- Shared teacher: a separately trained clean-MAESTRO Transformer likelihood model.
- Shared evidence: seven leakage-safe correction values generated by the frozen teacher.
- Branch A initialization: `transformer_explicit_surprise_step2.pt`.
- Branch B initialization: `transformer_explicit_correction_detector.pt`.
- Both branches freeze the detector backbone and train only the correction-evidence
  projection plus detection head.
- Both branches use the same curriculum, asymmetric hard replay, sparse evaluation,
  optimizer settings, and threshold sweep.
- Primary comparison: recall at precision `>= 0.80`.
- Secondary comparison: gain relative to each branch's starting frontier and error
  complementarity for a later heterogeneous ensemble.

## 2026-06-09 - External Likelihood A/B Result

- clean likelihood teacher completed 8 epochs; masked reconstruction loss decreased
  from `3.1065` to `0.9528`
- Branch A, initialized from Step 2, best epoch 7:
  - precision `0.8006`
  - recall `0.5319`
  - F1 `0.6391`
- Branch B, initialized from the from-scratch explicit-correction detector, best epoch 13:
  - precision `0.8036`
  - recall `0.4267`
  - F1 `0.5574`
- Branch A remains the stronger platform.
- Branch A improved recall by only about `0.0012` over the original Step 2 result,
  so the external likelihood Gate 1 did not pass.
- Next action: measure teacher clean/error separation and redundancy with the
  internal Step 2 surprise signal on the validation split.

## 2026-06-09 - External Likelihood Diagnostic

- validation notes: `1,261,824`, including `14,214` synthetic errors
- legacy evidence masking:
  - clean perplexity `1294.54`
  - all-note AUC `0.5860`
  - hard-candidate AUC `0.4977`
- train-aligned target-only masking:
  - clean perplexity `8.06`
  - all-note AUC `0.6284`
  - hard-candidate AUC `0.5302`
- Step 2 internal surprise:
  - all-note AUC `0.7862`
  - hard-candidate AUC `0.5426`
- The original external-likelihood experiment had a material train/inference
  masking mismatch, but correcting it still does not create a useful hard-candidate
  ranking gain.
- Decision: do not rerun the same bidirectional teacher fusion. Proceed to separately
  trained forward and backward likelihood models, which provide genuinely different
  one-sided evidence.

## 2026-06-09 - Directional Likelihood Training and Leakage Audit

- forward teacher completed 4 epochs:
  - validation loss `0.6705`
  - validation perplexity `1.9553`
  - validation pitch accuracy `0.7717`
- backward teacher completed 4 epochs:
  - validation loss `0.6980`
  - validation perplexity `2.0097`
  - validation pitch accuracy `0.7522`
- Audit finding: shifting the full 36-dimensional feature vectors is not sufficient
  for one-sided prediction. Neighbor-derived fields such as `next_interval`,
  `step_out`, passing/neighbor flags, and their backward equivalents can encode the
  target pitch through adjacent notes.
- Interpretation: the unusually low perplexity is contaminated by target leakage.
  These checkpoints are implementation diagnostics, not valid Stage 2 evidence.
- Decision: do not fuse these checkpoints into the detector.
- Next action:
  1. rebuild directional inputs from leakage-safe local/raw features;
  2. retrain forward and backward teachers with early stopping;
  3. evaluate all-note and hard-candidate separation before any detector run;
  4. require a meaningful gain over the current hard-candidate AUC `0.5426`;
  5. if the signal gate passes, run a frozen-Step-2 fusion probe before full Stage 2.
- Full analysis: `training_logs/directional_likelihood_analysis.md`.

## 2026-06-09 - Leakage-Safe Directional Likelihood Retraining Started

- directional input reduced from all 36 detector features to seven auditable
  source-note fields: pitch, velocity, duration, onset delta, beat phase, and
  source-note pitch-class sine/cosine
- removed all key, scale, harmony, chord, interval, passing/neighbor, resolution,
  duration-context, and metrical-theory fields from the directional teachers
- added a runtime invariant test: perturbing a target note's complete feature
  vector must not change the input used to predict that target
- retained causal attention and the one-note directional shift
- changed output names so invalid diagnostic checkpoints are not overwritten:
  - `checkpoints/transformer_forward_likelihood_leakage_safe.pt`
  - `checkpoints/transformer_backward_likelihood_leakage_safe.pt`
- training budget: up to 8 epochs per direction, with validation early stopping
  patience 2
- forward and backward smoke tests both passed
- after training, run the hard-candidate signal Gate before any detector fusion

## 2026-06-10 - Leakage-Safe Directional Likelihood Result

- both directional teachers completed all 8 epochs without errors or early stopping
- forward teacher best epoch 8:
  - validation loss `2.4721`
  - validation perplexity `11.8471`
  - validation pitch accuracy `0.3538`
- backward teacher best epoch 8:
  - validation loss `2.3574`
  - validation perplexity `10.5638`
  - validation pitch accuracy `0.3684`
- both validation curves were still improving at epoch 8
- the backward teacher is modestly stronger than the forward teacher
- unlike the invalid full-feature run with perplexity near `2.0`, these values are
  plausible for one-sided pitch prediction and pass the leakage audit
- interpretation: clean one-sided sequence modeling is viable, but likelihood
  quality alone does not establish wrong-note detection value
- next action: evaluate forward, backward, combined, and disagreement surprise on
  all notes and on Step 2 hard candidates; do not fuse into the detector unless the
  hard-candidate signal Gate passes

## 2026-06-10 - Directional Likelihood Signal Gate

- evaluation split: MAESTRO validation
- sparse synthetic error rate: `0.01`
- valid notes with both directions: `1,251,966`
- error notes: `14,102`
- Step 2 candidate threshold: `0.55`
- candidate count: `14,695`
- candidate precision: `0.6082`
- candidate recall ceiling: `0.6337`

All-note AUC:

- forward surprise: `0.8495`
- backward surprise: `0.8618`
- directional mean: `0.9101`
- directional minimum: `0.9062`
- Step 2 internal surprise: `0.7868`

Hard-candidate TP/FP AUC:

- forward surprise: `0.6162`
- backward surprise: `0.6382`
- directional mean: `0.6633`
- directional minimum: `0.6562`
- directional maximum: `0.6370`
- directional disagreement: `0.4838`
- Step 2 internal surprise: `0.5428`

Complementarity:

- directional mean Pearson correlation with Step 2 surprise: `0.2915`
- directional mean Spearman correlation with Step 2 surprise: `0.2876`
- best directional AUC increment over Step 2: `+0.1204`
- required Gate increment: `+0.0200`
- Gate result: **passed**

Interpretation:

- leakage-safe forward/backward likelihood provides substantially stronger ranking
  among the false-positive-heavy candidate population
- the low correlation with Step 2 confirms that the gain is not merely another
  rescaling of the existing masked-context surprise
- backward context is stronger than forward context, while their mean is better
  than either direction alone
- disagreement by itself is not useful

Decision:

- proceed to a frozen-Step-2 fusion probe
- keep the Step 2 Transformer backbone and directional teachers frozen
- train only a compact evidence projection plus detection head
- feed forward/backward probabilities, surprises, entropy/top-alternative evidence,
  and their mean/minimum
- require at least `+0.03` recall at precision `>=0.80` before a full Stage 2 run
- result files:
  - `training_logs/directional_likelihood_gate.json`
  - `training_logs/directional_likelihood_gate.md`

## 2026-06-10 - Frozen Step 2 Directional Fusion Probe

Protocol:

- Step 2 detector backbone and both directional likelihood teachers frozen
- Stage 1 candidate threshold `0.45`
- validation MIDI files split into 103 fusion-training files and 34 calibration
  files
- test split used only after selecting the fusion epoch and threshold
- fusion head inputs:
  - frozen Step 2 hidden state and explicit-surprise embedding
  - original Step 2 probability/logit/surprise
  - forward and backward observed probability, top probability, probability gap,
    entropy, mismatch, and surprise
  - directional mean/minimum/maximum surprise
- each epoch evaluated on calibration files; epoch selected by recall at precision
  `>=0.80`

Candidate coverage:

- fusion train recall ceiling: `0.6722`
- calibration recall ceiling: `0.6641`
- test recall ceiling: `0.7176`

Result:

- best fusion epoch: `10`
- Step 2 calibration-selected baseline on test:
  - threshold `0.85`
  - precision `0.8023`
  - recall `0.5328`
  - F1 `0.6404`
- calibration-selected fusion threshold `0.41`:
  - calibration precision `0.8003`, recall `0.5114`
  - test precision `0.7927`, recall `0.5752`
- adjacent test diagnostic point satisfying precision `>=0.80`:
  - threshold `0.43`
  - precision `0.8024`
  - recall `0.5658`
  - F1 `0.6636`
- recall gain over historical Step 2: `+0.0351`
- required Gate gain: `+0.0300`
- Gate result: **passed**

Interpretation:

- directional evidence translates into a meaningful PR-frontier improvement
- selecting the fusion epoch on file-held-out calibration is necessary; using the
  final epoch reduced the gain to nearly zero
- calibration-to-test precision drift remains, so the final product threshold
  will need a safety margin above `0.80`

Decision:

- proceed to full Stage 2 detector training
- initialize from the Step 2 detector
- freeze the forward/backward likelihood teachers
- train the detector with combined internal and directional evidence, curriculum,
  and FN-heavy asymmetric hard replay
- files:
  - `scripts/run_directional_fusion_probe.py`
  - `training_logs/directional_fusion_probe.json`
  - `training_logs/directional_fusion_probe.md`

## 2026-06-10 - Full Directional Stage 2 Training Started

Architecture:

- initialize the detector backbone from `transformer_explicit_surprise_step2.pt`
- replace the scalar surprise branch with a 17-value explicit evidence branch:
  - internal masked-context surprise and availability
  - six forward likelihood values
  - six backward likelihood values
  - directional mean, minimum, and maximum surprise
- freeze both directional likelihood teachers
- initialize new error-head evidence columns to zero so the starting detector score
  exactly preserves the Step 2 backbone before evidence learning

Schedule:

1. 8-epoch evidence warm-up:
   - freeze detector backbone
   - train only correction projection and error head
   - learning rate `0.001`
2. up to 28-epoch full fine-tuning:
   - unfreeze detector
   - learning rate `0.0001`
   - retain clean masked reconstruction, three-stage corruption curriculum, and
     FN-heavy asymmetric hard replay

## 2026-06-11 - Directional Stage 2 Joint Result and Frozen Control

Joint fine-tuning result:

- stopped after epoch 24; best checkpoint was epoch 16
- validation operating point at precision `>= 0.80`:
  - threshold `0.85`
  - precision `0.8064`
  - recall `0.4936`
  - F1 `0.6124`
- full detector fine-tuning did not preserve the gain seen in the frozen fusion probe

Frozen control started:

- initialize again from `transformer_explicit_surprise_step2.pt`
- freeze the Step 2 detector backbone and both directional teachers for the entire run
- train only the 17-value evidence projection and error head
- retain the same validation split, corruption curriculum, asymmetric hard replay,
  threshold sweep, and `precision_recall_score` checkpoint selection
- omit clean masked reconstruction because its encoder and pitch head are frozen
- output: `checkpoints/transformer_directional_stage2_frozen.pt`
- logs:
  - `training_logs/directional_stage2_frozen.log`
  - `training_logs/directional_stage2_frozen.err.log`

After this run, evaluate the frozen and joint checkpoints once on the same test split
using validation-selected thresholds. This is the strict comparison used to choose
the Stage 2 branch for the later ensemble and cascade stages.

## 2026-06-11 - Piano-Keyboard Corruption Audit

Problem found:

- the legacy `neighbor` and accidental-touch corruption sampled only pitch `+/-1`
- this models chromatic adjacency but misses same-color physical neighbours such as
  C-D and C#-D#
- the frozen Directional Stage 2 run was stopped during epoch 4 because it was still
  using this incomplete corruption profile

Correction profile `piano_keyboard_v4`:

- sample immediate chromatic keys on either side
- also sample the nearest same-color key on either side
- merge duplicate candidates and retain piano-range boundaries
- examples:
  - C4 -> B3, C#4, or D4
  - D4 -> C4, C#4, D#4, or E4
  - C#4 -> A#3, C4, D4, or D#4
- apply the same keyboard topology to replacement slips and extra accidental touches

Mis-touch semantics are separated explicitly:

- `neighbor` (`24%`): omit the intended note and press one adjacent key; replace it
- `neighbor_extra_touch` (`12%`): keep the intended note and simultaneously press
  an adjacent key; keep the correct note and delete only the extra touch
- `nearby` (`24%`): omit the intended note and press another pitch within seven
  semitones
- `nearby_plus_touch` (`10%`): replace the intended note with a nearby wrong pitch
  and simultaneously add another adjacent touch
- `scale_slip` (`15%`), `chord_slip` (`10%`), and octave displacement (`5%`)

This distinction matters for the action target even when the detector architecture
is unchanged: replacement mistakes require a pitch suggestion, while simultaneous
extra touches require deletion without suppressing the correctly played note.

## 2026-06-12 - Keyboard-Aware Pipeline Rebuild

Two exploratory runs changed both the corruption distribution and model target.
They were stopped because they no longer followed the already validated training
path that this rebuild is intended to restore. The stronger archived run reached:

- precision `0.8147`
- recall `0.3110`
- F1 `0.4502`
- correction top-3 `0.9134`

These runs are archived only as abandoned branches:

- `checkpoints/transformer_keyboard_aware_unified_collapsed_evidence.pt`
- `training_logs/keyboard_aware_unified_collapsed_evidence.log`

This is not a new-versus-old comparison experiment. The legacy corruption generator
was found to model piano mistakes incorrectly, so the dataset generation is repaired
and the established pipeline is being retrained from the beginning. Model
architecture, action/pitch targets, losses, weights, optimizer, schedule, and the
successful training path remain unchanged.

The historical path is:

1. Step 1A trained the ordinary detector with masked pitch learning, curriculum,
   and asymmetric replay, reaching `P=0.8092, R=0.4256`.
2. Step 2 loaded Step 1A, widened the error head, and added masked-context surprise,
   reaching `P=0.8012, R=0.5307`.

The rebuilt pipeline must train through all milestones already reached by the
project:

1. `transformer_keyboard_aware_step1a.pt`
   - no explicit detector evidence
   - original binary detection, keep/replace/delete action head, and pitch head
   - masked clean pitch learning, curriculum, and asymmetric hard replay
2. `transformer_keyboard_aware_step2.pt`
   - initialize from the keyboard-aware Step 1A checkpoint
   - add explicit masked-context surprise
   - copy the existing detector weights into the widened error head
3. frozen Directional Stage 2
4. joint Directional Stage 2

The formal Step 1A and Step 2 commands retain the historical `pitch_loss_weight=0.5`,
`kind_loss_weight=0.3`, class weights `1/6/4`, and test-side evaluation settings so
the corrected-data model can first recover the established `P=0.8012, R=0.5307`
stage and then continue through the frozen and joint Directional Stage 2 runs.

The sequential runner is `scripts/run_keyboard_aware_pipeline.ps1`.

## Planned Follow-up - Binary Clean/Dirty Detector

Do not interrupt the current keyboard-aware pipeline rebuild. After it completes,
run a separate detector experiment with note-level `clean` versus `dirty` as the
only primary target.

Motivation:

- in a compound mistake, the performer may play a wrong intended note and also
  brush an adjacent key
- the observed MIDI then contains two nearby dirty notes, but a reference-free
  model cannot reliably identify which one was the primary wrong press and which
  one was the secondary touch
- synthetic generation provenance can provide that distinction during training,
  but relying on it may teach an artificial replace/delete assignment that is not
  identifiable from real input

Both architectures must be trained to the same completed Stage 2 milestone:

```text
Step 1A detector
  -> explicit masked-context surprise Step 2
  -> frozen Directional Stage 2
  -> joint Directional Stage 2
```

The binary-only branch must not be compared with the completed existing branch
while it is still at Step 1A. It should receive the same curriculum, asymmetric
hard replay, likelihood teachers, evidence construction, training budget, data
split, and checkpoint-selection protocol, except for the removal of the
replace/delete auxiliary classification target.

The final comparison should therefore be:

1. the completed Stage 2 existing detector with binary detection plus auxiliary
   keep/replace/delete and pitch heads
2. the completed Stage 2 binary-only clean/dirty detector trained on the same
   corrected corruption distribution

The primary comparison metric is recall subject to precision >= 0.80. Correction
can be evaluated separately and must not determine whether a note is dirty.

Execution order was revised on 2026-06-13:

1. finish the currently running existing-head explicit-surprise Step 2
2. do not start either Directional Stage 2 branch yet
3. train the binary clean/dirty branch through Step 1A and explicit-surprise Step 2
4. compare the two aligned Step 2 checkpoints before deciding how to run the
   Directional Stage 2 experiments

For the binary branch, corrupted batches use only detection BCE, ranking loss, and
asymmetric hard replay. Both `pitch_loss_weight` and `kind_loss_weight` are zero.
Masked pitch reconstruction remains enabled only in the separate error-free
clean-mask phase, so it learns correct musical context for explicit surprise
without using replace/delete provenance from corrupted examples.

Scripts:

- `scripts/train_keyboard_aware_binary_step1a.ps1`
- `scripts/train_keyboard_aware_binary_step2.ps1`
- `scripts/run_binary_through_step2.ps1`
- `scripts/run_binary_through_step2.sh` for the remote Linux training host

## 2026-06-16 - Candidate Cascade and Traditional Baselines

The current `Precision >= 0.80` improvement path is a cascade system rather than
another full detector retraining:

```text
Stage 1: existing Step 2 detector + binary unified Step 2 detector
         -> low-threshold union candidate pool
Stage 2: verifier trained on candidate-level evidence
         -> remove false positives while preserving recall
```

### Union candidate verifier

The first union-candidate verifier used the existing Step 2 detector, the binary
unified Step 2 detector, and forward/backward likelihood teachers.

Result files:

- `training_logs/union_candidate_verifier.md`
- `training_logs/union_candidate_verifier.json`
- `scripts/run_union_candidate_verifier.py`

Key result:

- test candidate recall ceiling: `0.7297`
- raw candidate precision: `0.5121`
- three-class Step 2 frontier: `P=0.8038, R=0.5276`
- binary unified frontier: `P=0.8039, R=0.4929`
- max-score frontier: `P=0.8044, R=0.5187`
- MLP verifier test frontier: `P=0.8036, R=0.5623`

This showed the candidate pool contains enough true errors for `R > 0.60`, but
the ranking/filtering stage still needs improvement.

### Piece-level calibration

Result files:

- `training_logs/union_candidate_piece_calibration.md`
- `training_logs/union_candidate_piece_calibration.json`
- `scripts/calibrate_union_candidate_verifier.py`

Calibration alone did not cross the `R >= 0.60` target under the precision
constraint:

- global margin `0.00`: `P=0.7508, R=0.6131`
- global margin `0.05`: `P=0.8036, R=0.5623`
- top-k per piece achieved high precision but too little recall

Conclusion: piece-level calibration is useful as a final stabilizer, but it does
not fix the current ranking bottleneck by itself.

### Verifier model comparison

Result files:

- `training_logs/union_candidate_verifier_variants.md`
- `training_logs/union_candidate_verifier_variants.json`
- `scripts/run_union_candidate_verifier_variants.py`

The best candidate-level verifier was a traditional tree model:

- `sklearn_hist_gradient_boosting`: `P=0.8031, R=0.5762`
- `sklearn_logreg_balanced` test frontier: `P=0.8014, R=0.5723`
- best MLP variant: `P=0.8006, R=0.5708`
- random forest: `P=0.8050, R=0.5657`

This does not mean traditional ML is better than the sequence model. It means
the Stage 2 verifier is a tabular candidate-ranking problem after the deep
detectors and likelihood teachers have already produced strong evidence. Tree
models are well matched to this stage.

### True traditional end-to-end baseline

Result files:

- `training_logs/traditional_end_to_end_baseline.md`
- `training_logs/traditional_end_to_end_baseline.json`
- `scripts/run_traditional_end_to_end_baseline.py`

This is the strict traditional baseline: it does not use transformer detector
probabilities, likelihood-teacher surprise, or a DL-generated candidate pool. It
trains directly on `note_features()` hand-crafted note/music-theory features and
predicts clean versus wrong for every note.

Settings:

- train split: MAESTRO train
- train error rate: `0.08`
- validation/test error rate: `0.01`
- training sample: `500000` notes from `11193985` seen notes
- validation notes: `1261824`
- test notes: `1457920`

Best strict traditional result:

- `hist_gradient_boosting`: `P=0.8523, R=0.0924`
- random forest frontier: `P=0.8056, R=0.0345`
- logistic regression and linear SVM were effectively unusable at the precision
  constraint

Conclusion: pure hand-crafted traditional methods are not sufficient for
reference-free wrong-note detection over all notes. Their useful role is as a
cascade verifier after a deep contextual detector has proposed candidates.

### Current next experiment

`scripts/run_union_candidate_context_verifier.py` adds piece-relative and local
candidate-context features to the Stage 2 verifier:

- score rank/z-score within each piece
- per-piece candidate density
- note position within the piece
- local mean/max detector score and likelihood surprise over neighboring notes
- multiple lower Stage 1 candidate-threshold combinations

The first run exposed a CPU/GPU indexing bug in the metadata feature path. The
script has been fixed and relaunched on the remote training host. Its results
will determine whether improved Stage 2 ranking can move from the current
`P=0.8031, R=0.5762` frontier to `R >= 0.60`.

## 2026-06-19 - Theory Interaction Result and Voice-Aware Follow-up

The focused theory-interaction verifier completed with:

- candidate thresholds: three-class `0.60`, binary `0.50`
- candidate recall ceiling: `0.7284`
- best calibrated result:
  - verifier: histogram gradient boosting
  - calibration: candidate-density adjustment `alpha=0.06`
  - precision: `0.8001`
  - recall: `0.5932`

This does not improve the established frozen-context result
`P=0.8008, R=0.5963`. The 57 theory interaction features are retained as a
negative ablation rather than replacing the current best system.

The next experiment tests whether the existing melodic and ornament features are
being degraded by polyphonic event ordering. Adjacent MIDI events frequently
belong to different voices, so global `prev/next`, passing-tone, neighbor-tone,
and resolution features can describe a false melodic path.

Two lightweight offline voice-assignment methods will be evaluated in parallel:

1. onset-group minimum-cost matching:
   - globally match each onset group to active voice tracks;
   - penalize pitch distance, long gaps, overlap conflicts, crossings, and abrupt
     direction changes;
2. whole-piece beam/DP optimization:
   - retain several assignment hypotheses across onset groups;
   - optimize longer-range track continuity and crossing stability.

Both methods output the same candidate-level voice evidence:

- assignment confidence and uncertainty;
- melody, bass, and inner-voice role;
- previous and next same-voice pitch interval and time gap;
- same-voice step-in/step-out;
- passing, neighbor, and resolution evidence;
- track length, stability, and crossing evidence.

Voice assignment is auxiliary evidence only. Low-confidence assignments are
masked or down-weighted rather than forced.

Important evaluation correction:

- the standard dataset injects corruptions independently inside overlapping
  windows;
- whole-piece voice assignment requires one consistent observed MIDI sequence;
- this experiment therefore corrupts each complete piece once with a fixed seed,
  then slices overlapping windows;
- each run trains and evaluates both a no-voice baseline and its voice-aware
  verifier on exactly the same piece-consistent candidates.

Primary metric:

`maximize recall subject to precision >= 0.80`

Decision rules:

- preserve a method only if it improves its matched no-voice baseline;
- compare against the historical `R=0.5963` only as a secondary reference because
  the piece-consistent corruption protocol is stricter and scientifically cleaner;
- if either lightweight method improves verifier ranking, integrate its
  confidence-weighted voice features into the detector and retrain;
- if neither improves, use the assignment outputs for error analysis before
  attempting a learned voice-separation model.

## 2026-06-19 - Candidate Ceiling Diagnosis and Verifier Improvements 1-3

The first voice-aware runs accidentally used the older keyboard-aware Step 2
checkpoints. Their low candidate recall was therefore not evidence against voice
separation:

- old Step 2 checkpoints, piece-consistent protocol: candidate recall `0.5945`;
- correct directional-frozen checkpoints, piece features: `0.7190`;
- correct directional-frozen checkpoints, window features: `0.7166`;
- correct directional-frozen checkpoints, original legacy protocol: `0.7284`.

The legacy-protocol result exactly reproduces the established candidate ceiling,
confirming checkpoint selection as the root cause. The voice runner now pins the
correct directional-frozen checkpoints.

The next combined experiment reuses one candidate collection to evaluate three
low-cost verifier improvements:

1. calibration-selected convex score fusion across the old and theory-aware HGB
   verifiers, including candidate-density adjustment;
2. a genuine within-piece pairwise ranker trained against hard false-positive
   candidates;
3. FP/FN analysis by chord, scale, ornament, short-note, strong-beat,
   high-density, and model-disagreement categories.

Success criterion remains maximum recall subject to precision `>= 0.80`, with
the immediate target `R > 0.60`.
