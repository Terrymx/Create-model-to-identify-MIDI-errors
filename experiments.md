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
