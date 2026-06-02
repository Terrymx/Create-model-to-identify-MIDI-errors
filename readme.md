# MIDI Wrong-Note Detection

This project trains note-level models for detecting wrong notes in MIDI performances. It uses clean classical piano MIDI files, injects realistic synthetic note errors during training, and predicts which notes should be kept, replaced, or deleted.

The current experimental line uses a Transformer encoder with musical-context features. Earlier BiGRU experiments are kept as baselines.

## Task

Given a MIDI note sequence, the model predicts:

- whether each note is likely to be wrong;
- whether the note should be `keep`, `replace`, or `delete`;
- a replacement pitch candidate when the note should be replaced.

The long-term target is a practical sparse-error detector: high precision and high recall when a real piece contains only a small number of wrong notes.

## Dataset

The experiments use the MAESTRO MIDI dataset. Download the MIDI-only package from the official MAESTRO/Magenta dataset page and pass its extracted directory with `--data-root`.

The dataset itself is not included in this repository.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

## Transformer Training Example

Replace `<PATH_TO_MAESTRO>` with your local MAESTRO MIDI directory.

```powershell
python -u -m midi_error_detector.train `
  --model transformer `
  --data-root "<PATH_TO_MAESTRO>" `
  --clean-epochs 1 `
  --epochs 40 `
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
  --threshold-sweep 0.2 0.25 0.3 0.35 0.4 0.5 `
  --save-metric task_score `
  --lr-patience 4 `
  --lr-factor 0.5 `
  --lr-threshold 0.002 `
  --num-workers 0 `
  --output checkpoints\transformer_wrong_note_taskscore.pt
```

## Important Baseline

A previous 40-epoch BiGRU/default-model baseline used the same wrong-note task and reached:

- `task_score=0.7743`
- best detection threshold `0.5`
- precision `0.5994`
- recall `0.8343`
- F1 `0.6976`

That run had strong recall but weak precision, which motivated the later precision-first and sparse-error experiments.

## Current Experiment Line

The recent Transformer experiments explore:

- chord, scale, and degree features;
- low-error-rate evaluation for realistic sparse wrong-note settings;
- precision-first fine-tuning;
- high-confidence thresholding and top-K post-processing;
- melodic theory features such as passing tone, neighbor tone, and resolution;
- theory-weighted loss to recover recall without giving up too much precision.

Tracked experiment notes live in `experiments.md`.

## Local Notes

Do not put machine-specific paths, virtual environment paths, dataset paths, or private run commands in this public README.

Use `LOCAL_NOTES.md` for personal paths and local commands. That file is ignored by Git.

## Repository Hygiene

Training logs and large checkpoints are generated artifacts. Keep them local unless a specific result needs to be shared.

Typical ignored artifacts include:

- `training_logs/*.log`
- `training_logs/*.err.log`
- generated checkpoints
- local virtual environments
