from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from midi_error_detector.data import FEATURE_SIZE, MaestroWrongNoteDataset, corrupt_note_window
from midi_error_detector.harmony import harmony_scores_for_pitches
from midi_error_detector.model import build_wrong_note_model


class IndexedDataset(Dataset):
    def __init__(self, dataset: MaestroWrongNoteDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        item["dataset_idx"] = torch.tensor(idx, dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate harmony-aware post-processing under sparse errors.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--eval-split", default="test", choices=["validation", "test"])
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--threshold", type=float, default=0.97)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--output-json", default="training_logs/harmony_postprocess_eval.json")
    parser.add_argument("--output-md", default="training_logs/harmony_postprocess_eval.md")
    return parser.parse_args()


def checkpoint_args(checkpoint: dict) -> SimpleNamespace:
    saved = dict(checkpoint.get("args", {}))
    return SimpleNamespace(
        model=saved.get("model", "bigru"),
        input_size=int(saved.get("input_size", FEATURE_SIZE)),
        hidden_size=saved.get("hidden_size", 256),
        num_layers=saved.get("num_layers", 2),
        transformer_d_model=saved.get("transformer_d_model", 192),
        transformer_heads=saved.get("transformer_heads", 4),
        transformer_ffn_dim=saved.get("transformer_ffn_dim", 512),
        dropout=saved.get("dropout", 0.2),
    )


def make_model(path: str, device: torch.device) -> tuple[torch.nn.Module, SimpleNamespace]:
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint_args(checkpoint)
    model = build_wrong_note_model(
        model_type=args.model,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        transformer_d_model=args.transformer_d_model,
        transformer_heads=args.transformer_heads,
        transformer_ffn_dim=args.transformer_ffn_dim,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, args


def corrupted_notes_for_index(dataset: MaestroWrongNoteDataset, idx: int):
    file_id, start = dataset.index[idx]
    notes = dataset._get_file_notes(file_id)[start : start + dataset.window_size]
    rng = np.random.default_rng(dataset.seed + idx + dataset.epoch * 1_000_003)
    corrupted, _, _, _ = corrupt_note_window(notes, rng, dataset.error_rate)
    return corrupted[: dataset.window_size]


def new_stats() -> dict:
    return {"tp": 0, "fp": 0, "fn": 0}


def update_stats(stats: dict, predictions: torch.Tensor, error_targets: torch.Tensor, clean_targets: torch.Tensor) -> None:
    stats["tp"] += int((predictions & error_targets).sum().item())
    stats["fp"] += int((predictions & clean_targets).sum().item())
    stats["fn"] += int(((~predictions) & error_targets).sum().item())


def metrics(stats: dict) -> dict:
    precision = stats["tp"] / max(stats["tp"] + stats["fp"], 1)
    recall = stats["tp"] / max(stats["tp"] + stats["fn"], 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    f05 = 1.25 * precision * recall / max(0.25 * precision + recall, 1e-12)
    return {**stats, "precision": precision, "recall": recall, "f1": f1, "f0_5": f05}


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Harmony Post-Processing Evaluation",
        "",
        "| Strategy | TP | FP | FN | Precision | Recall | F1 | F0.5 | FP Drop | TP Drop |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    baseline = rows[0]
    for row in rows:
        lines.append(
            "| "
            f"{row['strategy']} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['f0_5']:.4f} | "
            f"{baseline['fp'] - row['fp']} | {baseline['tp'] - row['tp']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_args = make_model(args.checkpoint, device)
    dataset = MaestroWrongNoteDataset(
        root=args.data_root,
        split=args.eval_split,
        window_size=args.window_size,
        stride=args.stride,
        error_rate=args.error_rate,
        max_files=args.max_files,
        cache_notes=True,
        verbose=False,
    )
    dataset.set_epoch(0)
    loader = DataLoader(
        IndexedDataset(dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    strategies = {
        "baseline": new_stats(),
        "min_gain_-0.10": new_stats(),
        "min_gain_0.00": new_stats(),
        "min_gain_0.05": new_stats(),
        "protect_current_0.65_no_gain": new_stats(),
        "protect_current_0.75_no_gain": new_stats(),
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc="harmony eval", unit="batch", dynamic_ncols=True):
            features = batch["features"].to(device)
            if features.shape[-1] > model_args.input_size:
                model_features = features[..., : model_args.input_size]
            elif features.shape[-1] < model_args.input_size:
                model_features = torch.nn.functional.pad(features, (0, model_args.input_size - features.shape[-1]))
            else:
                model_features = features

            outputs = model(model_features)
            probabilities = torch.sigmoid(outputs["error_logits"])
            action_predictions = outputs["kind_logits"].argmax(dim=-1)
            top_pitches = outputs["pitch_logits"].argmax(dim=-1).cpu()
            valid_mask = batch["mask"].bool()
            error_targets = batch["is_error"].bool() & valid_mask
            clean_targets = (~batch["is_error"].bool()) & valid_mask
            base_predictions = (probabilities.cpu() >= args.threshold) & valid_mask & (action_predictions.cpu() != 0)

            keep_by_strategy = {name: base_predictions.clone() for name in strategies}
            for batch_idx, dataset_idx in enumerate(batch["dataset_idx"].tolist()):
                notes = corrupted_notes_for_index(dataset, int(dataset_idx))
                length = int(valid_mask[batch_idx].sum().item())
                candidate_pitches = [int(top_pitches[batch_idx, note_idx]) for note_idx in range(length)]
                current_scores, _, gains = harmony_scores_for_pitches(notes[:length], candidate_pitches)
                for note_idx in range(length):
                    if not bool(base_predictions[batch_idx, note_idx]):
                        continue
                    action_id = int(action_predictions.cpu()[batch_idx, note_idx])
                    if action_id != 1:
                        continue
                    gain = gains[note_idx]
                    current = current_scores[note_idx]
                    if gain < -0.10:
                        keep_by_strategy["min_gain_-0.10"][batch_idx, note_idx] = False
                    if gain < 0.0:
                        keep_by_strategy["min_gain_0.00"][batch_idx, note_idx] = False
                    if gain < 0.05:
                        keep_by_strategy["min_gain_0.05"][batch_idx, note_idx] = False
                    if current >= 0.65 and gain <= 0.0:
                        keep_by_strategy["protect_current_0.65_no_gain"][batch_idx, note_idx] = False
                    if current >= 0.75 and gain <= 0.0:
                        keep_by_strategy["protect_current_0.75_no_gain"][batch_idx, note_idx] = False

            for name, predictions in keep_by_strategy.items():
                update_stats(strategies[name], predictions, error_targets, clean_targets)

    rows = [{"strategy": name, **metrics(stats)} for name, stats in strategies.items()]
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), rows)
    print(json.dumps(rows, indent=2))
    print(f"wrote {output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
