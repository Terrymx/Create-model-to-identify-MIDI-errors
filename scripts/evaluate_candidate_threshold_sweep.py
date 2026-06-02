from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from midi_error_detector.data import FEATURE_SIZE, MaestroWrongNoteDataset
from midi_error_detector.model import build_wrong_note_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep first-stage candidate thresholds.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--eval-split", default="test", choices=["validation", "test"])
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.6, 0.7, 0.75, 0.8, 0.85])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--output-json", default="training_logs/candidate_threshold_sweep.json")
    parser.add_argument("--output-md", default="training_logs/candidate_threshold_sweep.md")
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


def f_beta(precision: float, recall: float, beta: float) -> float:
    beta2 = beta * beta
    return (1.0 + beta2) * precision * recall / max(beta2 * precision + recall, 1e-12)


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


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Candidate Threshold Sweep",
        "",
        f"- checkpoint: `{result['checkpoint']}`",
        f"- eval split: `{result['eval_split']}`",
        f"- error rate: `{result['error_rate']}`",
        f"- total notes: `{result['notes']}`",
        f"- total error notes: `{result['error_notes']}`",
        "",
        "| Threshold | Precision | Recall | F1 | F0.5 | TP | FP | FN | Candidates |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["thresholds"]:
        lines.append(
            "| "
            f"{row['threshold']:.2f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['f0_5']:.4f} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{row['candidates']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
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
        verbose=True,
    )
    dataset.set_epoch(0)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    thresholds = sorted(set(args.thresholds))
    stats = {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in thresholds}
    total_notes = 0
    total_errors = 0
    for batch in tqdm(loader, desc="threshold sweep", unit="batch", dynamic_ncols=True):
        features = batch["features"].to(device)
        if features.shape[-1] > model_args.input_size:
            features = features[..., : model_args.input_size]
        elif features.shape[-1] < model_args.input_size:
            features = torch.nn.functional.pad(features, (0, model_args.input_size - features.shape[-1]))

        outputs = model(features)
        probabilities = torch.sigmoid(outputs["error_logits"])
        action_predictions = outputs["kind_logits"].argmax(dim=-1)
        mask = batch["mask"].to(device).bool()
        targets = batch["is_error"].to(device).bool()
        clean_targets = (~targets) & mask
        error_targets = targets & mask
        total_notes += int(mask.sum().item())
        total_errors += int(error_targets.sum().item())
        for threshold, row in stats.items():
            predictions = (probabilities >= threshold) & (action_predictions != 0) & mask
            row["tp"] += int((predictions & error_targets).sum().item())
            row["fp"] += int((predictions & clean_targets).sum().item())
            row["fn"] += int(((~predictions) & error_targets).sum().item())

    rows = []
    for threshold, row in stats.items():
        precision = row["tp"] / max(row["tp"] + row["fp"], 1)
        recall = row["tp"] / max(row["tp"] + row["fn"], 1)
        rows.append(
            {
                "threshold": threshold,
                "tp": row["tp"],
                "fp": row["fp"],
                "fn": row["fn"],
                "candidates": row["tp"] + row["fp"],
                "precision": precision,
                "recall": recall,
                "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
                "f0_5": f_beta(precision, recall, 0.5),
            }
        )
    result = {
        "checkpoint": args.checkpoint,
        "eval_split": args.eval_split,
        "error_rate": args.error_rate,
        "notes": total_notes,
        "error_notes": total_errors,
        "thresholds": rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), result)
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {output_json}", flush=True)
    print(f"wrote {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
