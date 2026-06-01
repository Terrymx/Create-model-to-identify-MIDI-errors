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
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint under low wrong-note rates.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--version", default="v3.0.0")
    parser.add_argument("--eval-split", default="test", choices=["validation", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--error-rates", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    )
    parser.add_argument("--output-json", default="training_logs/low_error_precision_eval.json")
    parser.add_argument("--output-md", default="training_logs/low_error_precision_eval.md")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def checkpoint_args(ckpt: dict) -> SimpleNamespace:
    saved = dict(ckpt.get("args", {}))
    return SimpleNamespace(
        model=saved.get("model", "bigru"),
        hidden_size=saved.get("hidden_size", 256),
        num_layers=saved.get("num_layers", 2),
        transformer_d_model=saved.get("transformer_d_model", 192),
        transformer_heads=saved.get("transformer_heads", 4),
        transformer_ffn_dim=saved.get("transformer_ffn_dim", 512),
        dropout=saved.get("dropout", 0.2),
    )


def make_loader(args: argparse.Namespace, error_rate: float) -> DataLoader:
    dataset = MaestroWrongNoteDataset(
        root=args.data_root,
        split=args.eval_split,
        version=args.version,
        window_size=args.window_size,
        stride=args.stride,
        error_rate=error_rate,
        max_files=args.max_files,
        cache_notes=True,
        verbose=not args.quiet,
    )
    dataset.quiet = args.quiet
    dataset.set_epoch(0)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, thresholds: list[float]) -> dict:
    stats = {threshold: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for threshold in thresholds}
    totals = {
        "notes": 0.0,
        "error_notes": 0.0,
        "replace_notes": 0.0,
        "delete_notes": 0.0,
        "replace_pitch_top1": 0.0,
        "replace_pitch_top3": 0.0,
        "replace_kind_correct": 0.0,
        "delete_kind_correct": 0.0,
    }
    model.eval()
    for batch in tqdm(
        loader,
        desc="eval",
        unit="batch",
        dynamic_ncols=True,
        disable=getattr(loader.dataset, "quiet", False),
    ):
        features = batch["features"].to(device)
        is_error = batch["is_error"].to(device).bool()
        target_pitch = batch["target_pitch"].to(device)
        error_kind = batch["error_kind"].to(device)
        valid_mask = batch["mask"].to(device).bool()

        outputs = model(features)
        probabilities = torch.sigmoid(outputs["error_logits"])
        clean_targets = (~is_error) & valid_mask
        error_targets = is_error & valid_mask
        replace_targets = (error_kind == 1) & valid_mask
        delete_targets = (error_kind == 2) & valid_mask

        pitch_top3 = outputs["pitch_logits"].topk(k=3, dim=-1).indices
        pitch_top1 = outputs["pitch_logits"].argmax(dim=-1)
        kind_predictions = outputs["kind_logits"].argmax(dim=-1)

        totals["notes"] += float(valid_mask.sum())
        totals["error_notes"] += float(error_targets.sum())
        totals["replace_notes"] += float(replace_targets.sum())
        totals["delete_notes"] += float(delete_targets.sum())
        totals["replace_pitch_top1"] += float(((pitch_top1 == target_pitch) & replace_targets).sum())
        totals["replace_pitch_top3"] += float(
            ((pitch_top3 == target_pitch.unsqueeze(-1)).any(dim=-1) & replace_targets).sum()
        )
        totals["replace_kind_correct"] += float(((kind_predictions == error_kind) & replace_targets).sum())
        totals["delete_kind_correct"] += float(((kind_predictions == error_kind) & delete_targets).sum())

        for threshold, threshold_stats in stats.items():
            predictions = (probabilities >= threshold) & valid_mask
            threshold_stats["tp"] += float((predictions & error_targets).sum())
            threshold_stats["fp"] += float((predictions & clean_targets).sum())
            threshold_stats["fn"] += float(((~predictions) & error_targets).sum())

    threshold_rows = []
    for threshold in thresholds:
        row = stats[threshold]
        precision = row["tp"] / max(row["tp"] + row["fp"], 1.0)
        recall = row["tp"] / max(row["tp"] + row["fn"], 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        f05 = 1.25 * precision * recall / max((0.25 * precision) + recall, 1e-12)
        threshold_rows.append(
            {
                "threshold": threshold,
                "tp": int(row["tp"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f0_5": f05,
            }
        )

    replace_notes = max(totals["replace_notes"], 1.0)
    delete_notes = max(totals["delete_notes"], 1.0)
    return {
        "notes": int(totals["notes"]),
        "error_notes": int(totals["error_notes"]),
        "observed_error_rate": totals["error_notes"] / max(totals["notes"], 1.0),
        "replace_notes": int(totals["replace_notes"]),
        "delete_notes": int(totals["delete_notes"]),
        "replace_pitch_top1": totals["replace_pitch_top1"] / replace_notes,
        "replace_pitch_top3": totals["replace_pitch_top3"] / replace_notes,
        "replace_kind_acc": totals["replace_kind_correct"] / replace_notes,
        "delete_kind_acc": totals["delete_kind_correct"] / delete_notes,
        "thresholds": threshold_rows,
    }


def write_markdown(path: Path, results: list[dict]) -> None:
    lines = [
        "# Low Error Rate Precision Evaluation",
        "",
        "| Target Error Rate | Observed | Threshold | Precision | Recall | F1 | F0.5 | TP | FP | FN |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        for row in result["thresholds"]:
            lines.append(
                "| "
                f"{result['target_error_rate']:.3f} | "
                f"{result['observed_error_rate']:.4f} | "
                f"{row['threshold']:.2f} | "
                f"{row['precision']:.4f} | "
                f"{row['recall']:.4f} | "
                f"{row['f1']:.4f} | "
                f"{row['f0_5']:.4f} | "
                f"{row['tp']} | {row['fp']} | {row['fn']} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = checkpoint_args(ckpt)
    model = build_wrong_note_model(
        model_type=saved_args.model,
        input_size=FEATURE_SIZE,
        hidden_size=saved_args.hidden_size,
        num_layers=saved_args.num_layers,
        transformer_d_model=saved_args.transformer_d_model,
        transformer_heads=saved_args.transformer_heads,
        transformer_ffn_dim=saved_args.transformer_ffn_dim,
        dropout=saved_args.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    results = []
    for error_rate in args.error_rates:
        loader = make_loader(args, error_rate=error_rate)
        result = evaluate(model, loader, device, sorted(set(args.thresholds)))
        result["target_error_rate"] = error_rate
        results.append(result)
        best_f05 = max(result["thresholds"], key=lambda row: row["f0_5"])
        print(
            f"error_rate={error_rate:.3f} observed={result['observed_error_rate']:.4f} "
            f"best_f0.5_threshold={best_f05['threshold']:.2f} "
            f"precision={best_f05['precision']:.4f} recall={best_f05['recall']:.4f} "
            f"fp={best_f05['fp']} fn={best_f05['fn']}",
            flush=True,
        )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), results)
    print(f"wrote {output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
