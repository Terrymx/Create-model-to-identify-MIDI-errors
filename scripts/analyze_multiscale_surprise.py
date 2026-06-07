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
from midi_error_detector.train import build_explicit_surprise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare wrong-note surprise across context-window sizes.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--eval-split", default="validation", choices=["validation", "test"])
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--candidate-threshold", type=float, default=0.55)
    parser.add_argument("--scales", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--surprise-eval-groups", type=int, default=4)
    parser.add_argument("--output-json", default="training_logs/multiscale_surprise_validation.json")
    parser.add_argument("--output-md", default="training_logs/multiscale_surprise_validation.md")
    return parser.parse_args()


def saved_args(checkpoint: dict) -> SimpleNamespace:
    raw = dict(checkpoint.get("args", {}))
    return SimpleNamespace(
        model=raw.get("model", "transformer"),
        input_size=int(raw.get("input_size", FEATURE_SIZE)),
        hidden_size=int(raw.get("hidden_size", 256)),
        num_layers=int(raw.get("num_layers", 4)),
        transformer_d_model=int(raw.get("transformer_d_model", 192)),
        transformer_heads=int(raw.get("transformer_heads", 4)),
        transformer_ffn_dim=int(raw.get("transformer_ffn_dim", 512)),
        dropout=float(raw.get("dropout", 0.2)),
        explicit_surprise=bool(raw.get("explicit_surprise", False)),
        surprise_embedding_dim=int(raw.get("surprise_embedding_dim", 16)),
    )


def load_model(path: str, device: torch.device) -> tuple[torch.nn.Module, SimpleNamespace]:
    checkpoint = torch.load(path, map_location=device)
    args = saved_args(checkpoint)
    if not args.explicit_surprise:
        raise ValueError("The checkpoint must have explicit_surprise enabled.")
    model = build_wrong_note_model(
        model_type=args.model,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        transformer_d_model=args.transformer_d_model,
        transformer_heads=args.transformer_heads,
        transformer_ffn_dim=args.transformer_ffn_dim,
        dropout=args.dropout,
        explicit_surprise=True,
        surprise_embedding_dim=args.surprise_embedding_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, args


@torch.no_grad()
def centered_window_surprise(
    model: torch.nn.Module,
    features: torch.Tensor,
    mask: torch.Tensor,
    scale: int,
    eval_groups: int,
) -> torch.Tensor:
    """Compute surprise from overlapping blocks, preferring centered positions."""

    sequence_length = features.shape[1]
    if scale >= sequence_length:
        surprise, _ = build_explicit_surprise(
            model,
            features,
            mask.float(),
            training=False,
            train_mask_rate=0.0,
            eval_groups=eval_groups,
        )
        return surprise

    surprise = torch.zeros_like(mask, dtype=features.dtype)
    best_margin = torch.full_like(mask, -1, dtype=torch.long)
    stride = max(1, scale // 2)
    starts = list(range(0, max(sequence_length - scale, 0) + 1, stride))
    final_start = max(0, sequence_length - scale)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    for start in starts:
        end = min(start + scale, sequence_length)
        block_surprise_values, _ = build_explicit_surprise(
            model,
            features[:, start:end],
            mask[:, start:end].float(),
            training=False,
            train_mask_rate=0.0,
            eval_groups=eval_groups,
        )
        positions = torch.arange(end - start, device=features.device)
        margins = torch.minimum(positions, (end - start - 1) - positions)
        replace = margins.unsqueeze(0) > best_margin[:, start:end]
        surprise[:, start:end] = torch.where(replace, block_surprise_values, surprise[:, start:end])
        best_margin[:, start:end] = torch.where(replace, margins.unsqueeze(0), best_margin[:, start:end])
    return surprise


def rank_auc(values: torch.Tensor, labels: torch.Tensor) -> float:
    """Return tie-aware ROC AUC where larger values predict an error."""

    values = values.double()
    labels = labels.bool()
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    if positive_count == 0 or negative_count == 0:
        return 0.5
    order = torch.argsort(values)
    sorted_values = values[order]
    ranks = torch.empty(len(values), dtype=torch.double)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = float(ranks[labels].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def summarize(name: str, values: torch.Tensor, labels: torch.Tensor) -> dict:
    positive = values[labels]
    negative = values[~labels]
    return {
        "name": name,
        "auc": rank_auc(values, labels),
        "tp_mean": float(positive.mean()),
        "tp_std": float(positive.std(unbiased=False)),
        "fp_mean": float(negative.mean()),
        "fp_std": float(negative.std(unbiased=False)),
        "mean_gap": float(positive.mean() - negative.mean()),
    }


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Multi-Window Surprise Validation",
        "",
        f"- checkpoint: `{result['checkpoint']}`",
        f"- split: `{result['eval_split']}`",
        f"- candidate threshold: `{result['candidate_threshold']}`",
        f"- candidates: `{result['candidate_count']}`",
        f"- candidate precision: `{result['candidate_precision']:.4f}`",
        f"- candidate recall ceiling: `{result['candidate_recall']:.4f}`",
        "",
        "| Signal | AUC | TP Mean | FP Mean | Mean Gap |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result["signals"]:
        lines.append(
            f"| {row['name']} | {row['auc']:.4f} | {row['tp_mean']:.4f} | "
            f"{row['fp_mean']:.4f} | {row['mean_gap']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- best signal: `{result['best_signal']['name']}` (AUC `{result['best_signal']['auc']:.4f}`)",
            f"- 256-window AUC: `{result['wide_auc']:.4f}`",
            f"- incremental AUC: `{result['incremental_auc']:.4f}`",
            f"- recommendation: {result['recommendation']}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    scales = sorted(set(args.scales))
    if args.window_size not in scales:
        scales.append(args.window_size)
        scales.sort()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} scales={scales}", flush=True)
    model, model_args = load_model(args.checkpoint, device)
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

    collected = {scale: [] for scale in scales}
    labels = []
    total_errors = 0
    total_notes = 0
    candidate_count = 0
    candidate_positives = 0
    for batch in tqdm(loader, desc="multi-window validation", unit="batch", dynamic_ncols=True):
        full_features = batch["features"].to(device)
        features = full_features
        if features.shape[-1] > model_args.input_size:
            features = features[..., : model_args.input_size]
        elif features.shape[-1] < model_args.input_size:
            features = torch.nn.functional.pad(features, (0, model_args.input_size - features.shape[-1]))
        mask = batch["mask"].to(device).bool()
        target = batch["is_error"].to(device).bool()

        scale_surprises = {
            scale: centered_window_surprise(model, features, mask, scale, args.surprise_eval_groups)
            for scale in scales
        }
        wide_surprise = scale_surprises[args.window_size]
        outputs = model(features, surprise=wide_surprise, surprise_available=mask.float())
        error_probability = torch.sigmoid(outputs["error_logits"])
        candidate_mask = (error_probability >= args.candidate_threshold) & mask

        total_errors += int((target & mask).sum())
        total_notes += int(mask.sum())
        candidate_count += int(candidate_mask.sum())
        candidate_positives += int((candidate_mask & target).sum())
        if not bool(candidate_mask.any()):
            continue
        labels.append(target[candidate_mask].cpu())
        for scale in scales:
            collected[scale].append(scale_surprises[scale][candidate_mask].cpu())

    candidate_labels = torch.cat(labels).bool()
    surprise_by_scale = {scale: torch.cat(parts) for scale, parts in collected.items()}
    stacked = torch.stack([surprise_by_scale[scale] for scale in scales], dim=1)
    signals = [summarize(f"surprise_{scale}", surprise_by_scale[scale], candidate_labels) for scale in scales]
    signals.extend(
        [
            summarize("surprise_mean", stacked.mean(dim=1), candidate_labels),
            summarize("surprise_min", stacked.min(dim=1).values, candidate_labels),
            summarize("surprise_max", stacked.max(dim=1).values, candidate_labels),
            summarize("surprise_range", stacked.max(dim=1).values - stacked.min(dim=1).values, candidate_labels),
        ]
    )
    best_signal = max(signals, key=lambda row: row["auc"])
    wide_row = next(row for row in signals if row["name"] == f"surprise_{args.window_size}")
    incremental_auc = best_signal["auc"] - wide_row["auc"]
    recommendation = (
        "Implement multi-window surprise training; the scales add measurable separation."
        if incremental_auc >= 0.01
        else "Do not retrain yet; the tested scales are mostly redundant, so inspect scale alignment or move to the cascade probe."
    )
    result = {
        "checkpoint": args.checkpoint,
        "eval_split": args.eval_split,
        "error_rate": args.error_rate,
        "candidate_threshold": args.candidate_threshold,
        "scales": scales,
        "notes": total_notes,
        "error_notes": total_errors,
        "candidate_count": candidate_count,
        "candidate_positive_count": candidate_positives,
        "candidate_precision": candidate_positives / max(candidate_count, 1),
        "candidate_recall": candidate_positives / max(total_errors, 1),
        "signals": signals,
        "best_signal": best_signal,
        "wide_auc": wide_row["auc"],
        "incremental_auc": incremental_auc,
        "recommendation": recommendation,
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
