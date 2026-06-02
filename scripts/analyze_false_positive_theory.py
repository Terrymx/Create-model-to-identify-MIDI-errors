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

THEORY_FEATURES = {
    "chord_tone": 20,
    "step_in": 26,
    "step_out": 27,
    "same_direction": 28,
    "passing_tone": 29,
    "neighbor_tone": 30,
    "resolves_by_step": 31,
    "non_chord_resolution": 32,
    "short_or_normal_duration": 33,
    "downbeat_strength": 34,
    "subdivision_strength": 35,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze theory-feature patterns in false positives.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--eval-split", default="test", choices=["validation", "test"])
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.95, 0.97])
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--output-json", default="training_logs/false_positive_theory_analysis.json")
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


def new_bucket() -> dict:
    return {
        "count": 0,
        "feature_true": {name: 0 for name in THEORY_FEATURES},
        "sum_error_prob": 0.0,
        "sum_keep_prob": 0.0,
        "sum_replace_prob": 0.0,
        "sum_delete_prob": 0.0,
    }


def feature_mask(name: str, values: torch.Tensor) -> torch.Tensor:
    if name == "short_or_normal_duration":
        return values <= 0.35
    if name in {"downbeat_strength", "subdivision_strength"}:
        return values >= 0.5
    return values >= 0.5


def add_bucket(bucket: dict, features: torch.Tensor, error_prob: torch.Tensor, action_probs: torch.Tensor) -> None:
    count = int(features.shape[0])
    bucket["count"] += count
    if count == 0:
        return
    bucket["sum_error_prob"] += float(error_prob.sum().item())
    bucket["sum_keep_prob"] += float(action_probs[:, 0].sum().item())
    bucket["sum_replace_prob"] += float(action_probs[:, 1].sum().item())
    bucket["sum_delete_prob"] += float(action_probs[:, 2].sum().item())
    for name, index in THEORY_FEATURES.items():
        if index >= features.shape[1]:
            continue
        bucket["feature_true"][name] += int(feature_mask(name, features[:, index]).sum().item())


def finalize_bucket(bucket: dict) -> dict:
    count = max(bucket["count"], 1)
    return {
        "count": bucket["count"],
        "mean_error_prob": bucket["sum_error_prob"] / count,
        "mean_keep_prob": bucket["sum_keep_prob"] / count,
        "mean_replace_prob": bucket["sum_replace_prob"] / count,
        "mean_delete_prob": bucket["sum_delete_prob"] / count,
        "feature_rates": {name: value / count for name, value in bucket["feature_true"].items()},
    }


@torch.no_grad()
def analyze_checkpoint(
    label: str,
    path: str,
    loader: DataLoader,
    device: torch.device,
    thresholds: list[float],
) -> dict:
    model, model_args = make_model(path, device)
    threshold_stats = {
        threshold: {
            "raw_fp": new_bucket(),
            "raw_tp": new_bucket(),
            "reported_fp": new_bucket(),
            "reported_tp_bucket": new_bucket(),
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "reported_tp_count": 0,
            "reported_fp_count": 0,
            "reported_fn": 0,
        }
        for threshold in thresholds
    }
    total_clean = 0
    total_error = 0
    for batch in tqdm(loader, desc=label, unit="batch", dynamic_ncols=True):
        full_features = batch["features"].to(device)
        features = full_features
        if features.shape[-1] > model_args.input_size:
            features = features[..., : model_args.input_size]
        elif features.shape[-1] < model_args.input_size:
            features = torch.nn.functional.pad(features, (0, model_args.input_size - features.shape[-1]))

        valid_mask = batch["mask"].to(device).bool()
        is_error = batch["is_error"].to(device).bool()
        outputs = model(features)
        error_probabilities = torch.sigmoid(outputs["error_logits"])
        action_probabilities = torch.softmax(outputs["kind_logits"], dim=-1)
        actions = action_probabilities.argmax(dim=-1)

        clean_targets = (~is_error) & valid_mask
        error_targets = is_error & valid_mask
        total_clean += int(clean_targets.sum().item())
        total_error += int(error_targets.sum().item())

        flat_features = full_features.reshape(-1, full_features.shape[-1])
        flat_error_probabilities = error_probabilities.reshape(-1)
        flat_action_probabilities = action_probabilities.reshape(-1, 3)
        flat_clean_targets = clean_targets.reshape(-1)
        flat_error_targets = error_targets.reshape(-1)
        flat_actions = actions.reshape(-1)

        for threshold, stats in threshold_stats.items():
            predictions = (error_probabilities >= threshold) & valid_mask
            reported = predictions & (actions != 0)
            flat_predictions = predictions.reshape(-1)
            flat_reported = reported.reshape(-1)

            raw_fp = flat_predictions & flat_clean_targets
            raw_tp = flat_predictions & flat_error_targets
            raw_fn = (~flat_predictions) & flat_error_targets
            reported_fp = flat_reported & flat_clean_targets
            reported_tp = flat_reported & flat_error_targets
            reported_fn = (~flat_reported) & flat_error_targets

            stats["tp"] += int(raw_tp.sum().item())
            stats["fp"] += int(raw_fp.sum().item())
            stats["fn"] += int(raw_fn.sum().item())
            stats["reported_tp_count"] += int(reported_tp.sum().item())
            stats["reported_fp_count"] += int(reported_fp.sum().item())
            stats["reported_fn"] += int(reported_fn.sum().item())
            add_bucket(stats["raw_fp"], flat_features[raw_fp], flat_error_probabilities[raw_fp], flat_action_probabilities[raw_fp])
            add_bucket(stats["raw_tp"], flat_features[raw_tp], flat_error_probabilities[raw_tp], flat_action_probabilities[raw_tp])
            add_bucket(
                stats["reported_fp"],
                flat_features[reported_fp],
                flat_error_probabilities[reported_fp],
                flat_action_probabilities[reported_fp],
            )
            add_bucket(
                stats["reported_tp_bucket"],
                flat_features[reported_tp],
                flat_error_probabilities[reported_tp],
                flat_action_probabilities[reported_tp],
            )

    output = {
        "label": label,
        "checkpoint": path,
        "input_size": model_args.input_size,
        "total_clean": total_clean,
        "total_error": total_error,
        "thresholds": {},
    }
    for threshold, stats in threshold_stats.items():
        raw_precision = stats["tp"] / max(stats["tp"] + stats["fp"], 1)
        raw_recall = stats["tp"] / max(stats["tp"] + stats["fn"], 1)
        reported_precision = stats["reported_tp_count"] / max(stats["reported_tp_count"] + stats["reported_fp_count"], 1)
        reported_recall = stats["reported_tp_count"] / max(stats["reported_tp_count"] + stats["reported_fn"], 1)
        output["thresholds"][str(threshold)] = {
            "raw": {
                "tp": stats["tp"],
                "fp": stats["fp"],
                "fn": stats["fn"],
                "precision": raw_precision,
                "recall": raw_recall,
                "fp_analysis": finalize_bucket(stats["raw_fp"]),
                "tp_analysis": finalize_bucket(stats["raw_tp"]),
            },
            "reported_non_keep": {
                "tp": stats["reported_tp_count"],
                "fp": stats["reported_fp_count"],
                "fn": stats["reported_fn"],
                "precision": reported_precision,
                "recall": reported_recall,
                "fp_analysis": finalize_bucket(stats["reported_fp"]),
                "tp_analysis": finalize_bucket(stats["reported_tp_bucket"]),
            },
        }
    return output


def main() -> None:
    args = parse_args()
    if len(args.checkpoints) != len(args.labels):
        raise ValueError("--checkpoints and --labels must have the same length")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    results = [
        analyze_checkpoint(label, checkpoint, loader, device, sorted(set(args.thresholds)))
        for label, checkpoint in zip(args.labels, args.checkpoints)
    ]
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
