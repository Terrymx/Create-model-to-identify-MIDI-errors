from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from midi_error_detector.data import FEATURE_SIZE, MaestroWrongNoteDataset
from midi_error_detector.model import build_wrong_note_model
from midi_error_detector.train import (
    PITCH_CONTEXT_FEATURE_COLUMNS,
    build_explicit_correction_evidence,
    build_explicit_surprise,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure external likelihood quality and redundancy with the Step 2 surprise signal."
    )
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--eval-split", default="validation", choices=["validation", "test"])
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--candidate-threshold", type=float, default=0.55)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--output-json", default="training_logs/external_likelihood_diagnostic.json")
    parser.add_argument("--output-md", default="training_logs/external_likelihood_diagnostic.md")
    return parser.parse_args()


def checkpoint_args(checkpoint: dict) -> SimpleNamespace:
    raw = checkpoint.get("args", {})
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


def load_model(
    path: str,
    device: torch.device,
    require_explicit_surprise: bool,
) -> tuple[torch.nn.Module, SimpleNamespace]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = checkpoint_args(checkpoint)
    if require_explicit_surprise and not args.explicit_surprise:
        raise ValueError(f"{path} is not an explicit-surprise detector")
    model = build_wrong_note_model(
        model_type=args.model,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        transformer_d_model=args.transformer_d_model,
        transformer_heads=args.transformer_heads,
        transformer_ffn_dim=args.transformer_ffn_dim,
        dropout=args.dropout,
        explicit_surprise=require_explicit_surprise,
        surprise_embedding_dim=args.surprise_embedding_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, args


def adapt_features(features: torch.Tensor, input_size: int) -> torch.Tensor:
    if features.shape[-1] > input_size:
        return features[..., :input_size]
    if features.shape[-1] < input_size:
        return torch.nn.functional.pad(features, (0, input_size - features.shape[-1]))
    return features


def rank_auc(values: torch.Tensor, labels: torch.Tensor) -> float:
    values = values.double()
    labels = labels.bool()
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    if positive_count == 0 or negative_count == 0:
        return 0.5
    order = torch.argsort(values)
    sorted_values = values[order]
    ranks = torch.empty(len(values), dtype=torch.double)
    _, inverse, counts = torch.unique_consecutive(
        sorted_values,
        return_inverse=True,
        return_counts=True,
    )
    ends = counts.cumsum(0)
    starts = ends - counts
    average_rank = (starts.double() + 1.0 + ends.double()) / 2.0
    ranks[order] = average_rank[inverse]
    positive_rank_sum = float(ranks[labels].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double() - left.double().mean()
    right = right.double() - right.double().mean()
    denominator = torch.sqrt((left.square().sum()) * (right.square().sum())).clamp_min(1e-12)
    return float((left * right).sum() / denominator)


def average_ranks(values: torch.Tensor) -> torch.Tensor:
    values = values.double()
    order = torch.argsort(values)
    sorted_values = values[order]
    ranks = torch.empty(len(values), dtype=torch.double)
    _, inverse, counts = torch.unique_consecutive(
        sorted_values,
        return_inverse=True,
        return_counts=True,
    )
    ends = counts.cumsum(0)
    starts = ends - counts
    average_rank = (starts.double() + 1.0 + ends.double()) / 2.0
    ranks[order] = average_rank[inverse]
    return ranks


def js_divergence(clean: torch.Tensor, error: torch.Tensor, bins: int = 80) -> float:
    minimum = float(torch.minimum(clean.min(), error.min()))
    maximum = float(torch.maximum(clean.max(), error.max()))
    if maximum <= minimum:
        return 0.0
    clean_hist = torch.histc(clean.float(), bins=bins, min=minimum, max=maximum).double()
    error_hist = torch.histc(error.float(), bins=bins, min=minimum, max=maximum).double()
    clean_prob = (clean_hist + 1e-9) / (clean_hist.sum() + bins * 1e-9)
    error_prob = (error_hist + 1e-9) / (error_hist.sum() + bins * 1e-9)
    midpoint = 0.5 * (clean_prob + error_prob)
    return float(
        0.5 * (clean_prob * (clean_prob / midpoint).log()).sum()
        + 0.5 * (error_prob * (error_prob / midpoint).log()).sum()
    )


def signal_summary(values: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    clean = values[~labels]
    error = values[labels]
    return {
        "auc": rank_auc(values, labels),
        "clean_mean": float(clean.mean()),
        "clean_std": float(clean.std(unbiased=False)),
        "error_mean": float(error.mean()),
        "error_std": float(error.std(unbiased=False)),
        "mean_gap": float(error.mean() - clean.mean()),
        "js_divergence": js_divergence(clean, error),
    }


@torch.no_grad()
def build_train_aligned_surprise(
    model: torch.nn.Module,
    features: torch.Tensor,
    mask: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    """Mask only target positions, matching clean masked-pitch training."""

    valid = mask.bool()
    observed_pitch = torch.round(features[:, :, 0] * 127.0).long().clamp(0, 127)
    surprise = torch.zeros_like(mask)
    positions = torch.arange(features.shape[1], device=features.device).unsqueeze(0)
    group_count = max(1, groups)
    for group_index in range(group_count):
        selected = valid & ((positions % group_count) == group_index)
        if not selected.any():
            continue
        context_features = features.clone()
        context_features[:, :, PITCH_CONTEXT_FEATURE_COLUMNS] = torch.where(
            selected.unsqueeze(-1),
            torch.zeros_like(context_features[:, :, PITCH_CONTEXT_FEATURE_COLUMNS]),
            context_features[:, :, PITCH_CONTEXT_FEATURE_COLUMNS],
        )
        pitch_logits = model.predict_pitch(context_features)
        observed_log_probability = torch.log_softmax(pitch_logits, dim=-1).gather(
            dim=-1,
            index=observed_pitch.unsqueeze(-1),
        ).squeeze(-1)
        surprise = torch.where(selected, -observed_log_probability, surprise)
    return surprise


def write_markdown(path: Path, result: dict) -> None:
    all_notes = result["all_notes"]
    candidates = result["candidates"]
    lines = [
        "# External Likelihood Diagnostic",
        "",
        f"- split: `{result['eval_split']}`",
        f"- error rate: `{result['error_rate']}`",
        f"- valid notes: `{result['valid_notes']}`",
        f"- error notes: `{result['error_notes']}`",
        f"- legacy teacher clean perplexity: `{result['teacher_clean_perplexity']:.4f}`",
        f"- train-aligned teacher clean perplexity: `{result['aligned_teacher_clean_perplexity']:.4f}`",
        f"- teacher/base Pearson correlation: `{result['pearson_correlation']:.4f}`",
        f"- teacher/base Spearman correlation: `{result['spearman_correlation']:.4f}`",
        "",
        "| Population | Signal | AUC | Clean/FP Mean | Error/TP Mean | JS Divergence |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for population, rows in (("all notes", all_notes), ("candidates", candidates["signals"])):
        for name, row in rows.items():
            lines.append(
                f"| {population} | {name} | {row['auc']:.4f} | {row['clean_mean']:.4f} | "
                f"{row['error_mean']:.4f} | {row['js_divergence']:.4f} |"
            )
    lines.extend(
        [
            "",
            f"- candidate threshold: `{candidates['threshold']}`",
            f"- candidate precision: `{candidates['precision']:.4f}`",
            f"- candidate recall ceiling: `{candidates['recall']:.4f}`",
            f"- recommendation: {result['recommendation']}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    teacher, teacher_args = load_model(args.teacher_checkpoint, device, False)
    detector, detector_args = load_model(args.detector_checkpoint, device, True)
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

    teacher_parts: list[torch.Tensor] = []
    aligned_teacher_parts: list[torch.Tensor] = []
    base_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    candidate_parts: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="likelihood diagnostic", unit="batch", dynamic_ncols=True):
        raw_features = batch["features"].to(device)
        mask = batch["mask"].to(device).bool()
        labels = batch["is_error"].to(device).bool()
        teacher_features = adapt_features(raw_features, teacher_args.input_size)
        detector_features = adapt_features(raw_features, detector_args.input_size)

        _, teacher_surprise, _ = build_explicit_correction_evidence(
            teacher,
            teacher_features,
            mask.float(),
            groups=args.groups,
        )
        aligned_teacher_surprise = build_train_aligned_surprise(
            teacher,
            teacher_features,
            mask.float(),
            groups=args.groups,
        )
        base_surprise, base_available = build_explicit_surprise(
            detector,
            detector_features,
            mask.float(),
            training=False,
            train_mask_rate=0.0,
            eval_groups=args.groups,
        )
        outputs = detector(
            detector_features,
            surprise=base_surprise,
            surprise_available=base_available,
        )
        candidates = (torch.sigmoid(outputs["error_logits"]) >= args.candidate_threshold) & mask
        teacher_parts.append(teacher_surprise[mask].cpu())
        aligned_teacher_parts.append(aligned_teacher_surprise[mask].cpu())
        base_parts.append(base_surprise[mask].cpu())
        label_parts.append(labels[mask].cpu())
        candidate_parts.append(candidates[mask].cpu())

    teacher_values = torch.cat(teacher_parts)
    aligned_teacher_values = torch.cat(aligned_teacher_parts)
    base_values = torch.cat(base_parts)
    labels = torch.cat(label_parts).bool()
    candidate_mask = torch.cat(candidate_parts).bool()
    candidate_labels = labels[candidate_mask]
    candidate_teacher = teacher_values[candidate_mask]
    candidate_aligned_teacher = aligned_teacher_values[candidate_mask]
    candidate_base = base_values[candidate_mask]

    all_signals = {
        "external_teacher": signal_summary(teacher_values, labels),
        "external_teacher_train_aligned": signal_summary(aligned_teacher_values, labels),
        "step2_internal": signal_summary(base_values, labels),
    }
    candidate_signals = {
        "external_teacher": signal_summary(candidate_teacher, candidate_labels),
        "external_teacher_train_aligned": signal_summary(candidate_aligned_teacher, candidate_labels),
        "step2_internal": signal_summary(candidate_base, candidate_labels),
    }
    correlation = pearson(teacher_values, base_values)
    rank_correlation = pearson(average_ranks(teacher_values), average_ranks(base_values))
    teacher_increment = (
        candidate_signals["external_teacher_train_aligned"]["auc"]
        - candidate_signals["step2_internal"]["auc"]
    )
    if correlation >= 0.9 and teacher_increment < 0.01:
        recommendation = (
            "The external teacher is highly redundant with the Step 2 signal. Keep Branch A as the baseline "
            "and move to genuinely new structural evidence rather than expanding this likelihood path."
        )
    elif candidate_signals["external_teacher_train_aligned"]["auc"] < 0.6:
        recommendation = (
            "The teacher signal is too weak among hard candidates. Improve the clean-music model or its "
            "masking objective before another detector training run."
        )
    else:
        recommendation = (
            "The teacher contains complementary ranking information. Diagnose the fusion/head capacity "
            "before discarding the external likelihood path."
        )
    result = {
        "teacher_checkpoint": args.teacher_checkpoint,
        "detector_checkpoint": args.detector_checkpoint,
        "eval_split": args.eval_split,
        "error_rate": args.error_rate,
        "valid_notes": int(labels.numel()),
        "error_notes": int(labels.sum()),
        "teacher_clean_perplexity": math.exp(float(teacher_values[~labels].mean())),
        "aligned_teacher_clean_perplexity": math.exp(
            float(aligned_teacher_values[~labels].mean())
        ),
        "pearson_correlation": correlation,
        "spearman_correlation": rank_correlation,
        "all_notes": all_signals,
        "candidates": {
            "threshold": args.candidate_threshold,
            "count": int(candidate_mask.sum()),
            "precision": float(candidate_labels.float().mean()),
            "recall": float(candidate_labels.sum() / labels.sum().clamp_min(1)),
            "signals": candidate_signals,
            "teacher_auc_increment": teacher_increment,
        },
        "recommendation": recommendation,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
