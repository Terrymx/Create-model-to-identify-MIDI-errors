from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from midi_error_detector.data import FEATURE_SIZE, MaestroWrongNoteDataset
from midi_error_detector.model import build_wrong_note_model
from midi_error_detector.train import build_explicit_surprise


class CandidateVerifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a leakage-controlled candidate verifier.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="checkpoints/candidate_verifier_probe.pt")
    parser.add_argument("--output-json", default="training_logs/candidate_verifier_probe.json")
    parser.add_argument("--output-md", default="training_logs/candidate_verifier_probe.md")
    parser.add_argument("--candidate-threshold", type=float, default=0.45)
    parser.add_argument("--target-precision", type=float, default=0.80)
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--scales", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--calibration-file-fraction", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max-validation-files", type=int, default=None)
    parser.add_argument("--max-test-files", type=int, default=None)
    parser.add_argument("--surprise-eval-groups", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def checkpoint_args(checkpoint: dict) -> SimpleNamespace:
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


def load_base_model(path: str, device: torch.device) -> tuple[nn.Module, SimpleNamespace]:
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint_args(checkpoint)
    if not args.explicit_surprise:
        raise ValueError("Candidate verifier probe requires an explicit-surprise checkpoint.")
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


def make_dataset(args: argparse.Namespace, split: str, max_files: int | None) -> MaestroWrongNoteDataset:
    dataset = MaestroWrongNoteDataset(
        root=args.data_root,
        split=split,
        window_size=args.window_size,
        stride=args.stride,
        error_rate=args.error_rate,
        max_files=max_files,
        cache_notes=True,
        verbose=True,
    )
    dataset.set_epoch(0)
    return dataset


def split_indices_by_file(
    dataset: MaestroWrongNoteDataset,
    calibration_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    file_ids = sorted({file_id for file_id, _ in dataset.index})
    if len(file_ids) < 2:
        raise ValueError("At least two validation MIDI files are required.")
    rng = random.Random(seed)
    rng.shuffle(file_ids)
    calibration_count = min(len(file_ids) - 1, max(1, round(len(file_ids) * calibration_fraction)))
    calibration_files = set(file_ids[:calibration_count])
    train_files = set(file_ids[calibration_count:])
    train_indices = [index for index, (file_id, _) in enumerate(dataset.index) if file_id in train_files]
    calibration_indices = [index for index, (file_id, _) in enumerate(dataset.index) if file_id in calibration_files]
    return train_indices, calibration_indices, sorted(train_files), sorted(calibration_files)


def make_loader(dataset: MaestroWrongNoteDataset, indices: list[int], batch_size: int) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def centered_window_surprise(
    model: nn.Module,
    features: torch.Tensor,
    mask: torch.Tensor,
    scale: int,
    eval_groups: int,
) -> torch.Tensor:
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
        values, _ = build_explicit_surprise(
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
        surprise[:, start:end] = torch.where(replace, values, surprise[:, start:end])
        best_margin[:, start:end] = torch.where(replace, margins.unsqueeze(0), best_margin[:, start:end])
    return surprise


def local_mean(values: torch.Tensor, radius: int) -> torch.Tensor:
    width = radius * 2 + 1
    return torch.nn.functional.avg_pool1d(
        values.unsqueeze(1),
        kernel_size=width,
        stride=1,
        padding=radius,
    ).squeeze(1)


@torch.no_grad()
def collect_candidates(
    model: nn.Module,
    model_args: SimpleNamespace,
    loader: DataLoader,
    device: torch.device,
    scales: list[int],
    candidate_threshold: float,
    eval_groups: int,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    feature_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    stats = {"notes": 0, "error_notes": 0, "candidates": 0, "candidate_positives": 0}

    for batch in tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True):
        full_features = batch["features"].to(device)
        features = full_features
        if features.shape[-1] > model_args.input_size:
            features = features[..., : model_args.input_size]
        elif features.shape[-1] < model_args.input_size:
            features = torch.nn.functional.pad(features, (0, model_args.input_size - features.shape[-1]))
        mask = batch["mask"].to(device).bool()
        target = batch["is_error"].to(device).bool()

        surprises = {
            scale: centered_window_surprise(model, features, mask, scale, eval_groups)
            for scale in scales
        }
        wide_surprise = surprises[max(scales)]
        outputs = model(features, surprise=wide_surprise, surprise_available=mask.float())
        error_logit = outputs["error_logits"]
        error_probability = torch.sigmoid(error_logit)
        action_probability = torch.softmax(outputs["kind_logits"], dim=-1)
        pitch_probability = torch.softmax(outputs["pitch_logits"], dim=-1)
        observed_pitch = torch.round(full_features[..., 0] * 127.0).long().clamp(0, 127)
        observed_probability = pitch_probability.gather(-1, observed_pitch.unsqueeze(-1)).squeeze(-1)
        top2_probability, top2_pitch = pitch_probability.topk(k=2, dim=-1)
        entropy = -(pitch_probability * pitch_probability.clamp_min(1e-9).log()).sum(dim=-1) / math.log(128.0)
        pitch_shift = ((top2_pitch[..., 0].float() - observed_pitch.float()) / 24.0).clamp(-1.0, 1.0)

        stacked_surprise = torch.stack([surprises[scale] for scale in scales], dim=-1)
        surprise_mean = stacked_surprise.mean(dim=-1)
        surprise_min = stacked_surprise.min(dim=-1).values
        surprise_max = stacked_surprise.max(dim=-1).values
        surprise_range = surprise_max - surprise_min
        local_surprise_4 = local_mean(wide_surprise, radius=4)
        local_surprise_12 = local_mean(wide_surprise, radius=12)

        candidate_mask = (error_probability >= candidate_threshold) & mask
        stats["notes"] += int(mask.sum())
        stats["error_notes"] += int((target & mask).sum())
        stats["candidates"] += int(candidate_mask.sum())
        stats["candidate_positives"] += int((candidate_mask & target).sum())
        if not bool(candidate_mask.any()):
            continue

        scalar_features = torch.stack(
            [
                error_probability,
                error_logit.clamp(-12.0, 12.0) / 12.0,
                observed_probability,
                top2_probability[..., 0],
                top2_probability[..., 0] - top2_probability[..., 1],
                entropy,
                pitch_shift,
                surprise_mean.clamp(0.0, 12.0) / 12.0,
                surprise_min.clamp(0.0, 12.0) / 12.0,
                surprise_max.clamp(0.0, 12.0) / 12.0,
                surprise_range.clamp(0.0, 12.0) / 12.0,
                local_surprise_4.clamp(0.0, 12.0) / 12.0,
                local_surprise_12.clamp(0.0, 12.0) / 12.0,
            ],
            dim=-1,
        )
        normalized_scales = stacked_surprise.clamp(0.0, 12.0) / 12.0
        verifier_features = torch.cat(
            [scalar_features, normalized_scales, action_probability, full_features],
            dim=-1,
        )
        feature_rows.append(verifier_features[candidate_mask].cpu())
        label_rows.append(target[candidate_mask].float().cpu())

    if not feature_rows:
        raise RuntimeError(f"No candidates collected for {desc}.")
    stats["candidate_precision"] = stats["candidate_positives"] / max(stats["candidates"], 1)
    stats["candidate_recall_ceiling"] = stats["candidate_positives"] / max(stats["error_notes"], 1)
    return torch.cat(feature_rows), torch.cat(label_rows), stats


def standardize(
    train_features: torch.Tensor,
    *other_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0, unbiased=False).clamp_min(1e-5)
    train_standardized = (train_features - mean) / std
    others = [(features - mean) / std for features in other_features]
    return train_standardized, torch.stack([mean, std]), others


def train_verifier(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> CandidateVerifier:
    torch.manual_seed(args.seed)
    verifier = CandidateVerifier(train_features.shape[1], args.hidden_size, args.dropout).to(device)
    positives = float(train_labels.sum())
    negatives = float(len(train_labels) - positives)
    pos_weight = torch.tensor(negatives / max(positives, 1.0), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(verifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)

    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(train_labels), generator=generator)
        total_loss = 0.0
        verifier.train()
        for start in range(0, len(order), 4096):
            indices = order[start : start + 4096]
            features = train_features[indices].to(device)
            labels = train_labels[indices].to(device)
            loss = criterion(verifier(features), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(indices)
        print(f"verifier epoch={epoch}/{args.epochs} loss={total_loss / len(train_labels):.6f}", flush=True)
    return verifier


@torch.no_grad()
def predict(verifier: CandidateVerifier, features: torch.Tensor, device: torch.device) -> torch.Tensor:
    verifier.eval()
    scores = []
    for start in range(0, len(features), 8192):
        scores.append(torch.sigmoid(verifier(features[start : start + 8192].to(device))).cpu())
    return torch.cat(scores)


def metric_rows(scores: torch.Tensor, labels: torch.Tensor, total_errors: int) -> list[dict]:
    rows = []
    target = labels.bool()
    for step in range(1, 100):
        threshold = step / 100.0
        prediction = scores >= threshold
        tp = int((prediction & target).sum())
        fp = int((prediction & ~target).sum())
        fn = total_errors - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(total_errors, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        rows.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f0_5": 1.25 * precision * recall / max(0.25 * precision + recall, 1e-12),
            }
        )
    return rows


def select_operating_point(rows: list[dict], target_precision: float) -> dict:
    feasible = [row for row in rows if row["precision"] >= target_precision]
    if feasible:
        return max(feasible, key=lambda row: (row["recall"], row["precision"]))
    return max(rows, key=lambda row: (row["precision"], row["recall"]))


def write_markdown(path: Path, result: dict) -> None:
    selected = result["selected_test"]
    baseline = result["test_stage1"]
    lines = [
        "# Candidate Verifier Probe",
        "",
        f"- base checkpoint: `{result['base_checkpoint']}`",
        f"- Stage 1 threshold: `{result['candidate_threshold']}`",
        f"- verifier threshold selected on calibration files: `{result['selected_threshold']:.2f}`",
        f"- test candidate recall ceiling: `{result['test_stats']['candidate_recall_ceiling']:.4f}`",
        "",
        "| System | Precision | Recall | F1 | TP | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Stage 1 candidates | {baseline['precision']:.4f} | {baseline['recall']:.4f} | "
        f"{baseline['f1']:.4f} | {baseline['tp']} | {baseline['fp']} | {baseline['fn']} |",
        f"| Stage 1 + verifier | {selected['precision']:.4f} | {selected['recall']:.4f} | "
        f"{selected['f1']:.4f} | {selected['tp']} | {selected['fp']} | {selected['fn']} |",
        "",
        "## Test Threshold Sweep",
        "",
        "| Threshold | Precision | Recall | F1 | F0.5 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["test_thresholds"]:
        if row["threshold"] in {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, result["selected_threshold"]}:
            lines.append(
                f"| {row['threshold']:.2f} | {row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['f1']:.4f} | {row['f0_5']:.4f} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    scales = sorted(set(args.scales))
    if args.window_size not in scales:
        scales.append(args.window_size)
        scales.sort()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} scales={scales}", flush=True)
    base_model, model_args = load_base_model(args.checkpoint, device)

    validation = make_dataset(args, "validation", args.max_validation_files)
    train_indices, calibration_indices, train_files, calibration_files = split_indices_by_file(
        validation,
        args.calibration_file_fraction,
        args.seed,
    )
    test = make_dataset(args, "test", args.max_test_files)

    train_x, train_y, train_stats = collect_candidates(
        base_model,
        model_args,
        make_loader(validation, train_indices, args.batch_size),
        device,
        scales,
        args.candidate_threshold,
        args.surprise_eval_groups,
        "collect verifier train",
    )
    calibration_x, calibration_y, calibration_stats = collect_candidates(
        base_model,
        model_args,
        make_loader(validation, calibration_indices, args.batch_size),
        device,
        scales,
        args.candidate_threshold,
        args.surprise_eval_groups,
        "collect calibration",
    )
    test_x, test_y, test_stats = collect_candidates(
        base_model,
        model_args,
        make_loader(test, list(range(len(test))), args.batch_size),
        device,
        scales,
        args.candidate_threshold,
        args.surprise_eval_groups,
        "collect test",
    )
    print(f"train stats={train_stats}", flush=True)
    print(f"calibration stats={calibration_stats}", flush=True)
    print(f"test stats={test_stats}", flush=True)

    train_x, normalization, standardized = standardize(train_x, calibration_x, test_x)
    calibration_x, test_x = standardized
    verifier = train_verifier(train_x, train_y, args, device)
    calibration_rows = metric_rows(
        predict(verifier, calibration_x, device),
        calibration_y,
        calibration_stats["error_notes"],
    )
    selected_calibration = select_operating_point(calibration_rows, args.target_precision)
    selected_threshold = selected_calibration["threshold"]
    test_rows = metric_rows(predict(verifier, test_x, device), test_y, test_stats["error_notes"])
    selected_test = next(row for row in test_rows if row["threshold"] == selected_threshold)
    stage1_precision = test_stats["candidate_precision"]
    stage1_recall = test_stats["candidate_recall_ceiling"]
    test_stage1 = {
        "tp": test_stats["candidate_positives"],
        "fp": test_stats["candidates"] - test_stats["candidate_positives"],
        "fn": test_stats["error_notes"] - test_stats["candidate_positives"],
        "precision": stage1_precision,
        "recall": stage1_recall,
        "f1": 2.0 * stage1_precision * stage1_recall / max(stage1_precision + stage1_recall, 1e-12),
    }
    result = {
        "base_checkpoint": args.checkpoint,
        "candidate_threshold": args.candidate_threshold,
        "target_precision": args.target_precision,
        "scales": scales,
        "train_file_ids": train_files,
        "calibration_file_ids": calibration_files,
        "train_stats": train_stats,
        "calibration_stats": calibration_stats,
        "test_stats": test_stats,
        "selected_threshold": selected_threshold,
        "selected_calibration": selected_calibration,
        "selected_test": selected_test,
        "test_stage1": test_stage1,
        "calibration_thresholds": calibration_rows,
        "test_thresholds": test_rows,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": verifier.state_dict(),
            "normalization": normalization,
            "input_size": train_x.shape[1],
            "args": vars(args),
            "result": result,
        },
        output,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), result)
    print(json.dumps(result, indent=2), flush=True)
    print(f"saved {output}", flush=True)
    print(f"wrote {output_json}", flush=True)
    print(f"wrote {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
