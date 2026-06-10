from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from evaluate_directional_likelihood_gate import adapt_features, load_model
from midi_error_detector.data import MaestroWrongNoteDataset
from midi_error_detector.train import build_explicit_surprise


class DirectionalFusionHead(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen Step 2 directional evidence fusion probe.")
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--forward-checkpoint", required=True)
    parser.add_argument("--backward-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="checkpoints/directional_fusion_probe.pt")
    parser.add_argument("--output-json", default="training_logs/directional_fusion_probe.json")
    parser.add_argument("--output-md", default="training_logs/directional_fusion_probe.md")
    parser.add_argument("--candidate-threshold", type=float, default=0.45)
    parser.add_argument("--target-precision", type=float, default=0.80)
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--calibration-file-fraction", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--surprise-eval-groups", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-validation-files", type=int, default=None)
    parser.add_argument("--max-test-files", type=int, default=None)
    return parser.parse_args()


def make_dataset(
    args: argparse.Namespace,
    split: str,
    max_files: int | None,
) -> MaestroWrongNoteDataset:
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
        raise ValueError("At least two validation files are required.")
    rng = random.Random(seed)
    rng.shuffle(file_ids)
    calibration_count = min(len(file_ids) - 1, max(1, round(len(file_ids) * calibration_fraction)))
    calibration_files = set(file_ids[:calibration_count])
    train_files = set(file_ids[calibration_count:])
    train_indices = [idx for idx, (file_id, _) in enumerate(dataset.index) if file_id in train_files]
    calibration_indices = [idx for idx, (file_id, _) in enumerate(dataset.index) if file_id in calibration_files]
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
def directional_evidence(
    model: nn.Module,
    raw_features: torch.Tensor,
    mask: torch.Tensor,
    direction: str,
    safe_columns: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    features = raw_features[:, :, safe_columns]
    observed_pitch = torch.round(raw_features[:, :, 0] * 127.0).long().clamp(0, 127)
    oriented = features
    oriented_pitch = observed_pitch
    oriented_mask = mask.bool()
    if direction == "backward":
        oriented = oriented.flip(1)
        oriented_pitch = oriented_pitch.flip(1)
        oriented_mask = oriented_mask.flip(1)
    shifted = torch.zeros_like(oriented)
    shifted[:, 1:] = oriented[:, :-1]
    available = oriented_mask.clone()
    available[:, 0] = False
    probability = torch.softmax(model.predict_pitch(shifted, causal=True), dim=-1)
    observed_probability = probability.gather(-1, oriented_pitch.unsqueeze(-1)).squeeze(-1)
    top_probability, top_pitch = probability.max(dim=-1)
    surprise = -observed_probability.clamp_min(1e-9).log()
    entropy = -(probability * probability.clamp_min(1e-9).log()).sum(dim=-1) / math.log(128.0)
    evidence = torch.stack(
        [
            surprise.clamp(0.0, 12.0) / 12.0,
            observed_probability,
            top_probability,
            (top_probability - observed_probability).clamp(0.0, 1.0),
            entropy,
            (top_pitch != oriented_pitch).float(),
        ],
        dim=-1,
    )
    if direction == "backward":
        evidence = evidence.flip(1)
        surprise = surprise.flip(1)
        available = available.flip(1)
    return evidence, available


@torch.no_grad()
def collect_candidates(
    detector: nn.Module,
    detector_args,
    forward_model: nn.Module,
    forward_args,
    backward_model: nn.Module,
    backward_args,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    description: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    feature_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    base_score_rows: list[torch.Tensor] = []
    stats = {"notes": 0, "error_notes": 0, "candidates": 0, "candidate_positives": 0}
    for batch in tqdm(loader, desc=description, unit="batch", dynamic_ncols=True):
        raw_features = batch["features"].to(device)
        mask = batch["mask"].to(device).bool()
        labels = batch["is_error"].to(device).bool()
        detector_features = adapt_features(raw_features, detector_args.input_size)
        internal_surprise, internal_available = build_explicit_surprise(
            detector,
            detector_features,
            mask.float(),
            training=False,
            train_mask_rate=0.0,
            eval_groups=args.surprise_eval_groups,
        )
        encoded = detector.encode(detector_features)
        surprise_inputs = torch.stack(
            [
                internal_surprise.clamp(0.0, 12.0) / 12.0,
                internal_available.float(),
            ],
            dim=-1,
        )
        internal_embedding = detector.surprise_projection(surprise_inputs)
        base_outputs = detector(
            detector_features,
            surprise=internal_surprise,
            surprise_available=internal_available,
        )
        base_logit = base_outputs["error_logits"]
        base_probability = torch.sigmoid(base_logit)
        forward_evidence, forward_available = directional_evidence(
            forward_model,
            raw_features,
            mask,
            "forward",
            forward_args.safe_feature_columns,
        )
        backward_evidence, backward_available = directional_evidence(
            backward_model,
            raw_features,
            mask,
            "backward",
            backward_args.safe_feature_columns,
        )
        both_available = mask & forward_available & backward_available
        candidate_mask = both_available & (base_probability >= args.candidate_threshold)
        stats["notes"] += int(both_available.sum())
        stats["error_notes"] += int((labels & both_available).sum())
        stats["candidates"] += int(candidate_mask.sum())
        stats["candidate_positives"] += int((candidate_mask & labels).sum())
        if not bool(candidate_mask.any()):
            continue

        forward_surprise = forward_evidence[..., 0]
        backward_surprise = backward_evidence[..., 0]
        aggregate = torch.stack(
            [
                0.5 * (forward_surprise + backward_surprise),
                torch.minimum(forward_surprise, backward_surprise),
                torch.maximum(forward_surprise, backward_surprise),
            ],
            dim=-1,
        )
        scalar_base = torch.stack(
            [
                base_probability,
                base_logit.clamp(-12.0, 12.0) / 12.0,
                internal_surprise.clamp(0.0, 12.0) / 12.0,
            ],
            dim=-1,
        )
        fusion_features = torch.cat(
            [
                encoded,
                internal_embedding,
                scalar_base,
                forward_evidence,
                backward_evidence,
                aggregate,
            ],
            dim=-1,
        )
        feature_rows.append(fusion_features[candidate_mask].cpu())
        label_rows.append(labels[candidate_mask].float().cpu())
        base_score_rows.append(base_probability[candidate_mask].cpu())

    if not feature_rows:
        raise RuntimeError(f"No candidates collected for {description}")
    stats["candidate_precision"] = stats["candidate_positives"] / max(stats["candidates"], 1)
    stats["candidate_recall_ceiling"] = stats["candidate_positives"] / max(stats["error_notes"], 1)
    return (
        torch.cat(feature_rows),
        torch.cat(label_rows),
        torch.cat(base_score_rows),
        stats,
    )


def standardize(
    train_features: torch.Tensor,
    *others: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0, unbiased=False).clamp_min(1e-5)
    return (
        (train_features - mean) / std,
        torch.stack([mean, std]),
        [(features - mean) / std for features in others],
    )


def train_head(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    calibration_features: torch.Tensor,
    calibration_labels: torch.Tensor,
    calibration_total_errors: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DirectionalFusionHead, int, dict]:
    torch.manual_seed(args.seed)
    head = DirectionalFusionHead(train_features.shape[1], args.hidden_size, args.dropout).to(device)
    positives = float(train_labels.sum())
    negatives = float(len(train_labels) - positives)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / max(positives, 1.0), device=device)
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    best_epoch = 0
    best_row: dict | None = None
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(train_labels), generator=generator)
        total_loss = 0.0
        head.train()
        for start in range(0, len(order), 4096):
            indices = order[start : start + 4096]
            features = train_features[indices].to(device)
            labels = train_labels[indices].to(device)
            loss = criterion(head(features), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(indices)
        calibration_rows = metric_rows(
            predict(head, calibration_features, device),
            calibration_labels,
            calibration_total_errors,
        )
        selected = select_operating_point(calibration_rows, args.target_precision)
        feasible = selected["precision"] >= args.target_precision
        selection_key = (
            1 if feasible else 0,
            selected["recall"] if feasible else selected["precision"],
            selected["precision"] if feasible else selected["recall"],
        )
        if best_row is None:
            improved = True
        else:
            best_feasible = best_row["precision"] >= args.target_precision
            best_key = (
                1 if best_feasible else 0,
                best_row["recall"] if best_feasible else best_row["precision"],
                best_row["precision"] if best_feasible else best_row["recall"],
            )
            improved = selection_key > best_key
        if improved:
            best_epoch = epoch
            best_row = selected
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in head.state_dict().items()
            }
        print(
            f"fusion epoch={epoch}/{args.epochs} loss={total_loss / len(train_labels):.6f} "
            f"calibration_threshold={selected['threshold']:.2f} "
            f"calibration_precision={selected['precision']:.4f} "
            f"calibration_recall={selected['recall']:.4f} best_epoch={best_epoch}",
            flush=True,
        )
    if best_state is None or best_row is None:
        raise RuntimeError("No fusion checkpoint was selected")
    head.load_state_dict(best_state)
    return head, best_epoch, best_row


@torch.no_grad()
def predict(head: DirectionalFusionHead, features: torch.Tensor, device: torch.device) -> torch.Tensor:
    head.eval()
    return torch.cat(
        [
            torch.sigmoid(head(features[start : start + 8192].to(device))).cpu()
            for start in range(0, len(features), 8192)
        ]
    )


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
            }
        )
    return rows


def select_operating_point(rows: list[dict], target_precision: float) -> dict:
    feasible = [row for row in rows if row["precision"] >= target_precision]
    if feasible:
        return max(feasible, key=lambda row: (row["recall"], row["precision"]))
    return max(rows, key=lambda row: (row["precision"], row["recall"]))


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Directional Fusion Probe",
        "",
        f"- candidate threshold: `{result['candidate_threshold']}`",
        f"- target precision: `{result['target_precision']}`",
        f"- fusion epoch selected on calibration files: `{result['fusion_best_epoch']}`",
        f"- test candidate recall ceiling: `{result['test_stats']['candidate_recall_ceiling']:.4f}`",
        "",
        "| System | Selected on | Threshold | Precision | Recall | F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        f"| Step 2 | calibration | {result['base_selected_threshold']:.2f} | "
        f"{result['base_selected_test']['precision']:.4f} | "
        f"{result['base_selected_test']['recall']:.4f} | {result['base_selected_test']['f1']:.4f} |",
        f"| Directional fusion | calibration | {result['fusion_selected_threshold']:.2f} | "
        f"{result['fusion_selected_test']['precision']:.4f} | "
        f"{result['fusion_selected_test']['recall']:.4f} | {result['fusion_selected_test']['f1']:.4f} |",
        f"| Directional fusion | test diagnostic | {result['fusion_test_frontier']['threshold']:.2f} | "
        f"{result['fusion_test_frontier']['precision']:.4f} | "
        f"{result['fusion_test_frontier']['recall']:.4f} | {result['fusion_test_frontier']['f1']:.4f} |",
        "",
        f"- test-frontier recall gain over historical Step 2 `0.5307`: "
        f"`{result['recall_gain_over_historical_step2']:+.4f}`",
        f"- probe passed +0.03 recall Gate: `{result['probe_passed']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    detector, detector_args = load_model(
        args.detector_checkpoint,
        device,
        require_explicit_surprise=True,
    )
    forward_model, forward_args = load_model(
        args.forward_checkpoint,
        device,
        require_explicit_surprise=False,
    )
    backward_model, backward_args = load_model(
        args.backward_checkpoint,
        device,
        require_explicit_surprise=False,
    )
    validation = make_dataset(args, "validation", args.max_validation_files)
    train_indices, calibration_indices, train_files, calibration_files = split_indices_by_file(
        validation,
        args.calibration_file_fraction,
        args.seed,
    )
    test = make_dataset(args, "test", args.max_test_files)
    collection_args = (
        detector,
        detector_args,
        forward_model,
        forward_args,
        backward_model,
        backward_args,
    )
    train_x, train_y, train_base, train_stats = collect_candidates(
        *collection_args,
        make_loader(validation, train_indices, args.batch_size),
        device,
        args,
        "collect fusion train",
    )
    calibration_x, calibration_y, calibration_base, calibration_stats = collect_candidates(
        *collection_args,
        make_loader(validation, calibration_indices, args.batch_size),
        device,
        args,
        "collect fusion calibration",
    )
    test_x, test_y, test_base, test_stats = collect_candidates(
        *collection_args,
        make_loader(test, list(range(len(test))), args.batch_size),
        device,
        args,
        "collect fusion test",
    )
    print(f"train stats={train_stats}", flush=True)
    print(f"calibration stats={calibration_stats}", flush=True)
    print(f"test stats={test_stats}", flush=True)

    train_x, normalization, standardized = standardize(train_x, calibration_x, test_x)
    calibration_x, test_x = standardized
    head, fusion_best_epoch, fusion_selected_calibration = train_head(
        train_x,
        train_y,
        calibration_x,
        calibration_y,
        calibration_stats["error_notes"],
        args,
        device,
    )
    calibration_fusion = predict(head, calibration_x, device)
    test_fusion = predict(head, test_x, device)
    fusion_calibration_rows = metric_rows(
        calibration_fusion,
        calibration_y,
        calibration_stats["error_notes"],
    )
    fusion_selected_threshold = fusion_selected_calibration["threshold"]
    fusion_test_rows = metric_rows(test_fusion, test_y, test_stats["error_notes"])
    fusion_selected_test = next(
        row for row in fusion_test_rows if row["threshold"] == fusion_selected_threshold
    )
    fusion_test_frontier = select_operating_point(fusion_test_rows, args.target_precision)

    base_calibration_rows = metric_rows(
        calibration_base,
        calibration_y,
        calibration_stats["error_notes"],
    )
    base_selected_calibration = select_operating_point(base_calibration_rows, args.target_precision)
    base_selected_threshold = base_selected_calibration["threshold"]
    base_test_rows = metric_rows(test_base, test_y, test_stats["error_notes"])
    base_selected_test = next(
        row for row in base_test_rows if row["threshold"] == base_selected_threshold
    )
    recall_gain = fusion_test_frontier["recall"] - 0.5307
    result = {
        "detector_checkpoint": args.detector_checkpoint,
        "forward_checkpoint": args.forward_checkpoint,
        "backward_checkpoint": args.backward_checkpoint,
        "candidate_threshold": args.candidate_threshold,
        "target_precision": args.target_precision,
        "train_file_ids": train_files,
        "calibration_file_ids": calibration_files,
        "train_stats": train_stats,
        "calibration_stats": calibration_stats,
        "test_stats": test_stats,
        "base_selected_threshold": base_selected_threshold,
        "base_selected_calibration": base_selected_calibration,
        "base_selected_test": base_selected_test,
        "fusion_selected_threshold": fusion_selected_threshold,
        "fusion_best_epoch": fusion_best_epoch,
        "fusion_selected_calibration": fusion_selected_calibration,
        "fusion_selected_test": fusion_selected_test,
        "fusion_test_frontier": fusion_test_frontier,
        "recall_gain_over_historical_step2": recall_gain,
        "probe_passed": recall_gain >= 0.03,
        "base_calibration_thresholds": base_calibration_rows,
        "base_test_thresholds": base_test_rows,
        "fusion_calibration_thresholds": fusion_calibration_rows,
        "fusion_test_thresholds": fusion_test_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": head.state_dict(),
            "normalization": normalization,
            "input_size": train_x.shape[1],
            "args": vars(args),
            "result": result,
        },
        output,
    )
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
