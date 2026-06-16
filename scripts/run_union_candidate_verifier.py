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
from run_directional_fusion_probe import directional_evidence


class CandidateVerifier(nn.Module):
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
    parser = argparse.ArgumentParser(description="Union-candidate cascade verifier.")
    parser.add_argument("--threeclass-checkpoint", required=True)
    parser.add_argument("--binary-checkpoint", required=True)
    parser.add_argument("--forward-checkpoint", required=True)
    parser.add_argument("--backward-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="checkpoints/union_candidate_verifier.pt")
    parser.add_argument("--output-json", default="training_logs/union_candidate_verifier.json")
    parser.add_argument("--output-md", default="training_logs/union_candidate_verifier.md")
    parser.add_argument("--threeclass-candidate-threshold", type=float, default=0.45)
    parser.add_argument("--binary-candidate-threshold", type=float, default=0.45)
    parser.add_argument("--target-precision", type=float, default=0.80)
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--calibration-file-fraction", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--surprise-eval-groups", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-validation-files", type=int, default=None)
    parser.add_argument("--max-test-files", type=int, default=None)
    return parser.parse_args()


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
def detector_signals(
    model: nn.Module,
    model_args,
    raw_features: torch.Tensor,
    mask: torch.Tensor,
    groups: int,
) -> dict[str, torch.Tensor]:
    features = adapt_features(raw_features, model_args.input_size)
    surprise, available = build_explicit_surprise(
        model,
        features,
        mask.float(),
        training=False,
        train_mask_rate=0.0,
        eval_groups=groups,
    )
    outputs = model(features, surprise=surprise, surprise_available=available)
    logit = outputs["error_logits"]
    probability = torch.sigmoid(logit)
    observed_pitch = torch.round(raw_features[:, :, 0] * 127.0).long().clamp(0, 127)
    encoded_features: list[torch.Tensor] = [
        probability,
        logit.clamp(-12.0, 12.0) / 12.0,
        surprise.clamp(0.0, 12.0) / 12.0,
        available.float(),
    ]
    if outputs.get("kind_logits") is not None:
        kind_prob = torch.softmax(outputs["kind_logits"], dim=-1)
        pitch_prob = torch.softmax(outputs["pitch_logits"], dim=-1)
        observed_prob = pitch_prob.gather(-1, observed_pitch.unsqueeze(-1)).squeeze(-1)
        top_prob = pitch_prob.max(dim=-1).values
        entropy = -(pitch_prob * pitch_prob.clamp_min(1e-9).log()).sum(dim=-1) / math.log(128.0)
        encoded_features.extend(
            [
                kind_prob[..., 0],
                kind_prob[..., 1],
                kind_prob[..., 2],
                observed_prob,
                top_prob,
                entropy,
            ]
        )
    else:
        zeros = torch.zeros_like(probability)
        encoded_features.extend([zeros, zeros, zeros, zeros, zeros, zeros])
    if outputs.get("correction_logits") is not None:
        correction_prob = torch.softmax(outputs["correction_logits"], dim=-1)
        observed_prob = correction_prob.gather(-1, observed_pitch.unsqueeze(-1)).squeeze(-1)
        null_prob = correction_prob[..., 128]
        top_prob = correction_prob.max(dim=-1).values
        entropy = -(correction_prob * correction_prob.clamp_min(1e-9).log()).sum(dim=-1) / math.log(129.0)
        encoded_features.extend([observed_prob, null_prob, top_prob, entropy])
    else:
        zeros = torch.zeros_like(probability)
        encoded_features.extend([zeros, zeros, zeros, zeros])
    return {
        "features": torch.stack(encoded_features, dim=-1),
        "probability": probability,
        "available": available.bool(),
    }


@torch.no_grad()
def collect_candidates(
    three_model: nn.Module,
    three_args,
    binary_model: nn.Module,
    binary_args,
    forward_model: nn.Module,
    forward_args,
    backward_model: nn.Module,
    backward_args,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    description: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict]:
    feature_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    score_rows = {"threeclass": [], "binary": [], "max": []}
    stats = {
        "notes": 0,
        "error_notes": 0,
        "candidates": 0,
        "candidate_positives": 0,
        "threeclass_candidates": 0,
        "binary_candidates": 0,
        "overlap_candidates": 0,
    }
    for batch in tqdm(loader, desc=description, unit="batch", dynamic_ncols=True):
        raw_features = batch["features"].to(device)
        mask = batch["mask"].to(device).bool()
        labels = batch["is_error"].to(device).bool()
        three = detector_signals(
            three_model,
            three_args,
            raw_features,
            mask,
            args.surprise_eval_groups,
        )
        binary = detector_signals(
            binary_model,
            binary_args,
            raw_features,
            mask,
            args.surprise_eval_groups,
        )
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
        valid = mask & three["available"] & binary["available"] & forward_available & backward_available
        three_candidate = three["probability"] >= args.threeclass_candidate_threshold
        binary_candidate = binary["probability"] >= args.binary_candidate_threshold
        candidate_mask = valid & (three_candidate | binary_candidate)
        stats["notes"] += int(valid.sum())
        stats["error_notes"] += int((labels & valid).sum())
        stats["candidates"] += int(candidate_mask.sum())
        stats["candidate_positives"] += int((candidate_mask & labels).sum())
        stats["threeclass_candidates"] += int((candidate_mask & three_candidate).sum())
        stats["binary_candidates"] += int((candidate_mask & binary_candidate).sum())
        stats["overlap_candidates"] += int((candidate_mask & three_candidate & binary_candidate).sum())
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
        cross = torch.stack(
            [
                three["probability"],
                binary["probability"],
                torch.maximum(three["probability"], binary["probability"]),
                torch.minimum(three["probability"], binary["probability"]),
                (binary["probability"] - three["probability"]).clamp(-1.0, 1.0),
                three_candidate.float(),
                binary_candidate.float(),
                (three_candidate & binary_candidate).float(),
            ],
            dim=-1,
        )
        verifier_features = torch.cat(
            [
                raw_features,
                three["features"],
                binary["features"],
                cross,
                forward_evidence,
                backward_evidence,
                aggregate,
            ],
            dim=-1,
        )
        feature_rows.append(verifier_features[candidate_mask].cpu())
        label_rows.append(labels[candidate_mask].float().cpu())
        score_rows["threeclass"].append(three["probability"][candidate_mask].cpu())
        score_rows["binary"].append(binary["probability"][candidate_mask].cpu())
        score_rows["max"].append(torch.maximum(three["probability"], binary["probability"])[candidate_mask].cpu())

    if not feature_rows:
        raise RuntimeError(f"No candidates collected for {description}")
    stats["candidate_precision"] = stats["candidate_positives"] / max(stats["candidates"], 1)
    stats["candidate_recall_ceiling"] = stats["candidate_positives"] / max(stats["error_notes"], 1)
    return (
        torch.cat(feature_rows),
        torch.cat(label_rows),
        {name: torch.cat(parts) for name, parts in score_rows.items()},
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


@torch.no_grad()
def predict(model: CandidateVerifier, features: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    return torch.cat(
        [
            torch.sigmoid(model(features[start : start + 8192].to(device))).cpu()
            for start in range(0, len(features), 8192)
        ]
    )


def train_verifier(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    calibration_features: torch.Tensor,
    calibration_labels: torch.Tensor,
    calibration_total_errors: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[CandidateVerifier, int, dict]:
    torch.manual_seed(args.seed)
    model = CandidateVerifier(train_features.shape[1], args.hidden_size, args.dropout).to(device)
    positives = float(train_labels.sum())
    negatives = float(len(train_labels) - positives)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / max(positives, 1.0), device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    best_epoch = 0
    best_row: dict | None = None
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(train_labels), generator=generator)
        total_loss = 0.0
        model.train()
        for start in range(0, len(order), 4096):
            indices = order[start : start + 4096]
            features = train_features[indices].to(device)
            labels = train_labels[indices].to(device)
            loss = criterion(model(features), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        calibration_rows = metric_rows(
            predict(model, calibration_features, device),
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
                for name, value in model.state_dict().items()
            }
        print(
            f"verifier epoch={epoch}/{args.epochs} loss={total_loss / len(train_labels):.6f} "
            f"calibration_threshold={selected['threshold']:.2f} "
            f"calibration_precision={selected['precision']:.4f} "
            f"calibration_recall={selected['recall']:.4f} best_epoch={best_epoch}",
            flush=True,
        )
    if best_state is None or best_row is None:
        raise RuntimeError("No verifier checkpoint was selected")
    model.load_state_dict(best_state)
    return model, best_epoch, best_row


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Union Candidate Verifier",
        "",
        f"- threeclass candidate threshold: `{result['threeclass_candidate_threshold']}`",
        f"- binary candidate threshold: `{result['binary_candidate_threshold']}`",
        f"- target precision: `{result['target_precision']}`",
        f"- verifier epoch selected on calibration files: `{result['verifier_best_epoch']}`",
        f"- test candidate recall ceiling: `{result['test_stats']['candidate_recall_ceiling']:.4f}`",
        f"- test candidate precision: `{result['test_stats']['candidate_precision']:.4f}`",
        "",
        "| System | Selected on | Threshold | Precision | Recall | F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in result["test_frontiers"].items():
        lines.append(
            f"| {name} | test frontier | {row['threshold']:.2f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
        )
    lines.append(
        f"| verifier | calibration | {result['verifier_selected_threshold']:.2f} | "
        f"{result['verifier_selected_test']['precision']:.4f} | "
        f"{result['verifier_selected_test']['recall']:.4f} | "
        f"{result['verifier_selected_test']['f1']:.4f} |"
    )
    lines.append(
        f"| verifier | test frontier | {result['verifier_test_frontier']['threshold']:.2f} | "
        f"{result['verifier_test_frontier']['precision']:.4f} | "
        f"{result['verifier_test_frontier']['recall']:.4f} | "
        f"{result['verifier_test_frontier']['f1']:.4f} |"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    three_model, three_args = load_model(args.threeclass_checkpoint, device, require_explicit_surprise=True)
    binary_model, binary_args = load_model(args.binary_checkpoint, device, require_explicit_surprise=True)
    forward_model, forward_args = load_model(args.forward_checkpoint, device, require_explicit_surprise=False)
    backward_model, backward_args = load_model(args.backward_checkpoint, device, require_explicit_surprise=False)
    validation = make_dataset(args, "validation", args.max_validation_files)
    train_indices, calibration_indices, train_files, calibration_files = split_indices_by_file(
        validation,
        args.calibration_file_fraction,
        args.seed,
    )
    test = make_dataset(args, "test", args.max_test_files)
    collection_args = (
        three_model,
        three_args,
        binary_model,
        binary_args,
        forward_model,
        forward_args,
        backward_model,
        backward_args,
    )
    train_x, train_y, train_scores, train_stats = collect_candidates(
        *collection_args,
        make_loader(validation, train_indices, args.batch_size),
        device,
        args,
        "collect verifier train",
    )
    calibration_x, calibration_y, calibration_scores, calibration_stats = collect_candidates(
        *collection_args,
        make_loader(validation, calibration_indices, args.batch_size),
        device,
        args,
        "collect verifier calibration",
    )
    test_x, test_y, test_scores, test_stats = collect_candidates(
        *collection_args,
        make_loader(test, list(range(len(test))), args.batch_size),
        device,
        args,
        "collect verifier test",
    )
    print(f"train stats={train_stats}", flush=True)
    print(f"calibration stats={calibration_stats}", flush=True)
    print(f"test stats={test_stats}", flush=True)
    train_x, normalization, standardized = standardize(train_x, calibration_x, test_x)
    calibration_x, test_x = standardized
    verifier, best_epoch, selected_calibration = train_verifier(
        train_x,
        train_y,
        calibration_x,
        calibration_y,
        calibration_stats["error_notes"],
        args,
        device,
    )
    test_frontiers = {}
    for name in ("threeclass", "binary", "max"):
        rows = metric_rows(test_scores[name], test_y, test_stats["error_notes"])
        test_frontiers[name] = select_operating_point(rows, args.target_precision)
    calibration_verifier = predict(verifier, calibration_x, device)
    test_verifier = predict(verifier, test_x, device)
    calibration_rows = metric_rows(calibration_verifier, calibration_y, calibration_stats["error_notes"])
    test_rows = metric_rows(test_verifier, test_y, test_stats["error_notes"])
    selected_threshold = selected_calibration["threshold"]
    selected_test = next(row for row in test_rows if row["threshold"] == selected_threshold)
    test_frontier = select_operating_point(test_rows, args.target_precision)
    result = {
        "threeclass_checkpoint": args.threeclass_checkpoint,
        "binary_checkpoint": args.binary_checkpoint,
        "forward_checkpoint": args.forward_checkpoint,
        "backward_checkpoint": args.backward_checkpoint,
        "threeclass_candidate_threshold": args.threeclass_candidate_threshold,
        "binary_candidate_threshold": args.binary_candidate_threshold,
        "target_precision": args.target_precision,
        "train_file_ids": train_files,
        "calibration_file_ids": calibration_files,
        "train_stats": train_stats,
        "calibration_stats": calibration_stats,
        "test_stats": test_stats,
        "test_frontiers": test_frontiers,
        "verifier_best_epoch": best_epoch,
        "verifier_selected_threshold": selected_threshold,
        "verifier_selected_calibration": selected_calibration,
        "verifier_selected_test": selected_test,
        "verifier_test_frontier": test_frontier,
        "verifier_calibration_thresholds": calibration_rows,
        "verifier_test_thresholds": test_rows,
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
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
