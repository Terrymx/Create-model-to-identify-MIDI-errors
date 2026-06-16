from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evaluate_directional_likelihood_gate import load_model
from run_directional_fusion_probe import directional_evidence
from run_union_candidate_verifier import (
    CandidateVerifier,
    detector_signals,
    make_dataset,
    metric_rows,
    select_operating_point,
    split_indices_by_file,
)


class IndexedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        sample = self.dataset[index]
        file_id, _ = self.dataset.index[index]
        sample["file_id"] = torch.tensor(file_id, dtype=torch.long)
        return sample


def make_indexed_loader(dataset, indices: list[int], batch_size: int) -> DataLoader:
    return DataLoader(
        IndexedSubset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Piece-level calibration for union candidate verifier.")
    parser.add_argument("--verifier-checkpoint", default="checkpoints/union_candidate_verifier.pt")
    parser.add_argument("--output-json", default="training_logs/union_candidate_piece_calibration.json")
    parser.add_argument("--output-md", default="training_logs/union_candidate_piece_calibration.md")
    return parser.parse_args()


def filter_scores_by_strategy(
    scores: torch.Tensor,
    labels: torch.Tensor,
    file_ids: torch.Tensor,
    *,
    threshold: float,
    top_k: int | None,
) -> tuple[int, int, int]:
    selected = scores >= threshold
    if top_k is not None:
        topk_selected = torch.zeros_like(selected)
        for file_id in torch.unique(file_ids):
            piece_indices = torch.nonzero(file_ids == file_id, as_tuple=False).flatten()
            piece_scores = scores[piece_indices]
            eligible = piece_indices[piece_scores >= threshold]
            if len(eligible) == 0:
                continue
            keep_count = min(top_k, len(eligible))
            keep_order = torch.argsort(scores[eligible], descending=True)[:keep_count]
            topk_selected[eligible[keep_order]] = True
        selected = topk_selected
    targets = labels.bool()
    tp = int((selected & targets).sum())
    fp = int((selected & ~targets).sum())
    return tp, fp, int(selected.sum())


def metric_from_counts(tp: int, fp: int, total_errors: int) -> dict:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_errors, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": total_errors - tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def strategy_rows(
    scores: torch.Tensor,
    labels: torch.Tensor,
    file_ids: torch.Tensor,
    total_errors: int,
    *,
    top_k: int | None,
) -> list[dict]:
    rows = []
    for step in range(1, 100):
        threshold = step / 100.0
        tp, fp, selected = filter_scores_by_strategy(
            scores,
            labels,
            file_ids,
            threshold=threshold,
            top_k=top_k,
        )
        row = metric_from_counts(tp, fp, total_errors)
        row.update({"threshold": threshold, "top_k": top_k, "selected": selected})
        rows.append(row)
    return rows


def select_with_precision_margin(rows: list[dict], target_precision: float, margin: float) -> dict:
    target = target_precision + margin
    feasible = [row for row in rows if row["precision"] >= target]
    if feasible:
        return max(feasible, key=lambda row: (row["recall"], row["precision"]))
    return max(rows, key=lambda row: (row["precision"], row["recall"]))


def collect_with_file_ids(
    loader: DataLoader,
    *,
    ckpt_args: dict,
    verifier: CandidateVerifier,
    normalization: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    three_model, three_args = load_model(
        ckpt_args["threeclass_checkpoint"],
        device,
        require_explicit_surprise=True,
    )
    binary_model, binary_args = load_model(
        ckpt_args["binary_checkpoint"],
        device,
        require_explicit_surprise=True,
    )
    forward_model, forward_args = load_model(
        ckpt_args["forward_checkpoint"],
        device,
        require_explicit_surprise=False,
    )
    backward_model, backward_args = load_model(
        ckpt_args["backward_checkpoint"],
        device,
        require_explicit_surprise=False,
    )
    score_rows = []
    label_rows = []
    file_rows = []
    stats = {
        "notes": 0,
        "error_notes": 0,
        "candidates": 0,
        "candidate_positives": 0,
        "threeclass_candidates": 0,
        "binary_candidates": 0,
        "overlap_candidates": 0,
    }
    mean, std = normalization
    verifier.eval()
    for batch in tqdm(loader, desc="collect piece-calibration candidates", unit="batch", dynamic_ncols=True):
        batch_file_ids = batch.pop("file_id").to(device)
        raw_features = batch["features"].to(device)
        mask = batch["mask"].to(device).bool()
        labels = batch["is_error"].to(device).bool()
        three = detector_signals(
            three_model,
            three_args,
            raw_features,
            mask,
            ckpt_args.get("surprise_eval_groups", 4),
        )
        binary = detector_signals(
            binary_model,
            binary_args,
            raw_features,
            mask,
            ckpt_args.get("surprise_eval_groups", 4),
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
        three_candidate = three["probability"] >= ckpt_args["threeclass_candidate_threshold"]
        binary_candidate = binary["probability"] >= ckpt_args["binary_candidate_threshold"]
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
        )[candidate_mask].cpu()
        normalized = (verifier_features - mean) / std
        with torch.no_grad():
            scores = torch.sigmoid(verifier(normalized.to(device))).cpu()
        expanded_file_ids = batch_file_ids[:, None].expand_as(mask)[candidate_mask].cpu()
        score_rows.append(scores)
        label_rows.append(labels[candidate_mask].float().cpu())
        file_rows.append(expanded_file_ids)
    if not score_rows:
        raise RuntimeError("No candidates collected for calibration")
    stats["candidate_precision"] = stats["candidate_positives"] / max(stats["candidates"], 1)
    stats["candidate_recall_ceiling"] = stats["candidate_positives"] / max(stats["error_notes"], 1)
    return (
        torch.cat(score_rows),
        torch.cat(label_rows),
        torch.cat(file_rows),
        stats,
    )


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Union Candidate Piece Calibration",
        "",
        f"- target precision: `{result['target_precision']}`",
        f"- calibration total errors: `{result['calibration_stats']['error_notes']}`",
        f"- test total errors: `{result['test_stats']['error_notes']}`",
        "",
        "| Strategy | Cal threshold | Cal P | Cal R | Test P | Test R | Test F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in result["strategies"].items():
        cal = row["calibration"]
        test = row["test"]
        threshold = cal["threshold"]
        if cal.get("top_k") is not None:
            strategy_name = f"{name} top_k={cal['top_k']}"
        else:
            strategy_name = name
        lines.append(
            f"| {strategy_name} | {threshold:.2f} | {cal['precision']:.4f} | "
            f"{cal['recall']:.4f} | {test['precision']:.4f} | "
            f"{test['recall']:.4f} | {test['f1']:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.verifier_checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint["args"]
    verifier = CandidateVerifier(
        checkpoint["input_size"],
        int(ckpt_args.get("hidden_size", 128)),
        float(ckpt_args.get("dropout", 0.15)),
    ).to(device)
    verifier.load_state_dict(checkpoint["model_state_dict"])
    normalization = checkpoint["normalization"].cpu()

    data_args = argparse.Namespace(
        data_root=ckpt_args["data_root"],
        window_size=int(ckpt_args.get("window_size", 256)),
        stride=int(ckpt_args.get("stride", 128)),
        error_rate=float(ckpt_args.get("error_rate", 0.01)),
        max_validation_files=ckpt_args.get("max_validation_files"),
        max_test_files=ckpt_args.get("max_test_files"),
        calibration_file_fraction=float(ckpt_args.get("calibration_file_fraction", 0.25)),
        seed=int(ckpt_args.get("seed", 17)),
        batch_size=int(ckpt_args.get("batch_size", 8)),
    )
    validation = make_dataset(data_args, "validation", data_args.max_validation_files)
    _, calibration_indices, _, calibration_files = split_indices_by_file(
        validation,
        data_args.calibration_file_fraction,
        data_args.seed,
    )
    test = make_dataset(data_args, "test", data_args.max_test_files)
    calibration_scores, calibration_labels, calibration_files_per_candidate, calibration_stats = collect_with_file_ids(
        make_indexed_loader(validation, calibration_indices, data_args.batch_size),
        ckpt_args=ckpt_args,
        verifier=verifier,
        normalization=normalization,
        device=device,
    )
    test_scores, test_labels, test_files_per_candidate, test_stats = collect_with_file_ids(
        make_indexed_loader(test, list(range(len(test))), data_args.batch_size),
        ckpt_args=ckpt_args,
        verifier=verifier,
        normalization=normalization,
        device=device,
    )

    strategies = {}
    global_rows = metric_rows(calibration_scores, calibration_labels, calibration_stats["error_notes"])
    test_global_rows = metric_rows(test_scores, test_labels, test_stats["error_notes"])
    for margin in (0.0, 0.02, 0.05, 0.08, 0.10, 0.12):
        selected = select_with_precision_margin(global_rows, ckpt_args["target_precision"], margin)
        test_row = next(row for row in test_global_rows if row["threshold"] == selected["threshold"])
        strategies[f"global_margin_{margin:.2f}"] = {
            "calibration": selected,
            "test": test_row,
        }

    for top_k in (5, 10, 15, 20, 30):
        cal_rows = strategy_rows(
            calibration_scores,
            calibration_labels,
            calibration_files_per_candidate,
            calibration_stats["error_notes"],
            top_k=top_k,
        )
        test_rows = strategy_rows(
            test_scores,
            test_labels,
            test_files_per_candidate,
            test_stats["error_notes"],
            top_k=top_k,
        )
        selected = select_operating_point(cal_rows, ckpt_args["target_precision"])
        test_row = next(row for row in test_rows if row["threshold"] == selected["threshold"])
        strategies[f"topk_{top_k}"] = {
            "calibration": selected,
            "test": test_row,
        }

    result = {
        "verifier_checkpoint": args.verifier_checkpoint,
        "target_precision": ckpt_args["target_precision"],
        "calibration_files": calibration_files,
        "calibration_stats": calibration_stats,
        "test_stats": test_stats,
        "strategies": strategies,
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
