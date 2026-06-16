from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evaluate_directional_likelihood_gate import load_model
from midi_error_detector.data import MaestroWrongNoteDataset
from run_union_candidate_verifier import (
    detector_signals,
    metric_rows,
    select_operating_point,
    split_indices_by_file,
    standardize,
)
from run_directional_fusion_probe import directional_evidence


class IndexedSubset(Dataset):
    def __init__(self, dataset: MaestroWrongNoteDataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        dataset_idx = self.indices[idx]
        sample = self.dataset[dataset_idx]
        file_id, window_start = self.dataset.index[dataset_idx]
        sample["__file_id"] = torch.tensor(file_id, dtype=torch.long)
        sample["__window_start"] = torch.tensor(window_start, dtype=torch.long)
        sample["__file_note_count"] = torch.tensor(self.dataset._note_counts[file_id], dtype=torch.long)
        return sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Union-candidate verifier with piece-relative and local context features.")
    parser.add_argument("--threeclass-checkpoint", required=True)
    parser.add_argument("--binary-checkpoint", required=True)
    parser.add_argument("--forward-checkpoint", required=True)
    parser.add_argument("--backward-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-json", default="training_logs/union_candidate_context_verifier.json")
    parser.add_argument("--output-md", default="training_logs/union_candidate_context_verifier.md")
    parser.add_argument("--checkpoint-dir", default="checkpoints/union_candidate_context_verifier")
    parser.add_argument("--threeclass-candidate-thresholds", type=float, nargs="+", default=[0.45, 0.40, 0.35])
    parser.add_argument("--binary-candidate-thresholds", type=float, nargs="+", default=[0.45, 0.40, 0.35])
    parser.add_argument("--target-precision", type=float, default=0.80)
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--calibration-file-fraction", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--surprise-eval-groups", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
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


def make_loader(dataset: MaestroWrongNoteDataset, indices: list[int], batch_size: int) -> DataLoader:
    return DataLoader(
        IndexedSubset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _masked_local_stats(values: torch.Tensor, valid: torch.Tensor, radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch, length = values.shape
    means = torch.zeros_like(values)
    maxes = torch.zeros_like(values)
    for idx in range(length):
        left = max(0, idx - radius)
        right = min(length, idx + radius + 1)
        local_valid = valid[:, left:right]
        local_values = values[:, left:right]
        count = local_valid.sum(dim=1).clamp_min(1)
        means[:, idx] = (local_values * local_valid.float()).sum(dim=1) / count
        masked = local_values.masked_fill(~local_valid, -1e9)
        maxes[:, idx] = masked.max(dim=1).values.clamp_min(0.0)
    return means, maxes


def _rank_percentile(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    return ranks / float(len(values) - 1)


def append_piece_relative_features(
    features: torch.Tensor,
    file_ids: torch.Tensor,
    note_positions: torch.Tensor,
    file_note_counts: torch.Tensor,
    score_columns: torch.Tensor,
) -> torch.Tensor:
    x = features.numpy().astype(np.float32)
    files = file_ids.numpy().astype(np.int64)
    positions = note_positions.numpy().astype(np.float32)
    note_counts = file_note_counts.numpy().astype(np.float32)
    scores = score_columns.numpy().astype(np.float32)
    extras = np.zeros((len(x), 2 + scores.shape[1] * 4), dtype=np.float32)
    extras[:, 0] = positions / np.maximum(note_counts, 1.0)
    unique_files = np.unique(files)
    candidate_density_by_file = {file_id: float((files == file_id).sum()) / max(float(note_counts[files == file_id][0]), 1.0) for file_id in unique_files}
    for file_id in unique_files:
        mask = files == file_id
        extras[mask, 1] = candidate_density_by_file[file_id]
        file_scores = scores[mask]
        for col in range(scores.shape[1]):
            values = file_scores[:, col]
            mean = float(values.mean())
            std = float(max(values.std(), 1e-5))
            base = 2 + col * 4
            extras[mask, base] = _rank_percentile(values)
            extras[mask, base + 1] = (values - mean) / std
            extras[mask, base + 2] = float(values.max())
            extras[mask, base + 3] = mean
    return torch.from_numpy(np.concatenate([x, extras], axis=1))


@torch.no_grad()
def collect_context_candidates(
    models: tuple,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    three_threshold: float,
    binary_threshold: float,
    description: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    three_model, three_args, binary_model, binary_args, forward_model, forward_args, backward_model, backward_args = models
    feature_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    file_rows: list[torch.Tensor] = []
    position_rows: list[torch.Tensor] = []
    count_rows: list[torch.Tensor] = []
    score_column_rows: list[torch.Tensor] = []
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
        three = detector_signals(three_model, three_args, raw_features, mask, args.surprise_eval_groups)
        binary = detector_signals(binary_model, binary_args, raw_features, mask, args.surprise_eval_groups)
        forward_evidence, forward_available = directional_evidence(
            forward_model, raw_features, mask, "forward", forward_args.safe_feature_columns
        )
        backward_evidence, backward_available = directional_evidence(
            backward_model, raw_features, mask, "backward", backward_args.safe_feature_columns
        )
        valid = mask & three["available"] & binary["available"] & forward_available & backward_available
        three_candidate = three["probability"] >= three_threshold
        binary_candidate = binary["probability"] >= binary_threshold
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
        avg_surprise = 0.5 * (forward_surprise + backward_surprise)
        max_surprise = torch.maximum(forward_surprise, backward_surprise)
        max_prob = torch.maximum(three["probability"], binary["probability"])
        aggregate = torch.stack(
            [
                avg_surprise,
                torch.minimum(forward_surprise, backward_surprise),
                max_surprise,
            ],
            dim=-1,
        )
        cross = torch.stack(
            [
                three["probability"],
                binary["probability"],
                max_prob,
                torch.minimum(three["probability"], binary["probability"]),
                (binary["probability"] - three["probability"]).clamp(-1.0, 1.0),
                three_candidate.float(),
                binary_candidate.float(),
                (three_candidate & binary_candidate).float(),
            ],
            dim=-1,
        )

        context_sources = [
            three["probability"],
            binary["probability"],
            max_prob,
            avg_surprise,
            max_surprise,
            candidate_mask.float(),
        ]
        local_features = []
        for source in context_sources:
            mean4, max4 = _masked_local_stats(source, valid, 4)
            mean8, max8 = _masked_local_stats(source, valid, 8)
            local_features.extend([mean4, max4, mean8, max8, source - mean8])
        local_stack = torch.stack(local_features, dim=-1)
        verifier_features = torch.cat(
            [
                raw_features,
                three["features"],
                binary["features"],
                cross,
                forward_evidence,
                backward_evidence,
                aggregate,
                local_stack,
            ],
            dim=-1,
        )

        batch_size, length = candidate_mask.shape
        file_ids = batch["__file_id"].view(batch_size, 1).expand(batch_size, length)
        window_starts = batch["__window_start"].view(batch_size, 1).expand(batch_size, length)
        file_note_counts = batch["__file_note_count"].view(batch_size, 1).expand(batch_size, length)
        positions = window_starts + torch.arange(length).view(1, length)
        score_columns = torch.stack(
            [three["probability"], binary["probability"], max_prob, avg_surprise, max_surprise],
            dim=-1,
        )

        candidate_mask_cpu = candidate_mask.cpu()
        feature_rows.append(verifier_features[candidate_mask].cpu())
        label_rows.append(labels[candidate_mask].float().cpu())
        file_rows.append(file_ids[candidate_mask_cpu].cpu())
        position_rows.append(positions[candidate_mask_cpu].cpu())
        count_rows.append(file_note_counts[candidate_mask_cpu].cpu())
        score_column_rows.append(score_columns[candidate_mask].cpu())
        score_rows["threeclass"].append(three["probability"][candidate_mask].cpu())
        score_rows["binary"].append(binary["probability"][candidate_mask].cpu())
        score_rows["max"].append(max_prob[candidate_mask].cpu())

    if not feature_rows:
        raise RuntimeError(f"No candidates collected for {description}")
    stats["candidate_precision"] = stats["candidate_positives"] / max(stats["candidates"], 1)
    stats["candidate_recall_ceiling"] = stats["candidate_positives"] / max(stats["error_notes"], 1)
    return (
        torch.cat(feature_rows),
        torch.cat(label_rows),
        {name: torch.cat(parts) for name, parts in score_rows.items()},
        stats,
        torch.cat(file_rows),
        torch.cat(position_rows),
        torch.cat(count_rows),
        torch.cat(score_column_rows),
    )


def sklearn_probability(model, features: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(model.predict_proba(features)[:, 1].astype(np.float32))


def evaluate_model(model, calibration_x: torch.Tensor, calibration_y: torch.Tensor, calibration_total: int, test_x: torch.Tensor, test_y: torch.Tensor, test_total: int, target_precision: float) -> dict:
    calibration_scores = sklearn_probability(model, calibration_x.numpy())
    test_scores = sklearn_probability(model, test_x.numpy())
    calibration_rows = metric_rows(calibration_scores, calibration_y, calibration_total)
    test_rows = metric_rows(test_scores, test_y, test_total)
    selected_calibration = select_operating_point(calibration_rows, target_precision)
    selected_test = next(row for row in test_rows if row["threshold"] == selected_calibration["threshold"])
    test_frontier = select_operating_point(test_rows, target_precision)
    return {
        "selected_calibration": selected_calibration,
        "selected_test": selected_test,
        "test_frontier": test_frontier,
    }


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Union Candidate Context Verifier",
        "",
        f"- target precision: `{result['target_precision']}`",
        "",
        "| Candidate thresholds | Model | Candidate ceiling | Test P | Test R | Frontier P | Frontier R |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, block in result["runs"].items():
        for model_name, row in block["models"].items():
            selected = row["selected_test"]
            frontier = row["test_frontier"]
            lines.append(
                f"| {key} | {model_name} | {block['test_stats']['candidate_recall_ceiling']:.4f} | "
                f"{selected['precision']:.4f} | {selected['recall']:.4f} | "
                f"{frontier['precision']:.4f} | {frontier['recall']:.4f} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    models = (
        *load_model(args.threeclass_checkpoint, device, require_explicit_surprise=True),
        *load_model(args.binary_checkpoint, device, require_explicit_surprise=True),
        *load_model(args.forward_checkpoint, device, require_explicit_surprise=False),
        *load_model(args.backward_checkpoint, device, require_explicit_surprise=False),
    )
    validation = make_dataset(args, "validation", args.max_validation_files)
    train_indices, calibration_indices, train_files, calibration_files = split_indices_by_file(
        validation, args.calibration_file_fraction, args.seed
    )
    test = make_dataset(args, "test", args.max_test_files)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for three_threshold in args.threeclass_candidate_thresholds:
        for binary_threshold in args.binary_candidate_thresholds:
            run_key = f"three={three_threshold:.2f},binary={binary_threshold:.2f}"
            print(f"run={run_key}", flush=True)
            train_x, train_y, train_scores, train_stats, train_files_meta, train_pos, train_counts, train_score_columns = collect_context_candidates(
                models,
                make_loader(validation, train_indices, args.batch_size),
                device,
                args,
                three_threshold,
                binary_threshold,
                f"collect train {run_key}",
            )
            calibration_x, calibration_y, calibration_scores, calibration_stats, calibration_files_meta, calibration_pos, calibration_counts, calibration_score_columns = collect_context_candidates(
                models,
                make_loader(validation, calibration_indices, args.batch_size),
                device,
                args,
                three_threshold,
                binary_threshold,
                f"collect calibration {run_key}",
            )
            test_x, test_y, test_scores, test_stats, test_files_meta, test_pos, test_counts, test_score_columns = collect_context_candidates(
                models,
                make_loader(test, list(range(len(test))), args.batch_size),
                device,
                args,
                three_threshold,
                binary_threshold,
                f"collect test {run_key}",
            )
            train_x = append_piece_relative_features(train_x, train_files_meta, train_pos, train_counts, train_score_columns)
            calibration_x = append_piece_relative_features(calibration_x, calibration_files_meta, calibration_pos, calibration_counts, calibration_score_columns)
            test_x = append_piece_relative_features(test_x, test_files_meta, test_pos, test_counts, test_score_columns)
            train_x, normalization, standardized = standardize(train_x, calibration_x, test_x)
            calibration_x, test_x = standardized

            model_specs = [
                (
                    "hist_gradient_boosting",
                    HistGradientBoostingClassifier(
                        max_iter=500,
                        learning_rate=0.035,
                        max_leaf_nodes=31,
                        min_samples_leaf=20,
                        l2_regularization=0.03,
                        class_weight="balanced",
                        random_state=args.seed,
                    ),
                ),
                (
                    "hist_gradient_boosting_small_leaf",
                    HistGradientBoostingClassifier(
                        max_iter=500,
                        learning_rate=0.03,
                        max_leaf_nodes=63,
                        min_samples_leaf=10,
                        l2_regularization=0.01,
                        class_weight="balanced",
                        random_state=args.seed + 1,
                    ),
                ),
                (
                    "random_forest_balanced",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=args.seed,
                        n_jobs=-1,
                    ),
                ),
                (
                    "logistic_regression_balanced",
                    LogisticRegression(max_iter=3000, class_weight="balanced", random_state=args.seed),
                ),
            ]
            model_results = {}
            for model_name, model in model_specs:
                print(f"fitting {run_key} {model_name}", flush=True)
                model.fit(train_x.numpy(), train_y.numpy().astype(np.int64))
                model_results[model_name] = evaluate_model(
                    model,
                    calibration_x,
                    calibration_y,
                    calibration_stats["error_notes"],
                    test_x,
                    test_y,
                    test_stats["error_notes"],
                    args.target_precision,
                )
                dump(model, checkpoint_dir / f"{run_key.replace(',', '_').replace('=', '')}_{model_name}.joblib")
                frontier = model_results[model_name]["test_frontier"]
                print(
                    f"run={run_key} model={model_name} frontier_precision={frontier['precision']:.4f} "
                    f"frontier_recall={frontier['recall']:.4f}",
                    flush=True,
                )
            runs[run_key] = {
                "threeclass_candidate_threshold": three_threshold,
                "binary_candidate_threshold": binary_threshold,
                "train_stats": train_stats,
                "calibration_stats": calibration_stats,
                "test_stats": test_stats,
                "models": model_results,
                "train_file_ids": train_files,
                "calibration_file_ids": calibration_files,
                "normalization": normalization.tolist(),
            }
            partial = {
                "target_precision": args.target_precision,
                "runs": runs,
            }
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_json).write_text(json.dumps(partial, indent=2), encoding="utf-8")
            write_markdown(Path(args.output_md), partial)
    result = {
        "target_precision": args.target_precision,
        "runs": runs,
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
