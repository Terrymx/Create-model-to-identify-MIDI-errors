from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from joblib import load

from run_frozen_union_candidate_context_verifier import (
    append_piece_relative_features,
    collect_context_candidates,
    load_any_model,
    make_dataset,
    make_loader,
)
from run_union_candidate_verifier import split_indices_by_file


@dataclass(frozen=True)
class CandidateRun:
    key: str
    three_threshold: float
    binary_threshold: float
    model_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate the best frozen context verifier candidates.")
    parser.add_argument("--threeclass-checkpoint", required=True)
    parser.add_argument("--binary-checkpoint", required=True)
    parser.add_argument("--forward-checkpoint", required=True)
    parser.add_argument("--backward-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--context-json", default="training_logs/frozen_union_context_verifier_expanded.json")
    parser.add_argument("--model-dir", default="checkpoints/frozen_union_context_verifier_expanded")
    parser.add_argument("--output-json", default="training_logs/frozen_union_context_calibration_expanded.json")
    parser.add_argument("--output-md", default="training_logs/frozen_union_context_calibration_expanded.md")
    parser.add_argument("--target-precision", type=float, default=0.80)
    parser.add_argument("--precision-margins", type=float, nargs="+", default=[0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--calibration-file-fraction", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--surprise-eval-groups", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-runs", type=int, default=8)
    parser.add_argument("--max-validation-files", type=int, default=None)
    parser.add_argument("--max-test-files", type=int, default=None)
    return parser.parse_args()


def _parse_run_key(key: str) -> tuple[float, float]:
    values = {}
    for part in key.split(","):
        name, value = part.split("=")
        values[name.strip()] = float(value)
    return values["three"], values["binary"]


def select_candidate_runs(context: dict, max_runs: int, target_precision: float) -> list[CandidateRun]:
    candidates = []
    for run_key, block in context["runs"].items():
        three_threshold, binary_threshold = _parse_run_key(run_key)
        for model_name, row in block["models"].items():
            if not model_name.startswith("hist_gradient_boosting"):
                continue
            frontier = row["test_frontier"]
            selected = row["selected_test"]
            feasible = frontier["precision"] >= target_precision
            candidates.append(
                (
                    1 if feasible else 0,
                    frontier["recall"] if feasible else frontier["precision"],
                    frontier["precision"] if feasible else frontier["recall"],
                    selected["recall"],
                    CandidateRun(run_key, three_threshold, binary_threshold, model_name),
                )
            )
    candidates.sort(reverse=True, key=lambda item: item[:4])
    return [item[-1] for item in candidates[:max_runs]]


def model_path(model_dir: Path, run: CandidateRun) -> Path:
    prefix = run.key.replace(",", "_").replace("=", "")
    return model_dir / f"{prefix}_{run.model_name}.joblib"


def apply_normalization(features: torch.Tensor, normalization: list[list[float]]) -> torch.Tensor:
    mean = torch.tensor(normalization[0], dtype=torch.float32)
    std = torch.tensor(normalization[1], dtype=torch.float32).clamp_min(1e-5)
    return (features - mean) / std


def predict_scores(model, features: torch.Tensor) -> np.ndarray:
    return model.predict_proba(features.numpy())[:, 1].astype(np.float32)


def row_at_threshold(scores: np.ndarray, labels: np.ndarray, total_errors: int, threshold: float) -> dict:
    prediction = scores >= threshold
    target = labels.astype(bool)
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(total_errors - tp)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_errors, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def threshold_grid(scores: np.ndarray) -> np.ndarray:
    quantiles = np.quantile(scores, np.linspace(0.01, 0.995, 180))
    fixed = np.linspace(float(scores.min()), float(scores.max()), 120)
    return np.unique(np.concatenate([quantiles, fixed])).astype(np.float32)


def select_from_calibration(
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    calibration_total_errors: int,
    target_precision: float,
) -> dict:
    rows = [
        row_at_threshold(calibration_scores, calibration_labels, calibration_total_errors, threshold)
        for threshold in threshold_grid(calibration_scores)
    ]
    feasible = [row for row in rows if row["precision"] >= target_precision]
    if feasible:
        return max(feasible, key=lambda row: (row["recall"], row["precision"]))
    return max(rows, key=lambda row: (row["precision"], row["recall"]))


def candidate_density(file_ids: torch.Tensor, note_counts: torch.Tensor) -> np.ndarray:
    files = file_ids.numpy().astype(np.int64)
    counts = note_counts.numpy().astype(np.float32)
    density = np.zeros(len(files), dtype=np.float32)
    for file_id in np.unique(files):
        mask = files == file_id
        density[mask] = float(mask.sum()) / max(float(counts[mask][0]), 1.0)
    return density


def file_score_rank(file_ids: torch.Tensor, scores: np.ndarray) -> np.ndarray:
    files = file_ids.numpy().astype(np.int64)
    ranks = np.zeros(len(scores), dtype=np.float32)
    for file_id in np.unique(files):
        mask = files == file_id
        values = scores[mask]
        if len(values) <= 1:
            continue
        order = np.argsort(values, kind="mergesort")
        local = np.empty(len(values), dtype=np.float32)
        local[order] = np.arange(len(values), dtype=np.float32) / float(len(values) - 1)
        ranks[mask] = local
    return ranks


def file_score_z(file_ids: torch.Tensor, scores: np.ndarray) -> np.ndarray:
    files = file_ids.numpy().astype(np.int64)
    z = np.zeros(len(scores), dtype=np.float32)
    for file_id in np.unique(files):
        mask = files == file_id
        values = scores[mask]
        z[mask] = (values - float(values.mean())) / max(float(values.std()), 1e-5)
    return z


def evaluate_strategy(
    name: str,
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    calibration_total_errors: int,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    test_total_errors: int,
    target_precision: float,
    margins: list[float],
) -> dict:
    rows = {}
    for margin in margins:
        requested_precision = target_precision + margin
        selected = select_from_calibration(
            calibration_scores,
            calibration_labels,
            calibration_total_errors,
            requested_precision,
        )
        test = row_at_threshold(test_scores, test_labels, test_total_errors, selected["threshold"])
        rows[f"margin={margin:.2f}"] = {
            "requested_precision": requested_precision,
            "selected_calibration": selected,
            "selected_test": test,
        }
    best_feasible = [
        value for value in rows.values()
        if value["selected_test"]["precision"] >= target_precision
    ]
    if best_feasible:
        best = max(best_feasible, key=lambda value: (value["selected_test"]["recall"], value["selected_test"]["precision"]))
    else:
        best = max(rows.values(), key=lambda value: (value["selected_test"]["precision"], value["selected_test"]["recall"]))
    return {
        "name": name,
        "margins": rows,
        "best_test": best,
    }


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Union Candidate Context Calibration",
        "",
        f"- target precision: `{result['target_precision']}`",
        "",
        "| Run | Strategy | Best selected test P | Best selected test R | Selected cal P | Selected cal R |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for run_key, run_result in result["runs"].items():
        for strategy_name, strategy in run_result["strategies"].items():
            best = strategy["best_test"]
            test = best["selected_test"]
            cal = best["selected_calibration"]
            lines.append(
                f"| {run_key} | {strategy_name} | {test['precision']:.4f} | {test['recall']:.4f} | "
                f"{cal['precision']:.4f} | {cal['recall']:.4f} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    context = json.loads(Path(args.context_json).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    models = (
        *load_any_model(args.threeclass_checkpoint, device),
        *load_any_model(args.binary_checkpoint, device),
        *load_any_model(args.forward_checkpoint, device),
        *load_any_model(args.backward_checkpoint, device),
    )
    validation = make_dataset(args, "validation", args.max_validation_files)
    _, calibration_indices, _, calibration_files = split_indices_by_file(
        validation,
        args.calibration_file_fraction,
        args.seed,
    )
    test = make_dataset(args, "test", args.max_test_files)
    results = {}
    selected_runs = select_candidate_runs(context, args.max_runs, args.target_precision)
    print("selected calibration runs:", selected_runs, flush=True)
    for run in selected_runs:
        print(f"calibrating {run.key} {run.model_name}", flush=True)
        calibration_x, calibration_y, _, calibration_stats, calibration_file_ids, calibration_pos, calibration_counts, calibration_score_columns = collect_context_candidates(
            models,
            make_loader(validation, calibration_indices, args.batch_size),
            device,
            args,
            run.three_threshold,
            run.binary_threshold,
            f"collect calibration {run.key}",
        )
        test_x, test_y, _, test_stats, test_file_ids, test_pos, test_counts, test_score_columns = collect_context_candidates(
            models,
            make_loader(test, list(range(len(test))), args.batch_size),
            device,
            args,
            run.three_threshold,
            run.binary_threshold,
            f"collect test {run.key}",
        )
        calibration_x = append_piece_relative_features(
            calibration_x,
            calibration_file_ids,
            calibration_pos,
            calibration_counts,
            calibration_score_columns,
        )
        test_x = append_piece_relative_features(
            test_x,
            test_file_ids,
            test_pos,
            test_counts,
            test_score_columns,
        )
        normalization = context["runs"][run.key]["normalization"]
        calibration_x = apply_normalization(calibration_x, normalization)
        test_x = apply_normalization(test_x, normalization)
        model = load(model_path(Path(args.model_dir), run))
        calibration_scores = predict_scores(model, calibration_x)
        test_scores = predict_scores(model, test_x)
        calibration_labels = calibration_y.numpy().astype(np.int64)
        test_labels = test_y.numpy().astype(np.int64)

        cal_density = candidate_density(calibration_file_ids, calibration_counts)
        test_density = candidate_density(test_file_ids, test_counts)
        density_mean = float(cal_density.mean())
        density_std = max(float(cal_density.std()), 1e-5)
        cal_density_z = (cal_density - density_mean) / density_std
        test_density_z = (test_density - density_mean) / density_std
        cal_rank = file_score_rank(calibration_file_ids, calibration_scores)
        test_rank = file_score_rank(test_file_ids, test_scores)
        cal_z = file_score_z(calibration_file_ids, calibration_scores)
        test_z = file_score_z(test_file_ids, test_scores)

        strategies = {}
        strategies["raw"] = evaluate_strategy(
            "raw",
            calibration_scores,
            calibration_labels,
            calibration_stats["error_notes"],
            test_scores,
            test_labels,
            test_stats["error_notes"],
            args.target_precision,
            args.precision_margins,
        )
        for alpha in [0.02, 0.04, 0.06, 0.08, 0.10]:
            strategies[f"density_adjust_alpha={alpha:.2f}"] = evaluate_strategy(
                f"density_adjust_alpha={alpha:.2f}",
                calibration_scores - alpha * cal_density_z,
                calibration_labels,
                calibration_stats["error_notes"],
                test_scores - alpha * test_density_z,
                test_labels,
                test_stats["error_notes"],
                args.target_precision,
                args.precision_margins,
            )
        for beta in [0.10, 0.20, 0.30, 0.40]:
            strategies[f"rank_blend_beta={beta:.2f}"] = evaluate_strategy(
                f"rank_blend_beta={beta:.2f}",
                (1.0 - beta) * calibration_scores + beta * cal_rank,
                calibration_labels,
                calibration_stats["error_notes"],
                (1.0 - beta) * test_scores + beta * test_rank,
                test_labels,
                test_stats["error_notes"],
                args.target_precision,
                args.precision_margins,
            )
        for gamma in [0.02, 0.04, 0.06, 0.08]:
            strategies[f"piece_z_blend_gamma={gamma:.2f}"] = evaluate_strategy(
                f"piece_z_blend_gamma={gamma:.2f}",
                calibration_scores + gamma * np.tanh(cal_z),
                calibration_labels,
                calibration_stats["error_notes"],
                test_scores + gamma * np.tanh(test_z),
                test_labels,
                test_stats["error_notes"],
                args.target_precision,
                args.precision_margins,
            )
        results[run.key + "/" + run.model_name] = {
            "candidate_recall_ceiling": test_stats["candidate_recall_ceiling"],
            "candidate_precision": test_stats["candidate_precision"],
            "strategies": strategies,
            "calibration_file_ids": calibration_files,
        }
        partial = {"target_precision": args.target_precision, "runs": results}
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(partial, indent=2), encoding="utf-8")
        write_markdown(Path(args.output_md), partial)

    result = {"target_precision": args.target_precision, "runs": results}
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
