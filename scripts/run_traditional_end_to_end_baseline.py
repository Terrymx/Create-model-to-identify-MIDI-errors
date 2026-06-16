from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from torch.utils.data import DataLoader
from tqdm import tqdm

from midi_error_detector.data import MaestroWrongNoteDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end traditional ML wrong-note baselines.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-json", default="training_logs/traditional_end_to_end_baseline.json")
    parser.add_argument("--output-md", default="training_logs/traditional_end_to_end_baseline.md")
    parser.add_argument("--checkpoint-dir", default="checkpoints/traditional_end_to_end_baseline")
    parser.add_argument("--train-error-rate", type=float, default=0.08)
    parser.add_argument("--eval-error-rate", type=float, default=0.01)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--max-train-notes", type=int, default=500000)
    parser.add_argument("--max-validation-notes", type=int, default=0)
    parser.add_argument("--max-test-notes", type=int, default=0)
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-validation-files", type=int, default=None)
    parser.add_argument("--max-test-files", type=int, default=None)
    parser.add_argument("--target-precision", type=float, default=0.80)
    parser.add_argument("--sklearn-max-iter", type=int, default=3000)
    return parser.parse_args()


def make_dataset(args: argparse.Namespace, split: str, error_rate: float, max_files: int | None) -> MaestroWrongNoteDataset:
    dataset = MaestroWrongNoteDataset(
        root=args.data_root,
        split=split,
        window_size=args.window_size,
        stride=args.stride,
        error_rate=error_rate,
        seed=args.seed,
        max_files=max_files,
        cache_notes=True,
        verbose=True,
    )
    dataset.set_epoch(0)
    return dataset


def reservoir_collect(
    dataset: MaestroWrongNoteDataset,
    batch_size: int,
    max_notes: int,
    seed: int,
    description: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    kept_x: np.ndarray | None = None
    kept_y: np.ndarray | None = None
    seen = 0
    positives_seen = 0
    valid_seen = 0

    for batch in tqdm(loader, desc=description, unit="batch", dynamic_ncols=True):
        mask = batch["mask"].bool()
        x = batch["features"][mask].numpy().astype(np.float32)
        y = batch["is_error"][mask].numpy().astype(np.int64)
        valid_seen += len(y)
        positives_seen += int(y.sum())
        if max_notes <= 0:
            if kept_x is None:
                kept_x, kept_y = x, y
            else:
                kept_x = np.concatenate([kept_x, x], axis=0)
                kept_y = np.concatenate([kept_y, y], axis=0)
            continue

        if kept_x is None:
            kept_x = np.empty((max_notes, x.shape[1]), dtype=np.float32)
            kept_y = np.empty((max_notes,), dtype=np.int64)
        for row, label in zip(x, y):
            if seen < max_notes:
                kept_x[seen] = row
                kept_y[seen] = label
            else:
                replacement = int(rng.integers(0, seen + 1))
                if replacement < max_notes:
                    kept_x[replacement] = row
                    kept_y[replacement] = label
            seen += 1

    if kept_x is None or kept_y is None:
        raise RuntimeError(f"No notes collected for {description}")
    if max_notes > 0:
        kept = min(seen, max_notes)
        kept_x = kept_x[:kept]
        kept_y = kept_y[:kept]
    stats = {
        "windows": len(dataset),
        "notes_seen": valid_seen,
        "errors_seen": positives_seen,
        "error_rate": positives_seen / max(valid_seen, 1),
        "notes_kept": int(len(kept_y)),
        "errors_kept": int(kept_y.sum()),
        "kept_error_rate": float(kept_y.mean()) if len(kept_y) else 0.0,
    }
    return kept_x, kept_y, stats


def standardize(train_x: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, dict, list[np.ndarray]]:
    mean = train_x.mean(axis=0)
    std = np.maximum(train_x.std(axis=0), 1e-5)
    return (
        (train_x - mean) / std,
        {"mean": mean.tolist(), "std": std.tolist()},
        [(x - mean) / std for x in others],
    )


def scores_from_model(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1].astype(np.float32)
    if hasattr(model, "decision_function"):
        raw = np.clip(model.decision_function(x), -50.0, 50.0)
        return (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)
    return model.predict(x).astype(np.float32)


def metric_rows(scores: np.ndarray, labels: np.ndarray, total_errors: int | None = None) -> list[dict]:
    labels_bool = labels.astype(bool)
    total = int(labels_bool.sum()) if total_errors is None else int(total_errors)
    rows = []
    for step in range(1, 100):
        threshold = step / 100.0
        pred = scores >= threshold
        tp = int(np.logical_and(pred, labels_bool).sum())
        fp = int(np.logical_and(pred, ~labels_bool).sum())
        fn = total - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(total, 1)
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
        "# Traditional End-to-End Baseline",
        "",
        f"- train error rate: `{result['train_error_rate']}`",
        f"- eval error rate: `{result['eval_error_rate']}`",
        f"- target precision: `{result['target_precision']}`",
        f"- train notes kept: `{result['train_stats']['notes_kept']}` / seen `{result['train_stats']['notes_seen']}`",
        f"- validation notes: `{result['validation_stats']['notes_kept']}`",
        f"- test notes: `{result['test_stats']['notes_kept']}`",
        "",
        "| Model | Cal threshold | Cal P | Cal R | Test P | Test R | Test frontier P | Test frontier R |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in result["models"].items():
        cal = row["selected_calibration"]
        selected = row["selected_test"]
        frontier = row["test_frontier"]
        lines.append(
            f"| {name} | {cal['threshold']:.2f} | {cal['precision']:.4f} | {cal['recall']:.4f} | "
            f"{selected['precision']:.4f} | {selected['recall']:.4f} | "
            f"{frontier['precision']:.4f} | {frontier['recall']:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = make_dataset(args, "train", args.train_error_rate, args.max_train_files)
    validation = make_dataset(args, "validation", args.eval_error_rate, args.max_validation_files)
    test = make_dataset(args, "test", args.eval_error_rate, args.max_test_files)

    train_x, train_y, train_stats = reservoir_collect(
        train,
        args.batch_size,
        args.max_train_notes,
        args.seed,
        "collect traditional train",
    )
    validation_x, validation_y, validation_stats = reservoir_collect(
        validation,
        args.batch_size,
        args.max_validation_notes,
        args.seed + 1,
        "collect traditional validation",
    )
    test_x, test_y, test_stats = reservoir_collect(
        test,
        args.batch_size,
        args.max_test_notes,
        args.seed + 2,
        "collect traditional test",
    )
    print(f"train_stats={train_stats}", flush=True)
    print(f"validation_stats={validation_stats}", flush=True)
    print(f"test_stats={test_stats}", flush=True)

    train_x, normalization, standardized = standardize(train_x, validation_x, test_x)
    validation_x, test_x = standardized

    model_specs = [
        (
            "logistic_regression_plain",
            LogisticRegression(max_iter=args.sklearn_max_iter, random_state=args.seed),
        ),
        (
            "logistic_regression_balanced",
            LogisticRegression(max_iter=args.sklearn_max_iter, class_weight="balanced", random_state=args.seed),
        ),
        (
            "linear_svm_balanced",
            CalibratedClassifierCV(
                estimator=LinearSVC(class_weight="balanced", random_state=args.seed, max_iter=args.sklearn_max_iter),
                method="sigmoid",
                cv=3,
            ),
        ),
        (
            "random_forest_balanced",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=args.seed,
                n_jobs=-1,
            ),
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                l2_regularization=0.01,
                class_weight="balanced",
                random_state=args.seed,
            ),
        ),
    ]

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    models = {}
    try:
        from joblib import dump
    except ImportError:
        dump = None

    for name, model in model_specs:
        print(f"fitting {name}", flush=True)
        model.fit(train_x, train_y)
        validation_scores = scores_from_model(model, validation_x)
        test_scores = scores_from_model(model, test_x)
        validation_rows = metric_rows(validation_scores, validation_y)
        test_rows = metric_rows(test_scores, test_y)
        selected_calibration = select_operating_point(validation_rows, args.target_precision)
        selected_test = next(row for row in test_rows if row["threshold"] == selected_calibration["threshold"])
        test_frontier = select_operating_point(test_rows, args.target_precision)
        models[name] = {
            "selected_calibration": selected_calibration,
            "selected_test": selected_test,
            "test_frontier": test_frontier,
        }
        print(
            f"model={name} cal_threshold={selected_calibration['threshold']:.2f} "
            f"test_precision={selected_test['precision']:.4f} test_recall={selected_test['recall']:.4f} "
            f"frontier_precision={test_frontier['precision']:.4f} frontier_recall={test_frontier['recall']:.4f}",
            flush=True,
        )
        if dump is not None:
            dump(model, checkpoint_dir / f"{name}.joblib")

    result = {
        "train_error_rate": args.train_error_rate,
        "eval_error_rate": args.eval_error_rate,
        "target_precision": args.target_precision,
        "normalization": normalization,
        "train_stats": train_stats,
        "validation_stats": validation_stats,
        "test_stats": test_stats,
        "models": models,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
