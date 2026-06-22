from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def metric_row(
    scores: np.ndarray,
    labels: np.ndarray,
    total_errors: int,
    threshold: float,
) -> dict[str, float | int]:
    selected = scores >= threshold
    positive = labels.astype(bool)
    tp = int((selected & positive).sum())
    fp = int((selected & ~positive).sum())
    fn = int(total_errors - tp)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_errors, 1)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "selected": int(selected.sum()),
    }


def select_calibration_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    total_errors: int,
    target_precision: float,
) -> dict[str, float | int]:
    rows = [
        metric_row(scores, labels, total_errors, float(threshold))
        for threshold in np.unique(scores)
    ]
    feasible = [row for row in rows if row["precision"] >= target_precision]
    return max(
        feasible or rows,
        key=lambda row: (
            row["recall"] if row["precision"] >= target_precision else -1.0,
            row["precision"],
            -row["threshold"],
        ),
    )


def evaluate_calibrated_scores(
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    calibration_total_errors: int,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    test_total_errors: int,
    target_precision: float,
) -> dict:
    calibration = select_calibration_threshold(
        calibration_scores,
        calibration_labels,
        calibration_total_errors,
        target_precision,
    )
    return {
        "calibration": calibration,
        "test": metric_row(
            test_scores,
            test_labels,
            test_total_errors,
            float(calibration["threshold"]),
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--target-precision", type=float, default=0.80)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def load_canonical_split(
    cache_dir: Path,
    split: str,
    window_size: int,
) -> tuple[dict[str, np.ndarray], dict]:
    from run_counterfactual_global_search import canonical_candidate_indices

    with np.load(cache_dir / f"{split}.npz", allow_pickle=False) as loaded:
        metadata = json.loads(str(loaded["__metadata__"]))
        arrays = {
            name: loaded[name].copy()
            for name in loaded.files
            if name != "__metadata__"
        }
    rows = canonical_candidate_indices(arrays, window_size)
    return {name: values[rows] for name, values in arrays.items()}, metadata


def total_errors(metadata: dict) -> int:
    stats = metadata["stats"]
    return int(stats.get("unique_error_notes", stats["error_notes"]))


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Piece-consistent counterfactual B",
        "",
        "| System | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, block in result["systems"].items():
        row = block["test"]
        lines.append(
            f"| {name} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    from joblib import dump

    from run_counterfactual_edit_verifier import (
        build_b_variant_features,
        make_small_leaf,
    )

    args = parse_args()
    cache_dir = Path(args.cache_dir)
    train, train_meta = load_canonical_split(
        cache_dir, "train", args.window_size
    )
    calibration, calibration_meta = load_canonical_split(
        cache_dir, "calibration", args.window_size
    )
    test, test_meta = load_canonical_split(
        cache_dir, "test", args.window_size
    )
    calibration_total = total_errors(calibration_meta)
    test_total = total_errors(test_meta)
    feature_sets = {"A_matched_base": train["base_features"]}
    calibration_sets = {"A_matched_base": calibration["base_features"]}
    test_sets = {"A_matched_base": test["base_features"]}
    for variant in ("B1", "B2", "B3"):
        feature_sets[variant] = build_b_variant_features(
            train["base_features"],
            train["b_features"],
            train["b_ranking"],
            variant,
        )
        calibration_sets[variant] = build_b_variant_features(
            calibration["base_features"],
            calibration["b_features"],
            calibration["b_ranking"],
            variant,
        )
        test_sets[variant] = build_b_variant_features(
            test["base_features"],
            test["b_features"],
            test["b_ranking"],
            variant,
        )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    systems = {}
    for offset, name in enumerate(("A_matched_base", "B1", "B2", "B3")):
        model = make_small_leaf(args.seed + offset)
        model.fit(feature_sets[name], train["labels"].astype(np.int64))
        dump(model, checkpoint_dir / f"{name.lower()}_small_leaf.joblib")
        systems[name] = evaluate_calibrated_scores(
            model.predict_proba(calibration_sets[name])[:, 1],
            calibration["labels"],
            calibration_total,
            model.predict_proba(test_sets[name])[:, 1],
            test["labels"],
            test_total,
            args.target_precision,
        )
    result = {
        "protocol": "piece_consistent_unique_note",
        "target_precision": args.target_precision,
        "candidate_thresholds": {
            "threeclass": train_meta["threeclass_candidate_threshold"],
            "binary": train_meta["binary_candidate_threshold"],
        },
        "candidate_counts": {
            "train": len(train["labels"]),
            "calibration": len(calibration["labels"]),
            "test": len(test["labels"]),
        },
        "candidate_recall_ceiling": {
            "calibration": int(calibration["labels"].sum()) / calibration_total,
            "test": int(test["labels"].sum()) / test_total,
        },
        "systems": systems,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
