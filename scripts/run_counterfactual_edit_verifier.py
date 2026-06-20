from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import dump, load
from sklearn.ensemble import HistGradientBoostingClassifier

from build_counterfactual_candidate_cache import (
    aggregate_proposal_features,
    load_candidate_cache,
)
from calibrate_frozen_context_verifier import row_at_threshold, select_from_calibration


B_VARIANT_COLUMNS = {
    "B1": [0, 6, 7, 12, 13, 14, 15, 16],
    "B2": [0, 1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 15, 16],
    "B3": list(range(17)),
}


def build_b_variant_features(
    base_features: np.ndarray,
    proposal_features: np.ndarray,
    ranking_scores: np.ndarray,
    variant: str,
) -> np.ndarray:
    if variant not in B_VARIANT_COLUMNS:
        raise ValueError(f"Unknown B variant: {variant}")
    selected = proposal_features[:, :, B_VARIANT_COLUMNS[variant]]
    aggregated = aggregate_proposal_features(selected, ranking_scores)
    return np.concatenate([base_features, aggregated], axis=1).astype(np.float32)


def build_c_variant_features(
    base_features: np.ndarray,
    b_features: np.ndarray,
    b_ranking: np.ndarray,
    c_features: np.ndarray,
    c_ranking: np.ndarray,
    variant: str,
) -> np.ndarray:
    if variant == "C1":
        selected_c = c_features[:, :, :5]
    elif variant == "C2":
        selected_c = c_features[:, :, :15]
    else:
        raise ValueError(f"Unknown C variant: {variant}")
    b3 = build_b_variant_features(base_features, b_features, b_ranking, "B3")
    aggregated_c = aggregate_proposal_features(selected_c, c_ranking)
    return np.concatenate([b3, aggregated_c], axis=1).astype(np.float32)


def correction_metrics(
    proposals: np.ndarray,
    proposal_scores: np.ndarray,
    detected: np.ndarray,
    labels: np.ndarray,
    error_kinds: np.ndarray,
    target_pitch: np.ndarray,
) -> dict:
    replace = detected & labels.astype(bool) & (error_kinds == 1)
    count = int(replace.sum())
    if count == 0:
        return {
            "detected_replace_errors": 0,
            "top1_correct": 0,
            "top3_correct": 0,
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
        }
    order = np.argsort(-proposal_scores[replace], axis=1, kind="mergesort")
    ranked = np.take_along_axis(proposals[replace], order, axis=1)
    targets = target_pitch[replace]
    top1 = int((ranked[:, 0] == targets).sum())
    top3 = int((ranked[:, :3] == targets[:, None]).any(axis=1).sum())
    return {
        "detected_replace_errors": count,
        "top1_correct": top1,
        "top3_correct": top3,
        "top1_accuracy": top1 / count,
        "top3_accuracy": top3 / count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--old-context-json", required=True)
    parser.add_argument("--old-model-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--target-precision", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def load_split(cache_dir: Path, name: str) -> tuple[dict[str, np.ndarray], dict]:
    return load_candidate_cache(cache_dir / f"{name}.npz")


def normalize_from_context(
    features: np.ndarray,
    normalization: list[list[float]],
) -> np.ndarray:
    mean = np.asarray(normalization[0], dtype=np.float32)
    std = np.maximum(np.asarray(normalization[1], dtype=np.float32), 1e-5)
    return (features - mean) / std


def evaluate_score_rows(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    calibration: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    calibration_total_errors: int,
    test_total_errors: int,
    target_precision: float,
) -> dict:
    rows = []
    for margin in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]:
        selected = select_from_calibration(
            calibration_scores,
            calibration["labels"].astype(np.int64),
            calibration_total_errors,
            target_precision + margin,
        )
        test_row = row_at_threshold(
            test_scores,
            test["labels"].astype(np.int64),
            test_total_errors,
            selected["threshold"],
        )
        test_row["correction"] = correction_metrics(
            test["proposals"],
            test["b_ranking"],
            test_scores >= selected["threshold"],
            test["labels"],
            test["error_kind"],
            test["target_pitch"],
        )
        rows.append(
            {
                "requested_precision": target_precision + margin,
                "selected_calibration": selected,
                "selected_test": test_row,
            }
        )
    feasible = [
        row for row in rows if row["selected_test"]["precision"] >= target_precision
    ]
    return {
        "margins": rows,
        "best_feasible_test": max(
            feasible or rows,
            key=lambda row: (
                row["selected_test"]["recall"]
                if row["selected_test"]["precision"] >= target_precision
                else -1.0,
                row["selected_test"]["precision"],
            ),
        ),
    }


def make_small_leaf(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.03,
        max_leaf_nodes=63,
        min_samples_leaf=10,
        l2_regularization=0.01,
        class_weight="balanced",
        random_state=seed,
    )


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Counterfactual B Verifier",
        "",
        f"- candidate recall ceiling: `{result['test_stats']['candidate_recall_ceiling']:.4f}`",
        "",
        "| System | Precision | Recall | F1 | Replace Top-1 | Replace Top-3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, block in result["systems"].items():
        row = block["best_feasible_test"]["selected_test"]
        correction = row["correction"]
        lines.append(
            f"| {name} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {correction['top1_accuracy']:.4f} | "
            f"{correction['top3_accuracy']:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    train, train_meta = load_split(cache_dir, "train")
    calibration, calibration_meta = load_split(cache_dir, "calibration")
    test, test_meta = load_split(cache_dir, "test")
    calibration_total = int(calibration_meta["stats"]["error_notes"])
    test_total = int(test_meta["stats"]["error_notes"])

    context = json.loads(Path(args.old_context_json).read_text(encoding="utf-8"))
    normalization = context["runs"]["three=0.60,binary=0.50"]["normalization"]
    baseline_model = load(
        Path(args.old_model_dir)
        / "three0.60_binary0.50_hist_gradient_boosting_small_leaf.joblib"
    )
    calibration_baseline = baseline_model.predict_proba(
        normalize_from_context(calibration["base_features"], normalization)
    )[:, 1]
    test_baseline = baseline_model.predict_proba(
        normalize_from_context(test["base_features"], normalization)
    )[:, 1]
    systems = {
        "A_historical_old_small_leaf": evaluate_score_rows(
            calibration_baseline,
            test_baseline,
            calibration,
            test,
            calibration_total,
            test_total,
            args.target_precision,
        )
    }
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for offset, variant in enumerate(["B1", "B2", "B3"]):
        train_x = build_b_variant_features(
            train["base_features"],
            train["b_features"],
            train["b_ranking"],
            variant,
        )
        calibration_x = build_b_variant_features(
            calibration["base_features"],
            calibration["b_features"],
            calibration["b_ranking"],
            variant,
        )
        test_x = build_b_variant_features(
            test["base_features"],
            test["b_features"],
            test["b_ranking"],
            variant,
        )
        model = make_small_leaf(args.seed + offset)
        model.fit(train_x, train["labels"].astype(np.int64))
        dump(model, checkpoint_dir / f"{variant.lower()}_small_leaf.joblib")
        systems[variant] = evaluate_score_rows(
            model.predict_proba(calibration_x)[:, 1],
            model.predict_proba(test_x)[:, 1],
            calibration,
            test,
            calibration_total,
            test_total,
            args.target_precision,
        )
    if "c_features" in train:
        for offset, variant in enumerate(["C1", "C2"], start=3):
            train_x = build_c_variant_features(
                train["base_features"],
                train["b_features"],
                train["b_ranking"],
                train["c_features"],
                train["c_ranking"],
                variant,
            )
            calibration_x = build_c_variant_features(
                calibration["base_features"],
                calibration["b_features"],
                calibration["b_ranking"],
                calibration["c_features"],
                calibration["c_ranking"],
                variant,
            )
            test_x = build_c_variant_features(
                test["base_features"],
                test["b_features"],
                test["b_ranking"],
                test["c_features"],
                test["c_ranking"],
                variant,
            )
            model = make_small_leaf(args.seed + offset)
            model.fit(train_x, train["labels"].astype(np.int64))
            dump(model, checkpoint_dir / f"{variant.lower()}_small_leaf.joblib")
            systems[variant] = evaluate_score_rows(
                model.predict_proba(calibration_x)[:, 1],
                model.predict_proba(test_x)[:, 1],
                calibration,
                test,
                calibration_total,
                test_total,
                args.target_precision,
            )
    result = {
        "target_precision": args.target_precision,
        "train_stats": train_meta["stats"],
        "calibration_stats": calibration_meta["stats"],
        "test_stats": test_meta["stats"],
        "systems": systems,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
