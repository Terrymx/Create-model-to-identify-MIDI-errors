from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def comparison_variants() -> dict[str, tuple[str, str | None]]:
    return {
        "B2": ("B2", None),
        "B2_C_radius4": ("B2", "C1"),
        "B2_C_radius4_8_16": ("B2", "C2"),
        "B3_C_radius4": ("B3", "C1"),
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


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Piece-consistent counterfactual C",
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
        build_c_variant_features,
        make_small_leaf,
    )
    from run_counterfactual_piece_b import (
        evaluate_calibrated_scores,
        load_canonical_split,
        total_errors,
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
    split_arrays = {
        "train": train,
        "calibration": calibration,
        "test": test,
    }
    feature_sets: dict[str, dict[str, np.ndarray]] = {
        split_name: {}
        for split_name in split_arrays
    }
    for system_name, (b_variant, c_variant) in comparison_variants().items():
        for split_name, arrays in split_arrays.items():
            if c_variant is None:
                features = build_b_variant_features(
                    arrays["base_features"],
                    arrays["b_features"],
                    arrays["b_ranking"],
                    b_variant,
                )
            else:
                features = build_c_variant_features(
                    arrays["base_features"],
                    arrays["b_features"],
                    arrays["b_ranking"],
                    arrays["c_features"],
                    arrays["c_ranking"],
                    c_variant,
                    b_variant=b_variant,
                )
            feature_sets[split_name][system_name] = features

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    systems = {}
    for offset, system_name in enumerate(comparison_variants()):
        model = make_small_leaf(args.seed + offset)
        model.fit(
            feature_sets["train"][system_name],
            train["labels"].astype(np.int64),
        )
        dump(model, checkpoint_dir / f"{system_name.lower()}_small_leaf.joblib")
        systems[system_name] = evaluate_calibrated_scores(
            model.predict_proba(feature_sets["calibration"][system_name])[:, 1],
            calibration["labels"],
            calibration_total,
            model.predict_proba(feature_sets["test"][system_name])[:, 1],
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
