from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def comparison_variants() -> dict[str, tuple[str, str | None, str]]:
    return {
        "B2": ("B2", None, "primary"),
        "B2_C_radius4": ("B2", "C1", "primary"),
        "B2_C_radius4_8_16": ("B2", "C2", "primary"),
        "B3_C_radius4": ("B3", "C1", "primary"),
        "B3_C_radius4_8_12": ("B3", "C2", "alternate"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--alternate-cache-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--target-precision", type=float, default=0.80)
    parser.add_argument("--calibration-precision", type=float, default=0.81)
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
    cache_dirs = {
        "primary": Path(args.cache_dir),
        "alternate": Path(args.alternate_cache_dir),
    }
    sources = {
        source_name: {
            split_name: load_canonical_split(
                cache_dir,
                split_name,
                args.window_size,
            )
            for split_name in ("train", "calibration", "test")
        }
        for source_name, cache_dir in cache_dirs.items()
    }
    train, train_meta = sources["primary"]["train"]
    calibration, calibration_meta = sources["primary"]["calibration"]
    test, test_meta = sources["primary"]["test"]
    for split_name, primary in (
        ("train", train),
        ("calibration", calibration),
        ("test", test),
    ):
        alternate = sources["alternate"][split_name][0]
        for field in ("file_ids", "positions", "labels"):
            np.testing.assert_array_equal(primary[field], alternate[field])
    calibration_total = total_errors(calibration_meta)
    test_total = total_errors(test_meta)
    feature_sets: dict[str, dict[str, np.ndarray]] = {
        split_name: {} for split_name in ("train", "calibration", "test")
    }
    for system_name, (
        b_variant,
        c_variant,
        source_name,
    ) in comparison_variants().items():
        for split_name in feature_sets:
            arrays = sources[source_name][split_name][0]
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
    for system_name in comparison_variants():
        model = make_small_leaf(args.seed)
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
            args.calibration_precision,
        )
    result = {
        "protocol": "piece_consistent_unique_note",
        "target_precision": args.target_precision,
        "calibration_precision": args.calibration_precision,
        "c_radii": {
            source_name: sources[source_name]["train"][1]["c_radii"]
            for source_name in sources
        },
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
