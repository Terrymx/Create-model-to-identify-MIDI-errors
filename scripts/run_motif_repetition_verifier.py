from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import dump

from build_counterfactual_candidate_cache import load_candidate_cache
from motif_repetition_features import (
    MOTIF_FEATURE_NAMES,
    compute_motif_repetition_features,
)
from run_counterfactual_edit_verifier import (
    build_b_variant_features,
    build_c_variant_features,
    evaluate_score_rows,
    make_small_leaf,
)
from risk_control_selection import evaluate_fdr_score_rows
from voice_aware_dataset import PieceConsistentVoiceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--target-precision", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--max-validation-files", type=int, default=None)
    parser.add_argument("--max-test-files", type=int, default=None)
    parser.add_argument("--motif-radius", type=int, default=4)
    parser.add_argument("--motif-min-similarity", type=float, default=0.84)
    parser.add_argument("--motif-exclude-radius", type=int, default=16)
    return parser.parse_args()


def load_split(cache_dir: Path, name: str) -> tuple[dict[str, np.ndarray], dict]:
    return load_candidate_cache(cache_dir / f"{name}.npz")


def build_b2_motif_features(arrays: dict[str, np.ndarray]) -> np.ndarray:
    base = build_b_variant_features(
        arrays["base_features"],
        arrays["b_features"],
        arrays["b_ranking"],
        "B2",
    )
    return np.concatenate([base, arrays["motif_features"]], axis=1).astype(np.float32)


def build_c2_motif_features(arrays: dict[str, np.ndarray]) -> np.ndarray:
    base = build_c_variant_features(
        arrays["base_features"],
        arrays["b_features"],
        arrays["b_ranking"],
        arrays["c_features"],
        arrays["c_ranking"],
        "C2",
    )
    return np.concatenate([base, arrays["motif_features"]], axis=1).astype(np.float32)


def _piece_pitches(dataset: PieceConsistentVoiceDataset, file_id: int) -> np.ndarray:
    features = dataset._pieces[file_id].features
    return np.rint(features[:, 0] * 127.0).clip(0, 127).astype(np.int64)


def append_motif_features(
    arrays: dict[str, np.ndarray],
    dataset: PieceConsistentVoiceDataset,
    *,
    radius: int,
    min_similarity: float,
    exclude_radius: int,
) -> dict[str, np.ndarray]:
    motif = np.zeros((len(arrays["labels"]), len(MOTIF_FEATURE_NAMES)), dtype=np.float32)
    file_ids = arrays["file_ids"].astype(np.int64)
    positions = arrays["positions"].astype(np.int64)
    for file_id in np.unique(file_ids):
        row_mask = file_ids == file_id
        motif[row_mask] = compute_motif_repetition_features(
            piece_pitches=_piece_pitches(dataset, int(file_id)),
            candidate_positions=positions[row_mask],
            observed_pitch=arrays["observed_pitch"][row_mask],
            proposals=arrays["proposals"][row_mask],
            radius=radius,
            min_similarity=min_similarity,
            exclude_radius=exclude_radius,
        )
    enriched = dict(arrays)
    enriched["motif_features"] = motif
    return enriched


def motif_summary(arrays: dict[str, np.ndarray]) -> dict:
    motif = arrays["motif_features"]
    labels = arrays["labels"].astype(bool)
    gain_index = MOTIF_FEATURE_NAMES.index("proposal_consensus_gain")
    match_index = MOTIF_FEATURE_NAMES.index("motif_match_count")
    observed_index = MOTIF_FEATURE_NAMES.index("observed_pitch_consensus")
    proposal_index = MOTIF_FEATURE_NAMES.index("best_proposal_consensus")

    def mean_or_zero(values: np.ndarray) -> float:
        return float(values.mean()) if len(values) else 0.0

    return {
        "candidate_rows": int(len(motif)),
        "matched_rows": int((motif[:, match_index] > 0).sum()),
        "matched_fraction": float((motif[:, match_index] > 0).mean()) if len(motif) else 0.0,
        "positive_gain_fraction": float((motif[:, gain_index] > 0).mean()) if len(motif) else 0.0,
        "mean_gain_positive": mean_or_zero(motif[labels, gain_index]),
        "mean_gain_negative": mean_or_zero(motif[~labels, gain_index]),
        "mean_observed_consensus_positive": mean_or_zero(motif[labels, observed_index]),
        "mean_observed_consensus_negative": mean_or_zero(motif[~labels, observed_index]),
        "mean_proposal_consensus_positive": mean_or_zero(motif[labels, proposal_index]),
        "mean_proposal_consensus_negative": mean_or_zero(motif[~labels, proposal_index]),
    }


def train_and_predict(
    name: str,
    train_x: np.ndarray,
    calibration_x: np.ndarray,
    test_x: np.ndarray,
    train: dict[str, np.ndarray],
    seed: int,
    checkpoint_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    model = make_small_leaf(seed)
    model.fit(train_x, train["labels"].astype(np.int64))
    dump(model, checkpoint_dir / f"{name.lower()}_small_leaf.joblib")
    return (
        model.predict_proba(calibration_x)[:, 1],
        model.predict_proba(test_x)[:, 1],
    )


def add_threshold_and_fdr_systems(
    systems: dict,
    name: str,
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    calibration: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    calibration_total: int,
    test_total: int,
    target_precision: float,
) -> None:
    systems[name] = evaluate_score_rows(
        calibration_scores,
        test_scores,
        calibration,
        test,
        calibration_total,
        test_total,
        target_precision,
    )
    systems[f"{name}_fdr"] = evaluate_fdr_score_rows(
        calibration_scores,
        test_scores,
        calibration["labels"].astype(np.int64),
        test["labels"].astype(np.int64),
        calibration["file_ids"].astype(np.int64),
        test["file_ids"].astype(np.int64),
        calibration_total,
        test_total,
        target_precision,
    )


def train_and_evaluate(
    name: str,
    train_x: np.ndarray,
    calibration_x: np.ndarray,
    test_x: np.ndarray,
    train: dict[str, np.ndarray],
    calibration: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    calibration_total: int,
    test_total: int,
    target_precision: float,
    seed: int,
    checkpoint_dir: Path,
) -> dict:
    calibration_scores, test_scores = train_and_predict(
        name,
        train_x,
        calibration_x,
        test_x,
        train,
        seed,
        checkpoint_dir,
    )
    return evaluate_score_rows(
        calibration_scores,
        test_scores,
        calibration,
        test,
        calibration_total,
        test_total,
        target_precision,
    )


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Motif/Repetition Verifier",
        "",
        f"- target precision: `{result['target_precision']:.2f}`",
        f"- motif radius: `{result['motif']['radius']}`",
        f"- motif min similarity: `{result['motif']['min_similarity']}`",
        "",
        "| System | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, block in result["systems"].items():
        row = block["best_feasible_test"]["selected_test"]
        lines.append(
            f"| {name} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    train, train_meta = load_split(cache_dir, "train")
    calibration, calibration_meta = load_split(cache_dir, "calibration")
    test, test_meta = load_split(cache_dir, "test")
    validation_dataset = PieceConsistentVoiceDataset(
        root=args.data_root,
        split="validation",
        voice_method="onset_matching",
        window_size=args.window_size,
        stride=args.stride,
        error_rate=args.error_rate,
        seed=args.seed,
        max_files=args.max_validation_files,
        verbose=True,
    )
    test_dataset = PieceConsistentVoiceDataset(
        root=args.data_root,
        split="test",
        voice_method="onset_matching",
        window_size=args.window_size,
        stride=args.stride,
        error_rate=args.error_rate,
        seed=args.seed,
        max_files=args.max_test_files,
        verbose=True,
    )
    motif_kwargs = {
        "radius": args.motif_radius,
        "min_similarity": args.motif_min_similarity,
        "exclude_radius": args.motif_exclude_radius,
    }
    train = append_motif_features(train, validation_dataset, **motif_kwargs)
    calibration = append_motif_features(calibration, validation_dataset, **motif_kwargs)
    test = append_motif_features(test, test_dataset, **motif_kwargs)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    calibration_total = int(calibration_meta["stats"]["error_notes"])
    test_total = int(test_meta["stats"]["error_notes"])

    systems = {}
    calibration_scores, test_scores = train_and_predict(
        "B2",
        build_b_variant_features(train["base_features"], train["b_features"], train["b_ranking"], "B2"),
        build_b_variant_features(calibration["base_features"], calibration["b_features"], calibration["b_ranking"], "B2"),
        build_b_variant_features(test["base_features"], test["b_features"], test["b_ranking"], "B2"),
        train,
        args.seed,
        checkpoint_dir,
    )
    add_threshold_and_fdr_systems(
        systems,
        "B2",
        calibration_scores,
        test_scores,
        calibration,
        test,
        calibration_total,
        test_total,
        args.target_precision,
    )
    calibration_scores, test_scores = train_and_predict(
        "B2_motif",
        build_b2_motif_features(train),
        build_b2_motif_features(calibration),
        build_b2_motif_features(test),
        train,
        args.seed + 1,
        checkpoint_dir,
    )
    add_threshold_and_fdr_systems(
        systems,
        "B2_motif",
        calibration_scores,
        test_scores,
        calibration,
        test,
        calibration_total,
        test_total,
        args.target_precision,
    )
    if "c_features" in train:
        calibration_scores, test_scores = train_and_predict(
            "C2",
            build_c_variant_features(
                train["base_features"],
                train["b_features"],
                train["b_ranking"],
                train["c_features"],
                train["c_ranking"],
                "C2",
            ),
            build_c_variant_features(
                calibration["base_features"],
                calibration["b_features"],
                calibration["b_ranking"],
                calibration["c_features"],
                calibration["c_ranking"],
                "C2",
            ),
            build_c_variant_features(
                test["base_features"],
                test["b_features"],
                test["b_ranking"],
                test["c_features"],
                test["c_ranking"],
                "C2",
            ),
            train,
            args.seed + 2,
            checkpoint_dir,
        )
        add_threshold_and_fdr_systems(
            systems,
            "C2",
            calibration_scores,
            test_scores,
            calibration,
            test,
            calibration_total,
            test_total,
            args.target_precision,
        )
        calibration_scores, test_scores = train_and_predict(
            "C2_motif",
            build_c2_motif_features(train),
            build_c2_motif_features(calibration),
            build_c2_motif_features(test),
            train,
            args.seed + 3,
            checkpoint_dir,
        )
        add_threshold_and_fdr_systems(
            systems,
            "C2_motif",
            calibration_scores,
            test_scores,
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
        "motif": {
            **motif_kwargs,
            "feature_names": list(MOTIF_FEATURE_NAMES),
            "train_summary": motif_summary(train),
            "calibration_summary": motif_summary(calibration),
            "test_summary": motif_summary(test),
        },
        "systems": systems,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
