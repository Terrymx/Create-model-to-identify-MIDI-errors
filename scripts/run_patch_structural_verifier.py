from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from joblib import dump

from build_counterfactual_candidate_cache import load_candidate_cache
from e1_edit_energy_verifier import (
    E1Normalizer,
    build_e1_feature_tensors,
    evaluate_detection_with_external_correction,
    predict_e1,
    train_e1_model,
)
from patch_structural_features import (
    PATCH_STRUCTURAL_FEATURE_NAMES,
    compute_patch_structural_features,
)
from run_counterfactual_edit_verifier import evaluate_score_rows, make_small_leaf
from run_motif_repetition_verifier import (
    append_motif_features,
    build_c2_motif_features,
    motif_summary,
)
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
    parser.add_argument("--patch-radii", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--correction-weight", type=float, default=0.25)
    return parser.parse_args()


def load_split(cache_dir: Path, name: str) -> tuple[dict[str, np.ndarray], dict]:
    return load_candidate_cache(cache_dir / f"{name}.npz")


def _make_piece_dataset(args: argparse.Namespace, split: str) -> PieceConsistentVoiceDataset:
    return PieceConsistentVoiceDataset(
        root=args.data_root,
        split=split,
        voice_method="onset_matching",
        window_size=args.window_size,
        stride=args.stride,
        error_rate=args.error_rate,
        seed=args.seed,
        max_files=(
            args.max_validation_files if split == "validation" else args.max_test_files
        ),
        verbose=True,
    )


def append_patch_features(
    arrays: dict[str, np.ndarray],
    dataset: PieceConsistentVoiceDataset,
    edited_pitch: np.ndarray,
    radii: tuple[int, ...],
) -> dict[str, np.ndarray]:
    patch = np.zeros(
        (len(arrays["labels"]), len(PATCH_STRUCTURAL_FEATURE_NAMES) // 3 * len(radii)),
        dtype=np.float32,
    )
    file_ids = arrays["file_ids"].astype(np.int64)
    positions = arrays["positions"].astype(np.int64)
    for file_id in np.unique(file_ids):
        row_mask = file_ids == file_id
        patch[row_mask] = compute_patch_structural_features(
            piece_features=dataset._pieces[int(file_id)].features,
            candidate_positions=positions[row_mask],
            observed_pitch=arrays["observed_pitch"][row_mask],
            edited_pitch=edited_pitch[row_mask],
            radii=radii,
        )
    enriched = dict(arrays)
    enriched["patch_features"] = patch
    return enriched


def best_edited_pitch(arrays: dict[str, np.ndarray], proposal_scores: np.ndarray) -> np.ndarray:
    best = proposal_scores.argmax(axis=1)
    return arrays["proposals"][np.arange(len(best)), best].astype(np.float32)


def patch_summary(arrays: dict[str, np.ndarray]) -> dict:
    patch = arrays["patch_features"]
    labels = arrays["labels"].astype(bool)
    if len(patch) == 0:
        return {"candidate_rows": 0}
    return {
        "candidate_rows": int(len(patch)),
        "feature_count": int(patch.shape[1]),
        "mean_positive": float(patch[labels].mean()) if bool(labels.any()) else 0.0,
        "mean_negative": float(patch[~labels].mean()) if bool((~labels).any()) else 0.0,
        "std": float(patch.std()),
    }


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# P1 Patch Structural Verifier",
        "",
        f"- target precision: `{result['target_precision']:.2f}`",
        f"- patch radii: `{result['patch']['radii']}`",
        "",
        "| System | Precision | Recall | F1 | Replace Top-1 | Replace Top-3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, block in result["systems"].items():
        row = block["best_feasible_test"]["selected_test"]
        correction = row.get("correction", {})
        lines.append(
            f"| {name} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {correction.get('top1_accuracy', 0.0):.4f} | "
            f"{correction.get('top3_accuracy', 0.0):.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)
    train, train_meta = load_split(cache_dir, "train")
    calibration, calibration_meta = load_split(cache_dir, "calibration")
    test, test_meta = load_split(cache_dir, "test")
    validation_dataset = _make_piece_dataset(args, "validation")
    test_dataset = _make_piece_dataset(args, "test")
    motif_kwargs = {
        "radius": args.motif_radius,
        "min_similarity": args.motif_min_similarity,
        "exclude_radius": args.motif_exclude_radius,
    }
    train = append_motif_features(train, validation_dataset, **motif_kwargs)
    calibration = append_motif_features(calibration, validation_dataset, **motif_kwargs)
    test = append_motif_features(test, test_dataset, **motif_kwargs)
    train_x = build_c2_motif_features(train)
    calibration_x = build_c2_motif_features(calibration)
    test_x = build_c2_motif_features(test)
    calibration_total = int(calibration_meta["stats"]["error_notes"])
    test_total = int(test_meta["stats"]["error_notes"])
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    baseline = make_small_leaf(args.seed)
    baseline.fit(train_x, train["labels"].astype(np.int64))
    dump(baseline, checkpoint_dir / "c2_motif_hgb.joblib")
    calibration_baseline_scores = baseline.predict_proba(calibration_x)[:, 1]
    test_baseline_scores = baseline.predict_proba(test_x)[:, 1]

    raw_train_tensors = build_e1_feature_tensors(train, train_x)
    raw_calibration_tensors = build_e1_feature_tensors(calibration, calibration_x)
    raw_test_tensors = build_e1_feature_tensors(test, test_x)
    normalizer = E1Normalizer.fit(raw_train_tensors)
    train_tensors = normalizer.transform(raw_train_tensors)
    calibration_tensors = normalizer.transform(raw_calibration_tensors)
    test_tensors = normalizer.transform(raw_test_tensors)
    e1 = train_e1_model(
        train_tensors,
        train,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        correction_weight=args.correction_weight,
        device=device,
    )
    train_scores, train_logits, train_proposal_scores = predict_e1(
        e1, train_tensors, batch_size=args.batch_size, device=device
    )
    calibration_scores, calibration_logits, calibration_proposal_scores = predict_e1(
        e1, calibration_tensors, batch_size=args.batch_size, device=device
    )
    test_scores, test_logits, test_proposal_scores = predict_e1(
        e1, test_tensors, batch_size=args.batch_size, device=device
    )
    torch.save(
        {
            "model_state": e1.state_dict(),
            "normalizer": normalizer.to_jsonable(),
            "args": vars(args),
            "candidate_dim": train_tensors.candidate.shape[1],
            "proposal_dim": train_tensors.proposal.shape[2],
        },
        checkpoint_dir / "e1_selector.pt",
    )

    radii = tuple(int(value) for value in args.patch_radii)
    train = append_patch_features(
        train,
        validation_dataset,
        best_edited_pitch(train, train_proposal_scores),
        radii,
    )
    calibration = append_patch_features(
        calibration,
        validation_dataset,
        best_edited_pitch(calibration, calibration_proposal_scores),
        radii,
    )
    test = append_patch_features(
        test,
        test_dataset,
        best_edited_pitch(test, test_proposal_scores),
        radii,
    )
    train_patch_x = np.concatenate([train_x, train["patch_features"]], axis=1).astype(np.float32)
    calibration_patch_x = np.concatenate([calibration_x, calibration["patch_features"]], axis=1).astype(np.float32)
    test_patch_x = np.concatenate([test_x, test["patch_features"]], axis=1).astype(np.float32)

    systems = {
        "C2_motif_hgb": evaluate_detection_with_external_correction(
            calibration_baseline_scores,
            test_baseline_scores,
            test["b_ranking"],
            calibration,
            test,
            calibration_total,
            test_total,
            args.target_precision,
        ),
        "C2_motif_hgb_e1_correction": evaluate_detection_with_external_correction(
            calibration_baseline_scores,
            test_baseline_scores,
            test_proposal_scores,
            calibration,
            test,
            calibration_total,
            test_total,
            args.target_precision,
        ),
    }
    patch_model = make_small_leaf(args.seed + 10)
    patch_model.fit(train_patch_x, train["labels"].astype(np.int64))
    dump(patch_model, checkpoint_dir / "c2_motif_patch_hgb.joblib")
    systems["C2_motif_patch_hgb"] = evaluate_detection_with_external_correction(
        patch_model.predict_proba(calibration_patch_x)[:, 1],
        patch_model.predict_proba(test_patch_x)[:, 1],
        test_proposal_scores,
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
            "train_summary": motif_summary(train),
            "calibration_summary": motif_summary(calibration),
            "test_summary": motif_summary(test),
        },
        "patch": {
            "radii": list(radii),
            "feature_names": list(PATCH_STRUCTURAL_FEATURE_NAMES[: train["patch_features"].shape[1]]),
            "train_summary": patch_summary(train),
            "calibration_summary": patch_summary(calibration),
            "test_summary": patch_summary(test),
        },
        "e1": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "correction_weight": args.correction_weight,
            "device": str(device),
        },
        "systems": systems,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
