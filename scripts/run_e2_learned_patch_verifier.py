from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from joblib import dump

from build_counterfactual_candidate_cache import load_candidate_cache
from e1_edit_energy_verifier import evaluate_detection_with_external_correction
from e2_learned_patch_energy import (
    E2PatchNormalizer,
    E2PatchTensors,
    build_e2_hgb_features,
    build_patch_energy_tensors,
    predict_e2,
    train_e2_model,
)
from run_counterfactual_edit_verifier import make_small_leaf
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
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--max-validation-files", type=int, default=None)
    parser.add_argument("--max-test-files", type=int, default=None)
    parser.add_argument("--motif-radius", type=int, default=4)
    parser.add_argument("--motif-min-similarity", type=float, default=0.84)
    parser.add_argument("--motif-exclude-radius", type=int, default=16)
    parser.add_argument("--patch-radius", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--lr", type=float, default=0.0004)
    parser.add_argument("--correction-weight", type=float, default=0.35)
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


def build_split_patch_tensors(
    arrays: dict[str, np.ndarray],
    dataset: PieceConsistentVoiceDataset,
    candidate_features: np.ndarray,
    *,
    radius: int,
) -> E2PatchTensors:
    file_ids = arrays["file_ids"].astype(np.int64)
    positions = arrays["positions"].astype(np.int64)
    proposal_count = int(arrays["proposals"].shape[1])
    length = radius * 2 + 1
    patch_dim = int(dataset._pieces[0].features.shape[1]) + 3
    observed = np.zeros((len(file_ids), length, patch_dim), dtype=np.float32)
    edited = np.zeros((len(file_ids), proposal_count, length, patch_dim), dtype=np.float32)
    mask = np.zeros((len(file_ids), length), dtype=np.float32)

    for file_id in np.unique(file_ids):
        row_mask = file_ids == file_id
        rows = np.flatnonzero(row_mask)
        tensors = build_patch_energy_tensors(
            piece_features=dataset._pieces[int(file_id)].features,
            candidate_positions=positions[row_mask],
            observed_pitch=arrays["observed_pitch"][row_mask],
            proposals=arrays["proposals"][row_mask],
            radius=radius,
            candidate_features=candidate_features[row_mask],
        )
        observed[rows] = tensors.observed
        edited[rows] = tensors.edited
        mask[rows] = tensors.mask
    return E2PatchTensors(
        candidate=candidate_features.astype(np.float32),
        observed=observed,
        edited=edited,
        mask=mask,
    )


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# E2 Learned Patch Energy Verifier",
        "",
        f"- target precision: `{result['target_precision']:.2f}`",
        f"- patch radius: `{result['e2']['patch_radius']}`",
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

    raw_train_tensors = build_split_patch_tensors(
        train,
        validation_dataset,
        train_x,
        radius=args.patch_radius,
    )
    raw_calibration_tensors = build_split_patch_tensors(
        calibration,
        validation_dataset,
        calibration_x,
        radius=args.patch_radius,
    )
    raw_test_tensors = build_split_patch_tensors(
        test,
        test_dataset,
        test_x,
        radius=args.patch_radius,
    )
    normalizer = E2PatchNormalizer.fit(raw_train_tensors)
    train_tensors = normalizer.transform(raw_train_tensors)
    calibration_tensors = normalizer.transform(raw_calibration_tensors)
    test_tensors = normalizer.transform(raw_test_tensors)

    e2 = train_e2_model(
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
    train_scores, train_logits, train_proposal_scores = predict_e2(
        e2, train_tensors, batch_size=args.batch_size, device=device
    )
    calibration_scores, calibration_logits, calibration_proposal_scores = predict_e2(
        e2, calibration_tensors, batch_size=args.batch_size, device=device
    )
    test_scores, test_logits, test_proposal_scores = predict_e2(
        e2, test_tensors, batch_size=args.batch_size, device=device
    )
    torch.save(
        {
            "model_state": e2.state_dict(),
            "normalizer": normalizer.to_jsonable(),
            "args": vars(args),
            "candidate_dim": train_tensors.candidate.shape[1],
            "patch_feature_dim": train_tensors.observed.shape[2],
        },
        checkpoint_dir / "e2_patch_energy.pt",
    )

    train_e2_x = build_e2_hgb_features(train_x, train_scores, train_logits, train_proposal_scores)
    calibration_e2_x = build_e2_hgb_features(
        calibration_x,
        calibration_scores,
        calibration_logits,
        calibration_proposal_scores,
    )
    test_e2_x = build_e2_hgb_features(test_x, test_scores, test_logits, test_proposal_scores)
    e2_hgb = make_small_leaf(args.seed + 20)
    e2_hgb.fit(train_e2_x, train["labels"].astype(np.int64))
    dump(e2_hgb, checkpoint_dir / "c2_motif_e2_hgb.joblib")

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
        "C2_motif_hgb_e2_correction": evaluate_detection_with_external_correction(
            calibration_baseline_scores,
            test_baseline_scores,
            test_proposal_scores,
            calibration,
            test,
            calibration_total,
            test_total,
            args.target_precision,
        ),
        "E2_patch_energy": evaluate_detection_with_external_correction(
            calibration_scores,
            test_scores,
            test_proposal_scores,
            calibration,
            test,
            calibration_total,
            test_total,
            args.target_precision,
        ),
        "C2_motif_e2_hgb": evaluate_detection_with_external_correction(
            e2_hgb.predict_proba(calibration_e2_x)[:, 1],
            e2_hgb.predict_proba(test_e2_x)[:, 1],
            test_proposal_scores,
            calibration,
            test,
            calibration_total,
            test_total,
            args.target_precision,
        ),
    }
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
        "e2": {
            "patch_radius": args.patch_radius,
            "patch_length": args.patch_radius * 2 + 1,
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
