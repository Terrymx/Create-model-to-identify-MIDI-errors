from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from joblib import dump, load

from build_counterfactual_candidate_cache import load_candidate_cache
from e1_edit_energy_verifier import (
    E1EditEnergyNet,
    E1Normalizer,
    build_e1_feature_tensors,
    evaluate_detection_with_external_correction,
    predict_e1,
)
from proposal_trust_verifier import build_proposal_trust_features, evaluate_rescue_union
from proposal_trust_verifier import (
    build_high_confidence_slice_mask,
    mask_rescue_scores,
)
from run_counterfactual_edit_verifier import evaluate_score_rows, make_small_leaf
from motif_repetition_features import MOTIF_FEATURE_NAMES
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
    parser.add_argument("--c2-checkpoint-dir", required=True)
    parser.add_argument("--e1-checkpoint", required=True)
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
    parser.add_argument("--batch-size", type=int, default=512)
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


def _append_motif_to_splits(
    train: dict[str, np.ndarray],
    calibration: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    validation_dataset = _make_piece_dataset(args, "validation")
    test_dataset = _make_piece_dataset(args, "test")
    motif_kwargs = {
        "radius": args.motif_radius,
        "min_similarity": args.motif_min_similarity,
        "exclude_radius": args.motif_exclude_radius,
    }
    return (
        append_motif_features(train, validation_dataset, **motif_kwargs),
        append_motif_features(calibration, validation_dataset, **motif_kwargs),
        append_motif_features(test, test_dataset, **motif_kwargs),
    )


def _load_e1_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[E1EditEnergyNet, E1Normalizer]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    normalizer_payload = checkpoint["normalizer"]
    normalizer = E1Normalizer(
        candidate_mean=np.asarray(normalizer_payload["candidate_mean"], dtype=np.float32),
        candidate_std=np.asarray(normalizer_payload["candidate_std"], dtype=np.float32),
        proposal_mean=np.asarray(normalizer_payload["proposal_mean"], dtype=np.float32),
        proposal_std=np.asarray(normalizer_payload["proposal_std"], dtype=np.float32),
    )
    model = E1EditEnergyNet(
        candidate_dim=int(checkpoint["candidate_dim"]),
        proposal_dim=int(checkpoint["proposal_dim"]),
        hidden_dim=int(checkpoint["args"]["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, normalizer


def _score_e1(
    model: E1EditEnergyNet,
    normalizer: E1Normalizer,
    arrays: dict[str, np.ndarray],
    candidate_x: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = build_e1_feature_tensors(arrays, candidate_x)
    tensors = normalizer.transform(raw)
    return predict_e1(model, tensors, batch_size=batch_size, device=device)


def _write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Proposal-Selector Trust Verifier",
        "",
        f"- target precision: `{result['target_precision']:.2f}`",
        f"- motif radius: `{result['motif']['radius']}`",
        f"- motif min similarity: `{result['motif']['min_similarity']}`",
        f"- motif exclude radius: `{result['motif']['exclude_radius']}`",
        "",
        "| System | Precision | Recall | F1 | Rescued rows | Replace Top-1 | Replace Top-3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, block in result["systems"].items():
        row = block["best_feasible_test"]["selected_test"]
        correction = row.get("correction", {})
        lines.append(
            f"| {name} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row.get('rescued_rows', 0)} | "
            f"{correction.get('top1_accuracy', 0.0):.4f} | "
            f"{correction.get('top3_accuracy', 0.0):.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _best_narrow_rescue(
    *,
    calibration: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    calibration_c2_scores: np.ndarray,
    test_c2_scores: np.ndarray,
    calibration_e1_proposal_scores: np.ndarray,
    test_e1_proposal_scores: np.ndarray,
    calibration_rescue_scores: np.ndarray,
    test_rescue_scores: np.ndarray,
    calibration_total: int,
    test_total: int,
    target_precision: float,
) -> dict:
    motif_match_index = MOTIF_FEATURE_NAMES.index("motif_match_count")
    motif_gain_index = MOTIF_FEATURE_NAMES.index("proposal_consensus_gain")
    best_result: dict | None = None
    best_params: dict | None = None
    for c2_max in [0.20, 0.30, 0.40, 0.50, 0.60]:
        for e1_margin_min in [0.50, 1.00, 1.50, 2.00]:
            for e1_best_prob_min in [0.50, 0.60, 0.70, 0.80]:
                for require_b_agreement, require_c_agreement in [
                    (False, False),
                    (True, False),
                    (False, True),
                    (True, True),
                ]:
                    for motif_match_max, motif_gain_max in [(0.0, 0.0), (1.0, 0.0)]:
                        params = {
                            "c2_max": c2_max,
                            "e1_margin_min": e1_margin_min,
                            "e1_best_prob_min": e1_best_prob_min,
                            "require_b_agreement": require_b_agreement,
                            "require_c_agreement": require_c_agreement,
                            "motif_match_max": motif_match_max,
                            "motif_gain_max": motif_gain_max,
                        }
                        calibration_mask = build_high_confidence_slice_mask(
                            calibration,
                            calibration_c2_scores,
                            calibration_e1_proposal_scores,
                            motif_match_index=motif_match_index,
                            motif_gain_index=motif_gain_index,
                            **params,
                        )
                        test_mask = build_high_confidence_slice_mask(
                            test,
                            test_c2_scores,
                            test_e1_proposal_scores,
                            motif_match_index=motif_match_index,
                            motif_gain_index=motif_gain_index,
                            **params,
                        )
                        if int(calibration_mask.sum()) == 0:
                            continue
                        result = evaluate_rescue_union(
                            calibration_baseline_scores=calibration_c2_scores,
                            test_baseline_scores=test_c2_scores,
                            calibration_rescue_scores=mask_rescue_scores(
                                calibration_rescue_scores,
                                calibration_mask,
                            ),
                            test_rescue_scores=mask_rescue_scores(
                                test_rescue_scores,
                                test_mask,
                            ),
                            calibration=calibration,
                            test=test,
                            calibration_total_errors=calibration_total,
                            test_total_errors=test_total,
                            target_precision=target_precision,
                            test_proposal_scores=test_e1_proposal_scores,
                        )
                        row = result["best_feasible_test"]["selected_test"]
                        if row["precision"] < target_precision:
                            continue
                        key = (row["recall"], row["precision"], row["rescued_rows"])
                        if best_result is None:
                            best_result = result
                            best_params = {
                                **params,
                                "calibration_eligible": int(calibration_mask.sum()),
                                "test_eligible": int(test_mask.sum()),
                            }
                            continue
                        best_row = best_result["best_feasible_test"]["selected_test"]
                        best_key = (
                            best_row["recall"],
                            best_row["precision"],
                            best_row.get("rescued_rows", 0),
                        )
                        if key > best_key:
                            best_result = result
                            best_params = {
                                **params,
                                "calibration_eligible": int(calibration_mask.sum()),
                                "test_eligible": int(test_mask.sum()),
                            }
    if best_result is None:
        best_result = evaluate_rescue_union(
            calibration_baseline_scores=calibration_c2_scores,
            test_baseline_scores=test_c2_scores,
            calibration_rescue_scores=np.full_like(calibration_rescue_scores, -np.inf),
            test_rescue_scores=np.full_like(test_rescue_scores, -np.inf),
            calibration=calibration,
            test=test,
            calibration_total_errors=calibration_total,
            test_total_errors=test_total,
            target_precision=target_precision,
            test_proposal_scores=test_e1_proposal_scores,
        )
        best_params = {"calibration_eligible": 0, "test_eligible": 0}
    best_result = dict(best_result)
    best_result["slice_params"] = best_params
    return best_result


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)
    train, train_meta = load_split(cache_dir, "train")
    calibration, calibration_meta = load_split(cache_dir, "calibration")
    test, test_meta = load_split(cache_dir, "test")
    train, calibration, test = _append_motif_to_splits(train, calibration, test, args)
    train_x = build_c2_motif_features(train)
    calibration_x = build_c2_motif_features(calibration)
    test_x = build_c2_motif_features(test)
    calibration_total = int(calibration_meta["stats"]["error_notes"])
    test_total = int(test_meta["stats"]["error_notes"])
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    c2_model = load(Path(args.c2_checkpoint_dir) / "c2_motif_hgb.joblib")
    train_c2_scores = c2_model.predict_proba(train_x)[:, 1]
    calibration_c2_scores = c2_model.predict_proba(calibration_x)[:, 1]
    test_c2_scores = c2_model.predict_proba(test_x)[:, 1]

    e1_model, e1_normalizer = _load_e1_model(Path(args.e1_checkpoint), device=device)
    _train_e1_scores, _train_e1_logits, train_e1_proposal_scores = _score_e1(
        e1_model,
        e1_normalizer,
        train,
        train_x,
        batch_size=args.batch_size,
        device=device,
    )
    _calibration_e1_scores, _calibration_e1_logits, calibration_e1_proposal_scores = _score_e1(
        e1_model,
        e1_normalizer,
        calibration,
        calibration_x,
        batch_size=args.batch_size,
        device=device,
    )
    _test_e1_scores, _test_e1_logits, test_e1_proposal_scores = _score_e1(
        e1_model,
        e1_normalizer,
        test,
        test_x,
        batch_size=args.batch_size,
        device=device,
    )

    train_trust = build_proposal_trust_features(train, train_c2_scores, train_e1_proposal_scores)
    calibration_trust = build_proposal_trust_features(
        calibration,
        calibration_c2_scores,
        calibration_e1_proposal_scores,
    )
    test_trust = build_proposal_trust_features(test, test_c2_scores, test_e1_proposal_scores)
    train_full = np.concatenate([train_x, train_trust], axis=1).astype(np.float32)
    calibration_full = np.concatenate([calibration_x, calibration_trust], axis=1).astype(np.float32)
    test_full = np.concatenate([test_x, test_trust], axis=1).astype(np.float32)

    systems = {}
    systems["C2_motif_hgb"] = evaluate_detection_with_external_correction(
        calibration_c2_scores,
        test_c2_scores,
        test_e1_proposal_scores,
        calibration,
        test,
        calibration_total,
        test_total,
        args.target_precision,
    )

    trust_model = make_small_leaf(args.seed + 11)
    trust_model.fit(train_trust, train["labels"].astype(np.int64))
    dump(trust_model, checkpoint_dir / "proposal_trust_hgb.joblib")
    calibration_trust_scores = trust_model.predict_proba(calibration_trust)[:, 1]
    test_trust_scores = trust_model.predict_proba(test_trust)[:, 1]
    systems["proposal_trust_hgb"] = evaluate_detection_with_external_correction(
        calibration_trust_scores,
        test_trust_scores,
        test_e1_proposal_scores,
        calibration,
        test,
        calibration_total,
        test_total,
        args.target_precision,
    )
    systems["C2_motif_proposal_trust_rescue"] = evaluate_rescue_union(
        calibration_baseline_scores=calibration_c2_scores,
        test_baseline_scores=test_c2_scores,
        calibration_rescue_scores=calibration_trust_scores,
        test_rescue_scores=test_trust_scores,
        calibration=calibration,
        test=test,
        calibration_total_errors=calibration_total,
        test_total_errors=test_total,
        target_precision=args.target_precision,
        test_proposal_scores=test_e1_proposal_scores,
    )
    systems["C2_motif_narrow_trust_rescue"] = _best_narrow_rescue(
        calibration=calibration,
        test=test,
        calibration_c2_scores=calibration_c2_scores,
        test_c2_scores=test_c2_scores,
        calibration_e1_proposal_scores=calibration_e1_proposal_scores,
        test_e1_proposal_scores=test_e1_proposal_scores,
        calibration_rescue_scores=calibration_trust_scores,
        test_rescue_scores=test_trust_scores,
        calibration_total=calibration_total,
        test_total=test_total,
        target_precision=args.target_precision,
    )

    full_model = make_small_leaf(args.seed + 12)
    full_model.fit(train_full, train["labels"].astype(np.int64))
    dump(full_model, checkpoint_dir / "c2_motif_proposal_trust_hgb.joblib")
    calibration_full_scores = full_model.predict_proba(calibration_full)[:, 1]
    test_full_scores = full_model.predict_proba(test_full)[:, 1]
    systems["C2_motif_proposal_trust_hgb"] = evaluate_detection_with_external_correction(
        calibration_full_scores,
        test_full_scores,
        test_e1_proposal_scores,
        calibration,
        test,
        calibration_total,
        test_total,
        args.target_precision,
    )
    systems["C2_motif_full_trust_rescue"] = evaluate_rescue_union(
        calibration_baseline_scores=calibration_c2_scores,
        test_baseline_scores=test_c2_scores,
        calibration_rescue_scores=calibration_full_scores,
        test_rescue_scores=test_full_scores,
        calibration=calibration,
        test=test,
        calibration_total_errors=calibration_total,
        test_total_errors=test_total,
        target_precision=args.target_precision,
        test_proposal_scores=test_e1_proposal_scores,
    )
    systems["C2_motif_narrow_full_rescue"] = _best_narrow_rescue(
        calibration=calibration,
        test=test,
        calibration_c2_scores=calibration_c2_scores,
        test_c2_scores=test_c2_scores,
        calibration_e1_proposal_scores=calibration_e1_proposal_scores,
        test_e1_proposal_scores=test_e1_proposal_scores,
        calibration_rescue_scores=calibration_full_scores,
        test_rescue_scores=test_full_scores,
        calibration_total=calibration_total,
        test_total=test_total,
        target_precision=args.target_precision,
    )

    result = {
        "target_precision": args.target_precision,
        "train_stats": train_meta["stats"],
        "calibration_stats": calibration_meta["stats"],
        "test_stats": test_meta["stats"],
        "motif": {
            "radius": args.motif_radius,
            "min_similarity": args.motif_min_similarity,
            "exclude_radius": args.motif_exclude_radius,
            "train_summary": motif_summary(train),
            "calibration_summary": motif_summary(calibration),
            "test_summary": motif_summary(test),
        },
        "proposal_trust": {
            "feature_count": int(train_trust.shape[1]),
            "full_feature_count": int(train_full.shape[1]),
            "device": str(device),
        },
        "systems": systems,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_markdown(Path(args.output_md), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
