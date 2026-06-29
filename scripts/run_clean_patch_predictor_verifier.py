from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from joblib import dump
from torch.utils.data import DataLoader, TensorDataset

from build_counterfactual_candidate_cache import load_candidate_cache
from clean_patch_predictor import (
    CleanPatchBatch,
    CleanPatchPredictor,
    DenoisingPatchBatch,
    build_candidate_patch_batch,
    build_clean_patch_batch,
    build_denoising_patch_batch,
    build_patch_predictor_features,
    patch_denoising_loss,
    patch_negative_log_likelihood,
)
from e1_edit_energy_verifier import evaluate_detection_with_external_correction
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
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--max-validation-files", type=int, default=None)
    parser.add_argument("--max-test-files", type=int, default=None)
    parser.add_argument("--motif-radius", type=int, default=4)
    parser.add_argument("--motif-min-similarity", type=float, default=0.84)
    parser.add_argument("--motif-exclude-radius", type=int, default=16)
    parser.add_argument("--patch-radius", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--lr", type=float, default=0.0006)
    parser.add_argument("--max-clean-patches", type=int, default=240000)
    parser.add_argument(
        "--training-mode",
        choices=["clean", "denoising"],
        default="clean",
    )
    parser.add_argument("--contrastive-weight", type=float, default=0.75)
    parser.add_argument("--contrastive-margin", type=float, default=1.0)
    return parser.parse_args()


def load_split(cache_dir: Path, name: str) -> tuple[dict[str, np.ndarray], dict]:
    return load_candidate_cache(cache_dir / f"{name}.npz")


def make_piece_dataset(
    args: argparse.Namespace,
    split: str,
    *,
    error_rate: float,
) -> PieceConsistentVoiceDataset:
    return PieceConsistentVoiceDataset(
        root=args.data_root,
        split=split,
        voice_method="onset_matching",
        window_size=args.window_size,
        stride=args.stride,
        error_rate=error_rate,
        seed=args.seed,
        max_files=(args.max_validation_files if split == "validation" else args.max_test_files),
        verbose=True,
    )


def collect_clean_patch_batch(
    dataset: PieceConsistentVoiceDataset,
    *,
    radius: int,
    max_patches: int,
    seed: int,
) -> CleanPatchBatch:
    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int]] = []
    for file_id, piece in enumerate(dataset._pieces):
        pairs.extend((file_id, position) for position in range(len(piece.features)))
    if max_patches > 0 and len(pairs) > max_patches:
        selected = rng.choice(len(pairs), size=max_patches, replace=False)
        pairs = [pairs[int(index)] for index in selected]

    contexts: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for file_id in sorted({file_id for file_id, _ in pairs}):
        positions = np.asarray(
            [position for pair_file_id, position in pairs if pair_file_id == file_id],
            dtype=np.int64,
        )
        if len(positions) == 0:
            continue
        batch = build_clean_patch_batch(
            piece_features=dataset._pieces[file_id].features,
            center_positions=positions,
            radius=radius,
        )
        contexts.append(batch.context)
        masks.append(batch.mask)
        targets.append(batch.target_pitch)
    return CleanPatchBatch(
        context=np.concatenate(contexts, axis=0),
        mask=np.concatenate(masks, axis=0),
        target_pitch=np.concatenate(targets, axis=0),
    )


def collect_denoising_candidate_patch_batch(
    arrays: dict[str, np.ndarray],
    dataset: PieceConsistentVoiceDataset,
    *,
    radius: int,
) -> DenoisingPatchBatch:
    contexts: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    negative_masks: list[np.ndarray] = []
    file_ids = arrays["file_ids"].astype(np.int64)
    positions = arrays["positions"].astype(np.int64)
    for file_id in np.unique(file_ids):
        row_mask = file_ids == file_id
        batch = build_denoising_patch_batch(
            piece_features=dataset._pieces[int(file_id)].features,
            center_positions=positions[row_mask],
            target_pitch=arrays["target_pitch"][row_mask],
            observed_pitch=arrays["observed_pitch"][row_mask],
            radius=radius,
        )
        contexts.append(batch.context)
        masks.append(batch.mask)
        targets.append(batch.target_pitch)
        negatives.append(batch.negative_pitch)
        negative_masks.append(batch.negative_mask)
    return DenoisingPatchBatch(
        context=np.concatenate(contexts, axis=0),
        mask=np.concatenate(masks, axis=0),
        target_pitch=np.concatenate(targets, axis=0),
        negative_pitch=np.concatenate(negatives, axis=0),
        negative_mask=np.concatenate(negative_masks, axis=0),
    )


def train_patch_predictor(
    batch: CleanPatchBatch | DenoisingPatchBatch,
    *,
    seed: int,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    contrastive_weight: float,
    contrastive_margin: float,
    device: torch.device,
) -> CleanPatchPredictor:
    torch.manual_seed(seed)
    model = CleanPatchPredictor(
        patch_feature_dim=batch.context.shape[2],
        hidden_dim=hidden_dim,
    ).to(device)
    if isinstance(batch, DenoisingPatchBatch):
        dataset = TensorDataset(
            torch.from_numpy(batch.context).float(),
            torch.from_numpy(batch.mask).float(),
            torch.from_numpy(batch.target_pitch).long(),
            torch.from_numpy(batch.negative_pitch).long(),
            torch.from_numpy(batch.negative_mask).float(),
        )
    else:
        dataset = TensorDataset(
            torch.from_numpy(batch.context).float(),
            torch.from_numpy(batch.mask).float(),
            torch.from_numpy(batch.target_pitch).long(),
        )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_contrastive_rows = 0
        total_rows = 0
        for tensors in loader:
            context, mask, target = tensors[:3]
            context = context.to(device)
            mask = mask.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(context, mask)
            if len(tensors) == 5:
                negative = tensors[3].to(device)
                negative_mask = tensors[4].to(device)
                loss = patch_denoising_loss(
                    logits,
                    target,
                    negative,
                    negative_mask,
                    contrastive_weight=contrastive_weight,
                    margin=contrastive_margin,
                )
                total_contrastive_rows += int((negative_mask > 0.5).sum().detach().cpu())
            else:
                loss = patch_negative_log_likelihood(logits, target).mean()
            loss.backward()
            optimizer.step()
            rows = int(context.shape[0])
            total_loss += float(loss.detach().cpu()) * rows
            total_rows += rows
        print(
            f"epoch={epoch + 1}/{epochs} patch_loss={total_loss / max(total_rows, 1):.6f} "
            f"contrastive_rows={total_contrastive_rows}",
            flush=True,
        )
    return model


@torch.no_grad()
def _score_batch(
    model: CleanPatchPredictor,
    batch: CleanPatchBatch,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    for start in range(0, len(batch.context), batch_size):
        end = min(start + batch_size, len(batch.context))
        context = torch.from_numpy(batch.context[start:end]).float().to(device)
        mask = torch.from_numpy(batch.mask[start:end]).float().to(device)
        target = torch.from_numpy(batch.target_pitch[start:end]).long().to(device)
        logits = model(context, mask)
        energy = patch_negative_log_likelihood(logits, target)
        scores.append(energy.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(scores, axis=0)


def score_candidate_patch_features(
    arrays: dict[str, np.ndarray],
    dataset: PieceConsistentVoiceDataset,
    model: CleanPatchPredictor,
    *,
    radius: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    file_ids = arrays["file_ids"].astype(np.int64)
    positions = arrays["positions"].astype(np.int64)
    proposal_count = int(arrays["proposals"].shape[1])
    observed_energy = np.zeros(len(file_ids), dtype=np.float32)
    proposal_energy = np.zeros((len(file_ids), proposal_count), dtype=np.float32)
    proposal_mask = np.zeros((len(file_ids), proposal_count), dtype=np.float32)

    for file_id in np.unique(file_ids):
        row_mask = file_ids == file_id
        rows = np.flatnonzero(row_mask)
        observed, edited, mask = build_candidate_patch_batch(
            piece_features=dataset._pieces[int(file_id)].features,
            candidate_positions=positions[row_mask],
            observed_pitch=arrays["observed_pitch"][row_mask],
            proposals=arrays["proposals"][row_mask],
            radius=radius,
        )
        observed_energy[rows] = _score_batch(
            model,
            observed,
            batch_size=batch_size,
            device=device,
        )
        edited_energy = _score_batch(
            model,
            edited,
            batch_size=batch_size,
            device=device,
        ).reshape(len(rows), proposal_count)
        proposal_energy[rows] = edited_energy
        proposal_mask[rows] = mask
    features = build_patch_predictor_features(
        observed_energy=observed_energy,
        proposal_energy=proposal_energy,
        proposal_mask=proposal_mask,
    )
    proposal_scores = -proposal_energy
    return features, proposal_scores


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Clean-MIDI Patch Predictor Verifier",
        "",
        f"- target precision: `{result['target_precision']:.2f}`",
        f"- patch radius: `{result['patch_predictor']['patch_radius']}`",
        f"- training mode: `{result['patch_predictor']['training_mode']}`",
        f"- pretraining patches: `{result['patch_predictor']['pretraining_patches']}`",
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

    print("building validation verifier dataset", flush=True)
    validation_dataset = make_piece_dataset(args, "validation", error_rate=args.error_rate)
    if args.training_mode == "denoising":
        print("building denoising candidate pretraining dataset", flush=True)
        pretraining_patches = collect_denoising_candidate_patch_batch(
            train,
            validation_dataset,
            radius=args.patch_radius,
        )
        contrastive_rows = int((pretraining_patches.negative_mask > 0.5).sum())
        print(
            f"denoising candidate patches: {len(pretraining_patches.context)} "
            f"contrastive_rows={contrastive_rows}",
            flush=True,
        )
    else:
        print("building clean pretraining dataset", flush=True)
        clean_validation_dataset = make_piece_dataset(args, "validation", error_rate=0.0)
        pretraining_patches = collect_clean_patch_batch(
            clean_validation_dataset,
            radius=args.patch_radius,
            max_patches=args.max_clean_patches,
            seed=args.seed,
        )
        print(f"clean patches: {len(pretraining_patches.context)}", flush=True)

    patch_model = train_patch_predictor(
        pretraining_patches,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        contrastive_weight=args.contrastive_weight,
        contrastive_margin=args.contrastive_margin,
        device=device,
    )

    print("building test verifier dataset", flush=True)
    test_dataset = make_piece_dataset(args, "test", error_rate=args.error_rate)
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

    print("scoring patch predictor features", flush=True)
    train_patch_x, train_proposal_scores = score_candidate_patch_features(
        train,
        validation_dataset,
        patch_model,
        radius=args.patch_radius,
        batch_size=args.batch_size,
        device=device,
    )
    calibration_patch_x, calibration_proposal_scores = score_candidate_patch_features(
        calibration,
        validation_dataset,
        patch_model,
        radius=args.patch_radius,
        batch_size=args.batch_size,
        device=device,
    )
    test_patch_x, test_proposal_scores = score_candidate_patch_features(
        test,
        test_dataset,
        patch_model,
        radius=args.patch_radius,
        batch_size=args.batch_size,
        device=device,
    )

    calibration_total = int(calibration_meta["stats"]["error_notes"])
    test_total = int(test_meta["stats"]["error_notes"])
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    baseline = make_small_leaf(args.seed)
    baseline.fit(train_x, train["labels"].astype(np.int64))
    dump(baseline, checkpoint_dir / "c2_motif_hgb.joblib")
    patch_hgb = make_small_leaf(args.seed + 30)
    patch_hgb.fit(
        np.concatenate([train_x, train_patch_x], axis=1),
        train["labels"].astype(np.int64),
    )
    dump(patch_hgb, checkpoint_dir / "c2_motif_clean_patch_hgb.joblib")
    torch.save(
        {
            "model_state": patch_model.state_dict(),
            "args": vars(args),
            "patch_feature_dim": pretraining_patches.context.shape[2],
        },
        checkpoint_dir / "clean_patch_predictor.pt",
    )

    direct_calibration_scores = calibration_patch_x[:, 3]
    direct_test_scores = test_patch_x[:, 3]
    systems = {
        "C2_motif_hgb": evaluate_detection_with_external_correction(
            baseline.predict_proba(calibration_x)[:, 1],
            baseline.predict_proba(test_x)[:, 1],
            test["b_ranking"],
            calibration,
            test,
            calibration_total,
            test_total,
            args.target_precision,
        ),
        "PatchPredictor_direct": evaluate_detection_with_external_correction(
            direct_calibration_scores,
            direct_test_scores,
            test_proposal_scores,
            calibration,
            test,
            calibration_total,
            test_total,
            args.target_precision,
        ),
        "C2_motif_clean_patch_hgb": evaluate_detection_with_external_correction(
            patch_hgb.predict_proba(np.concatenate([calibration_x, calibration_patch_x], axis=1))[:, 1],
            patch_hgb.predict_proba(np.concatenate([test_x, test_patch_x], axis=1))[:, 1],
            test_proposal_scores,
            calibration,
            test,
            calibration_total,
            test_total,
            args.target_precision,
        ),
        "C2_motif_clean_patch_correction": evaluate_detection_with_external_correction(
            baseline.predict_proba(calibration_x)[:, 1],
            baseline.predict_proba(test_x)[:, 1],
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
        "patch_predictor": {
            "patch_radius": args.patch_radius,
            "patch_length": args.patch_radius * 2 + 1,
            "training_mode": args.training_mode,
            "pretraining_patches": int(len(pretraining_patches.context)),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "max_clean_patches": args.max_clean_patches,
            "contrastive_weight": args.contrastive_weight,
            "contrastive_margin": args.contrastive_margin,
            "device": str(device),
            "feature_names": [
                "observed_energy",
                "best_edited_energy",
                "mean_edited_energy",
                "best_gain",
                "mean_gain",
                "proposal_margin",
                "valid_proposal_count",
                "any_improved",
            ],
        },
        "systems": systems,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
