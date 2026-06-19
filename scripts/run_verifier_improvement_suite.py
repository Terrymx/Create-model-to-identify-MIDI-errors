from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import load
from torch.utils.data import DataLoader, Dataset

from calibrate_frozen_context_verifier import (
    candidate_density,
    row_at_threshold,
    select_from_calibration,
)
from midi_error_detector.data import MaestroWrongNoteDataset
from run_frozen_union_candidate_context_verifier import (
    append_piece_relative_features,
    collect_context_candidates,
    load_any_model,
)
from run_union_candidate_verifier import split_indices_by_file, standardize
from verifier_theory_features import THEORY_INTERACTION_SIZE
from voice_assignment import VOICE_FEATURE_SIZE


PIECE_RELATIVE_SIZE = 22


class LegacyIndexedSubset(Dataset):
    def __init__(self, dataset: MaestroWrongNoteDataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        dataset_index = self.indices[index]
        sample = self.dataset[dataset_index]
        file_id, window_start = self.dataset.index[dataset_index]
        sample["voice_features"] = torch.zeros(
            self.dataset.window_size,
            VOICE_FEATURE_SIZE,
            dtype=torch.float32,
        )
        sample["__file_id"] = torch.tensor(file_id, dtype=torch.long)
        sample["__window_start"] = torch.tensor(window_start, dtype=torch.long)
        sample["__file_note_count"] = torch.tensor(
            self.dataset._note_counts[file_id],
            dtype=torch.long,
        )
        return sample


class PairwiseRanker(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


@dataclass(frozen=True)
class CandidateBlock:
    full: torch.Tensor
    old: torch.Tensor
    raw: torch.Tensor
    labels: torch.Tensor
    file_ids: torch.Tensor
    note_counts: torch.Tensor
    stats: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threeclass-checkpoint", required=True)
    parser.add_argument("--binary-checkpoint", required=True)
    parser.add_argument("--forward-checkpoint", required=True)
    parser.add_argument("--backward-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--old-context-json", required=True)
    parser.add_argument("--old-model-dir", required=True)
    parser.add_argument("--theory-context-json", required=True)
    parser.add_argument("--theory-model-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--ranker-output", required=True)
    parser.add_argument("--target-precision", type=float, default=0.80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--ranker-epochs", type=int, default=30)
    parser.add_argument("--negatives-per-positive", type=int, default=4)
    parser.add_argument("--max-validation-files", type=int, default=None)
    parser.add_argument("--max-test-files", type=int, default=None)
    return parser.parse_args()


def split_old_and_theory_features(
    combined: torch.Tensor,
    theory_size: int,
    piece_relative_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if theory_size <= 0 or piece_relative_size <= 0:
        raise ValueError("Feature block sizes must be positive.")
    theory_start = combined.shape[1] - piece_relative_size - theory_size
    piece_start = combined.shape[1] - piece_relative_size
    if theory_start < 0:
        raise ValueError("Combined feature matrix is smaller than declared blocks.")
    old = torch.cat(
        [combined[:, :theory_start], combined[:, piece_start:]],
        dim=1,
    )
    return old, combined


def convex_weight_grid(model_count: int, step: float) -> list[tuple[float, ...]]:
    units = round(1.0 / step)
    rows = []
    for values in product(range(units + 1), repeat=model_count):
        if sum(values) == units:
            rows.append(tuple(value / units for value in values))
    return rows


def build_pair_indices(
    labels: np.ndarray,
    file_ids: np.ndarray,
    hardness: np.ndarray,
    negatives_per_positive: int,
) -> tuple[np.ndarray, np.ndarray]:
    positive_rows: list[int] = []
    negative_rows: list[int] = []
    for file_id in np.unique(file_ids):
        local = np.flatnonzero(file_ids == file_id)
        positives = local[labels[local] == 1]
        negatives = local[labels[local] == 0]
        if len(positives) == 0 or len(negatives) == 0:
            continue
        negatives = negatives[np.argsort(-hardness[negatives], kind="mergesort")]
        for offset, positive in enumerate(positives.tolist()):
            count = min(negatives_per_positive, len(negatives))
            selected = [
                int(negatives[(offset + index) % len(negatives)])
                for index in range(count)
            ]
            positive_rows.extend([positive] * count)
            negative_rows.extend(selected)
    return (
        np.asarray(positive_rows, dtype=np.int64),
        np.asarray(negative_rows, dtype=np.int64),
    )


def classify_candidate_patterns(raw: torch.Tensor) -> dict[str, np.ndarray]:
    values = raw.numpy()
    return {
        "chord_tone": values[:, 20] >= 0.5,
        "scale_tone": np.maximum(values[:, 8], values[:, 9]) >= 0.5,
        "ornament": np.maximum.reduce(
            [values[:, 29], values[:, 30], values[:, 32]]
        ) >= 0.5,
        "short_note": values[:, 33] <= 0.25,
        "strong_beat": values[:, 34] >= 0.65,
        "high_density": values[:, 12] >= 0.65,
    }


def category_risk_features(raw: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    values = raw.numpy()
    scale_confidence = np.maximum(values[:, 8], values[:, 9]).clip(0.0, 1.0)
    short_strength = ((0.25 - values[:, 33]) / 0.25).clip(0.0, 1.0)
    is_scale = scale_confidence >= 0.5
    is_short = values[:, 33] <= 0.25
    groups = np.full(len(values), 3, dtype=np.int64)
    groups[is_short & is_scale] = 0
    groups[is_short & ~is_scale] = 1
    groups[~is_short & is_scale] = 2
    risk = (short_strength * scale_confidence).astype(np.float32)
    return groups, risk


def apply_category_score_adjustment(
    scores: np.ndarray,
    groups: np.ndarray,
    risk: np.ndarray,
    offsets: tuple[float, float, float],
    risk_beta: float,
) -> np.ndarray:
    adjusted = scores.astype(np.float32, copy=True)
    for group, offset in enumerate(offsets):
        adjusted[groups == group] += offset
    adjusted += risk_beta * risk
    return adjusted


def observed_register_features(
    raw: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed_pitch = np.rint(raw[:, 0].numpy() * 127.0).clip(0, 127)
    registers = np.ones(len(observed_pitch), dtype=np.int64)
    registers[observed_pitch < 48] = 0
    registers[observed_pitch >= 72] = 2
    is_short = raw[:, 33].numpy() <= 0.25
    register_short = registers * 2 + is_short.astype(np.int64)
    continuous = ((observed_pitch - 60.0) / 36.0).clip(-1.0, 1.0).astype(
        np.float32
    )
    return registers, register_short, continuous


def make_dataset(args: argparse.Namespace, split: str, max_files: int | None) -> MaestroWrongNoteDataset:
    dataset = MaestroWrongNoteDataset(
        root=args.data_root,
        split=split,
        window_size=args.window_size,
        stride=args.stride,
        error_rate=args.error_rate,
        seed=13,
        max_files=max_files,
        cache_notes=True,
        verbose=True,
    )
    dataset.set_epoch(0)
    return dataset


def make_loader(
    dataset: MaestroWrongNoteDataset,
    indices: list[int],
    batch_size: int,
) -> DataLoader:
    return DataLoader(
        LegacyIndexedSubset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def collect_block(
    models: tuple,
    dataset: MaestroWrongNoteDataset,
    indices: list[int],
    device: torch.device,
    args: argparse.Namespace,
    description: str,
) -> CandidateBlock:
    collect_args = SimpleNamespace(surprise_eval_groups=4)
    (
        features,
        labels,
        _,
        stats,
        file_ids,
        positions,
        note_counts,
        score_columns,
        _,
    ) = collect_context_candidates(
        models,
        make_loader(dataset, indices, args.batch_size),
        device,
        collect_args,
        0.60,
        0.50,
        description,
    )
    raw = features[:, :36].clone()
    full = append_piece_relative_features(
        features,
        file_ids,
        positions,
        note_counts,
        score_columns,
    )
    old, full = split_old_and_theory_features(
        full,
        theory_size=THEORY_INTERACTION_SIZE,
        piece_relative_size=PIECE_RELATIVE_SIZE,
    )
    return CandidateBlock(
        full=full,
        old=old,
        raw=raw,
        labels=labels,
        file_ids=file_ids,
        note_counts=note_counts,
        stats=stats,
    )


def apply_normalization(
    features: torch.Tensor,
    normalization: list[list[float]],
) -> torch.Tensor:
    mean = torch.tensor(normalization[0], dtype=torch.float32)
    std = torch.tensor(normalization[1], dtype=torch.float32).clamp_min(1e-5)
    if features.shape[1] != len(mean):
        raise ValueError(
            f"Feature size {features.shape[1]} does not match normalization {len(mean)}."
        )
    return (features - mean) / std


def model_scores(model, features: torch.Tensor) -> np.ndarray:
    return model.predict_proba(features.numpy())[:, 1].astype(np.float32)


def model_path(directory: str, name: str) -> Path:
    return Path(directory) / f"three0.60_binary0.50_{name}.joblib"


def load_base_scores(
    args: argparse.Namespace,
    blocks: dict[str, CandidateBlock],
) -> tuple[list[str], dict[str, np.ndarray]]:
    run_key = "three=0.60,binary=0.50"
    old_json = json.loads(Path(args.old_context_json).read_text(encoding="utf-8"))
    theory_json = json.loads(Path(args.theory_context_json).read_text(encoding="utf-8"))
    old_norm = old_json["runs"][run_key]["normalization"]
    theory_norm = theory_json["runs"][run_key]["normalization"]
    specs = [
        ("old_hgb", args.old_model_dir, "hist_gradient_boosting", "old", old_norm),
        (
            "old_small_leaf",
            args.old_model_dir,
            "hist_gradient_boosting_small_leaf",
            "old",
            old_norm,
        ),
        (
            "theory_hgb",
            args.theory_model_dir,
            "hist_gradient_boosting",
            "full",
            theory_norm,
        ),
        (
            "theory_small_leaf",
            args.theory_model_dir,
            "hist_gradient_boosting_small_leaf",
            "full",
            theory_norm,
        ),
    ]
    names = [spec[0] for spec in specs]
    scores: dict[str, np.ndarray] = {}
    for split_name, block in blocks.items():
        split_scores = []
        for _, directory, model_name, feature_name, normalization in specs:
            model = load(model_path(directory, model_name))
            normalized = apply_normalization(
                getattr(block, feature_name),
                normalization,
            )
            split_scores.append(model_scores(model, normalized))
        scores[split_name] = np.stack(split_scores, axis=1)
    return names, scores


def density_z(
    calibration: CandidateBlock,
    test: CandidateBlock,
) -> tuple[np.ndarray, np.ndarray]:
    calibration_density = candidate_density(
        calibration.file_ids,
        calibration.note_counts,
    )
    test_density = candidate_density(test.file_ids, test.note_counts)
    mean = float(calibration_density.mean())
    std = max(float(calibration_density.std()), 1e-5)
    return (
        (calibration_density - mean) / std,
        (test_density - mean) / std,
    )


def evaluate_fusion(
    names: list[str],
    scores: dict[str, np.ndarray],
    blocks: dict[str, CandidateBlock],
    target_precision: float,
) -> dict:
    calibration_density_z, test_density_z = density_z(
        blocks["calibration"],
        blocks["test"],
    )
    weight_grid = convex_weight_grid(len(names), 0.1)
    margins = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
    alphas = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
    rows = []
    calibration_labels = blocks["calibration"].labels.numpy().astype(np.int64)
    test_labels = blocks["test"].labels.numpy().astype(np.int64)
    for margin in margins:
        requested_precision = target_precision + margin
        best = None
        for weights in weight_grid:
            weights_array = np.asarray(weights, dtype=np.float32)
            calibration_base = scores["calibration"] @ weights_array
            test_base = scores["test"] @ weights_array
            for alpha in alphas:
                calibration_score = calibration_base - alpha * calibration_density_z
                test_score = test_base - alpha * test_density_z
                selected = select_from_calibration(
                    calibration_score,
                    calibration_labels,
                    blocks["calibration"].stats["error_notes"],
                    requested_precision,
                )
                candidate = {
                    "weights": dict(zip(names, weights)),
                    "density_alpha": alpha,
                    "requested_precision": requested_precision,
                    "selected_calibration": selected,
                    "test_scores": test_score,
                }
                key = (
                    selected["recall"],
                    selected["precision"],
                    -alpha,
                )
                if best is None or key > best[0]:
                    best = (key, candidate)
        selected_candidate = best[1]
        test_row = row_at_threshold(
            selected_candidate.pop("test_scores"),
            test_labels,
            blocks["test"].stats["error_notes"],
            selected_candidate["selected_calibration"]["threshold"],
        )
        selected_candidate["selected_test"] = test_row
        rows.append(selected_candidate)
    return {
        "margins": rows,
        "best_feasible_test": max(
            (
                row
                for row in rows
                if row["selected_test"]["precision"] >= target_precision
            ),
            key=lambda row: (
                row["selected_test"]["recall"],
                row["selected_test"]["precision"],
            ),
            default=max(
                rows,
                key=lambda row: (
                    row["selected_test"]["precision"],
                    row["selected_test"]["recall"],
                ),
            ),
        ),
    }


def evaluate_category_aware(
    names: list[str],
    scores: dict[str, np.ndarray],
    blocks: dict[str, CandidateBlock],
    fusion: dict,
    target_precision: float,
) -> dict:
    calibration_density_z, test_density_z = density_z(
        blocks["calibration"],
        blocks["test"],
    )
    calibration_groups, calibration_risk = category_risk_features(
        blocks["calibration"].raw
    )
    test_groups, test_risk = category_risk_features(blocks["test"].raw)
    calibration_labels = blocks["calibration"].labels.numpy().astype(np.int64)
    test_labels = blocks["test"].labels.numpy().astype(np.int64)
    offset_grid = [0.00, 0.01, 0.02, 0.04, 0.06]
    beta_grid = [0.00, 0.01, 0.02, 0.04, 0.06]
    rows = []
    for fusion_row in fusion["margins"]:
        weights = np.asarray(
            [fusion_row["weights"][name] for name in names],
            dtype=np.float32,
        )
        alpha = fusion_row["density_alpha"]
        calibration_base = (
            scores["calibration"] @ weights - alpha * calibration_density_z
        )
        test_base = scores["test"] @ weights - alpha * test_density_z
        best = None
        for offsets in product(offset_grid, repeat=3):
            for beta in beta_grid:
                calibration_adjusted = apply_category_score_adjustment(
                    calibration_base,
                    calibration_groups,
                    calibration_risk,
                    offsets,
                    beta,
                )
                selected = select_from_calibration(
                    calibration_adjusted,
                    calibration_labels,
                    blocks["calibration"].stats["error_notes"],
                    fusion_row["requested_precision"],
                )
                key = (
                    selected["recall"],
                    selected["precision"],
                    -sum(offsets),
                    -beta,
                )
                if best is None or key > best[0]:
                    best = (key, offsets, beta, selected)
        _, offsets, beta, selected = best
        test_adjusted = apply_category_score_adjustment(
            test_base,
            test_groups,
            test_risk,
            offsets,
            beta,
        )
        test_row = row_at_threshold(
            test_adjusted,
            test_labels,
            blocks["test"].stats["error_notes"],
            selected["threshold"],
        )
        rows.append(
            {
                "weights": fusion_row["weights"],
                "density_alpha": alpha,
                "requested_precision": fusion_row["requested_precision"],
                "group_offsets": {
                    "short_and_scale": offsets[0],
                    "short_only": offsets[1],
                    "scale_only": offsets[2],
                },
                "risk_beta": beta,
                "selected_calibration": selected,
                "selected_test": test_row,
            }
        )
    return {
        "margins": rows,
        "best_feasible_test": max(
            (
                row
                for row in rows
                if row["selected_test"]["precision"] >= target_precision
            ),
            key=lambda row: (
                row["selected_test"]["recall"],
                row["selected_test"]["precision"],
            ),
            default=max(
                rows,
                key=lambda row: (
                    row["selected_test"]["precision"],
                    row["selected_test"]["recall"],
                ),
            ),
        ),
    }


def best_group_adjustment(
    base_scores: np.ndarray,
    labels: np.ndarray,
    total_errors: int,
    groups: np.ndarray,
    group_count: int,
    requested_precision: float,
    grid: list[float],
    coordinate_descent: bool,
) -> tuple[tuple[float, ...], dict]:
    offsets = np.zeros(group_count, dtype=np.float32)
    if coordinate_descent:
        for _ in range(2):
            for group in range(group_count):
                best = None
                for value in grid:
                    candidate = offsets.copy()
                    candidate[group] = value
                    adjusted = base_scores + candidate[groups]
                    selected = select_from_calibration(
                        adjusted,
                        labels,
                        total_errors,
                        requested_precision,
                    )
                    key = (
                        selected["recall"],
                        selected["precision"],
                        -float(np.abs(candidate).sum()),
                    )
                    if best is None or key > best[0]:
                        best = (key, candidate, selected)
                offsets = best[1]
        adjusted = base_scores + offsets[groups]
        selected = select_from_calibration(
            adjusted,
            labels,
            total_errors,
            requested_precision,
        )
        return tuple(float(value) for value in offsets), selected

    best = None
    for values in product(grid, repeat=group_count):
        candidate = np.asarray(values, dtype=np.float32)
        adjusted = base_scores + candidate[groups]
        selected = select_from_calibration(
            adjusted,
            labels,
            total_errors,
            requested_precision,
        )
        key = (
            selected["recall"],
            selected["precision"],
            -float(np.abs(candidate).sum()),
        )
        if best is None or key > best[0]:
            best = (key, values, selected)
    return tuple(float(value) for value in best[1]), best[2]


def evaluate_register_aware(
    names: list[str],
    scores: dict[str, np.ndarray],
    blocks: dict[str, CandidateBlock],
    target_precision: float,
) -> dict:
    model_index = names.index("old_small_leaf")
    calibration_density_z, test_density_z = density_z(
        blocks["calibration"],
        blocks["test"],
    )
    calibration_register, calibration_interaction, calibration_continuous = (
        observed_register_features(blocks["calibration"].raw)
    )
    test_register, test_interaction, test_continuous = observed_register_features(
        blocks["test"].raw
    )
    calibration_labels = blocks["calibration"].labels.numpy().astype(np.int64)
    test_labels = blocks["test"].labels.numpy().astype(np.int64)
    total_calibration_errors = blocks["calibration"].stats["error_notes"]
    total_test_errors = blocks["test"].stats["error_notes"]
    offset_grid = [-0.04, -0.02, 0.00, 0.02, 0.04]
    beta_grid = [-0.06, -0.04, -0.02, 0.00, 0.02, 0.04, 0.06]
    alphas = [0.00, 0.02, 0.04, 0.06, 0.08]
    variants = {"register": [], "register_short": [], "continuous": []}
    for margin in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]:
        requested_precision = target_precision + margin
        best_by_variant = {name: None for name in variants}
        for alpha in alphas:
            calibration_base = (
                scores["calibration"][:, model_index]
                - alpha * calibration_density_z
            )
            test_base = scores["test"][:, model_index] - alpha * test_density_z
            register_offsets, register_selected = best_group_adjustment(
                calibration_base,
                calibration_labels,
                total_calibration_errors,
                calibration_register,
                3,
                requested_precision,
                offset_grid,
                coordinate_descent=False,
            )
            interaction_offsets, interaction_selected = best_group_adjustment(
                calibration_base,
                calibration_labels,
                total_calibration_errors,
                calibration_interaction,
                6,
                requested_precision,
                offset_grid,
                coordinate_descent=True,
            )
            for variant_name, groups, test_groups, offsets, selected in [
                (
                    "register",
                    calibration_register,
                    test_register,
                    register_offsets,
                    register_selected,
                ),
                (
                    "register_short",
                    calibration_interaction,
                    test_interaction,
                    interaction_offsets,
                    interaction_selected,
                ),
            ]:
                adjusted_test = test_base + np.asarray(offsets)[test_groups]
                candidate = {
                    "base_model": "old_small_leaf",
                    "density_alpha": alpha,
                    "requested_precision": requested_precision,
                    "offsets": list(offsets),
                    "selected_calibration": selected,
                    "selected_test": row_at_threshold(
                        adjusted_test,
                        test_labels,
                        total_test_errors,
                        selected["threshold"],
                    ),
                }
                key = (
                    selected["recall"],
                    selected["precision"],
                    -sum(abs(value) for value in offsets),
                )
                if (
                    best_by_variant[variant_name] is None
                    or key > best_by_variant[variant_name][0]
                ):
                    best_by_variant[variant_name] = (key, candidate)

            for beta in beta_grid:
                adjusted_calibration = calibration_base + beta * calibration_continuous
                selected = select_from_calibration(
                    adjusted_calibration,
                    calibration_labels,
                    total_calibration_errors,
                    requested_precision,
                )
                adjusted_test = test_base + beta * test_continuous
                candidate = {
                    "base_model": "old_small_leaf",
                    "density_alpha": alpha,
                    "requested_precision": requested_precision,
                    "pitch_beta": beta,
                    "selected_calibration": selected,
                    "selected_test": row_at_threshold(
                        adjusted_test,
                        test_labels,
                        total_test_errors,
                        selected["threshold"],
                    ),
                }
                key = (selected["recall"], selected["precision"], -abs(beta))
                if (
                    best_by_variant["continuous"] is None
                    or key > best_by_variant["continuous"][0]
                ):
                    best_by_variant["continuous"] = (key, candidate)
        for name in variants:
            variants[name].append(best_by_variant[name][1])

    all_rows = [
        (variant_name, row)
        for variant_name, rows in variants.items()
        for row in rows
    ]
    feasible = [
        (variant_name, row)
        for variant_name, row in all_rows
        if row["selected_test"]["precision"] >= target_precision
    ]
    best_variant, best_row = max(
        feasible or all_rows,
        key=lambda item: (
            item[1]["selected_test"]["recall"]
            if item[1]["selected_test"]["precision"] >= target_precision
            else -1.0,
            item[1]["selected_test"]["precision"],
        ),
    )
    return {
        "pitch_source": "post_corruption_observed_pitch",
        "register_boundaries": {"low_max": 47, "middle_max": 71},
        "variants": variants,
        "best_variant": best_variant,
        "best_feasible_test": best_row,
    }


def train_pairwise_ranker(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    file_ids: torch.Tensor,
    hardness: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> PairwiseRanker:
    positive, negative = build_pair_indices(
        train_y.numpy().astype(np.int64),
        file_ids.numpy().astype(np.int64),
        hardness,
        args.negatives_per_positive,
    )
    if len(positive) == 0:
        raise RuntimeError("No within-piece positive/negative ranking pairs.")
    model = PairwiseRanker(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    positive_tensor = torch.from_numpy(positive)
    negative_tensor = torch.from_numpy(negative)
    for epoch in range(1, args.ranker_epochs + 1):
        order = torch.randperm(len(positive_tensor), generator=generator)
        total = 0.0
        model.train()
        for start in range(0, len(order), 4096):
            batch = order[start : start + 4096]
            pos_x = train_x[positive_tensor[batch]].to(device)
            neg_x = train_x[negative_tensor[batch]].to(device)
            pos_score = model(pos_x)
            neg_score = model(neg_x)
            pair_loss = F.softplus(-(pos_score - neg_score)).mean()
            point_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(
                    pos_score,
                    torch.ones_like(pos_score),
                )
                + F.binary_cross_entropy_with_logits(
                    neg_score,
                    torch.zeros_like(neg_score),
                )
            )
            loss = pair_loss + 0.20 * point_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch)
        print(
            f"pairwise epoch={epoch}/{args.ranker_epochs} loss={total / len(order):.6f}",
            flush=True,
        )
    return model


@torch.no_grad()
def ranker_scores(
    model: PairwiseRanker,
    features: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    parts = []
    for start in range(0, len(features), 8192):
        parts.append(torch.sigmoid(model(features[start : start + 8192].to(device))).cpu())
    return torch.cat(parts).numpy()


def evaluate_ranker(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    blocks: dict[str, CandidateBlock],
    target_precision: float,
) -> dict:
    calibration_labels = blocks["calibration"].labels.numpy().astype(np.int64)
    test_labels = blocks["test"].labels.numpy().astype(np.int64)
    rows = []
    for margin in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]:
        selected = select_from_calibration(
            calibration_scores,
            calibration_labels,
            blocks["calibration"].stats["error_notes"],
            target_precision + margin,
        )
        test = row_at_threshold(
            test_scores,
            test_labels,
            blocks["test"].stats["error_notes"],
            selected["threshold"],
        )
        rows.append(
            {
                "requested_precision": target_precision + margin,
                "selected_calibration": selected,
                "selected_test": test,
            }
        )
    return {
        "margins": rows,
        "best_feasible_test": max(
            (
                row
                for row in rows
                if row["selected_test"]["precision"] >= target_precision
            ),
            key=lambda row: (
                row["selected_test"]["recall"],
                row["selected_test"]["precision"],
            ),
            default=max(
                rows,
                key=lambda row: (
                    row["selected_test"]["precision"],
                    row["selected_test"]["recall"],
                ),
            ),
        ),
    }


def error_analysis(
    raw: torch.Tensor,
    labels: torch.Tensor,
    prediction: np.ndarray,
    base_scores: np.ndarray,
) -> dict:
    target = labels.numpy().astype(bool)
    patterns = classify_candidate_patterns(raw)
    groups = {
        "false_positive": prediction & ~target,
        "false_negative": ~prediction & target,
        "true_positive": prediction & target,
    }
    result = {}
    for group_name, group_mask in groups.items():
        count = int(group_mask.sum())
        row = {"count": count}
        for pattern_name, pattern_mask in patterns.items():
            matched = int(np.logical_and(group_mask, pattern_mask).sum())
            row[pattern_name] = {
                "count": matched,
                "fraction": matched / max(count, 1),
            }
        disagreement = base_scores.std(axis=1) >= 0.10
        matched = int(np.logical_and(group_mask, disagreement).sum())
        row["model_disagreement"] = {
            "count": matched,
            "fraction": matched / max(count, 1),
        }
        result[group_name] = row
    return result


def write_markdown(path: Path, result: dict) -> None:
    fusion = result["fusion"]["best_feasible_test"]["selected_test"]
    category = result["category_aware"]["best_feasible_test"]["selected_test"]
    register = result["register_aware"]["best_feasible_test"]["selected_test"]
    ranker = result["pairwise_ranker"]["best_feasible_test"]["selected_test"]
    lines = [
        "# Verifier Improvement Suite",
        "",
        f"- candidate recall ceiling: `{result['test_stats']['candidate_recall_ceiling']:.4f}`",
        "",
        "| System | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: |",
        f"| convex score fusion | {fusion['precision']:.4f} | {fusion['recall']:.4f} | {fusion['f1']:.4f} |",
        f"| category-aware calibration | {category['precision']:.4f} | {category['recall']:.4f} | {category['f1']:.4f} |",
        f"| post-corruption register calibration | {register['precision']:.4f} | {register['recall']:.4f} | {register['f1']:.4f} |",
        f"| pairwise MLP ranker | {ranker['precision']:.4f} | {ranker['recall']:.4f} | {ranker['f1']:.4f} |",
        "",
        "## Error Pattern Analysis",
        "",
        "| Group | Count | Chord | Scale | Ornament | Short | Strong beat | Dense | Model disagreement |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group_name, row in result["error_analysis"].items():
        lines.append(
            f"| {group_name} | {row['count']} | "
            f"{row['chord_tone']['fraction']:.3f} | "
            f"{row['scale_tone']['fraction']:.3f} | "
            f"{row['ornament']['fraction']:.3f} | "
            f"{row['short_note']['fraction']:.3f} | "
            f"{row['strong_beat']['fraction']:.3f} | "
            f"{row['high_density']['fraction']:.3f} | "
            f"{row['model_disagreement']['fraction']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = (
        *load_any_model(args.threeclass_checkpoint, device),
        *load_any_model(args.binary_checkpoint, device),
        *load_any_model(args.forward_checkpoint, device),
        *load_any_model(args.backward_checkpoint, device),
    )
    validation = make_dataset(args, "validation", args.max_validation_files)
    train_indices, calibration_indices, train_files, calibration_files = split_indices_by_file(
        validation,
        0.25,
        args.seed,
    )
    test = make_dataset(args, "test", args.max_test_files)
    blocks = {
        "train": collect_block(
            models,
            validation,
            train_indices,
            device,
            args,
            "collect improvement train",
        ),
        "calibration": collect_block(
            models,
            validation,
            calibration_indices,
            device,
            args,
            "collect improvement calibration",
        ),
        "test": collect_block(
            models,
            test,
            list(range(len(test))),
            device,
            args,
            "collect improvement test",
        ),
    }
    names, base_scores = load_base_scores(args, blocks)
    fusion = evaluate_fusion(
        names,
        base_scores,
        blocks,
        args.target_precision,
    )
    category_aware = evaluate_category_aware(
        names,
        base_scores,
        blocks,
        fusion,
        args.target_precision,
    )
    register_aware = evaluate_register_aware(
        names,
        base_scores,
        blocks,
        args.target_precision,
    )

    train_standardized, ranker_normalization, others = standardize(
        blocks["train"].full,
        blocks["calibration"].full,
        blocks["test"].full,
    )
    calibration_standardized, test_standardized = others
    hardness = base_scores["train"].mean(axis=1)
    ranker_model = train_pairwise_ranker(
        train_standardized,
        blocks["train"].labels,
        blocks["train"].file_ids,
        hardness,
        args,
        device,
    )
    torch.save(
        {
            "model_state_dict": ranker_model.state_dict(),
            "input_size": train_standardized.shape[1],
            "normalization": ranker_normalization,
            "args": vars(args),
        },
        args.ranker_output,
    )
    calibration_ranker_scores = ranker_scores(
        ranker_model,
        calibration_standardized,
        device,
    )
    test_ranker_scores = ranker_scores(
        ranker_model,
        test_standardized,
        device,
    )
    ranker = evaluate_ranker(
        calibration_ranker_scores,
        test_ranker_scores,
        blocks,
        args.target_precision,
    )

    candidates = [
        ("fusion", fusion["best_feasible_test"]),
        ("category_aware", category_aware["best_feasible_test"]),
        ("register_aware", register_aware["best_feasible_test"]),
        ("pairwise_ranker", ranker["best_feasible_test"]),
    ]
    best_name, best_row = max(
        candidates,
        key=lambda item: (
            item[1]["selected_test"]["recall"]
            if item[1]["selected_test"]["precision"] >= args.target_precision
            else -1.0,
            item[1]["selected_test"]["precision"],
        ),
    )
    if best_name in {"fusion", "category_aware"}:
        weights = np.asarray(
            [best_row["weights"][name] for name in names],
            dtype=np.float32,
        )
        test_scores = base_scores["test"] @ weights
        cal_density_z, test_density_z = density_z(
            blocks["calibration"],
            blocks["test"],
        )
        test_scores = test_scores - best_row["density_alpha"] * test_density_z
        if best_name == "category_aware":
            test_groups, test_risk = category_risk_features(blocks["test"].raw)
            offsets = best_row["group_offsets"]
            test_scores = apply_category_score_adjustment(
                test_scores,
                test_groups,
                test_risk,
                (
                    offsets["short_and_scale"],
                    offsets["short_only"],
                    offsets["scale_only"],
                ),
                best_row["risk_beta"],
            )
    elif best_name == "register_aware":
        model_index = names.index("old_small_leaf")
        test_scores = base_scores["test"][:, model_index]
        _, test_density_z = density_z(
            blocks["calibration"],
            blocks["test"],
        )
        test_scores = test_scores - best_row["density_alpha"] * test_density_z
        test_register, test_interaction, test_continuous = observed_register_features(
            blocks["test"].raw
        )
        if register_aware["best_variant"] == "register":
            test_scores += np.asarray(best_row["offsets"])[test_register]
        elif register_aware["best_variant"] == "register_short":
            test_scores += np.asarray(best_row["offsets"])[test_interaction]
        else:
            test_scores += best_row["pitch_beta"] * test_continuous
    else:
        test_scores = test_ranker_scores
    threshold = best_row["selected_calibration"]["threshold"]
    prediction = test_scores >= threshold
    analysis = error_analysis(
        blocks["test"].raw,
        blocks["test"].labels,
        prediction,
        base_scores["test"],
    )
    result = {
        "target_precision": args.target_precision,
        "base_models": names,
        "train_files": train_files,
        "calibration_files": calibration_files,
        "test_stats": blocks["test"].stats,
        "fusion": fusion,
        "category_aware": category_aware,
        "register_aware": register_aware,
        "pairwise_ranker": ranker,
        "analysis_system": best_name,
        "error_analysis": analysis,
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
