from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


PITCH_COLUMNS = (0, 6, 7)


@dataclass(frozen=True)
class CleanPatchBatch:
    context: np.ndarray
    mask: np.ndarray
    target_pitch: np.ndarray


@dataclass(frozen=True)
class DenoisingPatchBatch:
    context: np.ndarray
    mask: np.ndarray
    target_pitch: np.ndarray
    negative_pitch: np.ndarray
    negative_mask: np.ndarray


class CleanPatchPredictor(nn.Module):
    def __init__(
        self,
        *,
        patch_feature_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(patch_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.encoder = nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
        )

    def forward(self, context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(context)
        projected = projected * mask.unsqueeze(-1)
        encoded, _ = self.encoder(projected)
        encoded = encoded * mask.unsqueeze(-1)
        lengths = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = encoded.sum(dim=1) / lengths
        return self.head(pooled)


def patch_negative_log_likelihood(
    logits: torch.Tensor,
    target_pitch: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(logits, target_pitch.clamp(0, 127), reduction="none")


def patch_denoising_loss(
    logits: torch.Tensor,
    target_pitch: torch.Tensor,
    negative_pitch: torch.Tensor,
    negative_mask: torch.Tensor,
    *,
    contrastive_weight: float = 1.0,
    margin: float = 1.0,
) -> torch.Tensor:
    nll = patch_negative_log_likelihood(logits, target_pitch)
    target_logits = logits.gather(1, target_pitch.clamp(0, 127).unsqueeze(1)).squeeze(1)
    negative_logits = logits.gather(1, negative_pitch.clamp(0, 127).unsqueeze(1)).squeeze(1)
    ranking = F.relu(float(margin) - target_logits + negative_logits)
    weighted_ranking = ranking * negative_mask.float()
    if float(contrastive_weight) <= 0.0:
        return nll.mean()
    return nll.mean() + float(contrastive_weight) * weighted_ranking.mean()


def build_patch_predictor_features(
    *,
    observed_energy: np.ndarray,
    proposal_energy: np.ndarray,
    proposal_mask: np.ndarray,
) -> np.ndarray:
    observed = np.asarray(observed_energy, dtype=np.float32).reshape(-1)
    proposals = np.asarray(proposal_energy, dtype=np.float32)
    mask = np.asarray(proposal_mask, dtype=np.float32) > 0.5
    if proposals.shape != mask.shape:
        raise ValueError("proposal_energy and proposal_mask must have the same shape.")
    if len(observed) != proposals.shape[0]:
        raise ValueError("observed_energy must align with proposal rows.")

    valid_count = mask.sum(axis=1).astype(np.float32)
    safe_energy = np.where(mask, proposals, np.inf)
    best_edited = safe_energy.min(axis=1)
    best_edited = np.where(np.isfinite(best_edited), best_edited, observed)
    sum_edited = np.where(mask, proposals, 0.0).sum(axis=1)
    mean_edited = np.divide(
        sum_edited,
        np.maximum(valid_count, 1.0),
        out=np.zeros_like(sum_edited, dtype=np.float32),
    )
    mean_edited = np.where(valid_count > 0, mean_edited, observed)
    gains = observed[:, None] - proposals
    safe_gains = np.where(mask, gains, -np.inf)
    best_gain = safe_gains.max(axis=1)
    best_gain = np.where(np.isfinite(best_gain), best_gain, 0.0)
    mean_gain = np.divide(
        np.where(mask, gains, 0.0).sum(axis=1),
        np.maximum(valid_count, 1.0),
        out=np.zeros_like(observed, dtype=np.float32),
    )
    mean_gain = np.where(valid_count > 0, mean_gain, 0.0)
    sorted_energy = np.sort(safe_energy, axis=1)
    second_best = sorted_energy[:, 1] if proposals.shape[1] > 1 else np.inf
    margin = np.where(np.isfinite(second_best), second_best - best_edited, 0.0)
    any_improved = (best_gain > 0.0).astype(np.float32)
    features = np.stack(
        [
            observed,
            best_edited,
            mean_edited,
            best_gain,
            mean_gain,
            margin.astype(np.float32),
            valid_count,
            any_improved,
        ],
        axis=1,
    )
    return np.nan_to_num(features, copy=False).astype(np.float32)


def _copy_patch(
    piece_features: np.ndarray,
    center: int,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    length = radius * 2 + 1
    base_dim = int(piece_features.shape[1])
    patch = np.zeros((length, base_dim + 3), dtype=np.float32)
    mask = np.zeros(length, dtype=np.float32)
    denominator = float(max(radius, 1))
    for offset in range(-radius, radius + 1):
        source = int(center + offset)
        target = offset + radius
        if source < 0 or source >= len(piece_features):
            continue
        patch[target, :base_dim] = piece_features[source]
        patch[target, base_dim] = float(offset) / denominator
        patch[target, base_dim + 1] = 1.0 if offset == 0 else 0.0
        patch[target, base_dim + 2] = 1.0
        mask[target] = 1.0
    return patch, mask


def build_clean_patch_batch(
    *,
    piece_features: np.ndarray,
    center_positions: np.ndarray,
    radius: int,
) -> CleanPatchBatch:
    centers = np.asarray(center_positions, dtype=np.int64)
    row_count = len(centers)
    length = radius * 2 + 1
    feature_dim = int(piece_features.shape[1]) + 3
    context = np.zeros((row_count, length, feature_dim), dtype=np.float32)
    mask = np.zeros((row_count, length), dtype=np.float32)
    target_pitch = np.zeros(row_count, dtype=np.int64)

    for row, center in enumerate(centers):
        patch, patch_mask = _copy_patch(piece_features, int(center), radius)
        if 0 <= center < len(piece_features):
            target_pitch[row] = int(
                np.rint(float(piece_features[int(center), 0]) * 127.0)
            )
        patch[radius, list(PITCH_COLUMNS)] = 0.0
        context[row] = patch
        mask[row] = patch_mask

    return CleanPatchBatch(
        context=np.nan_to_num(context, copy=False).astype(np.float32),
        mask=mask,
        target_pitch=target_pitch,
    )


def build_denoising_patch_batch(
    *,
    piece_features: np.ndarray,
    center_positions: np.ndarray,
    target_pitch: np.ndarray,
    observed_pitch: np.ndarray,
    radius: int,
) -> DenoisingPatchBatch:
    clean = build_clean_patch_batch(
        piece_features=piece_features,
        center_positions=center_positions,
        radius=radius,
    )
    target = np.rint(np.asarray(target_pitch, dtype=np.float32)).astype(np.int64)
    negative = np.rint(np.asarray(observed_pitch, dtype=np.float32)).astype(np.int64)
    if len(target) != len(clean.context) or len(negative) != len(clean.context):
        raise ValueError("target_pitch and observed_pitch must align with center_positions.")
    negative_mask = (target != negative).astype(np.float32)
    return DenoisingPatchBatch(
        context=clean.context,
        mask=clean.mask,
        target_pitch=target,
        negative_pitch=negative,
        negative_mask=negative_mask,
    )


def build_candidate_patch_batch(
    *,
    piece_features: np.ndarray,
    candidate_positions: np.ndarray,
    observed_pitch: np.ndarray,
    proposals: np.ndarray,
    radius: int,
) -> tuple[CleanPatchBatch, CleanPatchBatch, np.ndarray]:
    candidate_positions = np.asarray(candidate_positions, dtype=np.int64)
    observed_pitch = np.asarray(observed_pitch, dtype=np.float32)
    proposals = np.asarray(proposals, dtype=np.float32)
    observed = build_clean_patch_batch(
        piece_features=piece_features,
        center_positions=candidate_positions,
        radius=radius,
    )
    observed_batch = CleanPatchBatch(
        context=observed.context,
        mask=observed.mask,
        target_pitch=np.rint(observed_pitch).astype(np.int64),
    )

    proposal_count = int(proposals.shape[1])
    edited_context = np.zeros(
        (
            len(candidate_positions) * proposal_count,
            observed.context.shape[1],
            observed.context.shape[2],
        ),
        dtype=np.float32,
    )
    edited_mask = np.zeros(
        (len(candidate_positions) * proposal_count, observed.mask.shape[1]),
        dtype=np.float32,
    )
    edited_targets = np.zeros(len(candidate_positions) * proposal_count, dtype=np.int64)
    proposal_mask = np.isfinite(proposals).astype(np.float32)

    output_row = 0
    for row, center in enumerate(candidate_positions):
        base = build_clean_patch_batch(
            piece_features=piece_features,
            center_positions=np.asarray([center], dtype=np.int64),
            radius=radius,
        )
        for proposal_index in range(proposal_count):
            edited_context[output_row] = base.context[0]
            edited_mask[output_row] = base.mask[0]
            edited_targets[output_row] = int(np.rint(float(proposals[row, proposal_index])))
            output_row += 1

    edited_batch = CleanPatchBatch(
        context=edited_context,
        mask=edited_mask,
        target_pitch=edited_targets,
    )
    return observed_batch, edited_batch, proposal_mask
