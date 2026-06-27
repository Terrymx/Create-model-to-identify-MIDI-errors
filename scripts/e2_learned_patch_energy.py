from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from e1_edit_energy_verifier import proposal_target_indices


@dataclass(frozen=True)
class E2PatchTensors:
    candidate: np.ndarray
    observed: np.ndarray
    edited: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class E2PatchNormalizer:
    candidate_mean: np.ndarray
    candidate_std: np.ndarray
    patch_mean: np.ndarray
    patch_std: np.ndarray

    @classmethod
    def fit(cls, tensors: E2PatchTensors) -> "E2PatchNormalizer":
        valid = tensors.mask.astype(bool)
        observed_valid = tensors.observed[valid]
        edited_valid = tensors.edited[
            np.broadcast_to(valid[:, None, :], tensors.edited.shape[:3])
        ]
        patch_values = np.concatenate([observed_valid, edited_valid], axis=0)
        if len(patch_values) == 0:
            patch_values = np.zeros((1, tensors.observed.shape[-1]), dtype=np.float32)
        candidate_mean = tensors.candidate.mean(axis=0, keepdims=True)
        candidate_std = tensors.candidate.std(axis=0, keepdims=True)
        patch_mean = patch_values.mean(axis=0, keepdims=True)
        patch_std = patch_values.std(axis=0, keepdims=True)
        return cls(
            candidate_mean=candidate_mean.astype(np.float32),
            candidate_std=np.maximum(candidate_std, 1e-5).astype(np.float32),
            patch_mean=patch_mean.astype(np.float32),
            patch_std=np.maximum(patch_std, 1e-5).astype(np.float32),
        )

    def transform(self, tensors: E2PatchTensors) -> E2PatchTensors:
        candidate = (tensors.candidate - self.candidate_mean) / self.candidate_std
        observed = (tensors.observed - self.patch_mean) / self.patch_std
        edited = (tensors.edited - self.patch_mean) / self.patch_std
        mask = tensors.mask.astype(np.float32)
        observed = observed * mask[:, :, None]
        edited = edited * mask[:, None, :, None]
        return E2PatchTensors(
            candidate=np.nan_to_num(candidate, copy=False).astype(np.float32),
            observed=np.nan_to_num(observed, copy=False).astype(np.float32),
            edited=np.nan_to_num(edited, copy=False).astype(np.float32),
            mask=mask,
        )

    def to_jsonable(self) -> dict:
        return {
            "candidate_mean": self.candidate_mean.tolist(),
            "candidate_std": self.candidate_std.tolist(),
            "patch_mean": self.patch_mean.tolist(),
            "patch_std": self.patch_std.tolist(),
        }


def _pitch_columns_for(pitch: float) -> tuple[float, float, float]:
    phase = 2.0 * np.pi * ((float(pitch) % 12.0) / 12.0)
    return float(pitch) / 127.0, float(np.sin(phase)), float(np.cos(phase))


def _set_center_pitch(patch: np.ndarray, center: int, pitch: float) -> None:
    patch[center, 0], patch[center, 6], patch[center, 7] = _pitch_columns_for(pitch)
    if center > 0 and patch[center - 1, -1] > 0.5:
        previous_pitch = patch[center - 1, 0] * 127.0
        patch[center, 4] = np.clip((pitch - previous_pitch) / 24.0, -1.0, 1.0)
    if center + 1 < patch.shape[0] and patch[center + 1, -1] > 0.5:
        next_pitch = patch[center + 1, 0] * 127.0
        patch[center + 1, 4] = np.clip((next_pitch - pitch) / 24.0, -1.0, 1.0)


def build_patch_energy_tensors(
    *,
    piece_features: np.ndarray,
    candidate_positions: np.ndarray,
    observed_pitch: np.ndarray,
    proposals: np.ndarray,
    radius: int,
    candidate_features: np.ndarray | None = None,
) -> E2PatchTensors:
    candidate_positions = np.asarray(candidate_positions, dtype=np.int64)
    observed_pitch = np.asarray(observed_pitch, dtype=np.float32)
    proposals = np.asarray(proposals, dtype=np.float32)
    row_count = len(candidate_positions)
    proposal_count = int(proposals.shape[1])
    length = radius * 2 + 1
    base_dim = int(piece_features.shape[1])
    feature_dim = base_dim + 3
    observed = np.zeros((row_count, length, feature_dim), dtype=np.float32)
    edited = np.zeros((row_count, proposal_count, length, feature_dim), dtype=np.float32)
    mask = np.zeros((row_count, length), dtype=np.float32)
    denominator = float(max(radius, 1))

    for row, position in enumerate(candidate_positions):
        for offset in range(-radius, radius + 1):
            source_index = int(position + offset)
            target_index = offset + radius
            if source_index < 0 or source_index >= len(piece_features):
                continue
            observed[row, target_index, :base_dim] = piece_features[source_index]
            observed[row, target_index, base_dim] = float(offset) / denominator
            observed[row, target_index, base_dim + 1] = 1.0 if offset == 0 else 0.0
            observed[row, target_index, base_dim + 2] = 1.0
            mask[row, target_index] = 1.0
        _set_center_pitch(observed[row], radius, float(observed_pitch[row]))
        for proposal_index in range(proposal_count):
            edited[row, proposal_index] = observed[row]
            _set_center_pitch(
                edited[row, proposal_index],
                radius,
                float(proposals[row, proposal_index]),
            )

    if candidate_features is None:
        candidate = np.zeros((row_count, 0), dtype=np.float32)
    else:
        candidate = np.asarray(candidate_features, dtype=np.float32)
    return E2PatchTensors(
        candidate=np.nan_to_num(candidate, copy=False).astype(np.float32),
        observed=np.nan_to_num(observed, copy=False).astype(np.float32),
        edited=np.nan_to_num(edited, copy=False).astype(np.float32),
        mask=mask,
    )


class E2PatchEnergyNet(nn.Module):
    def __init__(
        self,
        *,
        candidate_dim: int,
        patch_feature_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.patch_input = nn.Sequential(
            nn.Linear(patch_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.patch_encoder = nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        pair_dim = hidden_dim * 7
        self.proposal_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.detection_head = nn.Sequential(
            nn.Linear(hidden_dim * 7 + 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _encode_patch(self, patch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        projected = self.patch_input(patch)
        projected = projected * mask.unsqueeze(-1)
        encoded, _ = self.patch_encoder(projected)
        encoded = encoded * mask.unsqueeze(-1)
        lengths = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return encoded.sum(dim=1) / lengths

    def forward(
        self,
        candidate: torch.Tensor,
        observed: torch.Tensor,
        edited: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observed_embedding = self._encode_patch(observed, mask)
        batch, proposals, length, feature_dim = edited.shape
        edited_embedding = self._encode_patch(
            edited.reshape(batch * proposals, length, feature_dim),
            mask[:, None, :].expand(batch, proposals, length).reshape(batch * proposals, length),
        ).reshape(batch, proposals, -1)
        candidate_embedding = self.candidate_encoder(candidate)
        candidate_expanded = candidate_embedding[:, None, :].expand(-1, proposals, -1)
        observed_expanded = observed_embedding[:, None, :].expand(-1, proposals, -1)
        pair = torch.cat(
            [
                candidate_expanded,
                observed_expanded,
                edited_embedding,
                edited_embedding - observed_expanded,
            ],
            dim=2,
        )
        proposal_logits = self.proposal_head(pair).squeeze(-1)
        best_values = proposal_logits.max(dim=1).values
        margins = best_values - proposal_logits.topk(k=min(2, proposals), dim=1).values[:, -1]
        best_indices = proposal_logits.argmax(dim=1)
        best_edited = edited_embedding[torch.arange(batch, device=edited.device), best_indices]
        detection_features = torch.cat(
            [
                candidate_embedding,
                observed_embedding,
                best_edited,
                best_edited - observed_embedding,
                best_values[:, None],
                margins[:, None],
            ],
            dim=1,
        )
        candidate_logits = self.detection_head(detection_features).squeeze(1)
        return candidate_logits, proposal_logits


def _make_dataset(
    tensors: E2PatchTensors,
    labels: np.ndarray,
    target_indices: np.ndarray,
    target_mask: np.ndarray,
) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(tensors.candidate).float(),
        torch.from_numpy(tensors.observed).float(),
        torch.from_numpy(tensors.edited).float(),
        torch.from_numpy(tensors.mask).float(),
        torch.from_numpy(labels.astype(np.float32)),
        torch.from_numpy(target_indices.astype(np.int64)),
        torch.from_numpy(target_mask.astype(np.float32)),
    )


def train_e2_model(
    tensors: E2PatchTensors,
    arrays: dict[str, np.ndarray],
    *,
    seed: int,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    correction_weight: float,
    device: torch.device,
) -> E2PatchEnergyNet:
    torch.manual_seed(seed)
    target_indices, target_mask = proposal_target_indices(
        arrays["labels"],
        arrays["error_kind"],
        arrays["target_pitch"],
        arrays["proposals"],
    )
    dataset = _make_dataset(tensors, arrays["labels"], target_indices, target_mask)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    labels = arrays["labels"].astype(np.float32)
    positives = max(float(labels.sum()), 1.0)
    negatives = max(float(len(labels) - labels.sum()), 1.0)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    model = E2PatchEnergyNet(
        candidate_dim=tensors.candidate.shape[1],
        patch_feature_dim=tensors.observed.shape[2],
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ce = nn.CrossEntropyLoss(reduction="none")
    model.train()
    for _ in range(epochs):
        for candidate, observed, edited, mask, label, target_index, target_ok in loader:
            candidate = candidate.to(device)
            observed = observed.to(device)
            edited = edited.to(device)
            mask = mask.to(device)
            label = label.to(device)
            target_index = target_index.to(device)
            target_ok = target_ok.to(device)
            candidate_logits, proposal_logits = model(candidate, observed, edited, mask)
            detection_loss = bce(candidate_logits, label)
            proposal_loss = ce(proposal_logits, target_index)
            proposal_loss = (proposal_loss * target_ok).sum() / target_ok.sum().clamp_min(1.0)
            loss = detection_loss + correction_weight * proposal_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict_e2(
    model: E2PatchEnergyNet,
    tensors: E2PatchTensors,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = TensorDataset(
        torch.from_numpy(tensors.candidate).float(),
        torch.from_numpy(tensors.observed).float(),
        torch.from_numpy(tensors.edited).float(),
        torch.from_numpy(tensors.mask).float(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    candidate_logits: list[np.ndarray] = []
    proposal_logits: list[np.ndarray] = []
    for candidate, observed, edited, mask in loader:
        logits, proposal = model(
            candidate.to(device),
            observed.to(device),
            edited.to(device),
            mask.to(device),
        )
        candidate_logits.append(logits.detach().cpu().numpy())
        proposal_logits.append(proposal.detach().cpu().numpy())
    candidate_logit = np.concatenate(candidate_logits).astype(np.float32)
    proposal_logit = np.concatenate(proposal_logits).astype(np.float32)
    candidate_score = 1.0 / (1.0 + np.exp(-candidate_logit))
    return candidate_score.astype(np.float32), candidate_logit, proposal_logit


def build_e2_hgb_features(
    base_features: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_logits: np.ndarray,
    proposal_logits: np.ndarray,
) -> np.ndarray:
    sorted_logits = np.sort(proposal_logits, axis=1)
    best = sorted_logits[:, -1]
    second = sorted_logits[:, -2] if proposal_logits.shape[1] >= 2 else np.zeros_like(best)
    margin = best - second
    mean = proposal_logits.mean(axis=1)
    std = proposal_logits.std(axis=1)
    extras = np.stack(
        [
            candidate_scores,
            candidate_logits,
            best,
            margin,
            mean,
            std,
        ],
        axis=1,
    ).astype(np.float32)
    return np.concatenate([base_features.astype(np.float32), extras], axis=1)
