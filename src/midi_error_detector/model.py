"""BiGRU wrong-note detector and pitch corrector."""

from __future__ import annotations

import torch
from torch import nn


class BiGRUWrongNoteModel(nn.Module):
    """A note-level bidirectional GRU with detection, action, and pitch heads.

    The detection head predicts whether each input note is likely wrong.  The
    action head predicts whether to keep, replace, or delete the note.  The pitch
    head predicts top-k clean MIDI pitch candidates for replacement errors.
    """

    def __init__(self, input_size: int = 8, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.error_head = nn.Linear(hidden_size * 2, 1)
        self.kind_head = nn.Linear(hidden_size * 2, 3)
        self.pitch_head = nn.Linear(hidden_size * 2, 128)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded, _ = self.encoder(features)
        encoded = self.dropout(self.norm(encoded))
        return {
            "error_logits": self.error_head(encoded).squeeze(-1),
            "kind_logits": self.kind_head(encoded),
            "pitch_logits": self.pitch_head(encoded),
        }


def masked_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none", pos_weight=pos_weight)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def masked_pitch_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def masked_kind_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = nn.functional.cross_entropy(logits.transpose(1, 2), targets, reduction="none", weight=class_weight)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)
