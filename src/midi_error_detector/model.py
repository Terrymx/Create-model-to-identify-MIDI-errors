"""Wrong-note detector and pitch corrector models."""

from __future__ import annotations

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed positional encoding for note-window transformers."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_terms = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div_terms)
        encoding[:, 1::2] = torch.cos(positions * div_terms[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.shape[1], :].to(dtype=x.dtype)


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


class TransformerWrongNoteModel(nn.Module):
    """A compact Transformer encoder for note-level wrong-note detection.

    It keeps the same three output heads as the BiGRU model so training metrics,
    checkpoint selection, and inference can be compared directly.
    """

    def __init__(
        self,
        input_size: int = 16,
        d_model: int = 192,
        num_layers: int = 4,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.position = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.error_head = nn.Linear(d_model, 1)
        self.kind_head = nn.Linear(d_model, 3)
        self.pitch_head = nn.Linear(d_model, 128)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.input_projection(features)
        encoded = self.position(encoded)
        encoded = self.encoder(encoded)
        encoded = self.dropout(self.norm(encoded))
        return {
            "error_logits": self.error_head(encoded).squeeze(-1),
            "kind_logits": self.kind_head(encoded),
            "pitch_logits": self.pitch_head(encoded),
        }


def build_wrong_note_model(
    model_type: str,
    input_size: int,
    hidden_size: int = 256,
    num_layers: int = 2,
    transformer_d_model: int = 192,
    transformer_heads: int = 4,
    transformer_ffn_dim: int = 512,
    dropout: float = 0.2,
) -> nn.Module:
    if model_type == "bigru":
        return BiGRUWrongNoteModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    if model_type == "transformer":
        return TransformerWrongNoteModel(
            input_size=input_size,
            d_model=transformer_d_model,
            num_layers=num_layers,
            nhead=transformer_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model type: {model_type}")


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
