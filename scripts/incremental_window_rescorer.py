from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from incremental_global_context import (
    PieceEdit,
    apply_piece_edits,
    window_state_key,
)


class IncrementalWindowCache:
    def __init__(
        self,
        *,
        piece_id: int,
        piece_features: np.ndarray,
        window_size: int,
        compute: Callable[[np.ndarray, np.ndarray], Any],
    ) -> None:
        if piece_features.ndim != 2:
            raise ValueError("Piece features must have shape [notes, features].")
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        self.piece_id = int(piece_id)
        self.piece_features = piece_features
        self.window_size = int(window_size)
        self.compute = compute
        self.cache: dict[tuple, Any] = {}
        self.hits = 0
        self.misses = 0

    def get(
        self,
        window_start: int,
        edits: Iterable[PieceEdit],
    ) -> Any:
        edit_tuple = tuple(edits)
        key = window_state_key(
            self.piece_id,
            window_start,
            self.window_size,
            edit_tuple,
        )
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        piece = apply_piece_edits(self.piece_features, edit_tuple)
        sliced = piece[window_start : window_start + self.window_size]
        length = len(sliced)
        window = np.zeros(
            (self.window_size, piece.shape[1]),
            dtype=piece.dtype,
        )
        window[:length] = sliced
        mask = np.zeros(self.window_size, dtype=bool)
        mask[:length] = True
        value = self.compute(window, mask)
        self.cache[key] = value
        self.misses += 1
        return value
