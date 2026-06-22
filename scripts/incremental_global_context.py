from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, order=True)
class PieceEdit:
    position: int
    proposed_pitch: int


def state_key(
    piece_id: int,
    edits: Iterable[PieceEdit],
) -> tuple[int, tuple[tuple[int, int], ...]]:
    return (
        int(piece_id),
        tuple(
            sorted(
                (int(edit.position), int(edit.proposed_pitch))
                for edit in edits
            )
        ),
    )


def apply_piece_edits(
    features: np.ndarray,
    edits: Iterable[PieceEdit],
) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("Piece features must have shape [notes, features].")
    edited = features.copy()
    for edit in edits:
        position = int(edit.position)
        if position < 0 or position >= len(edited):
            raise IndexError(f"Edit position {position} is outside the piece.")
        pitch = float(np.clip(edit.proposed_pitch, 0, 127))
        phase = 2.0 * np.pi * (pitch % 12.0) / 12.0
        edited[position, 0] = pitch / 127.0
        edited[position, 6] = np.sin(phase)
        edited[position, 7] = np.cos(phase)
    return edited


def piece_window_starts(
    note_count: int,
    window_size: int,
    stride: int,
) -> tuple[int, ...]:
    if note_count <= 0:
        return ()
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive.")
    starts = list(range(0, max(1, note_count - window_size + 1), stride))
    final_start = max(0, note_count - window_size)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


def affected_window_starts(
    position: int,
    note_count: int,
    window_size: int,
    stride: int,
) -> tuple[int, ...]:
    if position < 0 or position >= note_count:
        raise IndexError("Position is outside the piece.")
    return tuple(
        start
        for start in piece_window_starts(note_count, window_size, stride)
        if start <= position < start + window_size
    )


def candidate_dependency_windows(
    position: int,
    note_count: int,
    window_size: int,
    stride: int,
    context_radius: int,
) -> tuple[int, ...]:
    if context_radius < 0:
        raise ValueError("context_radius must be non-negative.")
    left = max(0, position - context_radius)
    right = min(note_count - 1, position + context_radius)
    return tuple(
        start
        for start in piece_window_starts(note_count, window_size, stride)
        if start <= right and start + window_size - 1 >= left
    )


def window_state_key(
    piece_id: int,
    window_start: int,
    window_size: int,
    edits: Iterable[PieceEdit],
) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    window_end = window_start + window_size
    relevant = tuple(
        sorted(
            (int(edit.position), int(edit.proposed_pitch))
            for edit in edits
            if window_start <= int(edit.position) < window_end
        )
    )
    return int(piece_id), int(window_start), relevant
