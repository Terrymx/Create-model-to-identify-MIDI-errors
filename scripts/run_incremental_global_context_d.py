from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from incremental_global_context import PieceEdit


@dataclass(frozen=True)
class IncrementalBeamState:
    edits: tuple[PieceEdit, ...]
    score: float


def _edit_key(edits: tuple[PieceEdit, ...]) -> tuple[tuple[int, int], ...]:
    return tuple((edit.position, edit.proposed_pitch) for edit in edits)


def incremental_beam_search(
    *,
    candidate_positions: np.ndarray,
    proposals: dict[int, tuple[int, int]],
    score_all: Callable[[tuple[PieceEdit, ...]], np.ndarray],
    score_floor: float,
    beam_width: int,
    max_edits: int,
) -> IncrementalBeamState:
    if beam_width <= 0 or max_edits < 0:
        raise ValueError("Invalid beam width or edit budget.")
    beam = [IncrementalBeamState((), 0.0)]
    visited_positions: set[int] = set()
    while True:
        expanded: dict[tuple[tuple[int, int], ...], IncrementalBeamState] = {}
        changed = False
        for state in beam:
            scores = score_all(state.edits)
            available = [
                row
                for row, position in enumerate(candidate_positions.tolist())
                if position not in visited_positions
                and all(edit.position != position for edit in state.edits)
            ]
            if not available:
                expanded[_edit_key(state.edits)] = state
                continue
            row = max(
                available,
                key=lambda index: (
                    float(scores[index]),
                    -int(candidate_positions[index]),
                    -index,
                ),
            )
            position = int(candidate_positions[row])
            keep = IncrementalBeamState(state.edits, state.score)
            expanded[_edit_key(keep.edits)] = keep
            if scores[row] >= score_floor and len(state.edits) < max_edits:
                for pitch in proposals[row]:
                    edits = tuple(
                        sorted(
                            (*state.edits, PieceEdit(position, int(pitch))),
                            key=lambda edit: (edit.position, edit.proposed_pitch),
                        )
                    )
                    refreshed = score_all(edits)
                    edit_score = float(
                        sum(
                            max(0.0, refreshed[index] - score_floor)
                            for index, candidate_position in enumerate(
                                candidate_positions.tolist()
                            )
                            if any(
                                edit.position == candidate_position
                                for edit in edits
                            )
                        )
                    )
                    candidate = IncrementalBeamState(edits, edit_score)
                    key = _edit_key(edits)
                    previous = expanded.get(key)
                    if previous is None or candidate.score > previous.score:
                        expanded[key] = candidate
                changed = True
            visited_positions.add(position)
        beam = sorted(
            expanded.values(),
            key=lambda state: (-state.score, _edit_key(state.edits)),
        )[:beam_width]
        if not changed or len(visited_positions) >= len(candidate_positions):
            return beam[0]
