from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True, order=True)
class EditCandidate:
    candidate_index: int
    position: int
    proposed_pitch: int
    utility: float


@dataclass(frozen=True)
class BeamState:
    edits: tuple[EditCandidate, ...]
    score: float


def edit_set_key(edits: Iterable[EditCandidate]) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        sorted(
            (
                int(edit.position),
                int(edit.proposed_pitch),
                int(edit.candidate_index),
            )
            for edit in edits
        )
    )


def _default_score(
    edits: Sequence[EditCandidate],
    edit_penalty: float,
    conflict_distance: int,
    conflict_penalty: float,
) -> float:
    score = sum(float(edit.utility) - edit_penalty for edit in edits)
    if conflict_distance <= 0 or conflict_penalty <= 0.0:
        return score
    positions = sorted(int(edit.position) for edit in edits)
    conflicts = sum(
        1
        for left_index, left in enumerate(positions)
        for right in positions[left_index + 1 :]
        if right - left <= conflict_distance
    )
    return score - conflict_penalty * conflicts


def beam_search_edits(
    candidates: Sequence[EditCandidate],
    *,
    beam_width: int,
    max_edits: int,
    proposal_floor: float = float("-inf"),
    edit_penalty: float = 0.0,
    conflict_distance: int = 0,
    conflict_penalty: float = 0.0,
    score_callback: Callable[[tuple[EditCandidate, ...]], float] | None = None,
) -> BeamState:
    if beam_width <= 0:
        raise ValueError("beam_width must be positive.")
    if max_edits < 0:
        raise ValueError("max_edits must be non-negative.")
    if conflict_distance < 0:
        raise ValueError("conflict_distance must be non-negative.")

    grouped: dict[int, list[EditCandidate]] = {}
    for candidate in candidates:
        if candidate.utility >= proposal_floor:
            grouped.setdefault(int(candidate.position), []).append(candidate)
    for choices in grouped.values():
        choices.sort(
            key=lambda edit: (
                -float(edit.utility),
                int(edit.proposed_pitch),
                int(edit.candidate_index),
            )
        )

    def score(edits: tuple[EditCandidate, ...]) -> float:
        if score_callback is not None:
            return float(score_callback(edits))
        return _default_score(
            edits,
            edit_penalty,
            conflict_distance,
            conflict_penalty,
        )

    beam = [BeamState(edits=(), score=score(()))]
    for position in sorted(grouped):
        expanded: dict[tuple[tuple[int, int, int], ...], BeamState] = {}
        for state in beam:
            options: list[EditCandidate | None] = [None, *grouped[position]]
            for option in options:
                if option is None:
                    edits = state.edits
                    candidate_score = state.score
                elif len(state.edits) < max_edits:
                    edits = tuple(
                        sorted(
                            (*state.edits, option),
                            key=lambda edit: (
                                edit.position,
                                edit.proposed_pitch,
                                edit.candidate_index,
                            ),
                        )
                    )
                    if score_callback is None:
                        conflicts = sum(
                            1
                            for existing in state.edits
                            if abs(existing.position - option.position)
                            <= conflict_distance
                        )
                        candidate_score = (
                            state.score
                            + float(option.utility)
                            - edit_penalty
                            - conflict_penalty * conflicts
                        )
                    else:
                        candidate_score = score(edits)
                else:
                    continue
                key = edit_set_key(edits)
                candidate_state = BeamState(edits=edits, score=candidate_score)
                previous = expanded.get(key)
                if previous is None or candidate_state.score > previous.score:
                    expanded[key] = candidate_state
        beam = sorted(
            expanded.values(),
            key=lambda state: (-state.score, edit_set_key(state.edits)),
        )[:beam_width]
    return beam[0]
