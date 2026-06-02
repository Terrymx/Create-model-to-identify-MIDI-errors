"""Lightweight harmony scoring for precision-first post-processing."""

from __future__ import annotations

import numpy as np

from .data import (
    NoteEvent,
    _CONSONANT_INTERVAL_CLASSES,
    _MAJOR_SCALE_PCS,
    _MINOR_SCALE_PCS,
    _TRIAD_INTERVALS,
    _estimate_key_tonics,
)


def _pitch_harmony_score(
    pitch: int,
    idx: int,
    starts: np.ndarray,
    pitches: np.ndarray,
    durations: np.ndarray,
    major_tonic: int,
    minor_tonic: int,
    onset_tolerance: float,
) -> float:
    local = np.flatnonzero(np.abs(starts - starts[idx]) <= onset_tolerance)
    local_pitches = pitches[local].copy()
    local_positions = {int(note_idx): pos for pos, note_idx in enumerate(local.tolist())}
    if idx in local_positions:
        local_pitches[local_positions[idx]] = pitch

    pitch_class = int(pitch) % 12
    major_scale = float(((pitch_class - major_tonic) % 12) in _MAJOR_SCALE_PCS)
    minor_scale = float(((pitch_class - minor_tonic) % 12) in _MINOR_SCALE_PCS)
    scale_score = max(major_scale, minor_scale)

    chord_score = 0.0
    consonance_score = 0.0
    if len(local_pitches) >= 2:
        root = int(local_pitches.min()) % 12
        relative_pc = (pitch_class - root) % 12
        chord_score = float(any(relative_pc in intervals for intervals in _TRIAD_INTERVALS.values()))
        intervals = np.abs(local_pitches - pitch)
        intervals = intervals[intervals > 0]
        if len(intervals) > 0:
            consonance_score = float(any(int(interval % 12) in _CONSONANT_INTERVAL_CLASSES for interval in intervals))

    duration_weight = float(np.clip(durations[idx] / max(float(np.median(durations[max(0, idx - 2) : idx + 3])), 0.03), 0.0, 2.0) / 2.0)
    return 0.45 * chord_score + 0.35 * scale_score + 0.15 * consonance_score + 0.05 * duration_weight


def harmony_scores_for_pitches(
    notes: list[NoteEvent],
    candidate_pitches: list[int],
    onset_tolerance: float = 0.08,
) -> tuple[list[float], list[float], list[float]]:
    """Return current score, candidate score, and candidate-current gain per note."""

    if len(notes) != len(candidate_pitches):
        raise ValueError("notes and candidate_pitches must have the same length")
    if not notes:
        return [], [], []

    starts = np.asarray([note.start for note in notes], dtype=np.float32)
    pitches = np.asarray([note.pitch for note in notes], dtype=np.float32)
    durations = np.asarray([note.duration for note in notes], dtype=np.float32)
    major_tonic, minor_tonic = _estimate_key_tonics(pitches, durations)

    current_scores: list[float] = []
    candidate_scores: list[float] = []
    gains: list[float] = []
    for idx, (note, candidate_pitch) in enumerate(zip(notes, candidate_pitches)):
        current = _pitch_harmony_score(
            note.pitch,
            idx,
            starts,
            pitches,
            durations,
            major_tonic,
            minor_tonic,
            onset_tolerance,
        )
        candidate = _pitch_harmony_score(
            int(candidate_pitch),
            idx,
            starts,
            pitches,
            durations,
            major_tonic,
            minor_tonic,
            onset_tolerance,
        )
        current_scores.append(current)
        candidate_scores.append(candidate)
        gains.append(candidate - current)
    return current_scores, candidate_scores, gains
