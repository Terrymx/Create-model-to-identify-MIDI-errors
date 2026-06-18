from __future__ import annotations

from dataclasses import dataclass
from itertools import count

import numpy as np
from scipy.optimize import linear_sum_assignment

from midi_error_detector.data import NoteEvent


VOICE_FEATURE_SIZE = 18


@dataclass(frozen=True)
class VoiceAssignment:
    voice_ids: np.ndarray
    confidence: np.ndarray


@dataclass
class _Track:
    voice_id: int
    last_pitch: int
    last_start: float
    last_end: float
    previous_pitch: int | None
    pitch_sum: float
    pitch_square_sum: float
    length: int

    @property
    def mean_pitch(self) -> float:
        return self.pitch_sum / max(self.length, 1)

    @property
    def pitch_std(self) -> float:
        mean = self.mean_pitch
        return float(max(self.pitch_square_sum / max(self.length, 1) - mean * mean, 0.0) ** 0.5)

    def clone(self) -> "_Track":
        return _Track(**vars(self))

    def append(self, note: NoteEvent) -> None:
        self.previous_pitch = self.last_pitch
        self.last_pitch = note.pitch
        self.last_start = note.start
        self.last_end = note.end
        self.pitch_sum += note.pitch
        self.pitch_square_sum += note.pitch * note.pitch
        self.length += 1


@dataclass
class _BeamState:
    tracks: dict[int, _Track]
    voice_ids: list[int]
    confidence: list[float]
    cost: float
    next_voice_id: int


def _onset_groups(notes: list[NoteEvent], tolerance: float = 0.03) -> list[list[int]]:
    groups: list[list[int]] = []
    for index, note in enumerate(notes):
        if not groups or abs(note.start - notes[groups[-1][0]].start) > tolerance:
            groups.append([index])
        else:
            groups[-1].append(index)
    return groups


def _track_note_cost(track: _Track, note: NoteEvent) -> float:
    pitch_cost = abs(note.pitch - track.last_pitch) / 12.0
    gap = max(0.0, note.start - track.last_end)
    gap_cost = min(gap / 2.0, 1.5) * 0.25
    overlap = max(0.0, track.last_end - note.start)
    overlap_cost = min(overlap / max(note.duration, 0.05), 2.0) * 0.75
    if track.previous_pitch is None:
        direction_cost = 0.0
    else:
        previous_direction = track.last_pitch - track.previous_pitch
        new_direction = note.pitch - track.last_pitch
        direction_cost = 0.15 if previous_direction * new_direction < 0 and abs(new_direction) > 5 else 0.0
    register_cost = min(abs(note.pitch - track.mean_pitch) / 24.0, 1.0) * 0.15
    return pitch_cost + gap_cost + overlap_cost + direction_cost + register_cost


def _matching_matrix(
    tracks: list[_Track],
    notes: list[NoteEvent],
    new_voice_cost: float,
) -> np.ndarray:
    row_count = len(notes)
    column_count = len(tracks) + row_count
    costs = np.full((row_count, column_count), new_voice_cost + 4.0, dtype=np.float64)
    for row, note in enumerate(notes):
        for column, track in enumerate(tracks):
            costs[row, column] = _track_note_cost(track, note)
        costs[row, len(tracks) + row] = new_voice_cost
    return costs


def _assignment_confidence(costs: np.ndarray, row: int, column: int) -> float:
    selected = float(costs[row, column])
    alternatives = np.delete(costs[row], column)
    second = float(alternatives.min()) if len(alternatives) else selected + 1.0
    margin = max(0.0, second - selected)
    absolute = np.exp(-max(selected, 0.0))
    return float(np.clip(0.35 * absolute + 0.65 * (1.0 - np.exp(-margin)), 0.05, 1.0))


def _candidate_matchings(costs: np.ndarray, limit: int) -> list[tuple[np.ndarray, float]]:
    rows, columns = linear_sum_assignment(costs)
    base = np.full(costs.shape[0], -1, dtype=np.int64)
    base[rows] = columns
    candidates: list[tuple[np.ndarray, float]] = [
        (base, float(costs[rows, columns].sum()))
    ]
    for row, column in zip(rows.tolist(), columns.tolist()):
        perturbed = costs.copy()
        perturbed[row, column] = costs.max() + 100.0
        alt_rows, alt_columns = linear_sum_assignment(perturbed)
        assignment = np.full(costs.shape[0], -1, dtype=np.int64)
        assignment[alt_rows] = alt_columns
        candidates.append((assignment, float(costs[alt_rows, alt_columns].sum())))
    unique: dict[tuple[int, ...], tuple[np.ndarray, float]] = {}
    for assignment, cost_value in candidates:
        key = tuple(assignment.tolist())
        current = unique.get(key)
        if current is None or cost_value < current[1]:
            unique[key] = (assignment, cost_value)
    return sorted(unique.values(), key=lambda item: item[1])[:limit]


def _apply_matching(
    state: _BeamState,
    group_notes: list[NoteEvent],
    tracks: list[_Track],
    costs: np.ndarray,
    assignment: np.ndarray,
    assignment_cost: float,
) -> _BeamState:
    next_tracks = {voice_id: track.clone() for voice_id, track in state.tracks.items()}
    next_voice_id = state.next_voice_id
    group_voice_ids: list[int] = []
    group_confidence: list[float] = []
    used_track_ids: set[int] = set()
    for row, note in enumerate(group_notes):
        column = int(assignment[row])
        if column < len(tracks):
            voice_id = tracks[column].voice_id
            if voice_id in used_track_ids:
                raise RuntimeError("A voice track cannot receive two notes from one onset group.")
            next_tracks[voice_id].append(note)
        else:
            voice_id = next_voice_id
            next_voice_id += 1
            next_tracks[voice_id] = _Track(
                voice_id=voice_id,
                last_pitch=note.pitch,
                last_start=note.start,
                last_end=note.end,
                previous_pitch=None,
                pitch_sum=float(note.pitch),
                pitch_square_sum=float(note.pitch * note.pitch),
                length=1,
            )
        used_track_ids.add(voice_id)
        group_voice_ids.append(voice_id)
        group_confidence.append(_assignment_confidence(costs, row, column))
    return _BeamState(
        tracks=next_tracks,
        voice_ids=state.voice_ids + group_voice_ids,
        confidence=state.confidence + group_confidence,
        cost=state.cost + assignment_cost,
        next_voice_id=next_voice_id,
    )


def _initial_state() -> _BeamState:
    return _BeamState({}, [], [], 0.0, 0)


def _expand_state(
    state: _BeamState,
    group_notes: list[NoteEvent],
    alternatives: int,
    max_gap: float = 4.0,
    new_voice_cost: float = 0.90,
) -> list[_BeamState]:
    onset = group_notes[0].start
    active = [
        track
        for track in state.tracks.values()
        if onset - track.last_end <= max_gap
    ]
    active.sort(key=lambda track: (track.mean_pitch, track.voice_id))
    costs = _matching_matrix(active, group_notes, new_voice_cost)
    return [
        _apply_matching(state, group_notes, active, costs, assignment, cost_value)
        for assignment, cost_value in _candidate_matchings(costs, alternatives)
    ]


def _assign_onset_matching(notes: list[NoteEvent]) -> VoiceAssignment:
    state = _initial_state()
    for group in _onset_groups(notes):
        group_notes = [notes[index] for index in group]
        state = _expand_state(state, group_notes, alternatives=1)[0]
    return VoiceAssignment(
        np.asarray(state.voice_ids, dtype=np.int64),
        np.asarray(state.confidence, dtype=np.float32),
    )


def _assign_global_beam(notes: list[NoteEvent], beam_width: int) -> VoiceAssignment:
    beam = [_initial_state()]
    alternatives = max(2, min(6, beam_width))
    for group in _onset_groups(notes):
        group_notes = [notes[index] for index in group]
        expanded = [
            candidate
            for state in beam
            for candidate in _expand_state(state, group_notes, alternatives=alternatives)
        ]
        expanded.sort(
            key=lambda state: (
                state.cost + 0.015 * len(state.tracks),
                len(state.tracks),
                tuple(state.voice_ids),
            )
        )
        beam = expanded[: max(1, beam_width)]
    best = min(beam, key=lambda state: state.cost + 0.015 * len(state.tracks))
    return VoiceAssignment(
        np.asarray(best.voice_ids, dtype=np.int64),
        np.asarray(best.confidence, dtype=np.float32),
    )


def assign_voices(
    notes: list[NoteEvent],
    method: str,
    beam_width: int = 12,
) -> VoiceAssignment:
    if not notes:
        return VoiceAssignment(
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
        )
    if method == "onset_matching":
        return _assign_onset_matching(notes)
    if method == "global_beam":
        return _assign_global_beam(notes, beam_width)
    raise ValueError(f"Unknown voice assignment method: {method}")


def build_voice_features(
    notes: list[NoteEvent],
    assignment: VoiceAssignment,
) -> np.ndarray:
    note_count = len(notes)
    if assignment.voice_ids.shape != (note_count,) or assignment.confidence.shape != (note_count,):
        raise ValueError("Voice assignment must align one-to-one with notes.")
    features = np.zeros((note_count, VOICE_FEATURE_SIZE), dtype=np.float32)
    if note_count == 0:
        return features

    voice_indices: dict[int, list[int]] = {}
    for index, voice_id in enumerate(assignment.voice_ids.tolist()):
        voice_indices.setdefault(int(voice_id), []).append(index)
    voice_mean_pitch = {
        voice_id: float(np.mean([notes[index].pitch for index in indices]))
        for voice_id, indices in voice_indices.items()
    }
    ordered_voices = sorted(voice_mean_pitch, key=lambda voice_id: voice_mean_pitch[voice_id])
    bass_voice = ordered_voices[0]
    melody_voice = ordered_voices[-1]

    features[:, 0] = assignment.confidence
    for index, voice_id in enumerate(assignment.voice_ids.tolist()):
        features[index, 1] = float(voice_id == melody_voice)
        features[index, 2] = float(voice_id == bass_voice)
        features[index, 3] = float(voice_id not in {melody_voice, bass_voice})

    for voice_id, indices in voice_indices.items():
        pitches = np.asarray([notes[index].pitch for index in indices], dtype=np.float32)
        starts = np.asarray([notes[index].start for index in indices], dtype=np.float32)
        previous_interval = np.diff(pitches, prepend=pitches[0])
        next_interval = np.diff(pitches, append=pitches[-1])
        previous_gap = np.diff(starts, prepend=starts[0])
        next_gap = np.diff(starts, append=starts[-1])
        previous_interval[0] = 0.0
        next_interval[-1] = 0.0
        previous_gap[0] = 0.0
        next_gap[-1] = 0.0
        step_in = (np.abs(previous_interval) <= 2.0) & (previous_interval != 0.0)
        step_out = (np.abs(next_interval) <= 2.0) & (next_interval != 0.0)
        same_direction = previous_interval * next_interval > 0.0
        opposite_direction = previous_interval * next_interval < 0.0
        passing = step_in & step_out & same_direction
        neighbor = step_in & step_out & opposite_direction
        resolves = step_out
        track_length = min(len(indices) / 32.0, 1.0)
        pitch_stability = float(np.clip(1.0 - pitches.std() / 12.0, 0.0, 1.0))

        for local_index, note_index in enumerate(indices):
            confidence = float(assignment.confidence[note_index])
            features[note_index, 4] = previous_interval[local_index] / 24.0
            features[note_index, 5] = next_interval[local_index] / 24.0
            features[note_index, 6] = min(previous_gap[local_index] / 4.0, 1.0)
            features[note_index, 7] = min(next_gap[local_index] / 4.0, 1.0)
            features[note_index, 8] = float(step_in[local_index])
            features[note_index, 9] = float(step_out[local_index])
            features[note_index, 10] = float(same_direction[local_index])
            features[note_index, 11] = float(passing[local_index])
            features[note_index, 12] = float(neighbor[local_index])
            features[note_index, 13] = float(resolves[local_index])
            features[note_index, 14] = track_length
            features[note_index, 15] = pitch_stability
            features[note_index, 17] = 1.0 - confidence

    for group in _onset_groups(notes):
        ordered = sorted(group, key=lambda index: notes[index].pitch)
        voice_order = [int(assignment.voice_ids[index]) for index in ordered]
        for left, right in zip(voice_order, voice_order[1:]):
            if voice_mean_pitch[left] > voice_mean_pitch[right]:
                for index in group:
                    features[index, 16] = 1.0
                break

    weighted_columns = list(range(1, 17))
    features[:, weighted_columns] *= assignment.confidence[:, None]
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
