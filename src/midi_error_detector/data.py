"""MAESTRO MIDI loading and synthetic wrong-note generation.

The MAESTRO dataset contains aligned MIDI/audio performances plus metadata with a
suggested train/validation/test split.  This module uses only the MIDI archive and
turns each piece into windows of note events.  During training, clean note events
are corrupted online so the model sees an input pitch sequence, an error label for
each input note, and the clean target pitch that should replace wrong notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import pretty_midi
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

ErrorKind = Literal["clean", "neighbor", "nearby", "nearby_plus_touch", "delete_touch"]
KIND_TO_ID = {"clean": 0, "replace": 1, "delete": 2}
FEATURE_SIZE = 16
_MAJOR_PROFILE = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
_MINOR_PROFILE = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)
_MAJOR_SCALE_PCS = {0, 2, 4, 5, 7, 9, 11}
_MINOR_SCALE_PCS = {0, 2, 3, 5, 7, 8, 10}
_CONSONANT_INTERVAL_CLASSES = {0, 3, 4, 5, 7, 8, 9}


@dataclass(frozen=True)
class NoteEvent:
    """A single piano note event extracted from a MIDI file."""

    pitch: int
    velocity: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def load_maestro_metadata(root: str | Path, split: str, version: str = "v3.0.0") -> pd.DataFrame:
    """Load MAESTRO metadata and return rows for one split.

    Args:
        root: Directory containing the extracted MAESTRO MIDI archive.
        split: One of ``train``, ``validation``, or ``test``.
        version: MAESTRO version string used in the metadata filename.
    """

    root = Path(root)
    metadata_path = root / f"maestro-{version}.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Could not find {metadata_path}. Download and unzip the MIDI-only archive from "
            "https://magenta.tensorflow.org/datasets/maestro first."
        )
    metadata = pd.read_csv(metadata_path)
    if split not in set(metadata["split"]):
        raise ValueError(f"Unknown split {split!r}; expected one of {sorted(metadata['split'].unique())}")
    return metadata[metadata["split"] == split].reset_index(drop=True)


def extract_note_events(midi_path: str | Path) -> list[NoteEvent]:
    """Read a MIDI file and return sorted, non-drum note events."""

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes: list[NoteEvent] = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            if note.end <= note.start:
                continue
            notes.append(NoteEvent(note.pitch, note.velocity, note.start, note.end))
    notes.sort(key=lambda n: (n.start, n.pitch, n.end))
    return notes


def _clip_pitch(pitch: int, low: int, high: int) -> int:
    return int(min(high, max(low, pitch)))


def _random_pitch_shift(rng: np.random.Generator, max_shift: int) -> int:
    shift = 0
    while shift == 0:
        shift = int(rng.integers(-max_shift, max_shift + 1))
    return shift


def corrupt_note_window(
    notes: Iterable[NoteEvent],
    rng: np.random.Generator,
    error_rate: float = 0.08,
    nearby_max_shift: int = 7,
    piano_low: int = 21,
    piano_high: int = 108,
) -> tuple[list[NoteEvent], np.ndarray, np.ndarray, list[ErrorKind]]:
    """Inject synthetic performance/transcription errors into a clean note window.

    Error types:
    * ``neighbor``: replace a note by an adjacent semitone, modelling a slipped
      black/white neighbouring key.
    * ``nearby``: replace a note by a random pitch within ``nearby_max_shift``
      semitones, modelling a musically nearby wrong note.
    * ``nearby_plus_touch``: replace a note and add an extra adjacent accidental
      touch at almost the same onset.

    Returns corrupted notes, binary error labels, clean target pitches, and error
    kind strings aligned to the corrupted notes. Replacement mistakes use the
    original clean pitch as target; extra accidental touches are labelled as
    ``delete_touch`` so the model can learn that they should be removed instead
    of replaced.
    """

    corrupted: list[NoteEvent] = []
    error_labels: list[int] = []
    target_pitches: list[int] = []
    kinds: list[ErrorKind] = []

    for note in notes:
        if rng.random() >= error_rate:
            corrupted.append(note)
            error_labels.append(0)
            target_pitches.append(note.pitch)
            kinds.append("clean")
            continue

        error_type = rng.choice(["neighbor", "nearby", "nearby_plus_touch"], p=[0.45, 0.40, 0.15])
        if error_type == "neighbor":
            new_pitch = _clip_pitch(note.pitch + int(rng.choice([-1, 1])), piano_low, piano_high)
        else:
            new_pitch = _clip_pitch(note.pitch + _random_pitch_shift(rng, nearby_max_shift), piano_low, piano_high)

        corrupted.append(NoteEvent(new_pitch, note.velocity, note.start, note.end))
        error_labels.append(1)
        target_pitches.append(note.pitch)
        kinds.append(error_type)  # type: ignore[arg-type]

        if error_type == "nearby_plus_touch":
            touch_pitch = _clip_pitch(note.pitch + int(rng.choice([-1, 1])), piano_low, piano_high)
            touch_start = max(0.0, note.start + float(rng.normal(0.0, 0.015)))
            touch_end = max(touch_start + 0.03, min(note.end, touch_start + max(0.05, note.duration * 0.35)))
            corrupted.append(NoteEvent(touch_pitch, max(1, int(note.velocity * 0.7)), touch_start, touch_end))
            error_labels.append(1)
            target_pitches.append(touch_pitch)
            kinds.append("delete_touch")

    order = sorted(range(len(corrupted)), key=lambda i: (corrupted[i].start, corrupted[i].pitch, corrupted[i].end))
    corrupted = [corrupted[i] for i in order]
    return (
        corrupted,
        np.asarray([error_labels[i] for i in order], dtype=np.float32),
        np.asarray([target_pitches[i] for i in order], dtype=np.int64),
        [kinds[i] for i in order],
    )


def _estimate_key_tonics(pitches: np.ndarray, durations: np.ndarray) -> tuple[int, int]:
    """Estimate major/minor tonic pitch classes from a window-level pitch histogram."""

    pitch_classes = (pitches.astype(np.int64) % 12).tolist()
    weights = np.maximum(durations, 0.05)
    histogram = np.zeros(12, dtype=np.float32)
    for pitch_class, weight in zip(pitch_classes, weights):
        histogram[pitch_class] += float(weight)
    if histogram.sum() <= 0.0:
        return 0, 0
    histogram = histogram / histogram.sum()
    major_scores = [float(np.dot(histogram, np.roll(_MAJOR_PROFILE, tonic))) for tonic in range(12)]
    minor_scores = [float(np.dot(histogram, np.roll(_MINOR_PROFILE, tonic))) for tonic in range(12)]
    return int(np.argmax(major_scores)), int(np.argmax(minor_scores))


def _harmonic_context_features(
    starts: np.ndarray,
    pitches: np.ndarray,
    onset_tolerance: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return simple vertical-context features for notes that start together."""

    note_count = len(pitches)
    density = np.zeros(note_count, dtype=np.float32)
    nearest_interval = np.zeros(note_count, dtype=np.float32)
    consonance = np.zeros(note_count, dtype=np.float32)
    for idx in range(note_count):
        same_onset = np.flatnonzero(np.abs(starts - starts[idx]) <= onset_tolerance)
        same_onset = same_onset[same_onset != idx]
        if len(same_onset) == 0:
            continue
        intervals = np.abs(pitches[same_onset] - pitches[idx])
        interval_classes = (intervals.astype(np.int64) % 12).tolist()
        density[idx] = min(len(same_onset) / 6.0, 1.0)
        nearest_interval[idx] = min(float(intervals.min()) / 24.0, 1.0)
        consonance[idx] = float(any(interval_class in _CONSONANT_INTERVAL_CLASSES for interval_class in interval_classes))
    return density, nearest_interval, consonance


def note_features(notes: list[NoteEvent]) -> np.ndarray:
    """Convert note events into normalized model features.

    Besides raw performance features, this includes lightweight music-theory cues:
    estimated key/scale membership, profile strength for major/minor pitch classes,
    and simple same-onset harmonic context.
    """

    if not notes:
        return np.zeros((0, FEATURE_SIZE), dtype=np.float32)

    starts = np.asarray([n.start for n in notes], dtype=np.float32)
    pitches = np.asarray([n.pitch for n in notes], dtype=np.float32)
    velocities = np.asarray([n.velocity for n in notes], dtype=np.float32)
    durations = np.asarray([n.duration for n in notes], dtype=np.float32)
    delta_start = np.diff(starts, prepend=starts[0])
    pitch_delta = np.diff(pitches, prepend=pitches[0])
    abs_pitch_delta = np.abs(pitch_delta)

    pitch_class = (pitches % 12.0) / 12.0
    pitch_class_int = pitches.astype(np.int64) % 12
    major_tonic, minor_tonic = _estimate_key_tonics(pitches, durations)
    major_relative_pc = (pitch_class_int - major_tonic) % 12
    minor_relative_pc = (pitch_class_int - minor_tonic) % 12
    major_profile = _MAJOR_PROFILE / _MAJOR_PROFILE.max()
    minor_profile = _MINOR_PROFILE / _MINOR_PROFILE.max()
    in_major_scale = np.asarray([pc in _MAJOR_SCALE_PCS for pc in major_relative_pc], dtype=np.float32)
    in_minor_scale = np.asarray([pc in _MINOR_SCALE_PCS for pc in minor_relative_pc], dtype=np.float32)
    harmonic_density, nearest_harmonic_interval, harmonic_consonance = _harmonic_context_features(starts, pitches)

    return np.stack(
        [
            pitches / 127.0,
            velocities / 127.0,
            np.clip(np.log1p(durations) / np.log1p(10.0), 0.0, 1.0),
            np.clip(np.log1p(np.maximum(delta_start, 0.0)) / np.log1p(10.0), 0.0, 1.0),
            np.clip(pitch_delta / 24.0, -1.0, 1.0),
            np.sin(2.0 * np.pi * (starts % 1.0)),
            np.sin(2.0 * np.pi * pitch_class),
            np.cos(2.0 * np.pi * pitch_class),
            in_major_scale,
            in_minor_scale,
            major_profile[major_relative_pc],
            minor_profile[minor_relative_pc],
            harmonic_density,
            nearest_harmonic_interval,
            harmonic_consonance,
            np.clip(abs_pitch_delta / 12.0, 0.0, 1.0),
        ],
        axis=1,
    ).astype(np.float32)


class MaestroWrongNoteDataset(Dataset):
    """Windowed MAESTRO MIDI dataset with online synthetic wrong-note labels."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        version: str = "v3.0.0",
        window_size: int = 256,
        stride: int = 128,
        error_rate: float = 0.08,
        seed: int = 13,
        max_files: int | None = None,
        cache_notes: bool = True,
        verbose: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.error_rate = error_rate
        self.seed = seed
        self.epoch = 0
        self.cache_notes = cache_notes
        self.verbose = verbose

        metadata = load_maestro_metadata(self.root, split, version)
        if max_files is not None:
            metadata = metadata.iloc[:max_files]

        self.files: list[Path] = [self.root / name for name in metadata["midi_filename"].tolist()]
        self.index: list[tuple[int, int]] = []
        self._note_counts: list[int] = []
        self._notes_cache: list[list[NoteEvent] | None] = [None] * len(self.files)
        midi_iterator = tqdm(
            list(enumerate(self.files)),
            desc=f"loading {split} MIDI",
            unit="file",
            dynamic_ncols=True,
            disable=not self.verbose,
        )
        for file_id, midi_path in midi_iterator:
            notes = extract_note_events(midi_path)
            note_count = len(notes)
            if self.cache_notes:
                self._notes_cache[file_id] = notes
            self._note_counts.append(note_count)
            if note_count == 0:
                continue
            for start in range(0, max(1, note_count - window_size + 1), stride):
                self.index.append((file_id, start))
            if note_count > window_size and self.index[-1] != (file_id, note_count - window_size):
                self.index.append((file_id, note_count - window_size))

    def __len__(self) -> int:
        return len(self.index)

    def set_epoch(self, epoch: int) -> None:
        """Change the deterministic corruption seed for training-time resampling."""

        self.epoch = epoch

    def _get_file_notes(self, file_id: int) -> list[NoteEvent]:
        cached = self._notes_cache[file_id]
        if cached is not None:
            return cached
        notes = extract_note_events(self.files[file_id])
        if self.cache_notes:
            self._notes_cache[file_id] = notes
        return notes

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        file_id, start = self.index[idx]
        notes = self._get_file_notes(file_id)[start : start + self.window_size]
        rng = np.random.default_rng(self.seed + idx + self.epoch * 1_000_003)
        corrupted, is_error, target_pitch, kinds = corrupt_note_window(notes, rng, self.error_rate)

        # Extra touches can make the sequence slightly longer; trim/pad to a fixed window.
        corrupted = corrupted[: self.window_size]
        is_error = is_error[: self.window_size]
        target_pitch = target_pitch[: self.window_size]
        kind_ids = np.asarray(
            [KIND_TO_ID["clean" if kind == "clean" else "delete" if kind == "delete_touch" else "replace"] for kind in kinds],
            dtype=np.int64,
        )[: self.window_size]
        length = len(corrupted)
        features = note_features(corrupted)

        pad = self.window_size - length
        if pad > 0:
            features = np.pad(features, ((0, pad), (0, 0)))
            is_error = np.pad(is_error, (0, pad))
            target_pitch = np.pad(target_pitch, (0, pad))
            kind_ids = np.pad(kind_ids, (0, pad))

        mask = np.zeros(self.window_size, dtype=np.float32)
        mask[:length] = 1.0
        return {
            "features": torch.from_numpy(features),
            "is_error": torch.from_numpy(is_error.astype(np.float32)),
            "target_pitch": torch.from_numpy(target_pitch.astype(np.int64)),
            "error_kind": torch.from_numpy(kind_ids.astype(np.int64)),
            "mask": torch.from_numpy(mask),
        }
