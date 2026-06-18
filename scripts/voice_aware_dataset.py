from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from midi_error_detector.data import (
    KIND_TO_ID,
    NULL_CORRECTION_ID,
    MaestroWrongNoteDataset,
    NoteEvent,
    corrupt_note_window,
    note_features,
    theory_detection_weights,
)
from voice_assignment import assign_voices, build_voice_features


@dataclass(frozen=True)
class PieceObservation:
    notes: list[NoteEvent]
    features: np.ndarray
    is_error: np.ndarray
    target_pitch: np.ndarray
    correction_target: np.ndarray
    error_kind: np.ndarray
    det_weight: np.ndarray
    voice_features: np.ndarray


def build_piece_observation(
    notes: list[NoteEvent],
    error_rate: float,
    seed: int,
    voice_method: str,
    beam_width: int = 12,
) -> PieceObservation:
    rng = np.random.default_rng(seed)
    corrupted, is_error, target_pitch, kinds = corrupt_note_window(
        notes,
        rng,
        error_rate,
    )
    kind_ids = np.asarray(
        [
            KIND_TO_ID[
                "clean"
                if kind == "clean"
                else "delete"
                if kind == "delete_touch"
                else "replace"
            ]
            for kind in kinds
        ],
        dtype=np.int64,
    )
    correction_target = target_pitch.copy()
    correction_target[kind_ids == KIND_TO_ID["delete"]] = NULL_CORRECTION_ID
    features = note_features(corrupted)
    det_weight = theory_detection_weights(features, is_error)
    assignment = assign_voices(
        corrupted,
        method=voice_method,
        beam_width=beam_width,
    )
    voice_features = build_voice_features(corrupted, assignment)
    return PieceObservation(
        notes=corrupted,
        features=features,
        is_error=is_error.astype(np.float32),
        target_pitch=target_pitch.astype(np.int64),
        correction_target=correction_target.astype(np.int64),
        error_kind=kind_ids,
        det_weight=det_weight.astype(np.float32),
        voice_features=voice_features.astype(np.float32),
    )


def _slice_and_pad(
    values: np.ndarray,
    start: int,
    window_size: int,
) -> np.ndarray:
    sliced = values[start : start + window_size]
    pad = window_size - len(sliced)
    if pad <= 0:
        return sliced
    if sliced.ndim == 1:
        return np.pad(sliced, (0, pad))
    return np.pad(sliced, ((0, pad), (0, 0)))


def slice_piece_observation(
    piece: PieceObservation,
    start: int,
    window_size: int,
) -> dict[str, np.ndarray]:
    length = max(0, min(window_size, len(piece.notes) - start))
    mask = np.zeros(window_size, dtype=np.float32)
    mask[:length] = 1.0
    return {
        "features": _slice_and_pad(piece.features, start, window_size),
        "is_error": _slice_and_pad(piece.is_error, start, window_size),
        "target_pitch": _slice_and_pad(piece.target_pitch, start, window_size),
        "correction_target": _slice_and_pad(piece.correction_target, start, window_size),
        "error_kind": _slice_and_pad(piece.error_kind, start, window_size),
        "det_weight": _slice_and_pad(piece.det_weight, start, window_size),
        "voice_features": _slice_and_pad(piece.voice_features, start, window_size),
        "mask": mask,
    }


class PieceConsistentVoiceDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        voice_method: str,
        version: str = "v3.0.0",
        window_size: int = 256,
        stride: int = 128,
        error_rate: float = 0.01,
        seed: int = 13,
        beam_width: int = 12,
        max_files: int | None = None,
        verbose: bool = True,
    ) -> None:
        source = MaestroWrongNoteDataset(
            root=root,
            split=split,
            version=version,
            window_size=window_size,
            stride=stride,
            error_rate=0.0,
            seed=seed,
            max_files=max_files,
            cache_notes=True,
            verbose=verbose,
        )
        self.files = source.files
        self.window_size = window_size
        self.stride = stride
        self.error_rate = error_rate
        self.seed = seed
        self.voice_method = voice_method
        self.beam_width = beam_width
        self.index: list[tuple[int, int]] = []
        self._note_counts: list[int] = []
        self._pieces: list[PieceObservation] = []
        iterator = tqdm(
            range(len(self.files)),
            desc=f"building {split} piece-consistent voices ({voice_method})",
            unit="file",
            dynamic_ncols=True,
            disable=not verbose,
        )
        for file_id in iterator:
            piece = build_piece_observation(
                source._get_file_notes(file_id),
                error_rate=error_rate,
                seed=seed + file_id,
                voice_method=voice_method,
                beam_width=beam_width,
            )
            self._pieces.append(piece)
            note_count = len(piece.notes)
            self._note_counts.append(note_count)
            if note_count == 0:
                continue
            starts = list(range(0, max(1, note_count - window_size + 1), stride))
            final_start = max(0, note_count - window_size)
            if not starts or starts[-1] != final_start:
                starts.append(final_start)
            self.index.extend((file_id, start) for start in starts)

    def __len__(self) -> int:
        return len(self.index)

    def set_epoch(self, epoch: int) -> None:
        if epoch != 0:
            raise ValueError("Piece-consistent verifier datasets are fixed at epoch 0.")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        file_id, start = self.index[idx]
        sample = slice_piece_observation(
            self._pieces[file_id],
            start=start,
            window_size=self.window_size,
        )
        return {
            key: torch.from_numpy(value)
            for key, value in sample.items()
        }
