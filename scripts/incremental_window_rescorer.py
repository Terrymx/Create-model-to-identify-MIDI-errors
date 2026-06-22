from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from incremental_global_context import (
    PieceEdit,
    apply_piece_edits,
    window_state_key,
)


@dataclass(frozen=True)
class CandidateContext:
    candidate_index: int
    window_start: int
    local_position: int
    proposals: tuple[int, int]


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


class IncrementalCandidateRescorer:
    def __init__(
        self,
        *,
        window_cache: IncrementalWindowCache,
        score_candidate: Callable[[Any, int, np.ndarray], float],
    ) -> None:
        self.window_cache = window_cache
        self.score_candidate = score_candidate

    def score(
        self,
        candidate: CandidateContext,
        edits: Iterable[PieceEdit],
    ) -> float:
        window = self.window_cache.get(candidate.window_start, edits)
        proposals = np.asarray(candidate.proposals, dtype=np.int64)
        return float(
            self.score_candidate(
                window,
                int(candidate.local_position),
                proposals,
            )
        )


class FrozenWindowComputer:
    def __init__(self, models: tuple, device, surprise_groups: int = 4) -> None:
        self.models = models
        self.device = device
        self.surprise_groups = surprise_groups

    def __call__(self, window_features: np.ndarray, mask: np.ndarray) -> dict:
        import torch

        from counterfactual_edit_features import directional_pitch_distribution
        from run_frozen_union_candidate_context_verifier import (
            _masked_local_stats,
            append_theory_features,
            detector_signals,
            directional_evidence,
        )

        (
            three_model,
            three_args,
            binary_model,
            binary_args,
            forward_model,
            forward_args,
            backward_model,
            backward_args,
        ) = self.models
        raw = torch.from_numpy(window_features).unsqueeze(0).to(self.device)
        valid_mask = torch.from_numpy(mask).unsqueeze(0).to(self.device).bool()
        with torch.no_grad():
            three = detector_signals(
                three_model,
                three_args,
                forward_model,
                forward_args,
                backward_model,
                backward_args,
                raw,
                valid_mask,
                self.surprise_groups,
            )
            binary = detector_signals(
                binary_model,
                binary_args,
                forward_model,
                forward_args,
                backward_model,
                backward_args,
                raw,
                valid_mask,
                self.surprise_groups,
            )
            forward_evidence, forward_available = directional_evidence(
                forward_model,
                raw,
                valid_mask,
                "forward",
                forward_args.safe_feature_columns,
            )
            backward_evidence, backward_available = directional_evidence(
                backward_model,
                raw,
                valid_mask,
                "backward",
                backward_args.safe_feature_columns,
            )
            forward_probability, _ = directional_pitch_distribution(
                forward_model,
                raw,
                valid_mask,
                "forward",
                forward_args.safe_feature_columns,
            )
            backward_probability, _ = directional_pitch_distribution(
                backward_model,
                raw,
                valid_mask,
                "backward",
                backward_args.safe_feature_columns,
            )
            available = (
                valid_mask
                & three["available"]
                & binary["available"]
                & forward_available
                & backward_available
            )
            three_candidate = three["probability"] >= 0.30
            binary_candidate = binary["probability"] >= 0.25
            candidate_mask = available & (three_candidate | binary_candidate)
            forward_surprise = forward_evidence[..., 0]
            backward_surprise = backward_evidence[..., 0]
            avg_surprise = 0.5 * (forward_surprise + backward_surprise)
            max_surprise = torch.maximum(forward_surprise, backward_surprise)
            max_prob = torch.maximum(three["probability"], binary["probability"])
            aggregate = torch.stack(
                [
                    avg_surprise,
                    torch.minimum(forward_surprise, backward_surprise),
                    max_surprise,
                ],
                dim=-1,
            )
            cross = torch.stack(
                [
                    three["probability"],
                    binary["probability"],
                    max_prob,
                    torch.minimum(three["probability"], binary["probability"]),
                    (binary["probability"] - three["probability"]).clamp(-1.0, 1.0),
                    three_candidate.float(),
                    binary_candidate.float(),
                    (three_candidate & binary_candidate).float(),
                ],
                dim=-1,
            )
            context_sources = [
                three["probability"],
                binary["probability"],
                max_prob,
                avg_surprise,
                max_surprise,
                candidate_mask.float(),
            ]
            local_features = []
            for source in context_sources:
                mean4, max4 = _masked_local_stats(source, available, 4)
                mean8, max8 = _masked_local_stats(source, available, 8)
                local_features.extend(
                    [mean4, max4, mean8, max8, source - mean8]
                )
            verifier = torch.cat(
                [
                    raw,
                    three["features"],
                    binary["features"],
                    cross,
                    forward_evidence,
                    backward_evidence,
                    aggregate,
                    torch.stack(local_features, dim=-1),
                ],
                dim=-1,
            )
            verifier = append_theory_features(
                verifier,
                raw,
                available,
                three["probability"],
                binary["probability"],
                binary["features"][..., -1],
                forward_surprise,
                backward_surprise,
            )
        return {
            "raw": raw[0].cpu(),
            "mask": valid_mask[0].cpu(),
            "verifier": verifier[0].cpu(),
            "score_columns": torch.stack(
                [
                    three["probability"],
                    binary["probability"],
                    max_prob,
                    avg_surprise,
                    max_surprise,
                ],
                dim=-1,
            )[0].cpu(),
            "forward_probability": forward_probability[0].cpu(),
            "backward_probability": backward_probability[0].cpu(),
        }
