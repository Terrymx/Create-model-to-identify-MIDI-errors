from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from counterfactual_edit_features import (
    apply_observed_pitch_edit,
    counterfactual_target_features,
    directional_pitch_distribution,
    local_edit_impact_features,
)
from incremental_global_context import PieceEdit
from incremental_window_rescorer import FrozenWindowComputer, IncrementalWindowCache
from run_frozen_union_candidate_context_verifier import (
    append_piece_relative_features,
)
from run_counterfactual_edit_verifier import build_c_variant_features
from run_verifier_improvement_suite import (
    PIECE_RELATIVE_SIZE,
    split_old_and_theory_features,
)
from verifier_theory_features import THEORY_INTERACTION_SIZE


def _observed_log_probability(
    probability: torch.Tensor,
    raw: torch.Tensor,
) -> torch.Tensor:
    pitch = torch.round(raw[..., 0] * 127.0).long().clamp(0, 127)
    return probability.gather(-1, pitch.unsqueeze(-1)).squeeze(-1).clamp_min(
        1e-9
    ).log()


class IncrementalPieceScorer:
    def __init__(
        self,
        *,
        piece_id: int,
        piece_features: np.ndarray,
        candidate_arrays: dict[str, np.ndarray],
        models: tuple,
        verifier_model,
        device: torch.device,
        window_size: int = 256,
    ) -> None:
        self.piece_id = int(piece_id)
        self.arrays = candidate_arrays
        self.models = models
        self.verifier_model = verifier_model
        self.device = device
        self.window_cache = IncrementalWindowCache(
            piece_id=piece_id,
            piece_features=piece_features,
            window_size=window_size,
            compute=FrozenWindowComputer(models, device),
        )
        self.score_cache: dict[tuple[tuple[int, int], ...], np.ndarray] = {}

    def score_all(self, edits: Iterable[PieceEdit]) -> np.ndarray:
        edit_tuple = tuple(sorted(edits))
        key = tuple((edit.position, edit.proposed_pitch) for edit in edit_tuple)
        if key in self.score_cache:
            return self.score_cache[key]
        window_outputs = [
            self.window_cache.get(int(start), edit_tuple)
            for start in self.arrays["window_starts"]
        ]
        local_positions = self.arrays["local_positions"].astype(np.int64)
        verifier_rows = torch.stack(
            [
                output["verifier"][position]
                for output, position in zip(window_outputs, local_positions)
            ]
        )
        score_columns = torch.stack(
            [
                output["score_columns"][position]
                for output, position in zip(window_outputs, local_positions)
            ]
        )
        combined = append_piece_relative_features(
            verifier_rows,
            torch.from_numpy(self.arrays["file_ids"].astype(np.int64)),
            torch.from_numpy(self.arrays["positions"].astype(np.int64)),
            torch.from_numpy(self.arrays["file_note_counts"].astype(np.int64)),
            score_columns,
        )
        base_features, _ = split_old_and_theory_features(
            combined,
            theory_size=THEORY_INTERACTION_SIZE,
            piece_relative_size=PIECE_RELATIVE_SIZE,
        )
        b_features = self._dynamic_b_features(window_outputs, local_positions)
        c_features = self._dynamic_c_features(
            window_outputs,
            local_positions,
            edit_tuple,
        )
        features = build_c_variant_features(
            base_features.numpy(),
            b_features,
            self.arrays["b_ranking"],
            c_features,
            self.arrays["c_ranking"],
            "C2",
            b_variant="B2",
        )
        scores = self.verifier_model.predict_proba(features)[:, 1]
        self.score_cache[key] = scores
        return scores

    def _dynamic_b_features(
        self,
        outputs: list[dict],
        local_positions: np.ndarray,
    ) -> np.ndarray:
        rows = []
        for row, (output, position) in enumerate(zip(outputs, local_positions)):
            forward = output["forward_probability"][position].to(self.device)
            backward = output["backward_probability"][position].to(self.device)
            raw = output["raw"][position].to(self.device)
            observed = torch.round(raw[0] * 127.0).long().view(1)
            proposals = torch.from_numpy(
                self.arrays["proposals"][row].astype(np.int64)
            ).to(self.device)
            count = len(proposals)
            rows.append(
                counterfactual_target_features(
                    forward.view(1, -1).repeat(count, 1),
                    backward.view(1, -1).repeat(count, 1),
                    observed.repeat(count),
                    proposals,
                ).cpu()
            )
        return torch.stack(rows).numpy().astype(np.float32)

    def _dynamic_c_features(
        self,
        outputs: list[dict],
        local_positions: np.ndarray,
        edits: tuple[PieceEdit, ...],
    ) -> np.ndarray:
        (
            _,
            _,
            _,
            _,
            forward_model,
            forward_args,
            backward_model,
            backward_args,
        ) = self.models
        rows = []
        for row, (output, position) in enumerate(zip(outputs, local_positions)):
            proposal_rows = []
            raw = output["raw"].unsqueeze(0).to(self.device)
            mask = output["mask"].unsqueeze(0).to(self.device).bool()
            original_forward = output["forward_probability"].unsqueeze(0).to(
                self.device
            )
            original_backward = output["backward_probability"].unsqueeze(0).to(
                self.device
            )
            position_tensor = torch.tensor([position], device=self.device)
            for proposed in self.arrays["c_proposals"][row]:
                edited = apply_observed_pitch_edit(
                    raw,
                    position_tensor,
                    torch.tensor([int(proposed)], device=self.device),
                )
                edited_forward, _ = directional_pitch_distribution(
                    forward_model,
                    edited,
                    mask,
                    "forward",
                    forward_args.safe_feature_columns,
                )
                edited_backward, _ = directional_pitch_distribution(
                    backward_model,
                    edited,
                    mask,
                    "backward",
                    backward_args.safe_feature_columns,
                )
                proposal_rows.append(
                    local_edit_impact_features(
                        _observed_log_probability(original_forward, raw),
                        _observed_log_probability(original_backward, raw),
                        _observed_log_probability(edited_forward, edited),
                        _observed_log_probability(edited_backward, edited),
                        position_tensor,
                        mask,
                        radii=(4, 8, 16),
                    )[0].cpu()
                )
            rows.append(torch.stack(proposal_rows))
        return torch.stack(rows).numpy().astype(np.float32)
