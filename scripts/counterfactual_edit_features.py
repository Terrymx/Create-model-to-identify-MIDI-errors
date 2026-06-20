from __future__ import annotations

import torch


COUNTERFACTUAL_TARGET_FEATURE_NAMES = (
    "forward_log_gain",
    "backward_log_gain",
    "mean_log_gain",
    "min_log_gain",
    "max_log_gain",
    "absolute_gain_disagreement",
    "forward_observed_probability",
    "forward_proposed_probability",
    "backward_observed_probability",
    "backward_proposed_probability",
    "both_directions_improve",
    "either_direction_improves",
    "absolute_pitch_distance_normalized",
    "octave_relation",
    "ascending_edit",
    "descending_edit",
    "chromatic_neighbor",
)


def build_replacement_proposals(
    detector_probability: torch.Tensor,
    forward_probability: torch.Tensor,
    backward_probability: torch.Tensor,
    observed_pitch: torch.Tensor,
    source_top_k: int = 3,
    max_proposals: int = 4,
) -> torch.Tensor:
    distributions = (
        detector_probability,
        forward_probability,
        backward_probability,
    )
    if any(values.shape != detector_probability.shape for values in distributions):
        raise ValueError("All proposal distributions must have equal shape.")
    if detector_probability.ndim != 2 or detector_probability.shape[1] != 128:
        raise ValueError("Proposal distributions must have shape [rows, 128].")
    if observed_pitch.ndim != 1 or len(observed_pitch) != len(detector_probability):
        raise ValueError("Observed pitches must align with proposal rows.")
    if source_top_k <= 0 or max_proposals <= 0:
        raise ValueError("Proposal limits must be positive.")

    observed = observed_pitch.long().clamp(0, 127)
    rows: list[list[int]] = []
    for row_index in range(len(observed)):
        observed_value = int(observed[row_index])
        union: set[int] = set()
        for distribution in distributions:
            ranked = sorted(
                (pitch for pitch in range(128) if pitch != observed_value),
                key=lambda pitch: (-float(distribution[row_index, pitch]), pitch),
            )
            union.update(ranked[:source_top_k])
        ranked_union = sorted(
            union,
            key=lambda pitch: (
                -max(
                    float(distribution[row_index, pitch])
                    for distribution in distributions
                ),
                pitch,
            ),
        )
        if len(ranked_union) < max_proposals:
            remaining = sorted(
                (
                    pitch
                    for pitch in range(128)
                    if pitch != observed_value and pitch not in union
                ),
                key=lambda pitch: (
                    -max(
                        float(distribution[row_index, pitch])
                        for distribution in distributions
                    ),
                    pitch,
                ),
            )
            ranked_union.extend(remaining[: max_proposals - len(ranked_union)])
        rows.append(ranked_union[:max_proposals])
    return torch.tensor(rows, dtype=torch.long, device=detector_probability.device)


def counterfactual_target_features(
    forward_probability: torch.Tensor,
    backward_probability: torch.Tensor,
    observed_pitch: torch.Tensor,
    proposed_pitch: torch.Tensor,
) -> torch.Tensor:
    if forward_probability.shape != backward_probability.shape:
        raise ValueError("Forward and backward distributions must have equal shape.")
    if forward_probability.ndim != 2 or forward_probability.shape[1] != 128:
        raise ValueError("Directional distributions must have shape [rows, 128].")
    observed = observed_pitch.long().clamp(0, 127)
    proposed = proposed_pitch.long().clamp(0, 127)
    if observed.shape != proposed.shape or observed.ndim != 1:
        raise ValueError("Observed and proposed pitch must be one-dimensional and aligned.")
    if len(observed) != len(forward_probability):
        raise ValueError("Pitch rows must align with probability rows.")

    forward_observed = forward_probability.gather(1, observed[:, None]).squeeze(1)
    forward_proposed = forward_probability.gather(1, proposed[:, None]).squeeze(1)
    backward_observed = backward_probability.gather(1, observed[:, None]).squeeze(1)
    backward_proposed = backward_probability.gather(1, proposed[:, None]).squeeze(1)
    forward_gain = (
        forward_proposed.clamp_min(1e-9).log()
        - forward_observed.clamp_min(1e-9).log()
    )
    backward_gain = (
        backward_proposed.clamp_min(1e-9).log()
        - backward_observed.clamp_min(1e-9).log()
    )
    pitch_delta = proposed - observed
    absolute_distance = pitch_delta.abs().float()
    return torch.stack(
        [
            forward_gain,
            backward_gain,
            0.5 * (forward_gain + backward_gain),
            torch.minimum(forward_gain, backward_gain),
            torch.maximum(forward_gain, backward_gain),
            (forward_gain - backward_gain).abs(),
            forward_observed,
            forward_proposed,
            backward_observed,
            backward_proposed,
            ((forward_gain > 0) & (backward_gain > 0)).float(),
            ((forward_gain > 0) | (backward_gain > 0)).float(),
            (absolute_distance / 24.0).clamp(0.0, 1.0),
            ((absolute_distance > 0) & (pitch_delta.remainder(12) == 0)).float(),
            (pitch_delta > 0).float(),
            (pitch_delta < 0).float(),
            (absolute_distance == 1).float(),
        ],
        dim=1,
    )
