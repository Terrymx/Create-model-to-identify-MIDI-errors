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


def correction_pitch_distribution(
    outputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    correction_logits = outputs.get("correction_logits")
    if correction_logits is not None:
        probability = torch.softmax(correction_logits, dim=-1)[..., :128]
        return probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    pitch_logits = outputs.get("pitch_logits")
    if pitch_logits is None:
        raise ValueError("Model outputs contain no pitch or correction logits.")
    return torch.softmax(pitch_logits, dim=-1)


def apply_observed_pitch_edit(
    raw_features: torch.Tensor,
    positions: torch.Tensor,
    proposed_pitch: torch.Tensor,
) -> torch.Tensor:
    if raw_features.ndim != 3:
        raise ValueError("Raw features must have shape [batch, length, features].")
    if positions.ndim != 1 or proposed_pitch.ndim != 1:
        raise ValueError("Positions and pitches must be one-dimensional.")
    if len(positions) != len(raw_features) or len(proposed_pitch) != len(raw_features):
        raise ValueError("One edit position and pitch are required per batch row.")
    edited = raw_features.clone()
    row = torch.arange(len(edited), device=edited.device)
    position = positions.long().clamp(0, edited.shape[1] - 1)
    pitch = proposed_pitch.float().clamp(0, 127)
    phase = 2.0 * torch.pi * torch.remainder(pitch, 12.0) / 12.0
    edited[row, position, 0] = pitch / 127.0
    edited[row, position, 6] = torch.sin(phase)
    edited[row, position, 7] = torch.cos(phase)
    return edited


def local_edit_impact_features(
    original_forward_log_probability: torch.Tensor,
    original_backward_log_probability: torch.Tensor,
    edited_forward_log_probability: torch.Tensor,
    edited_backward_log_probability: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
    radii: tuple[int, ...] = (4, 8, 16),
) -> torch.Tensor:
    shapes = {
        tuple(original_forward_log_probability.shape),
        tuple(original_backward_log_probability.shape),
        tuple(edited_forward_log_probability.shape),
        tuple(edited_backward_log_probability.shape),
        tuple(mask.shape),
    }
    if len(shapes) != 1:
        raise ValueError("Likelihood and mask tensors must have equal shapes.")
    forward_delta = edited_forward_log_probability - original_forward_log_probability
    backward_delta = edited_backward_log_probability - original_backward_log_probability
    combined_delta = 0.5 * (forward_delta + backward_delta)
    rows = []
    for row_index, center_value in enumerate(positions.tolist()):
        values = []
        for radius in radii:
            left = max(0, int(center_value) - radius)
            right = min(mask.shape[1], int(center_value) + radius + 1)
            valid = mask[row_index, left:right].bool()
            count = int(valid.sum())
            if count == 0:
                values.extend([0.0] * 5)
                continue
            forward_sum = float(forward_delta[row_index, left:right][valid].sum())
            backward_sum = float(backward_delta[row_index, left:right][valid].sum())
            combined = combined_delta[row_index, left:right][valid]
            combined_sum = float(combined.sum())
            improved_fraction = float((combined > 0).float().mean())
            agreement = float(
                (forward_sum >= 0 and backward_sum >= 0)
                or (forward_sum < 0 and backward_sum < 0)
            )
            values.extend(
                [
                    forward_sum,
                    backward_sum,
                    combined_sum,
                    improved_fraction,
                    agreement,
                ]
            )
        rows.append(values)
    return torch.tensor(
        rows,
        dtype=original_forward_log_probability.dtype,
        device=original_forward_log_probability.device,
    )


@torch.no_grad()
def directional_pitch_distribution(
    model: torch.nn.Module,
    raw_features: torch.Tensor,
    mask: torch.Tensor,
    direction: str,
    safe_columns: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if direction not in {"forward", "backward"}:
        raise ValueError("Direction must be 'forward' or 'backward'.")
    directional = raw_features[:, :, safe_columns]
    oriented_mask = mask.bool()
    if direction == "backward":
        directional = directional.flip(1)
        oriented_mask = oriented_mask.flip(1)
    shifted = torch.zeros_like(directional)
    shifted[:, 1:] = directional[:, :-1]
    available = oriented_mask.clone()
    available[:, 0] = False
    probability = torch.softmax(model.predict_pitch(shifted, causal=True), dim=-1)
    if direction == "backward":
        probability = probability.flip(1)
        available = available.flip(1)
    return probability, available


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
