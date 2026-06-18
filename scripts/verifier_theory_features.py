from __future__ import annotations

import torch


THEORY_COLUMNS = {
    "harmonic_density": 12,
    "harmonic_consonance": 14,
    "major_scale_distance": 18,
    "minor_scale_distance": 19,
    "chord_tone": 20,
    "step_in": 26,
    "step_out": 27,
    "passing_tone": 29,
    "neighbor_tone": 30,
    "resolves_by_step": 31,
    "non_chord_resolution": 32,
    "local_duration": 33,
    "downbeat_strength": 34,
    "subdivision_strength": 35,
}

LOCAL_THEORY_SOURCE_COUNT = 9
LOCAL_STATS_PER_SOURCE = 5
INTERACTION_COUNT = 12
LOCAL_THEORY_SIZE = LOCAL_THEORY_SOURCE_COUNT * LOCAL_STATS_PER_SOURCE
THEORY_INTERACTION_SIZE = LOCAL_THEORY_SIZE + INTERACTION_COUNT

ORNAMENT_PROTECTION_INDEX = LOCAL_THEORY_SIZE
UNRESOLVED_NON_CHORD_INDEX = LOCAL_THEORY_SIZE + 1
ACCIDENTAL_TOUCH_INDEX = LOCAL_THEORY_SIZE + 2


def _masked_local_stats(
    values: torch.Tensor,
    valid: torch.Tensor,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    means = torch.zeros_like(values)
    maxes = torch.zeros_like(values)
    length = values.shape[1]
    for index in range(length):
        left = max(0, index - radius)
        right = min(length, index + radius + 1)
        local_valid = valid[:, left:right]
        local_values = values[:, left:right]
        count = local_valid.sum(dim=1).clamp_min(1)
        means[:, index] = (
            local_values * local_valid.to(local_values.dtype)
        ).sum(dim=1) / count
        masked = local_values.masked_fill(~local_valid, float("-inf"))
        local_max = masked.max(dim=1).values
        maxes[:, index] = torch.where(
            torch.isfinite(local_max),
            local_max,
            torch.zeros_like(local_max),
        )
    return means, maxes


def _column(raw_features: torch.Tensor, name: str) -> torch.Tensor:
    return raw_features[..., THEORY_COLUMNS[name]].clamp(0.0, 1.0)


def build_theory_interaction_features(
    raw_features: torch.Tensor,
    valid: torch.Tensor,
    three_probability: torch.Tensor,
    binary_probability: torch.Tensor,
    delete_probability: torch.Tensor,
    forward_surprise: torch.Tensor,
    backward_surprise: torch.Tensor,
) -> torch.Tensor:
    valid = valid.bool()
    chord_tone = _column(raw_features, "chord_tone")
    passing_tone = _column(raw_features, "passing_tone")
    neighbor_tone = _column(raw_features, "neighbor_tone")
    non_chord_resolution = _column(raw_features, "non_chord_resolution")
    local_duration = _column(raw_features, "local_duration")
    downbeat = _column(raw_features, "downbeat_strength")
    subdivision = _column(raw_features, "subdivision_strength")
    harmonic_density = _column(raw_features, "harmonic_density")
    harmonic_consonance = _column(raw_features, "harmonic_consonance")
    scale_distance = torch.minimum(
        _column(raw_features, "major_scale_distance"),
        _column(raw_features, "minor_scale_distance"),
    )

    theory_sources = [
        chord_tone,
        passing_tone,
        neighbor_tone,
        non_chord_resolution,
        local_duration,
        downbeat,
        subdivision,
        scale_distance,
        harmonic_consonance,
    ]
    local_features: list[torch.Tensor] = []
    for source in theory_sources:
        mean4, max4 = _masked_local_stats(source, valid, 4)
        mean8, max8 = _masked_local_stats(source, valid, 8)
        local_features.extend([mean4, max4, mean8, max8, source - mean8])

    step_in = _column(raw_features, "step_in")
    step_out = _column(raw_features, "step_out")
    resolves = torch.maximum(
        _column(raw_features, "resolves_by_step"),
        non_chord_resolution,
    )
    ornament = torch.maximum(
        torch.maximum(passing_tone, neighbor_tone),
        non_chord_resolution,
    )
    ornament_protection = torch.maximum(
        ornament,
        torch.minimum(step_in, step_out) * resolves,
    )
    unresolved_non_chord = (
        (1.0 - chord_tone) * scale_distance * (1.0 - resolves)
    )

    shortness = (1.0 - local_duration).clamp(0.0, 1.0)
    pitch_delta = raw_features[..., 4].abs()
    nearby_pitch = (1.0 - pitch_delta * 8.0).clamp(0.0, 1.0)
    accidental_touch = (
        shortness
        * nearby_pitch
        * delete_probability.clamp(0.0, 1.0)
        * (0.5 + 0.5 * harmonic_density)
    )

    three_probability = three_probability.clamp(0.0, 1.0)
    binary_probability = binary_probability.clamp(0.0, 1.0)
    delete_probability = delete_probability.clamp(0.0, 1.0)
    forward_surprise = forward_surprise.clamp(0.0, 1.0)
    backward_surprise = backward_surprise.clamp(0.0, 1.0)
    max_surprise = torch.maximum(forward_surprise, backward_surprise)
    min_surprise = torch.minimum(forward_surprise, backward_surprise)
    consensus = torch.minimum(three_probability, binary_probability)
    disagreement = (three_probability - binary_probability).abs()
    theory_support = torch.maximum(chord_tone, ornament_protection)

    interactions = [
        ornament_protection,
        unresolved_non_chord,
        accidental_touch,
        downbeat * unresolved_non_chord * max_surprise,
        consensus,
        disagreement,
        consensus * theory_support,
        disagreement * unresolved_non_chord,
        min_surprise * (1.0 - ornament_protection),
        (forward_surprise - backward_surprise).abs(),
        shortness * max_surprise,
        delete_probability * nearby_pitch * (1.0 - ornament_protection),
    ]
    result = torch.stack(local_features + interactions, dim=-1)
    result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result * valid.unsqueeze(-1).to(result.dtype)
