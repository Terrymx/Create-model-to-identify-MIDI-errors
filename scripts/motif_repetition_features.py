from __future__ import annotations

import numpy as np


MOTIF_FEATURE_NAMES = (
    "motif_match_count",
    "motif_support",
    "best_motif_similarity",
    "mean_motif_similarity",
    "expected_pitch_stability",
    "observed_pitch_consensus",
    "best_proposal_consensus",
    "proposal_consensus_gain",
    "observed_abs_delta",
    "best_proposal_abs_delta",
    "has_positive_consensus_gain",
    "best_proposal_pitch",
)


def motif_feature_size() -> int:
    return len(MOTIF_FEATURE_NAMES)


def _context_signature(
    pitches: np.ndarray,
    center: int,
    radius: int,
) -> tuple[np.ndarray, float] | None:
    if center - radius < 0 or center + radius >= len(pitches):
        return None
    offsets = [
        index
        for index in range(center - radius, center + radius + 1)
        if index != center
    ]
    context = pitches[offsets].astype(np.float32)
    if context.size == 0:
        return None
    anchor = float(context[0])
    return context - anchor, anchor


def _signature_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.size == 0:
        return 0.0
    mean_abs_delta = float(np.mean(np.abs(left - right)))
    return max(0.0, 1.0 - mean_abs_delta / 12.0)


def _precompute_signatures(
    pitches: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.arange(radius, max(radius, len(pitches) - radius), dtype=np.int64)
    if centers.size == 0:
        return (
            centers,
            np.zeros((0, radius * 2), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    signatures = np.zeros((len(centers), radius * 2), dtype=np.float32)
    anchors = np.zeros(len(centers), dtype=np.float32)
    for row, center in enumerate(centers.tolist()):
        signature = _context_signature(pitches, center, radius)
        if signature is None:
            continue
        signatures[row], anchors[row] = signature
    return centers, signatures, anchors


def _pitch_consensus(pitch: float, expected_pitches: np.ndarray) -> tuple[float, float]:
    if expected_pitches.size == 0:
        return 0.0, 0.0
    distances = np.abs(expected_pitches.astype(np.float32) - float(pitch))
    best_distance = float(np.min(distances))
    return float(np.exp(-best_distance / 2.0)), best_distance


def compute_motif_repetition_features(
    *,
    piece_pitches: np.ndarray,
    candidate_positions: np.ndarray,
    observed_pitch: np.ndarray,
    proposals: np.ndarray,
    radius: int = 3,
    min_similarity: float = 0.85,
    exclude_radius: int | None = None,
) -> np.ndarray:
    """Compute same-piece motif/repetition evidence for candidate notes.

    Matching is transposition-invariant and excludes the candidate center pitch
    from the context signature.  For each matching center, the matched center
    pitch is transposed by the surrounding-context anchor difference to estimate
    the consensus pitch expected at the candidate position.
    """
    pitches = np.asarray(piece_pitches, dtype=np.float32)
    positions = np.asarray(candidate_positions, dtype=np.int64)
    observed = np.asarray(observed_pitch, dtype=np.float32)
    proposal_values = np.asarray(proposals, dtype=np.float32)
    if proposal_values.ndim == 1:
        proposal_values = proposal_values[:, None]
    if len(positions) != len(observed) or len(positions) != len(proposal_values):
        raise ValueError("candidate_positions, observed_pitch, and proposals must align.")
    if radius < 1:
        raise ValueError("radius must be positive.")
    exclusion = radius if exclude_radius is None else int(exclude_radius)
    features = np.zeros((len(positions), motif_feature_size()), dtype=np.float32)
    all_centers, all_signatures, all_anchors = _precompute_signatures(pitches, radius)
    for row, center in enumerate(positions.tolist()):
        current = _context_signature(pitches, center, radius)
        if current is None:
            continue
        current_signature, current_anchor = current
        if all_centers.size == 0:
            continue
        similarities = 1.0 - np.mean(
            np.abs(all_signatures - current_signature[None, :]),
            axis=1,
        ) / 12.0
        similarities = np.maximum(similarities, 0.0).astype(np.float32)
        keep = (similarities >= min_similarity) & (
            np.abs(all_centers - center) > exclusion
        )
        if not bool(keep.any()):
            continue
        similarity_array = similarities[keep]
        expected_array = pitches[all_centers[keep]] + (
            current_anchor - all_anchors[keep]
        )
        observed_consensus, observed_delta = _pitch_consensus(
            observed[row],
            expected_array,
        )
        proposal_consensus = []
        proposal_delta = []
        for pitch in proposal_values[row]:
            consensus, delta = _pitch_consensus(pitch, expected_array)
            proposal_consensus.append(consensus)
            proposal_delta.append(delta)
        proposal_consensus_array = np.asarray(proposal_consensus, dtype=np.float32)
        proposal_delta_array = np.asarray(proposal_delta, dtype=np.float32)
        best_index = int(np.argmax(proposal_consensus_array))
        best_consensus = float(proposal_consensus_array[best_index])
        best_delta = float(proposal_delta_array[best_index])
        gain = best_consensus - observed_consensus
        features[row] = np.asarray(
            [
                float(len(expected_array)),
                min(1.0, float(len(expected_array)) / 3.0),
                float(np.max(similarity_array)),
                float(np.mean(similarity_array)),
                float(1.0 / (1.0 + np.std(expected_array))),
                observed_consensus,
                best_consensus,
                gain,
                observed_delta,
                best_delta,
                1.0 if gain > 0.0 else 0.0,
                float(proposal_values[row, best_index]),
            ],
            dtype=np.float32,
        )
    return features
