from __future__ import annotations

import numpy as np


_CONSONANT_INTERVAL_CLASSES = {0, 3, 4, 5, 7, 8, 9}
_BASE_FEATURE_NAMES = (
    "count",
    "pitch_std",
    "pitch_range",
    "observed_abs_z",
    "edited_abs_z",
    "z_gain",
    "observed_pc_support",
    "edited_pc_support",
    "pc_support_gain",
    "observed_nearest_distance",
    "edited_nearest_distance",
    "nearest_distance_gain",
    "observed_mean_interval",
    "edited_mean_interval",
    "mean_interval_gain",
    "observed_consonant_fraction",
    "edited_consonant_fraction",
    "consonance_gain",
    "duration_center",
    "delta_start_center",
    "downbeat_strength",
    "subdivision_strength",
)
_DEFAULT_RADII = (4, 8, 16)
PATCH_STRUCTURAL_FEATURE_NAMES = tuple(
    f"r{radius}_{name}" for radius in _DEFAULT_RADII for name in _BASE_FEATURE_NAMES
)


def patch_structural_feature_size(radii: tuple[int, ...] = _DEFAULT_RADII) -> int:
    return len(_BASE_FEATURE_NAMES) * len(radii)


def _pc_support(pitch: float, context_pitches: np.ndarray) -> float:
    if context_pitches.size == 0:
        return 0.0
    pitch_class = int(round(float(pitch))) % 12
    context_pc = np.rint(context_pitches).astype(np.int64) % 12
    return float((context_pc == pitch_class).mean())


def _nearest_distance(pitch: float, context_pitches: np.ndarray) -> float:
    if context_pitches.size == 0:
        return 1.0
    distance = np.min(np.abs(context_pitches.astype(np.float32) - float(pitch)))
    return float(np.clip(distance / 12.0, 0.0, 1.0))


def _mean_interval(pitch: float, context_pitches: np.ndarray) -> float:
    if context_pitches.size == 0:
        return 1.0
    return float(np.clip(np.mean(np.abs(context_pitches - float(pitch))) / 24.0, 0.0, 1.0))


def _consonant_fraction(pitch: float, context_pitches: np.ndarray) -> float:
    if context_pitches.size == 0:
        return 0.0
    intervals = np.abs(np.rint(context_pitches).astype(np.int64) - int(round(float(pitch))))
    interval_classes = intervals % 12
    return float(np.isin(interval_classes, list(_CONSONANT_INTERVAL_CLASSES)).mean())


def _radius_feature_names(radius: int) -> tuple[str, ...]:
    return tuple(f"r{radius}_{name}" for name in _BASE_FEATURE_NAMES)


def feature_names_for_radii(radii: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(name for radius in radii for name in _radius_feature_names(radius))


def compute_patch_structural_features(
    *,
    piece_features: np.ndarray,
    candidate_positions: np.ndarray,
    observed_pitch: np.ndarray,
    edited_pitch: np.ndarray,
    radii: tuple[int, ...] = _DEFAULT_RADII,
) -> np.ndarray:
    features = np.asarray(piece_features, dtype=np.float32)
    positions = np.asarray(candidate_positions, dtype=np.int64)
    observed = np.asarray(observed_pitch, dtype=np.float32)
    edited = np.asarray(edited_pitch, dtype=np.float32)
    if len(positions) != len(observed) or len(positions) != len(edited):
        raise ValueError("candidate_positions, observed_pitch, and edited_pitch must align.")
    output = np.zeros((len(positions), patch_structural_feature_size(radii)), dtype=np.float32)
    if features.size == 0:
        return output
    piece_pitches = np.rint(features[:, 0] * 127.0).clip(0, 127).astype(np.float32)
    for row, center in enumerate(positions.tolist()):
        values: list[float] = []
        center_index = int(center)
        for radius in radii:
            left = max(0, center_index - radius)
            right = min(len(piece_pitches), center_index + radius + 1)
            if center_index < 0 or center_index >= len(piece_pitches) or right <= left:
                values.extend([0.0] * len(_BASE_FEATURE_NAMES))
                continue
            local_indices = np.arange(left, right)
            context_indices = local_indices[local_indices != center_index]
            context_pitches = piece_pitches[context_indices]
            if context_pitches.size == 0:
                values.extend([0.0] * len(_BASE_FEATURE_NAMES))
                continue
            mean = float(context_pitches.mean())
            std = float(context_pitches.std())
            safe_std = max(std, 1.0)
            observed_z = abs(float(observed[row]) - mean) / safe_std
            edited_z = abs(float(edited[row]) - mean) / safe_std
            observed_pc = _pc_support(observed[row], context_pitches)
            edited_pc = _pc_support(edited[row], context_pitches)
            observed_nearest = _nearest_distance(observed[row], context_pitches)
            edited_nearest = _nearest_distance(edited[row], context_pitches)
            observed_interval = _mean_interval(observed[row], context_pitches)
            edited_interval = _mean_interval(edited[row], context_pitches)
            observed_consonance = _consonant_fraction(observed[row], context_pitches)
            edited_consonance = _consonant_fraction(edited[row], context_pitches)
            center_features = features[center_index]
            values.extend(
                [
                    min(float(context_pitches.size) / float(max(radius * 2, 1)), 1.0),
                    min(std / 12.0, 1.0),
                    min(float(context_pitches.max() - context_pitches.min()) / 36.0, 1.0),
                    min(observed_z / 4.0, 1.0),
                    min(edited_z / 4.0, 1.0),
                    np.clip((observed_z - edited_z) / 4.0, -1.0, 1.0),
                    observed_pc,
                    edited_pc,
                    edited_pc - observed_pc,
                    observed_nearest,
                    edited_nearest,
                    observed_nearest - edited_nearest,
                    observed_interval,
                    edited_interval,
                    observed_interval - edited_interval,
                    observed_consonance,
                    edited_consonance,
                    edited_consonance - observed_consonance,
                    float(center_features[2]) if center_features.shape[0] > 2 else 0.0,
                    float(center_features[3]) if center_features.shape[0] > 3 else 0.0,
                    float(center_features[34]) if center_features.shape[0] > 34 else 0.0,
                    float(center_features[35]) if center_features.shape[0] > 35 else 0.0,
                ]
            )
        output[row] = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(output, copy=False).astype(np.float32)
