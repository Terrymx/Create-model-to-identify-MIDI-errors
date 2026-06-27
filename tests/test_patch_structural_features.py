from __future__ import annotations

import unittest

import numpy as np

from patch_structural_features import (
    PATCH_STRUCTURAL_FEATURE_NAMES,
    compute_patch_structural_features,
    feature_names_for_radii,
)


def _features_from_pitches(pitches: list[int]) -> np.ndarray:
    features = np.zeros((len(pitches), 36), dtype=np.float32)
    pitches_array = np.asarray(pitches, dtype=np.float32)
    features[:, 0] = pitches_array / 127.0
    features[:, 2] = 0.2
    features[:, 3] = 0.1
    phase = 2.0 * np.pi * (pitches_array % 12.0) / 12.0
    features[:, 6] = np.sin(phase)
    features[:, 7] = np.cos(phase)
    features[:, 34] = 0.5
    features[:, 35] = 0.25
    return features


class PatchStructuralFeatureTests(unittest.TestCase):
    def test_feature_dimension_is_stable(self) -> None:
        features = _features_from_pitches([60, 62, 64, 65, 67, 69, 71, 72])

        result = compute_patch_structural_features(
            piece_features=features,
            candidate_positions=np.asarray([3, 4], dtype=np.int64),
            observed_pitch=np.asarray([65, 67], dtype=np.float32),
            edited_pitch=np.asarray([64, 69], dtype=np.float32),
            radii=(2, 4),
        )

        self.assertEqual(result.shape, (2, len(PATCH_STRUCTURAL_FEATURE_NAMES) // 3 * 2))
        self.assertTrue(np.isfinite(result).all())

    def test_consensus_gain_is_positive_when_edit_matches_local_pitch_class(self) -> None:
        features = _features_from_pitches([60, 64, 67, 61, 72, 76, 79])

        result = compute_patch_structural_features(
            piece_features=features,
            candidate_positions=np.asarray([3], dtype=np.int64),
            observed_pitch=np.asarray([61], dtype=np.float32),
            edited_pitch=np.asarray([60], dtype=np.float32),
            radii=(3,),
        )

        names = feature_names_for_radii((3,))
        pc_gain_index = names.index("r3_pc_support_gain")
        nearest_gain_index = names.index("r3_nearest_distance_gain")
        self.assertGreater(result[0, pc_gain_index], 0.0)
        self.assertGreater(result[0, nearest_gain_index], 0.0)

    def test_edge_candidate_returns_finite_features(self) -> None:
        features = _features_from_pitches([60, 62])

        result = compute_patch_structural_features(
            piece_features=features,
            candidate_positions=np.asarray([0], dtype=np.int64),
            observed_pitch=np.asarray([60], dtype=np.float32),
            edited_pitch=np.asarray([61], dtype=np.float32),
            radii=(4,),
        )

        self.assertEqual(result.shape[0], 1)
        self.assertTrue(np.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
