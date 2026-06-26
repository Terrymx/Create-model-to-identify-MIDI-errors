from __future__ import annotations

import unittest

import numpy as np

from motif_repetition_features import (
    MOTIF_FEATURE_NAMES,
    compute_motif_repetition_features,
)


class MotifRepetitionFeatureTests(unittest.TestCase):
    def test_transposed_repetition_prefers_consensus_proposal(self) -> None:
        # Two identical interval patterns appear at positions 3 and 10, transposed
        # up by five semitones.  The observed center pitch at 10 is wrong by -1,
        # while proposal 70 matches the transposed first occurrence.
        pitches = np.asarray(
            [60, 62, 64, 65, 67, 69, 71, 65, 67, 69, 69, 72, 74, 76],
            dtype=np.int64,
        )
        features = compute_motif_repetition_features(
            piece_pitches=pitches,
            candidate_positions=np.asarray([10], dtype=np.int64),
            observed_pitch=np.asarray([69], dtype=np.int64),
            proposals=np.asarray([[68, 70, 71, 72]], dtype=np.int64),
            radius=2,
            min_similarity=0.99,
            exclude_radius=3,
        )

        index = {name: i for i, name in enumerate(MOTIF_FEATURE_NAMES)}
        self.assertEqual(features.shape, (1, len(MOTIF_FEATURE_NAMES)))
        self.assertGreaterEqual(features[0, index["motif_match_count"]], 1.0)
        self.assertGreater(features[0, index["best_proposal_consensus"]], 0.99)
        self.assertLess(features[0, index["observed_pitch_consensus"]], 0.99)
        self.assertGreater(features[0, index["proposal_consensus_gain"]], 0.0)
        self.assertEqual(features[0, index["best_proposal_pitch"]], 70.0)

    def test_self_neighborhood_is_excluded_from_matches(self) -> None:
        pitches = np.asarray([60, 62, 64, 65, 67, 69, 71], dtype=np.int64)
        features = compute_motif_repetition_features(
            piece_pitches=pitches,
            candidate_positions=np.asarray([3], dtype=np.int64),
            observed_pitch=np.asarray([65], dtype=np.int64),
            proposals=np.asarray([[64, 66]], dtype=np.int64),
            radius=2,
            min_similarity=0.90,
            exclude_radius=4,
        )

        index = {name: i for i, name in enumerate(MOTIF_FEATURE_NAMES)}
        self.assertEqual(features[0, index["motif_match_count"]], 0.0)
        self.assertEqual(features[0, index["motif_support"]], 0.0)
        self.assertEqual(features[0, index["proposal_consensus_gain"]], 0.0)

    def test_empty_or_edge_context_falls_back_to_zero_features(self) -> None:
        pitches = np.asarray([60, 62, 64], dtype=np.int64)
        features = compute_motif_repetition_features(
            piece_pitches=pitches,
            candidate_positions=np.asarray([0, 2], dtype=np.int64),
            observed_pitch=np.asarray([60, 64], dtype=np.int64),
            proposals=np.asarray([[59, 61], [63, 65]], dtype=np.int64),
            radius=2,
        )

        self.assertEqual(features.shape, (2, len(MOTIF_FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.all(features == 0.0))


if __name__ == "__main__":
    unittest.main()
