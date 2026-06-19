from __future__ import annotations

import unittest

import numpy as np
import torch

from run_verifier_improvement_suite import (
    build_pair_indices,
    classify_candidate_patterns,
    convex_weight_grid,
    split_old_and_theory_features,
)


class VerifierImprovementSuiteTests(unittest.TestCase):
    def test_split_old_and_theory_features_preserves_piece_relative_suffix(self) -> None:
        base = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        theory = torch.full((2, 3), 100.0)
        piece = torch.full((2, 4), 200.0)
        combined = torch.cat([base, theory, piece], dim=1)

        old, full = split_old_and_theory_features(
            combined,
            theory_size=3,
            piece_relative_size=4,
        )

        self.assertTrue(torch.equal(full, combined))
        self.assertTrue(torch.equal(old, torch.cat([base, piece], dim=1)))

    def test_convex_weight_grid_contains_single_models_and_normalized_mixtures(self) -> None:
        weights = convex_weight_grid(model_count=3, step=0.5)

        self.assertIn((1.0, 0.0, 0.0), weights)
        self.assertIn((0.0, 1.0, 0.0), weights)
        self.assertIn((0.0, 0.0, 1.0), weights)
        self.assertIn((0.5, 0.5, 0.0), weights)
        self.assertTrue(all(abs(sum(row) - 1.0) < 1e-8 for row in weights))

    def test_pair_indices_stay_within_piece_and_use_hard_negatives(self) -> None:
        labels = np.asarray([1, 0, 0, 1, 0, 0], dtype=np.int64)
        files = np.asarray([4, 4, 4, 9, 9, 9], dtype=np.int64)
        hardness = np.asarray([0.8, 0.9, 0.2, 0.7, 0.1, 0.95], dtype=np.float32)

        positive, negative = build_pair_indices(
            labels,
            files,
            hardness,
            negatives_per_positive=1,
        )

        self.assertEqual(len(positive), 2)
        self.assertTrue(np.all(files[positive] == files[negative]))
        self.assertEqual(negative.tolist(), [1, 5])

    def test_candidate_pattern_classification_reports_overlapping_music_types(self) -> None:
        raw = torch.zeros(3, 36)
        raw[:, 33] = 0.5
        raw[0, 20] = 1.0
        raw[0, 8] = 1.0
        raw[1, 29] = 1.0
        raw[1, 33] = 0.1
        raw[2, 34] = 1.0
        raw[2, 12] = 0.9

        rows = classify_candidate_patterns(raw)

        self.assertEqual(rows["chord_tone"].tolist(), [True, False, False])
        self.assertEqual(rows["scale_tone"].tolist(), [True, False, False])
        self.assertEqual(rows["ornament"].tolist(), [False, True, False])
        self.assertEqual(rows["short_note"].tolist(), [False, True, False])
        self.assertEqual(rows["strong_beat"].tolist(), [False, False, True])
        self.assertEqual(rows["high_density"].tolist(), [False, False, True])


if __name__ == "__main__":
    unittest.main()
