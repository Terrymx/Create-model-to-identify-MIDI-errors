from __future__ import annotations

import unittest

import numpy as np

from run_counterfactual_edit_verifier import (
    build_b_variant_features,
    build_c_variant_features,
    correction_metrics,
)


class CounterfactualEditVerifierTests(unittest.TestCase):
    def test_b_variants_have_increasing_feature_sets(self) -> None:
        base = np.zeros((3, 5), dtype=np.float32)
        proposals = np.zeros((3, 4, 17), dtype=np.float32)
        ranking = np.zeros((3, 4), dtype=np.float32)

        b1 = build_b_variant_features(base, proposals, ranking, "B1")
        b2 = build_b_variant_features(base, proposals, ranking, "B2")
        b3 = build_b_variant_features(base, proposals, ranking, "B3")

        self.assertEqual(b1.shape[0], 3)
        self.assertLess(b1.shape[1], b2.shape[1])
        self.assertLess(b2.shape[1], b3.shape[1])
        np.testing.assert_array_equal(b1[:, :5], base)

    def test_correction_metrics_use_detected_replace_errors_only(self) -> None:
        proposals = np.asarray(
            [[61, 62, 63], [70, 71, 72], [80, 81, 82]],
            dtype=np.int64,
        )
        proposal_scores = np.asarray(
            [[0.2, 0.9, 0.1], [0.8, 0.7, 0.6], [0.9, 0.2, 0.1]],
            dtype=np.float32,
        )
        detected = np.asarray([True, True, False])
        labels = np.asarray([1, 1, 1])
        kinds = np.asarray([1, 2, 1])
        targets = np.asarray([62, 70, 80])

        metrics = correction_metrics(
            proposals,
            proposal_scores,
            detected,
            labels,
            kinds,
            targets,
        )

        self.assertEqual(metrics["detected_replace_errors"], 1)
        self.assertEqual(metrics["top1_correct"], 1)
        self.assertEqual(metrics["top3_correct"], 1)

    def test_c_variants_extend_b3_with_increasing_radii(self) -> None:
        base = np.zeros((2, 5), dtype=np.float32)
        b_features = np.zeros((2, 4, 17), dtype=np.float32)
        b_ranking = np.zeros((2, 4), dtype=np.float32)
        c_features = np.zeros((2, 2, 15), dtype=np.float32)
        c_ranking = np.zeros((2, 2), dtype=np.float32)

        c1 = build_c_variant_features(
            base, b_features, b_ranking, c_features, c_ranking, "C1"
        )
        c2 = build_c_variant_features(
            base, b_features, b_ranking, c_features, c_ranking, "C2"
        )

        self.assertLess(c1.shape[1], c2.shape[1])
        self.assertGreater(c1.shape[1], base.shape[1])

    def test_c_variant_can_extend_b2_instead_of_b3(self) -> None:
        base = np.zeros((2, 5), dtype=np.float32)
        b_features = np.zeros((2, 4, 17), dtype=np.float32)
        b_ranking = np.zeros((2, 4), dtype=np.float32)
        c_features = np.zeros((2, 2, 15), dtype=np.float32)
        c_ranking = np.zeros((2, 2), dtype=np.float32)

        b2_c1 = build_c_variant_features(
            base,
            b_features,
            b_ranking,
            c_features,
            c_ranking,
            "C1",
            b_variant="B2",
        )
        b3_c1 = build_c_variant_features(
            base,
            b_features,
            b_ranking,
            c_features,
            c_ranking,
            "C1",
            b_variant="B3",
        )

        self.assertLess(b2_c1.shape[1], b3_c1.shape[1])


if __name__ == "__main__":
    unittest.main()
