from __future__ import annotations

import unittest

import numpy as np

from proposal_trust_verifier import (
    build_high_confidence_slice_mask,
    build_proposal_trust_features,
    build_trust_rescue_scores,
    evaluate_rescue_union,
    mask_rescue_scores,
)


class ProposalTrustVerifierTests(unittest.TestCase):
    def _arrays(self) -> dict[str, np.ndarray]:
        return {
            "observed_pitch": np.asarray([60, 65], dtype=np.float32),
            "proposals": np.asarray([[61, 64, 67], [62, 65, 69]], dtype=np.float32),
            "b_ranking": np.asarray([[0.2, 0.7, 0.3], [0.8, 0.1, 0.4]], dtype=np.float32),
            "c_ranking": np.asarray([[0.1, 0.6], [0.9, 0.2]], dtype=np.float32),
            "c_proposal_slots": np.asarray([[0, 1], [0, 2]], dtype=np.int64),
            "labels": np.asarray([1, 0], dtype=np.int64),
        }

    def test_build_proposal_trust_features_summarizes_e1_best_proposal(self) -> None:
        arrays = self._arrays()
        c2_scores = np.asarray([0.25, 0.75], dtype=np.float32)
        e1_scores = np.asarray([[0.1, 2.0, 0.5], [1.5, 0.3, 1.0]], dtype=np.float32)

        features = build_proposal_trust_features(arrays, c2_scores, e1_scores)

        self.assertEqual(features.shape, (2, 15))
        self.assertAlmostEqual(float(features[0, 0]), 0.25)
        self.assertAlmostEqual(float(features[0, 1]), 2.0)
        self.assertAlmostEqual(float(features[0, 2]), 1.5)
        self.assertAlmostEqual(float(features[0, 3]), 0.7)
        self.assertAlmostEqual(float(features[0, 4]), 0.6)
        self.assertAlmostEqual(float(features[0, 5]), 1.0)
        self.assertAlmostEqual(float(features[0, 6]), 1.0)
        self.assertAlmostEqual(float(features[0, 9]), 4.0 / 24.0)

    def test_build_trust_rescue_scores_only_boosts_baseline_rejected_rows(self) -> None:
        baseline_scores = np.asarray([0.90, 0.20, 0.30], dtype=np.float32)
        rescue_scores = np.asarray([0.10, 0.80, 0.40], dtype=np.float32)

        combined = build_trust_rescue_scores(
            baseline_scores,
            rescue_scores,
            baseline_threshold=0.70,
            rescue_threshold=0.60,
        )

        np.testing.assert_array_equal(
            combined >= 0.5,
            np.asarray([True, True, False]),
        )

    def test_high_confidence_slice_requires_strong_e1_and_weak_motif(self) -> None:
        arrays = self._arrays()
        arrays["motif_features"] = np.asarray(
            [
                [0.0, 0.0],
                [2.0, 0.5],
            ],
            dtype=np.float32,
        )
        c2_scores = np.asarray([0.25, 0.20], dtype=np.float32)
        e1_scores = np.asarray([[0.1, 2.0, 0.5], [1.5, 0.3, 1.0]], dtype=np.float32)

        mask = build_high_confidence_slice_mask(
            arrays,
            c2_scores,
            e1_scores,
            c2_max=0.30,
            e1_margin_min=1.0,
            e1_best_prob_min=0.60,
            require_b_agreement=True,
            require_c_agreement=True,
            motif_match_index=0,
            motif_gain_index=1,
            motif_match_max=0.0,
            motif_gain_max=0.0,
        )

        np.testing.assert_array_equal(mask, np.asarray([True, False]))

    def test_mask_rescue_scores_blocks_ineligible_high_scores(self) -> None:
        scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
        mask = np.asarray([True, False, True])

        masked = mask_rescue_scores(scores, mask)

        self.assertEqual(float(masked[1]), float("-inf"))
        self.assertGreater(float(masked[0]), 0.0)

    def test_evaluate_rescue_union_selects_threshold_from_calibration(self) -> None:
        calibration = {"labels": np.asarray([1, 0, 1, 0], dtype=np.int64)}
        test = {
            "labels": np.asarray([1, 0, 1, 0], dtype=np.int64),
            "proposals": np.zeros((4, 3), dtype=np.int64),
            "b_ranking": np.zeros((4, 3), dtype=np.float32),
            "error_kind": np.ones(4, dtype=np.int64),
            "target_pitch": np.zeros(4, dtype=np.int64),
        }
        calibration_baseline = np.asarray([0.9, 0.1, 0.2, 0.1], dtype=np.float32)
        test_baseline = np.asarray([0.9, 0.1, 0.2, 0.1], dtype=np.float32)
        calibration_rescue = np.asarray([0.2, 0.2, 0.9, 0.1], dtype=np.float32)
        test_rescue = np.asarray([0.2, 0.2, 0.9, 0.1], dtype=np.float32)

        result = evaluate_rescue_union(
            calibration_baseline_scores=calibration_baseline,
            test_baseline_scores=test_baseline,
            calibration_rescue_scores=calibration_rescue,
            test_rescue_scores=test_rescue,
            calibration=calibration,
            test=test,
            calibration_total_errors=2,
            test_total_errors=2,
            target_precision=0.80,
        )

        row = result["best_feasible_test"]["selected_test"]
        self.assertEqual(row["tp"], 2)
        self.assertEqual(row["fp"], 0)
        self.assertEqual(row["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
