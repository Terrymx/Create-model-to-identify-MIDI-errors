from __future__ import annotations

import unittest

import numpy as np

from risk_control_selection import (
    apply_score_probability_bins,
    evaluate_fdr_score_rows,
    fit_score_probability_bins,
    row_from_prediction,
    select_piecewise_fdr,
)


class RiskControlSelectionTests(unittest.TestCase):
    def test_quantile_calibrator_is_monotone_in_score(self) -> None:
        scores = np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], dtype=np.float32)
        labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)

        calibrator = fit_score_probability_bins(scores, labels, bin_count=3)
        probabilities = apply_score_probability_bins(calibrator, scores)

        self.assertTrue(np.all(np.diff(probabilities) >= -1e-6))
        self.assertLess(probabilities[0], probabilities[-1])

    def test_piecewise_fdr_selects_largest_safe_prefix_per_piece(self) -> None:
        probabilities = np.asarray([0.95, 0.85, 0.55, 0.90, 0.60], dtype=np.float32)
        file_ids = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)

        selected = select_piecewise_fdr(probabilities, file_ids, max_fdr=0.20)

        np.testing.assert_array_equal(
            selected,
            np.asarray([True, True, False, True, False]),
        )

    def test_fdr_evaluation_uses_calibration_q_on_test_scores(self) -> None:
        calibration_scores = np.asarray(
            [0.95, 0.90, 0.82, 0.80, 0.70, 0.20],
            dtype=np.float32,
        )
        calibration_labels = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int64)
        calibration_files = np.asarray([0, 0, 1, 1, 1, 1], dtype=np.int64)
        test_scores = np.asarray([0.96, 0.88, 0.81, 0.30], dtype=np.float32)
        test_labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
        test_files = np.asarray([0, 0, 0, 1], dtype=np.int64)

        result = evaluate_fdr_score_rows(
            calibration_scores,
            test_scores,
            calibration_labels,
            test_labels,
            calibration_files,
            test_files,
            calibration_total_errors=3,
            test_total_errors=2,
            target_precision=0.80,
            q_grid=np.asarray([0.05, 0.10, 0.20, 0.30], dtype=np.float32),
            bin_count=3,
        )

        row = result["best_feasible_test"]["selected_test"]
        self.assertGreaterEqual(row["precision"], 0.80)
        self.assertGreater(row["recall"], 0.0)
        self.assertIn("selected_fdr", result["best_feasible_test"])

    def test_row_from_prediction_reports_selected_count(self) -> None:
        row = row_from_prediction(
            np.asarray([True, False, True]),
            np.asarray([1, 1, 0]),
            total_errors=2,
        )

        self.assertEqual(row["selected"], 2)
        self.assertEqual(row["tp"], 1)
        self.assertEqual(row["fp"], 1)


if __name__ == "__main__":
    unittest.main()
