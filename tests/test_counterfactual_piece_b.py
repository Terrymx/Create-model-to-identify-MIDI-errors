from __future__ import annotations

import unittest

import numpy as np

from run_counterfactual_piece_b import evaluate_calibrated_scores


class CounterfactualPieceBTests(unittest.TestCase):
    def test_calibration_threshold_is_applied_unchanged_to_test(self) -> None:
        calibration_scores = np.asarray([0.9, 0.8, 0.7, 0.1])
        calibration_labels = np.asarray([1, 1, 0, 0])
        test_scores = np.asarray([0.85, 0.75, 0.65])
        test_labels = np.asarray([1, 0, 1])

        result = evaluate_calibrated_scores(
            calibration_scores,
            calibration_labels,
            calibration_total_errors=2,
            test_scores=test_scores,
            test_labels=test_labels,
            test_total_errors=3,
            target_precision=0.80,
        )

        self.assertEqual(
            result["calibration"]["threshold"],
            result["test"]["threshold"],
        )
        self.assertEqual(result["test"]["tp"], 1)
        self.assertEqual(result["test"]["fp"], 0)


if __name__ == "__main__":
    unittest.main()
