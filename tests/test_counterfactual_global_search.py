from __future__ import annotations

import unittest

import numpy as np

from run_counterfactual_global_search import (
    _global_structures,
    _score_floor_grid,
    global_feature_spec,
    canonical_candidate_indices,
    detection_metrics,
    select_global_predictions,
)


class CounterfactualGlobalSearchTests(unittest.TestCase):
    def test_canonical_rows_prefer_candidate_farthest_from_window_edge(self) -> None:
        arrays = {
            "file_ids": np.asarray([0, 0, 0, 1]),
            "positions": np.asarray([140, 140, 200, 10]),
            "local_positions": np.asarray([12, 140, 72, 10]),
            "dataset_indices": np.asarray([2, 3, 4, 5]),
        }

        rows = canonical_candidate_indices(arrays, window_size=256)

        np.testing.assert_array_equal(rows, np.asarray([1, 2, 3]))

    def test_global_selection_returns_original_row_mask(self) -> None:
        arrays = {
            "file_ids": np.asarray([0, 0, 0, 1]),
            "positions": np.asarray([10, 20, 30, 5]),
            "file_note_counts": np.asarray([100, 100, 100, 40]),
            "proposals": np.asarray([[61], [62], [63], [70]]),
        }
        scores = np.asarray([0.90, 0.85, 0.20, 0.80])

        selected = select_global_predictions(
            arrays,
            scores,
            score_floor=0.50,
            beam_width=4,
            max_edit_rate=0.01,
            conflict_distance=0,
            conflict_penalty=0.0,
        )

        np.testing.assert_array_equal(
            selected,
            np.asarray([True, False, False, True]),
        )

    def test_metrics_include_errors_outside_candidate_set(self) -> None:
        labels = np.asarray([1, 0, 1])
        selected = np.asarray([True, True, False])

        metrics = detection_metrics(selected, labels, total_errors=4)

        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 3)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.25)

    def test_recalibration_uses_dense_floors_and_wider_d1_budgets(self) -> None:
        floors = _score_floor_grid(np.linspace(0.0, 1.0, 1001))
        structures = _global_structures("D1")

        self.assertGreaterEqual(len(floors), 100)
        self.assertTrue(any(rate == 0.03 for _, rate, _, _ in structures))
        self.assertTrue(
            all(distance == 0 and penalty == 0.0 for _, _, distance, penalty in structures)
        )

    def test_global_feature_spec_uses_selected_b2_multiscale_c(self) -> None:
        self.assertEqual(
            global_feature_spec("B2", "C2"),
            {
                "b_variant": "B2",
                "c_variant": "C2",
                "label": "B2_C_radius4_8_16",
            },
        )


if __name__ == "__main__":
    unittest.main()
