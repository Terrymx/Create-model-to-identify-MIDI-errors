from __future__ import annotations

import unittest

import numpy as np

from sweep_piece_consistent_detector_thresholds import (
    canonical_note_rows,
    union_candidate_row,
)


class PieceConsistentDetectorSweepTests(unittest.TestCase):
    def test_canonical_rows_prefer_larger_window_edge_margin(self) -> None:
        file_ids = np.asarray([0, 0, 0, 1])
        positions = np.asarray([140, 140, 200, 5])
        local_positions = np.asarray([12, 140, 72, 5])

        rows = canonical_note_rows(
            file_ids,
            positions,
            local_positions,
            window_size=256,
        )

        np.testing.assert_array_equal(rows, np.asarray([1, 2, 3]))

    def test_union_row_counts_unique_errors_and_candidates(self) -> None:
        labels = np.asarray([1, 0, 1, 0])
        three = np.asarray([0.7, 0.2, 0.1, 0.9])
        binary = np.asarray([0.1, 0.6, 0.2, 0.1])

        row = union_candidate_row(
            labels,
            three,
            binary,
            three_threshold=0.6,
            binary_threshold=0.5,
        )

        self.assertEqual(row["candidates"], 3)
        self.assertEqual(row["candidate_errors"], 1)
        self.assertAlmostEqual(row["candidate_precision"], 1 / 3)
        self.assertAlmostEqual(row["candidate_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
