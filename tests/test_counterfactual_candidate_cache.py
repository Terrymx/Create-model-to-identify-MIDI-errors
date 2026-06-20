from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_counterfactual_candidate_cache import (
    aggregate_proposal_features,
    load_candidate_cache,
    save_candidate_cache,
)
from build_counterfactual_local_cache import top_proposal_indices


class CounterfactualCandidateCacheTests(unittest.TestCase):
    def test_aggregate_keeps_flattened_rows_and_summary(self) -> None:
        features = np.asarray(
            [
                [[1.0, 4.0], [3.0, 2.0]],
                [[5.0, 0.0], [1.0, 6.0]],
            ],
            dtype=np.float32,
        )
        ranking = np.asarray([[0.2, 0.9], [0.8, 0.1]], dtype=np.float32)

        aggregated = aggregate_proposal_features(features, ranking)

        self.assertEqual(aggregated.shape, (2, 10))
        np.testing.assert_allclose(aggregated[0, :4], [1.0, 4.0, 3.0, 2.0])
        np.testing.assert_allclose(aggregated[0, 4:6], [3.0, 4.0])
        np.testing.assert_allclose(aggregated[0, 6:8], [2.0, 3.0])
        np.testing.assert_allclose(aggregated[0, 8:10], [3.0, 2.0])

    def test_cache_round_trip_preserves_arrays_and_metadata(self) -> None:
        arrays = {
            "base_features": np.arange(12, dtype=np.float32).reshape(3, 4),
            "labels": np.asarray([0, 1, 1], dtype=np.int64),
            "proposals": np.asarray([[60, 61], [62, 63], [64, 65]], dtype=np.int64),
        }
        metadata = {"split": "calibration", "candidate_recall": 0.72}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"

            save_candidate_cache(path, arrays, metadata)
            loaded_arrays, loaded_metadata = load_candidate_cache(path)

        self.assertEqual(loaded_metadata, metadata)
        for name, values in arrays.items():
            np.testing.assert_array_equal(loaded_arrays[name], values)

    def test_top_proposals_are_selected_deterministically(self) -> None:
        scores = np.asarray(
            [[0.1, 0.9, 0.4, 0.8], [0.5, 0.5, 0.2, 0.1]],
            dtype=np.float32,
        )

        selected = top_proposal_indices(scores, count=2)

        np.testing.assert_array_equal(selected, [[1, 3], [0, 1]])


if __name__ == "__main__":
    unittest.main()
