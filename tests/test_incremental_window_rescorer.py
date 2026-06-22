from __future__ import annotations

import unittest

import numpy as np

from incremental_global_context import PieceEdit
from incremental_window_rescorer import IncrementalWindowCache


class IncrementalWindowRescorerTests(unittest.TestCase):
    def test_cache_reuses_same_window_state_and_ignores_outside_edits(self) -> None:
        calls = []

        def compute(window_features, mask):
            calls.append(window_features.copy())
            return {"pitch_sum": float(window_features[:, 0].sum())}

        features = np.zeros((400, 10), dtype=np.float32)
        cache = IncrementalWindowCache(
            piece_id=2,
            piece_features=features,
            window_size=256,
            compute=compute,
        )

        first = cache.get(0, [PieceEdit(20, 61)])
        second = cache.get(
            0,
            [PieceEdit(20, 61), PieceEdit(300, 70)],
        )

        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 1)

    def test_cache_miss_when_edit_inside_window_changes(self) -> None:
        calls = []

        def compute(window_features, mask):
            calls.append(1)
            return float(window_features[20, 0])

        cache = IncrementalWindowCache(
            piece_id=1,
            piece_features=np.zeros((300, 10), dtype=np.float32),
            window_size=256,
            compute=compute,
        )

        before = cache.get(0, [])
        after = cache.get(0, [PieceEdit(20, 61)])

        self.assertNotEqual(before, after)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
