from __future__ import annotations

import unittest

import numpy as np

from incremental_global_context import PieceEdit
from incremental_window_rescorer import (
    CandidateContext,
    IncrementalCandidateRescorer,
    IncrementalWindowCache,
)


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

    def test_candidate_rescorer_uses_updated_window_and_two_proposals(self) -> None:
        calls = []

        def window_compute(window_features, mask):
            calls.append(float(window_features[20, 0]))
            return window_features

        def score_candidate(window, local_position, proposals):
            return float(window[local_position, 0]) + float(proposals.max()) / 1000

        cache = IncrementalWindowCache(
            piece_id=0,
            piece_features=np.zeros((300, 10), dtype=np.float32),
            window_size=256,
            compute=window_compute,
        )
        rescorer = IncrementalCandidateRescorer(
            window_cache=cache,
            score_candidate=score_candidate,
        )
        candidate = CandidateContext(
            candidate_index=7,
            window_start=0,
            local_position=20,
            proposals=(61, 62),
        )

        before = rescorer.score(candidate, [])
        after = rescorer.score(candidate, [PieceEdit(20, 61)])

        self.assertGreater(after, before)
        self.assertEqual(calls[0], 0.0)
        self.assertAlmostEqual(calls[1], 61 / 127, places=6)


if __name__ == "__main__":
    unittest.main()
