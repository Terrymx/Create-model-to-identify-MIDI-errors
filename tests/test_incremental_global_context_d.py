from __future__ import annotations

import unittest

import numpy as np

from incremental_global_context import PieceEdit
from run_incremental_global_context_d import (
    IncrementalBeamState,
    incremental_beam_search,
)


class IncrementalGlobalContextDTests(unittest.TestCase):
    def test_later_choice_uses_scores_after_prior_edit(self) -> None:
        proposals = {
            0: (61, 62),
            1: (70, 71),
        }

        def score_all(edits):
            positions = {edit.position for edit in edits}
            if 10 in positions:
                return np.asarray([0.9, 0.8])
            return np.asarray([0.9, 0.2])

        result = incremental_beam_search(
            candidate_positions=np.asarray([10, 20]),
            proposals=proposals,
            score_all=score_all,
            score_floor=0.5,
            beam_width=4,
            max_edits=2,
        )

        self.assertEqual(
            [edit.position for edit in result.edits],
            [10, 20],
        )

    def test_one_edit_per_position_and_deterministic_pruning(self) -> None:
        proposals = {0: (61, 62), 1: (70, 71)}

        def score_all(edits):
            return np.asarray([0.8, 0.7])

        first = incremental_beam_search(
            candidate_positions=np.asarray([10, 20]),
            proposals=proposals,
            score_all=score_all,
            score_floor=0.5,
            beam_width=2,
            max_edits=1,
        )
        second = incremental_beam_search(
            candidate_positions=np.asarray([10, 20]),
            proposals=proposals,
            score_all=score_all,
            score_floor=0.5,
            beam_width=2,
            max_edits=1,
        )

        self.assertIsInstance(first, IncrementalBeamState)
        self.assertEqual(first, second)
        self.assertEqual(len({edit.position for edit in first.edits}), len(first.edits))
        self.assertEqual(len(first.edits), 1)


if __name__ == "__main__":
    unittest.main()
