from __future__ import annotations

import unittest

from counterfactual_beam_search import (
    EditCandidate,
    beam_search_edits,
    edit_set_key,
)


class CounterfactualBeamSearchTests(unittest.TestCase):
    def test_selects_positive_edits_and_keeps_each_note_once(self) -> None:
        candidates = [
            EditCandidate(0, 10, 61, 1.2),
            EditCandidate(1, 10, 62, 0.9),
            EditCandidate(2, 20, 70, 0.8),
            EditCandidate(3, 30, 80, -0.2),
        ]

        state = beam_search_edits(candidates, beam_width=4, max_edits=3)

        self.assertEqual(
            [(edit.position, edit.proposed_pitch) for edit in state.edits],
            [(10, 61), (20, 70)],
        )

    def test_edit_budget_and_conflict_penalty_change_selection(self) -> None:
        candidates = [
            EditCandidate(0, 10, 61, 1.0),
            EditCandidate(1, 11, 62, 0.9),
            EditCandidate(2, 30, 70, 0.8),
        ]

        state = beam_search_edits(
            candidates,
            beam_width=8,
            max_edits=2,
            conflict_distance=2,
            conflict_penalty=0.5,
        )

        self.assertEqual([edit.position for edit in state.edits], [10, 30])

    def test_pruning_and_keys_are_deterministic(self) -> None:
        candidates = [
            EditCandidate(2, 20, 70, 1.0),
            EditCandidate(0, 10, 61, 1.0),
            EditCandidate(1, 15, 65, 1.0),
        ]

        first = beam_search_edits(candidates, beam_width=2, max_edits=2)
        second = beam_search_edits(
            list(reversed(candidates)),
            beam_width=2,
            max_edits=2,
        )

        self.assertEqual(edit_set_key(first.edits), edit_set_key(second.edits))
        self.assertEqual(edit_set_key(first.edits), tuple(sorted(edit_set_key(first.edits))))

    def test_score_callback_can_rescore_whole_edit_sets(self) -> None:
        candidates = [
            EditCandidate(0, 10, 61, 0.6),
            EditCandidate(1, 20, 70, 0.5),
        ]

        def rescore(edits):
            positions = {edit.position for edit in edits}
            return 2.0 if positions == {10, 20} else sum(edit.utility for edit in edits)

        state = beam_search_edits(
            candidates,
            beam_width=4,
            max_edits=2,
            score_callback=rescore,
        )

        self.assertEqual([edit.position for edit in state.edits], [10, 20])
        self.assertEqual(state.score, 2.0)


if __name__ == "__main__":
    unittest.main()
