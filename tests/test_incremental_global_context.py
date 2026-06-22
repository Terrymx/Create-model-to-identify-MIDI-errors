from __future__ import annotations

import unittest

import numpy as np

from incremental_global_context import (
    PieceEdit,
    affected_window_starts,
    apply_piece_edits,
    candidate_dependency_windows,
    state_key,
    window_state_key,
)


class IncrementalGlobalContextTests(unittest.TestCase):
    def test_state_key_is_sorted_and_piece_scoped(self) -> None:
        edits = [PieceEdit(20, 70), PieceEdit(10, 61)]

        self.assertEqual(
            state_key(3, edits),
            (3, ((10, 61), (20, 70))),
        )

    def test_apply_piece_edits_updates_only_pitch_columns(self) -> None:
        features = np.zeros((30, 10), dtype=np.float32)
        features[10, 2] = 0.75

        edited = apply_piece_edits(features, [PieceEdit(10, 61)])

        self.assertAlmostEqual(float(edited[10, 0]), 61 / 127)
        self.assertNotEqual(float(edited[10, 6]), 0.0)
        self.assertNotEqual(float(edited[10, 7]), 0.0)
        self.assertEqual(float(edited[10, 2]), 0.75)
        np.testing.assert_array_equal(edited[:10], features[:10])

    def test_affected_windows_include_final_non_stride_window(self) -> None:
        self.assertEqual(
            affected_window_starts(
                position=299,
                note_count=300,
                window_size=256,
                stride=128,
            ),
            (44,),
        )
        self.assertEqual(
            affected_window_starts(
                position=100,
                note_count=300,
                window_size=256,
                stride=128,
            ),
            (0, 44),
        )

    def test_candidate_dependencies_expand_by_context_radius(self) -> None:
        self.assertEqual(
            candidate_dependency_windows(
                position=150,
                note_count=500,
                window_size=256,
                stride=128,
                context_radius=16,
            ),
            (0, 128),
        )

    def test_window_key_ignores_edits_outside_window(self) -> None:
        edits = [
            PieceEdit(20, 61),
            PieceEdit(140, 70),
            PieceEdit(400, 80),
        ]

        self.assertEqual(
            window_state_key(2, 128, 256, edits),
            (2, 128, ((140, 70),)),
        )


if __name__ == "__main__":
    unittest.main()
