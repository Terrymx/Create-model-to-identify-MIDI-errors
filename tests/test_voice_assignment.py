from __future__ import annotations

import unittest

import numpy as np
import torch

from midi_error_detector.data import NoteEvent
from voice_assignment import (
    VOICE_FEATURE_SIZE,
    assign_voices,
    build_voice_features,
)
from voice_aware_dataset import build_piece_observation, slice_piece_observation
from run_frozen_union_candidate_context_verifier import append_voice_features


def two_parallel_voices() -> list[NoteEvent]:
    return [
        NoteEvent(48, 80, 0.0, 0.8),
        NoteEvent(72, 90, 0.0, 0.7),
        NoteEvent(50, 80, 1.0, 1.8),
        NoteEvent(74, 90, 1.0, 1.7),
        NoteEvent(52, 80, 2.0, 2.8),
        NoteEvent(76, 90, 2.0, 2.7),
    ]


class VoiceAssignmentTests(unittest.TestCase):
    def test_onset_matching_preserves_parallel_voice_tracks(self) -> None:
        notes = two_parallel_voices()

        result = assign_voices(notes, method="onset_matching")

        self.assertEqual(result.voice_ids[0], result.voice_ids[2])
        self.assertEqual(result.voice_ids[2], result.voice_ids[4])
        self.assertEqual(result.voice_ids[1], result.voice_ids[3])
        self.assertEqual(result.voice_ids[3], result.voice_ids[5])
        self.assertNotEqual(result.voice_ids[0], result.voice_ids[1])
        self.assertTrue(np.all(np.isfinite(result.confidence)))

    def test_notes_at_same_onset_use_distinct_tracks(self) -> None:
        notes = [
            NoteEvent(48, 80, 0.0, 1.0),
            NoteEvent(60, 80, 0.0, 1.0),
            NoteEvent(72, 80, 0.0, 1.0),
        ]

        for method in ("onset_matching", "global_beam"):
            with self.subTest(method=method):
                result = assign_voices(notes, method=method)
                self.assertEqual(len(set(result.voice_ids.tolist())), 3)

    def test_voice_features_have_roles_and_same_voice_motion(self) -> None:
        notes = two_parallel_voices()
        result = assign_voices(notes, method="onset_matching")

        features = build_voice_features(notes, result)

        self.assertEqual(features.shape, (len(notes), VOICE_FEATURE_SIZE))
        self.assertTrue(np.isfinite(features).all())
        # First onset: low note is bass and high note is melody.
        self.assertGreater(features[0, 2], 0.5)
        self.assertGreater(features[1, 1], 0.5)
        # Later notes have a +2-semitone incoming interval, confidence weighted.
        self.assertAlmostEqual(
            float(features[2, 4]),
            (2.0 / 24.0) * float(result.confidence[2]),
            places=5,
        )
        self.assertAlmostEqual(
            float(features[3, 4]),
            (2.0 / 24.0) * float(result.confidence[3]),
            places=5,
        )

    def test_global_beam_returns_same_public_feature_contract(self) -> None:
        notes = two_parallel_voices()
        result = assign_voices(notes, method="global_beam", beam_width=8)

        features = build_voice_features(notes, result)

        self.assertEqual(features.shape, (len(notes), VOICE_FEATURE_SIZE))
        self.assertTrue(np.all(result.confidence >= 0.0))
        self.assertTrue(np.all(result.confidence <= 1.0))

    def test_piece_consistent_corruption_matches_across_overlapping_windows(self) -> None:
        notes = [
            NoteEvent(48 + index, 80, float(index), float(index) + 0.8)
            for index in range(12)
        ]
        piece = build_piece_observation(
            notes,
            error_rate=0.25,
            seed=29,
            voice_method="onset_matching",
        )

        first = slice_piece_observation(piece, start=0, window_size=8)
        second = slice_piece_observation(piece, start=4, window_size=8)

        self.assertTrue(np.array_equal(first["features"][4:8], second["features"][0:4]))
        self.assertTrue(np.array_equal(first["is_error"][4:8], second["is_error"][0:4]))
        self.assertTrue(np.array_equal(first["voice_features"][4:8], second["voice_features"][0:4]))
        self.assertEqual(first["mask"].sum(), 8)
        self.assertEqual(second["mask"].sum(), 8)

    def test_append_voice_features_preserves_existing_feature_prefix(self) -> None:
        base = torch.randn(2, 5, 13)
        voice = torch.randn(2, 5, VOICE_FEATURE_SIZE)
        valid = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, True],
            ]
        )

        result = append_voice_features(base, voice, valid)

        self.assertTrue(torch.equal(result[..., :13], base))
        self.assertEqual(result.shape[-1], 13 + VOICE_FEATURE_SIZE)
        self.assertTrue(torch.equal(result[0, 3:, 13:], torch.zeros(2, VOICE_FEATURE_SIZE)))


if __name__ == "__main__":
    unittest.main()
