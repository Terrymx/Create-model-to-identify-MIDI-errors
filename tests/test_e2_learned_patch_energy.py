from __future__ import annotations

import unittest

import numpy as np
import torch

from e2_learned_patch_energy import (
    E2PatchEnergyNet,
    build_patch_energy_tensors,
)


def _features_from_pitches(pitches: list[int]) -> np.ndarray:
    features = np.zeros((len(pitches), 36), dtype=np.float32)
    pitches_array = np.asarray(pitches, dtype=np.float32)
    features[:, 0] = pitches_array / 127.0
    features[:, 2] = 0.2
    features[:, 3] = 0.1
    features[:, 4] = np.diff(pitches_array, prepend=pitches_array[0]) / 24.0
    phase = 2.0 * np.pi * (pitches_array % 12.0) / 12.0
    features[:, 6] = np.sin(phase)
    features[:, 7] = np.cos(phase)
    features[:, 34] = 0.5
    features[:, 35] = 0.25
    return features


class E2LearnedPatchEnergyTests(unittest.TestCase):
    def test_build_patch_energy_tensors_shapes_observed_and_edited_proposals(self) -> None:
        features = _features_from_pitches([60, 62, 64, 65, 67, 69])

        tensors = build_patch_energy_tensors(
            piece_features=features,
            candidate_positions=np.asarray([2, 4], dtype=np.int64),
            observed_pitch=np.asarray([64, 67], dtype=np.float32),
            proposals=np.asarray([[63, 65, 67], [66, 68, 70]], dtype=np.float32),
            radius=2,
        )

        self.assertEqual(tensors.observed.shape, (2, 5, 39))
        self.assertEqual(tensors.edited.shape, (2, 3, 5, 39))
        self.assertEqual(tensors.mask.shape, (2, 5))
        self.assertTrue(np.isfinite(tensors.observed).all())
        self.assertTrue(np.isfinite(tensors.edited).all())

    def test_edited_patch_updates_center_pitch_without_mutating_observed(self) -> None:
        features = _features_from_pitches([60, 62, 64, 65, 67])

        tensors = build_patch_energy_tensors(
            piece_features=features,
            candidate_positions=np.asarray([2], dtype=np.int64),
            observed_pitch=np.asarray([64], dtype=np.float32),
            proposals=np.asarray([[63, 67]], dtype=np.float32),
            radius=1,
        )

        center = 1
        self.assertAlmostEqual(float(tensors.observed[0, center, 0]), 64 / 127.0)
        self.assertAlmostEqual(float(tensors.edited[0, 0, center, 0]), 63 / 127.0)
        self.assertAlmostEqual(float(tensors.edited[0, 1, center, 0]), 67 / 127.0)
        self.assertNotEqual(float(tensors.edited[0, 0, center, 6]), float(tensors.observed[0, center, 6]))

    def test_model_forward_returns_detection_and_proposal_logits(self) -> None:
        model = E2PatchEnergyNet(candidate_dim=7, patch_feature_dim=39, hidden_dim=8)
        candidate = torch.zeros((4, 7), dtype=torch.float32)
        observed = torch.zeros((4, 5, 39), dtype=torch.float32)
        edited = torch.zeros((4, 3, 5, 39), dtype=torch.float32)
        mask = torch.ones((4, 5), dtype=torch.float32)

        candidate_logits, proposal_logits = model(candidate, observed, edited, mask)

        self.assertEqual(tuple(candidate_logits.shape), (4,))
        self.assertEqual(tuple(proposal_logits.shape), (4, 3))


if __name__ == "__main__":
    unittest.main()
