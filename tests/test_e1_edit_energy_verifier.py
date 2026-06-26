from __future__ import annotations

import unittest

import numpy as np
import torch

from e1_edit_energy_verifier import (
    E1EditEnergyNet,
    build_e1_feature_tensors,
    proposal_target_indices,
)


class E1EditEnergyVerifierTest(unittest.TestCase):
    def test_proposal_target_indices_only_marks_true_replacements(self) -> None:
        labels = np.asarray([1, 1, 0, 1], dtype=np.int64)
        error_kind = np.asarray([1, 2, 1, 1], dtype=np.int64)
        target_pitch = np.asarray([64, 66, 67, 70], dtype=np.int64)
        proposals = np.asarray(
            [
                [60, 64, 65, 67],
                [66, 67, 68, 69],
                [67, 68, 69, 70],
                [60, 61, 62, 63],
            ],
            dtype=np.int64,
        )

        target_indices, mask = proposal_target_indices(
            labels,
            error_kind,
            target_pitch,
            proposals,
        )

        np.testing.assert_array_equal(target_indices, np.asarray([1, 0, 0, 0]))
        np.testing.assert_array_equal(mask, np.asarray([True, False, False, False]))

    def test_build_e1_feature_tensors_aligns_candidate_and_proposals(self) -> None:
        arrays = {
            "observed_pitch": np.asarray([60, 61], dtype=np.float32),
            "proposals": np.asarray([[62, 64, 65, 67], [59, 60, 62, 64]], dtype=np.float32),
            "b_features": np.ones((2, 4, 17), dtype=np.float32),
            "b_ranking": np.asarray([[0.1, 0.4, 0.2, 0.3], [0.3, 0.2, 0.1, 0.0]], dtype=np.float32),
            "c_features": np.ones((2, 4, 15), dtype=np.float32) * 2,
            "c_ranking": np.asarray([[0.0, 0.5, 0.2, 0.1], [0.4, 0.2, 0.1, 0.0]], dtype=np.float32),
        }
        candidate = np.zeros((2, 6), dtype=np.float32)

        tensors = build_e1_feature_tensors(arrays, candidate)

        self.assertEqual(tensors.candidate.shape, (2, 6))
        self.assertEqual(tensors.proposal.shape[0:2], (2, 4))
        self.assertGreater(tensors.proposal.shape[2], 34)
        self.assertAlmostEqual(float(tensors.proposal[0, 1, -3]), 60 / 127.0)
        self.assertAlmostEqual(float(tensors.proposal[0, 1, -2]), 64 / 127.0)
        self.assertAlmostEqual(float(tensors.proposal[0, 1, -1]), 4 / 24.0)

    def test_model_forward_returns_candidate_and_proposal_logits(self) -> None:
        model = E1EditEnergyNet(candidate_dim=6, proposal_dim=10, hidden_dim=8)
        candidate = torch.zeros((3, 6), dtype=torch.float32)
        proposal = torch.zeros((3, 4, 10), dtype=torch.float32)

        candidate_logits, proposal_logits = model(candidate, proposal)

        self.assertEqual(tuple(candidate_logits.shape), (3,))
        self.assertEqual(tuple(proposal_logits.shape), (3, 4))


if __name__ == "__main__":
    unittest.main()
