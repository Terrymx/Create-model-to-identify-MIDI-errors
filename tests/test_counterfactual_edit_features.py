from __future__ import annotations

import unittest

import torch

from counterfactual_edit_features import (
    build_replacement_proposals,
    counterfactual_target_features,
)


class CounterfactualEditFeatureTests(unittest.TestCase):
    def test_gain_is_positive_when_proposal_is_more_likely(self) -> None:
        forward = torch.full((1, 128), 1e-4)
        backward = torch.full((1, 128), 1e-4)
        forward[0, 60], forward[0, 62] = 0.10, 0.40
        backward[0, 60], backward[0, 62] = 0.20, 0.30

        features = counterfactual_target_features(
            forward,
            backward,
            torch.tensor([60]),
            torch.tensor([62]),
        )

        self.assertGreater(float(features[0, 0]), 0.0)
        self.assertGreater(float(features[0, 1]), 0.0)
        self.assertGreater(float(features[0, 5]), 0.0)

    def test_directional_disagreement_and_pitch_relation_are_reported(self) -> None:
        forward = torch.full((1, 128), 1e-4)
        backward = torch.full((1, 128), 1e-4)
        forward[0, 60], forward[0, 72] = 0.10, 0.50
        backward[0, 60], backward[0, 72] = 0.50, 0.10

        features = counterfactual_target_features(
            forward,
            backward,
            torch.tensor([60]),
            torch.tensor([72]),
        )

        self.assertGreater(float(features[0, 5]), 0.0)
        self.assertEqual(float(features[0, 12]), 0.5)
        self.assertEqual(float(features[0, 13]), 1.0)

    def test_near_zero_probabilities_remain_finite(self) -> None:
        forward = torch.zeros((1, 128))
        backward = torch.zeros((1, 128))

        features = counterfactual_target_features(
            forward,
            backward,
            torch.tensor([60]),
            torch.tensor([61]),
        )

        self.assertTrue(torch.isfinite(features).all())

    def test_api_has_no_clean_target_argument(self) -> None:
        with self.assertRaises(TypeError):
            counterfactual_target_features(
                torch.ones((1, 128)) / 128,
                torch.ones((1, 128)) / 128,
                torch.tensor([60]),
                torch.tensor([61]),
                clean_target_pitch=torch.tensor([64]),
            )

    def test_proposals_union_sources_exclude_observed_and_deduplicate(self) -> None:
        detector = torch.zeros((1, 128))
        forward = torch.zeros((1, 128))
        backward = torch.zeros((1, 128))
        detector[0, 60], detector[0, 61], detector[0, 62] = 0.9, 0.8, 0.7
        forward[0, 61], forward[0, 63], forward[0, 64] = 0.95, 0.7, 0.6
        backward[0, 63], backward[0, 65], backward[0, 66] = 0.9, 0.8, 0.7

        proposals = build_replacement_proposals(
            detector,
            forward,
            backward,
            torch.tensor([60]),
            source_top_k=3,
            max_proposals=4,
        )

        self.assertEqual(proposals.shape, (1, 4))
        self.assertNotIn(60, proposals[0].tolist())
        self.assertEqual(len(set(proposals[0].tolist())), 4)
        self.assertEqual(proposals[0].tolist(), [61, 63, 65, 62])

    def test_proposal_ties_are_deterministic_and_api_rejects_clean_target(self) -> None:
        scores = torch.ones((1, 128))
        first = build_replacement_proposals(
            scores,
            scores,
            scores,
            torch.tensor([0]),
            source_top_k=2,
            max_proposals=2,
        )
        second = build_replacement_proposals(
            scores,
            scores,
            scores,
            torch.tensor([0]),
            source_top_k=2,
            max_proposals=2,
        )

        self.assertTrue(torch.equal(first, second))
        self.assertTrue(((first >= 0) & (first <= 127)).all())
        with self.assertRaises(TypeError):
            build_replacement_proposals(
                scores,
                scores,
                scores,
                torch.tensor([0]),
                clean_target_pitch=torch.tensor([64]),
            )


if __name__ == "__main__":
    unittest.main()
