from __future__ import annotations

import unittest

import torch

from counterfactual_edit_features import (
    apply_observed_pitch_edit,
    build_replacement_proposals,
    correction_pitch_distribution,
    counterfactual_target_features,
    directional_pitch_distribution,
    local_edit_impact_features,
)


class CounterfactualEditFeatureTests(unittest.TestCase):
    class _FakeDirectionalModel:
        def predict_pitch(self, features: torch.Tensor, causal: bool) -> torch.Tensor:
            self.features = features.clone()
            rows, length, _ = features.shape
            logits = torch.zeros(rows, length, 128)
            logits[..., 60] = 1.0
            return logits

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

    def test_directional_distribution_does_not_read_target_features(self) -> None:
        raw = torch.zeros((1, 4, 8))
        raw[0, :, 0] = torch.tensor([50, 51, 52, 53]) / 127.0
        mask = torch.ones((1, 4), dtype=torch.bool)
        model = self._FakeDirectionalModel()

        baseline, available = directional_pitch_distribution(
            model,
            raw,
            mask,
            "forward",
            [0, 1, 2, 3, 5, 6, 7],
        )
        perturbed = raw.clone()
        perturbed[0, 2, :] = 99.0
        changed, _ = directional_pitch_distribution(
            model,
            perturbed,
            mask,
            "forward",
            [0, 1, 2, 3, 5, 6, 7],
        )

        self.assertTrue(torch.equal(baseline[:, 2], changed[:, 2]))
        self.assertFalse(bool(available[0, 0]))
        self.assertTrue(bool(available[0, 2]))

    def test_correction_distribution_is_preferred_and_renormalized(self) -> None:
        outputs = {
            "pitch_logits": torch.zeros((1, 1, 128)),
            "correction_logits": torch.zeros((1, 1, 129)),
        }
        outputs["pitch_logits"][0, 0, 20] = 10.0
        outputs["correction_logits"][0, 0, 70] = 5.0
        outputs["correction_logits"][0, 0, 128] = 8.0

        probability = correction_pitch_distribution(outputs)

        self.assertEqual(int(probability[0, 0].argmax()), 70)
        self.assertAlmostEqual(float(probability[0, 0].sum()), 1.0, places=6)

    def test_observed_pitch_edit_changes_only_safe_pitch_columns(self) -> None:
        raw = torch.arange(36, dtype=torch.float32).reshape(1, 1, 36)

        edited = apply_observed_pitch_edit(raw, torch.tensor([0]), torch.tensor([61]))

        changed = torch.nonzero(
            edited[0, 0] != raw[0, 0],
            as_tuple=False,
        ).flatten().tolist()
        self.assertEqual(changed, [0, 6, 7])
        self.assertAlmostEqual(float(edited[0, 0, 0]), 61 / 127.0, places=6)

    def test_local_impact_respects_radii_mask_and_directional_agreement(self) -> None:
        original_forward = torch.zeros((1, 7))
        original_backward = torch.zeros((1, 7))
        edited_forward = torch.tensor([[0.0, 0.0, 1.0, 2.0, 1.0, -1.0, 8.0]])
        edited_backward = torch.tensor([[0.0, 0.0, 0.5, 1.0, 0.5, -2.0, 8.0]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)

        features = local_edit_impact_features(
            original_forward,
            original_backward,
            edited_forward,
            edited_backward,
            torch.tensor([3]),
            mask,
            radii=(1, 2),
        )

        self.assertEqual(features.shape, (1, 10))
        self.assertGreater(float(features[0, 0]), 0.0)
        self.assertGreater(float(features[0, 1]), 0.0)
        self.assertEqual(float(features[0, 4]), 1.0)
        self.assertLess(float(features[0, 8]), 8.0)


if __name__ == "__main__":
    unittest.main()
