from __future__ import annotations

import unittest

import torch

from verifier_theory_features import (
    ACCIDENTAL_TOUCH_INDEX,
    ORNAMENT_PROTECTION_INDEX,
    THEORY_INTERACTION_SIZE,
    build_theory_interaction_features,
)
from run_frozen_union_candidate_context_verifier import append_theory_features


def base_inputs() -> tuple[torch.Tensor, ...]:
    raw = torch.zeros(1, 5, 36)
    valid = torch.tensor([[True, True, True, True, False]])
    three = torch.full((1, 5), 0.6)
    binary = torch.full((1, 5), 0.6)
    delete = torch.zeros(1, 5)
    forward = torch.zeros(1, 5)
    backward = torch.zeros(1, 5)
    return raw, valid, three, binary, delete, forward, backward


class TheoryInteractionFeatureTests(unittest.TestCase):
    def test_features_have_stable_finite_shape(self) -> None:
        values = build_theory_interaction_features(*base_inputs())
        self.assertEqual(values.shape, (1, 5, THEORY_INTERACTION_SIZE))
        self.assertTrue(bool(torch.isfinite(values).all()))

    def test_ornament_protection_increases_for_resolving_passing_note(self) -> None:
        raw, valid, three, binary, delete, forward, backward = base_inputs()
        raw[0, 2, 26:33] = torch.tensor([1, 1, 1, 1, 0, 1, 1])

        values = build_theory_interaction_features(
            raw,
            valid,
            three,
            binary,
            delete,
            forward,
            backward,
        )

        self.assertGreater(
            float(values[0, 2, ORNAMENT_PROTECTION_INDEX]),
            float(values[0, 1, ORNAMENT_PROTECTION_INDEX]),
        )

    def test_accidental_touch_uses_short_neighbor_delete_evidence(self) -> None:
        raw, valid, three, binary, delete, forward, backward = base_inputs()
        raw[0, 2, 4] = 1.0 / 24.0
        raw[0, 2, 33] = 0.05
        delete[0, 2] = 0.95

        values = build_theory_interaction_features(
            raw,
            valid,
            three,
            binary,
            delete,
            forward,
            backward,
        )

        self.assertGreater(
            float(values[0, 2, ACCIDENTAL_TOUCH_INDEX]),
            float(values[0, 1, ACCIDENTAL_TOUCH_INDEX]),
        )

    def test_padding_does_not_change_valid_local_statistics(self) -> None:
        raw, valid, three, binary, delete, forward, backward = base_inputs()
        first = build_theory_interaction_features(
            raw,
            valid,
            three,
            binary,
            delete,
            forward,
            backward,
        )
        raw[0, 4] = 999.0
        second = build_theory_interaction_features(
            raw,
            valid,
            three,
            binary,
            delete,
            forward,
            backward,
        )

        self.assertTrue(torch.allclose(first[:, :4], second[:, :4]))

    def test_append_preserves_prefix_and_adds_theory_columns(self) -> None:
        base = torch.randn(1, 5, 11)
        raw, valid, three, binary, delete, forward, backward = base_inputs()

        result = append_theory_features(
            base,
            raw,
            valid,
            three,
            binary,
            delete,
            forward,
            backward,
        )

        self.assertTrue(torch.equal(result[..., :11], base))
        self.assertEqual(result.shape[-1], 11 + THEORY_INTERACTION_SIZE)


if __name__ == "__main__":
    unittest.main()
