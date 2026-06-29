from __future__ import annotations

import unittest

import numpy as np
import torch

from clean_patch_predictor import (
    CleanPatchBatch,
    CleanPatchPredictor,
    DenoisingPatchBatch,
    build_candidate_patch_batch,
    build_clean_patch_batch,
    build_denoising_patch_batch,
    build_patch_predictor_features,
    patch_denoising_loss,
    patch_negative_log_likelihood,
)


def _features_from_pitches(pitches: list[int]) -> np.ndarray:
    features = np.zeros((len(pitches), 36), dtype=np.float32)
    pitches_array = np.asarray(pitches, dtype=np.float32)
    features[:, 0] = pitches_array / 127.0
    features[:, 2] = 0.25
    features[:, 3] = 0.125
    features[:, 4] = np.diff(pitches_array, prepend=pitches_array[0]) / 24.0
    phase = 2.0 * np.pi * (pitches_array % 12.0) / 12.0
    features[:, 6] = np.sin(phase)
    features[:, 7] = np.cos(phase)
    return features


class CleanPatchPredictorTests(unittest.TestCase):
    def test_build_clean_patch_batch_masks_center_pitch_columns(self) -> None:
        features = _features_from_pitches([60, 62, 64, 65, 67])

        batch = build_clean_patch_batch(
            piece_features=features,
            center_positions=np.asarray([2], dtype=np.int64),
            radius=1,
        )

        self.assertIsInstance(batch, CleanPatchBatch)
        self.assertEqual(batch.context.shape, (1, 3, 39))
        self.assertEqual(batch.mask.shape, (1, 3))
        self.assertEqual(batch.target_pitch.shape, (1,))
        self.assertEqual(int(batch.target_pitch[0]), 64)
        self.assertAlmostEqual(float(batch.context[0, 1, 0]), 0.0)
        self.assertAlmostEqual(float(batch.context[0, 1, 6]), 0.0)
        self.assertAlmostEqual(float(batch.context[0, 1, 7]), 0.0)
        self.assertAlmostEqual(float(batch.context[0, 0, 0]), 62 / 127.0)

    def test_build_candidate_patch_batch_returns_observed_and_edited_contexts(self) -> None:
        features = _features_from_pitches([60, 62, 64, 65, 67])

        observed, edited, mask = build_candidate_patch_batch(
            piece_features=features,
            candidate_positions=np.asarray([2], dtype=np.int64),
            observed_pitch=np.asarray([64], dtype=np.float32),
            proposals=np.asarray([[63, 67]], dtype=np.float32),
            radius=1,
        )

        self.assertEqual(observed.context.shape, (1, 3, 39))
        self.assertEqual(edited.context.shape, (2, 3, 39))
        self.assertEqual(mask.shape, (1, 2))
        self.assertEqual(observed.target_pitch.tolist(), [64])
        self.assertEqual(edited.target_pitch.tolist(), [63, 67])

    def test_clean_patch_predictor_returns_pitch_logits_and_energy(self) -> None:
        model = CleanPatchPredictor(patch_feature_dim=39, hidden_dim=8)
        batch = CleanPatchBatch(
            context=np.zeros((2, 3, 39), dtype=np.float32),
            mask=np.ones((2, 3), dtype=np.float32),
            target_pitch=np.asarray([60, 64], dtype=np.int64),
        )

        logits = model(
            torch.from_numpy(batch.context).float(),
            torch.from_numpy(batch.mask).float(),
        )
        energy = patch_negative_log_likelihood(
            logits,
            torch.from_numpy(batch.target_pitch).long(),
        )

        self.assertEqual(tuple(logits.shape), (2, 128))
        self.assertEqual(tuple(energy.shape), (2,))
        self.assertTrue(torch.isfinite(energy).all())
        self.assertLessEqual(float(energy.min()), float(energy.max()))

    def test_build_patch_predictor_features_summarizes_proposal_energy_gain(self) -> None:
        features = build_patch_predictor_features(
            observed_energy=np.asarray([5.0, 2.0], dtype=np.float32),
            proposal_energy=np.asarray(
                [
                    [4.0, 6.0, 7.0],
                    [3.0, 1.5, 9.0],
                ],
                dtype=np.float32,
            ),
            proposal_mask=np.asarray(
                [
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )

        self.assertEqual(features.shape, (2, 8))
        self.assertAlmostEqual(float(features[0, 0]), 5.0)
        self.assertAlmostEqual(float(features[0, 1]), 4.0)
        self.assertAlmostEqual(float(features[0, 3]), 1.0)
        self.assertAlmostEqual(float(features[0, 6]), 2.0)
        self.assertAlmostEqual(float(features[0, 7]), 1.0)
        self.assertAlmostEqual(float(features[1, 3]), 0.5)

    def test_build_denoising_patch_batch_keeps_clean_target_and_observed_negative(self) -> None:
        features = _features_from_pitches([60, 62, 64, 65, 67])

        batch = build_denoising_patch_batch(
            piece_features=features,
            center_positions=np.asarray([2, 3], dtype=np.int64),
            target_pitch=np.asarray([64, 69], dtype=np.float32),
            observed_pitch=np.asarray([64, 65], dtype=np.float32),
            radius=1,
        )

        self.assertIsInstance(batch, DenoisingPatchBatch)
        self.assertEqual(batch.context.shape, (2, 3, 39))
        self.assertEqual(batch.target_pitch.tolist(), [64, 69])
        self.assertEqual(batch.negative_pitch.tolist(), [64, 65])
        self.assertEqual(batch.negative_mask.tolist(), [0.0, 1.0])
        self.assertAlmostEqual(float(batch.context[1, 1, 0]), 0.0)

    def test_patch_denoising_loss_adds_margin_when_negative_beats_target(self) -> None:
        logits = torch.zeros((2, 128), dtype=torch.float32)
        logits[0, 64] = 4.0
        logits[0, 60] = 1.0
        logits[1, 69] = 1.0
        logits[1, 65] = 4.0
        target = torch.asarray([64, 69], dtype=torch.long)
        negative = torch.asarray([60, 65], dtype=torch.long)
        negative_mask = torch.asarray([1.0, 1.0], dtype=torch.float32)

        plain = patch_negative_log_likelihood(logits, target).mean()
        denoising = patch_denoising_loss(
            logits,
            target,
            negative,
            negative_mask,
            contrastive_weight=1.0,
            margin=1.0,
        )

        self.assertGreater(float(denoising), float(plain))


if __name__ == "__main__":
    unittest.main()
