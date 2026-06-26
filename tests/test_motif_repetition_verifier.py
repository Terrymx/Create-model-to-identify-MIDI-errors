from __future__ import annotations

import unittest

import numpy as np

from motif_repetition_features import motif_feature_size
from run_counterfactual_edit_verifier import (
    build_b_variant_features,
    build_c_variant_features,
)
from run_motif_repetition_verifier import (
    build_b2_motif_features,
    build_c2_motif_features,
)


class MotifRepetitionVerifierTests(unittest.TestCase):
    def _arrays(self) -> dict[str, np.ndarray]:
        rows, proposals = 3, 4
        base = np.arange(rows * 5, dtype=np.float32).reshape(rows, 5)
        b = np.arange(rows * proposals * 17, dtype=np.float32).reshape(
            rows,
            proposals,
            17,
        )
        c = np.arange(rows * proposals * 15, dtype=np.float32).reshape(
            rows,
            proposals,
            15,
        )
        b_ranking = b[..., 2]
        c_ranking = c[..., 0]
        motif = np.full((rows, motif_feature_size()), 0.25, dtype=np.float32)
        return {
            "base_features": base,
            "b_features": b,
            "b_ranking": b_ranking,
            "c_features": c,
            "c_ranking": c_ranking,
            "motif_features": motif,
        }

    def test_b2_motif_appends_features_without_changing_prefix(self) -> None:
        arrays = self._arrays()
        baseline = build_b_variant_features(
            arrays["base_features"],
            arrays["b_features"],
            arrays["b_ranking"],
            "B2",
        )

        combined = build_b2_motif_features(arrays)

        np.testing.assert_allclose(combined[:, : baseline.shape[1]], baseline)
        np.testing.assert_allclose(combined[:, baseline.shape[1] :], arrays["motif_features"])

    def test_c2_motif_appends_features_without_changing_prefix(self) -> None:
        arrays = self._arrays()
        baseline = build_c_variant_features(
            arrays["base_features"],
            arrays["b_features"],
            arrays["b_ranking"],
            arrays["c_features"],
            arrays["c_ranking"],
            "C2",
        )

        combined = build_c2_motif_features(arrays)

        np.testing.assert_allclose(combined[:, : baseline.shape[1]], baseline)
        np.testing.assert_allclose(combined[:, baseline.shape[1] :], arrays["motif_features"])


if __name__ == "__main__":
    unittest.main()
