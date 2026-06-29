from __future__ import annotations

import unittest

import numpy as np

from proposal_trust_audit import group_summary, target_rank_summary


class ProposalTrustAuditTests(unittest.TestCase):
    def test_group_summary_reports_positive_and_target_present_rates(self) -> None:
        arrays = {
            "labels": np.asarray([1, 0, 1, 0], dtype=np.int64),
            "target_pitch": np.asarray([64, 65, 67, 69], dtype=np.int64),
            "proposals": np.asarray(
                [[60, 64, 65], [65, 66, 67], [60, 61, 62], [69, 70, 71]],
                dtype=np.int64,
            ),
        }
        mask = np.asarray([True, True, True, False])
        baseline_scores = np.asarray([0.2, 0.8, 0.4, 0.1], dtype=np.float32)
        rescue_scores = np.asarray([0.9, 0.6, 0.3, 0.2], dtype=np.float32)

        summary = group_summary("slice", mask, arrays, baseline_scores, rescue_scores)

        self.assertEqual(summary["name"], "slice")
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["positives"], 2)
        self.assertAlmostEqual(summary["positive_rate"], 2 / 3)
        self.assertAlmostEqual(summary["target_present_fraction"], 2 / 3)
        self.assertAlmostEqual(summary["mean_baseline_score"], (0.2 + 0.8 + 0.4) / 3)

    def test_target_rank_summary_counts_topk_among_target_present_rows(self) -> None:
        proposals = np.asarray([[60, 64, 65], [65, 66, 67], [70, 71, 72]], dtype=np.int64)
        target_pitch = np.asarray([64, 67, 69], dtype=np.int64)
        proposal_scores = np.asarray(
            [[0.2, 0.9, 0.1], [0.8, 0.7, 0.6], [0.1, 0.2, 0.3]],
            dtype=np.float32,
        )

        summary = target_rank_summary(proposals, target_pitch, proposal_scores)

        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["target_present"], 2)
        self.assertAlmostEqual(summary["top1"], 0.5)
        self.assertAlmostEqual(summary["top3"], 1.0)


if __name__ == "__main__":
    unittest.main()
