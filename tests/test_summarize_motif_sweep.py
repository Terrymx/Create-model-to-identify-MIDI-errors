from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from summarize_motif_sweep import best_rows, iter_result_rows


def _write_result(path: Path, radius: int, similarity: float, recall: float) -> None:
    payload = {
        "motif": {
            "radius": radius,
            "min_similarity": similarity,
            "exclude_radius": 16,
        },
        "systems": {
            "C2_motif": {
                "best_feasible_test": {
                    "selected_test": {
                        "precision": 0.801,
                        "recall": recall,
                        "f1": 0.69,
                        "tp": 10,
                        "fp": 2,
                        "fn": 4,
                    }
                }
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class SummarizeMotifSweepTest(unittest.TestCase):
    def test_extracts_and_ranks_best_feasible_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_result(root / "low.json", radius=3, similarity=0.84, recall=0.59)
            _write_result(root / "high.json", radius=4, similarity=0.82, recall=0.62)

            rows = list(iter_result_rows([root / "low.json", root / "high.json"]))
            ranked = best_rows(rows, min_precision=0.80)

        self.assertEqual([row["source"] for row in ranked], ["high.json", "low.json"])
        self.assertEqual(ranked[0]["system"], "C2_motif")
        self.assertEqual(ranked[0]["radius"], 4)
        self.assertAlmostEqual(ranked[0]["min_similarity"], 0.82)


if __name__ == "__main__":
    unittest.main()
