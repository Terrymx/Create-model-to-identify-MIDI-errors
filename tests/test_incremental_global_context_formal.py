from __future__ import annotations

import unittest

from run_incremental_global_context_formal import select_calibration_config


class IncrementalGlobalContextFormalTests(unittest.TestCase):
    def test_selects_highest_recall_at_required_precision(self) -> None:
        rows = [
            {"precision": 0.82, "recall": 0.55, "config": {"name": "a"}},
            {"precision": 0.81, "recall": 0.58, "config": {"name": "b"}},
            {"precision": 0.79, "recall": 0.62, "config": {"name": "c"}},
        ]

        selected = select_calibration_config(rows, required_precision=0.81)

        self.assertEqual(selected["config"]["name"], "b")


if __name__ == "__main__":
    unittest.main()
