from __future__ import annotations

import unittest

from run_counterfactual_piece_c import comparison_variants


class CounterfactualPieceCTests(unittest.TestCase):
    def test_comparison_contains_approved_c_combinations(self) -> None:
        variants = comparison_variants()

        self.assertEqual(
            variants,
            {
                "B2": ("B2", None),
                "B2_C_radius4": ("B2", "C1"),
                "B2_C_radius4_8_16": ("B2", "C2"),
                "B3_C_radius4": ("B3", "C1"),
            },
        )


if __name__ == "__main__":
    unittest.main()
