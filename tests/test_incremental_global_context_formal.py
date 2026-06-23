from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from run_incremental_global_context_formal import (
    aggregate_config_checkpoints,
    atomic_write_json,
    finalize_calibration_grid,
    load_selected_calibration,
    parse_args,
    partition_piece_ids,
    select_calibration_config,
)


class IncrementalGlobalContextFormalTests(unittest.TestCase):
    def test_partition_piece_ids_is_disjoint_and_complete(self) -> None:
        piece_ids = [14, 24, 30, 31, 35, 37, 39, 49, 53]

        partitions = [
            partition_piece_ids(piece_ids, worker_index=index, worker_count=4)
            for index in range(4)
        ]

        flattened = [piece_id for part in partitions for piece_id in part]
        self.assertEqual(sorted(flattened), piece_ids)
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(partitions[0], [14, 35, 53])
        self.assertEqual(partitions[1], [24, 37])

    def test_partition_piece_ids_rejects_invalid_worker(self) -> None:
        with self.assertRaises(ValueError):
            partition_piece_ids([1, 2], worker_index=4, worker_count=4)

    def test_parse_args_accepts_parallel_phase_controls(self) -> None:
        args = parse_args(
            [
                "--cache-dir",
                "cache",
                "--data-root",
                "data",
                "--threeclass-checkpoint",
                "three.pt",
                "--binary-checkpoint",
                "binary.pt",
                "--forward-checkpoint",
                "forward.pt",
                "--backward-checkpoint",
                "backward.pt",
                "--verifier-checkpoint",
                "verifier.joblib",
                "--progress-dir",
                "progress",
                "--output-json",
                "result.json",
                "--phase",
                "calibration",
                "--worker-index",
                "3",
                "--worker-count",
                "4",
            ]
        )

        self.assertEqual(args.phase, "calibration")
        self.assertEqual(args.worker_index, 3)
        self.assertEqual(args.worker_count, 4)

    def test_aggregate_config_checkpoints_reads_every_expected_piece(self) -> None:
        config = {"score_floor": 0.5, "edit_rate": 0.01}
        with tempfile.TemporaryDirectory() as temp_dir:
            progress_dir = Path(temp_dir)
            rows = {
                1: {
                    "piece_id": 1,
                    "tp": 3,
                    "fp": 1,
                    "errors": 5,
                    "seconds": 2.0,
                    "window_hits": 4,
                    "window_misses": 2,
                    "edits": [],
                },
                2: {
                    "piece_id": 2,
                    "tp": 2,
                    "fp": 0,
                    "errors": 4,
                    "seconds": 3.0,
                    "window_hits": 5,
                    "window_misses": 1,
                    "edits": [],
                },
            }
            for piece_id, row in rows.items():
                path = (
                    progress_dir
                    / "calibration"
                    / "floor0.500000_rate0.0100"
                    / f"{piece_id}.json"
                )
                atomic_write_json(path, row)

            result = aggregate_config_checkpoints(
                progress_dir=progress_dir,
                split="calibration",
                config=config,
                expected_piece_ids=[1, 2],
            )

        self.assertEqual(result["tp"], 5)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["pieces"], 2)
        self.assertAlmostEqual(result["seconds"], 5.0)

    def test_aggregate_config_checkpoints_rejects_missing_piece(self) -> None:
        config = {"score_floor": 0.5, "edit_rate": 0.01}
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "missing"):
                aggregate_config_checkpoints(
                    progress_dir=Path(temp_dir),
                    split="calibration",
                    config=config,
                    expected_piece_ids=[1],
                )

    def test_atomic_write_json_leaves_complete_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "result.json"
            atomic_write_json(path, {"piece_id": 7, "tp": 2})

            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded, {"piece_id": 7, "tp": 2})

    def test_finalize_calibration_grid_persists_selected_config(self) -> None:
        configs = [
            {"score_floor": 0.5, "edit_rate": 0.01},
            {"score_floor": 0.6, "edit_rate": 0.01},
        ]
        rows = [
            {
                "piece_id": 1,
                "tp": 5,
                "fp": 2,
                "errors": 10,
                "seconds": 1.0,
                "window_hits": 1,
                "window_misses": 1,
                "edits": [],
            },
            {
                "piece_id": 1,
                "tp": 4,
                "fp": 0,
                "errors": 10,
                "seconds": 1.0,
                "window_hits": 1,
                "window_misses": 1,
                "edits": [],
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            progress_dir = Path(temp_dir)
            for config, row in zip(configs, rows):
                path = (
                    progress_dir
                    / "calibration"
                    / (
                        f"floor{config['score_floor']:.6f}_"
                        f"rate{config['edit_rate']:.4f}"
                    )
                    / "1.json"
                )
                atomic_write_json(path, row)

            grid, selected = finalize_calibration_grid(
                progress_dir=progress_dir,
                configs=configs,
                expected_piece_ids=[1],
                required_precision=0.8,
            )
            loaded = load_selected_calibration(progress_dir)

        self.assertEqual(len(grid), 2)
        self.assertEqual(selected["config"], configs[1])
        self.assertEqual(loaded, selected)

    def test_finalize_calibration_grid_rejects_partial_grid(self) -> None:
        configs = [
            {"score_floor": 0.5, "edit_rate": 0.01},
            {"score_floor": 0.6, "edit_rate": 0.01},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            progress_dir = Path(temp_dir)
            atomic_write_json(
                (
                    progress_dir
                    / "calibration"
                    / "floor0.500000_rate0.0100"
                    / "1.json"
                ),
                {
                    "piece_id": 1,
                    "tp": 1,
                    "fp": 0,
                    "errors": 2,
                    "seconds": 1.0,
                    "window_hits": 1,
                    "window_misses": 1,
                    "edits": [],
                },
            )

            with self.assertRaisesRegex(RuntimeError, "missing"):
                finalize_calibration_grid(
                    progress_dir=progress_dir,
                    configs=configs,
                    expected_piece_ids=[1],
                    required_precision=0.8,
                )

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
