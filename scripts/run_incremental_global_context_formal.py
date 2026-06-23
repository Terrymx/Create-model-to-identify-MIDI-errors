from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np


def select_calibration_config(
    rows: list[dict],
    required_precision: float,
) -> dict:
    feasible = [row for row in rows if row["precision"] >= required_precision]
    return max(
        feasible or rows,
        key=lambda row: (
            row["recall"] if row["precision"] >= required_precision else -1.0,
            row["precision"],
        ),
    )


def metric_row(tp: int, fp: int, total_errors: int) -> dict:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_errors, 1)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(total_errors - tp),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "selected": int(tp + fp),
    }


def partition_piece_ids(
    piece_ids: list[int],
    worker_index: int,
    worker_count: int,
) -> list[int]:
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if worker_index < 0 or worker_index >= worker_count:
        raise ValueError(
            f"worker_index must be in [0, {worker_count}), got {worker_index}"
        )
    return sorted(int(piece_id) for piece_id in piece_ids)[
        worker_index::worker_count
    ]


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_piece_configurations(
    *,
    configs: list[dict],
    make_scorer: Callable[[], object],
    run_search: Callable[[object, dict], dict],
) -> list[dict]:
    if not configs:
        return []
    scorer = make_scorer()
    return [run_search(scorer, config) for config in configs]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--threeclass-checkpoint", required=True)
    parser.add_argument("--binary-checkpoint", required=True)
    parser.add_argument("--forward-checkpoint", required=True)
    parser.add_argument("--backward-checkpoint", required=True)
    parser.add_argument("--verifier-checkpoint", required=True)
    parser.add_argument("--progress-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--calibration-precision", type=float, default=0.81)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("calibration", "test", "finalize"),
    )
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    args = parser.parse_args(argv)
    partition_piece_ids([], args.worker_index, args.worker_count)
    return args


def _load_raw_split(cache_dir: Path, split: str) -> dict[str, np.ndarray]:
    with np.load(cache_dir / f"{split}.npz", allow_pickle=False) as loaded:
        return {
            name: loaded[name].copy()
            for name in loaded.files
            if name != "__metadata__"
        }


def _baseline(
    arrays: dict[str, np.ndarray],
    verifier_model,
    total_errors: int,
    precision_target: float,
) -> tuple[dict, float]:
    from run_counterfactual_edit_verifier import build_c_variant_features
    from run_counterfactual_global_search import canonical_candidate_indices
    from run_counterfactual_piece_b import select_calibration_threshold

    rows = canonical_candidate_indices(arrays, 256)
    features = build_c_variant_features(
        arrays["base_features"],
        arrays["b_features"],
        arrays["b_ranking"],
        arrays["c_features"],
        arrays["c_ranking"],
        "C2",
        b_variant="B2",
    )
    scores = verifier_model.predict_proba(features)[:, 1][rows]
    selected = select_calibration_threshold(
        scores,
        arrays["labels"][rows],
        total_errors,
        precision_target,
    )
    return selected, float(selected["threshold"])


def _baseline_at_threshold(
    arrays: dict[str, np.ndarray],
    verifier_model,
    total_errors: int,
    threshold: float,
) -> dict:
    from run_counterfactual_edit_verifier import build_c_variant_features
    from run_counterfactual_global_search import canonical_candidate_indices

    rows = canonical_candidate_indices(arrays, 256)
    features = build_c_variant_features(
        arrays["base_features"],
        arrays["b_features"],
        arrays["b_ranking"],
        arrays["c_features"],
        arrays["c_ranking"],
        "C2",
        b_variant="B2",
    )
    scores = verifier_model.predict_proba(features)[:, 1][rows]
    selected = scores >= threshold
    labels = arrays["labels"][rows].astype(bool)
    return {
        **metric_row(
            int((selected & labels).sum()),
            int((selected & ~labels).sum()),
            total_errors,
        ),
        "threshold": float(threshold),
    }


def _config_name(config: dict) -> str:
    return f"floor{config['score_floor']:.6f}_rate{config['edit_rate']:.4f}"


def _piece_result_path(
    progress_dir: Path,
    split: str,
    config: dict,
    piece_id: int,
) -> Path:
    return progress_dir / split / _config_name(config) / f"{piece_id}.json"


def aggregate_config_checkpoints(
    *,
    progress_dir: Path,
    split: str,
    config: dict,
    expected_piece_ids: list[int],
) -> dict:
    totals = {
        "tp": 0,
        "fp": 0,
        "errors": 0,
        "seconds": 0.0,
        "window_hits": 0,
        "window_misses": 0,
        "pieces": 0,
    }
    missing = []
    for piece_id in sorted(int(value) for value in expected_piece_ids):
        result_path = _piece_result_path(
            progress_dir, split, config, piece_id
        )
        if not result_path.exists():
            missing.append(piece_id)
            continue
        row = json.loads(result_path.read_text(encoding="utf-8"))
        totals["tp"] += int(row["tp"])
        totals["fp"] += int(row["fp"])
        totals["errors"] += int(row["errors"])
        totals["seconds"] += float(row["seconds"])
        totals["window_hits"] += int(row["window_hits"])
        totals["window_misses"] += int(row["window_misses"])
        totals["pieces"] += 1
    if missing:
        raise RuntimeError(
            f"missing {len(missing)} {split} checkpoints for "
            f"{_config_name(config)}: {missing[:10]}"
        )
    metrics = metric_row(totals["tp"], totals["fp"], totals["errors"])
    metrics.update(
        {
            "config": config,
            "seconds": totals["seconds"],
            "pieces": totals["pieces"],
            "window_hits": totals["window_hits"],
            "window_misses": totals["window_misses"],
        }
    )
    return metrics


def _selected_calibration_path(progress_dir: Path) -> Path:
    return progress_dir / "selected_calibration.json"


def finalize_calibration_grid(
    *,
    progress_dir: Path,
    configs: list[dict],
    expected_piece_ids: list[int],
    required_precision: float,
) -> tuple[list[dict], dict]:
    rows = [
        aggregate_config_checkpoints(
            progress_dir=progress_dir,
            split="calibration",
            config=config,
            expected_piece_ids=expected_piece_ids,
        )
        for config in configs
    ]
    selected = select_calibration_config(rows, required_precision)
    atomic_write_json(_selected_calibration_path(progress_dir), selected)
    return rows, selected


def load_selected_calibration(progress_dir: Path) -> dict:
    path = _selected_calibration_path(progress_dir)
    if not path.exists():
        raise RuntimeError(
            f"selected calibration does not exist: {path}; run finalize first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_piece_search(
    *,
    piece_id: int,
    arrays: dict[str, np.ndarray],
    dataset,
    models: tuple,
    verifier_model,
):
    import torch

    from incremental_piece_scorer import IncrementalPieceScorer
    from run_counterfactual_global_search import canonical_candidate_indices

    full_rows = np.flatnonzero(arrays["file_ids"] == piece_id)
    piece_arrays = {
        name: values[full_rows]
        for name, values in arrays.items()
    }
    piece_arrays["window_starts"] = (
        piece_arrays["positions"] - piece_arrays["local_positions"]
    ).astype(np.int64)
    canonical_rows = canonical_candidate_indices(piece_arrays, 256)
    canonical = {
        name: values[canonical_rows]
        for name, values in piece_arrays.items()
    }
    scorer = IncrementalPieceScorer(
        piece_id=int(piece_id),
        piece_features=dataset._pieces[int(piece_id)].features,
        candidate_arrays=piece_arrays,
        output_rows=canonical_rows,
        models=models,
        verifier_model=verifier_model,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    candidate_positions = canonical["positions"].astype(np.int64)
    return {
        "piece_id": int(piece_id),
        "scorer": scorer,
        "candidate_positions": candidate_positions,
        "proposals": {
            index: tuple(
                int(value)
                for value in canonical["c_proposals"][index, :2]
            )
            for index in range(len(candidate_positions))
        },
        "labels": canonical["labels"].astype(bool),
        "note_count": int(dataset._note_counts[int(piece_id)]),
        "errors": int(dataset._pieces[int(piece_id)].is_error.sum()),
    }


def _search_prepared_piece(
    prepared: dict,
    config: dict,
    beam_width: int,
) -> dict:
    from run_incremental_global_context_d import incremental_beam_search

    scorer = prepared["scorer"]
    hits_before = scorer.window_cache.hits
    misses_before = scorer.window_cache.misses
    start = time.perf_counter()
    state = incremental_beam_search(
        candidate_positions=prepared["candidate_positions"],
        proposals=prepared["proposals"],
        score_all=scorer.score_all,
        score_floor=float(config["score_floor"]),
        beam_width=beam_width,
        max_edits=max(
            1,
            int(math.ceil(prepared["note_count"] * config["edit_rate"])),
        ),
    )
    elapsed = time.perf_counter() - start
    selected_positions = {edit.position for edit in state.edits}
    selected = np.asarray(
        [
            int(position) in selected_positions
            for position in prepared["candidate_positions"]
        ],
        dtype=bool,
    )
    labels = prepared["labels"]
    return {
        "piece_id": prepared["piece_id"],
        "tp": int((selected & labels).sum()),
        "fp": int((selected & ~labels).sum()),
        "errors": prepared["errors"],
        "seconds": elapsed,
        "window_hits": scorer.window_cache.hits - hits_before,
        "window_misses": scorer.window_cache.misses - misses_before,
        "edits": [
            {
                "position": edit.position,
                "proposed_pitch": edit.proposed_pitch,
            }
            for edit in state.edits
        ],
    }


def _run_piece_configs(
    *,
    piece_ids: list[int],
    configs: list[dict],
    arrays: dict[str, np.ndarray],
    dataset,
    models: tuple,
    verifier_model,
    beam_width: int,
    progress_dir: Path,
) -> None:
    for piece_id in sorted(int(value) for value in piece_ids):
        pending = [
            config
            for config in configs
            if not _piece_result_path(
                progress_dir, "calibration", config, piece_id
            ).exists()
        ]
        if not pending:
            continue
        prepared = _prepare_piece_search(
            piece_id=piece_id,
            arrays=arrays,
            dataset=dataset,
            models=models,
            verifier_model=verifier_model,
        )

        def run_search(scorer, config):
            if scorer is not prepared["scorer"]:
                raise RuntimeError("piece scorer identity changed")
            row = _search_prepared_piece(prepared, config, beam_width)
            atomic_write_json(
                _piece_result_path(
                    progress_dir, "calibration", config, piece_id
                ),
                row,
            )
            print(
                json.dumps(
                    {
                        "split": "calibration",
                        "config": _config_name(config),
                        "piece": piece_id,
                        "shared_score_states": len(scorer.score_cache),
                    }
                ),
                flush=True,
            )
            return row

        run_piece_configurations(
            configs=pending,
            make_scorer=lambda: prepared["scorer"],
            run_search=run_search,
        )


def _run_config(
    *,
    split: str,
    arrays: dict[str, np.ndarray],
    dataset,
    models: tuple,
    verifier_model,
    config: dict,
    beam_width: int,
    progress_dir: Path,
    piece_ids: list[int] | None = None,
) -> dict:
    import torch

    from incremental_piece_scorer import IncrementalPieceScorer
    from run_counterfactual_global_search import canonical_candidate_indices
    from run_incremental_global_context_d import incremental_beam_search

    totals = {
        "tp": 0,
        "fp": 0,
        "errors": 0,
        "seconds": 0.0,
        "window_hits": 0,
        "window_misses": 0,
        "pieces": 0,
    }
    selected_piece_ids = (
        sorted(np.unique(arrays["file_ids"]).tolist())
        if piece_ids is None
        else sorted(int(piece_id) for piece_id in piece_ids)
    )
    for piece_id in selected_piece_ids:
        result_path = _piece_result_path(
            progress_dir, split, config, int(piece_id)
        )
        if result_path.exists():
            row = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            full_rows = np.flatnonzero(arrays["file_ids"] == piece_id)
            piece_arrays = {
                name: values[full_rows]
                for name, values in arrays.items()
            }
            piece_arrays["window_starts"] = (
                piece_arrays["positions"]
                - piece_arrays["local_positions"]
            ).astype(np.int64)
            canonical_rows = canonical_candidate_indices(piece_arrays, 256)
            canonical = {
                name: values[canonical_rows]
                for name, values in piece_arrays.items()
            }
            scorer = IncrementalPieceScorer(
                piece_id=int(piece_id),
                piece_features=dataset._pieces[int(piece_id)].features,
                candidate_arrays=piece_arrays,
                output_rows=canonical_rows,
                models=models,
                verifier_model=verifier_model,
                device=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
            )
            candidate_positions = canonical["positions"].astype(np.int64)
            proposals = {
                index: tuple(
                    int(value)
                    for value in canonical["c_proposals"][index, :2]
                )
                for index in range(len(candidate_positions))
            }
            note_count = int(dataset._note_counts[int(piece_id)])
            start = time.perf_counter()
            state = incremental_beam_search(
                candidate_positions=candidate_positions,
                proposals=proposals,
                score_all=scorer.score_all,
                score_floor=float(config["score_floor"]),
                beam_width=beam_width,
                max_edits=max(
                    1,
                    int(math.ceil(note_count * config["edit_rate"])),
                ),
            )
            elapsed = time.perf_counter() - start
            selected_positions = {edit.position for edit in state.edits}
            selected = np.asarray(
                [
                    int(position) in selected_positions
                    for position in candidate_positions
                ],
                dtype=bool,
            )
            labels = canonical["labels"].astype(bool)
            row = {
                "piece_id": int(piece_id),
                "tp": int((selected & labels).sum()),
                "fp": int((selected & ~labels).sum()),
                "errors": int(dataset._pieces[int(piece_id)].is_error.sum()),
                "seconds": elapsed,
                "window_hits": scorer.window_cache.hits,
                "window_misses": scorer.window_cache.misses,
                "edits": [
                    {
                        "position": edit.position,
                        "proposed_pitch": edit.proposed_pitch,
                    }
                    for edit in state.edits
                ],
            }
            atomic_write_json(result_path, row)
        totals["tp"] += row["tp"]
        totals["fp"] += row["fp"]
        totals["errors"] += row["errors"]
        totals["seconds"] += row["seconds"]
        totals["window_hits"] += row["window_hits"]
        totals["window_misses"] += row["window_misses"]
        totals["pieces"] += 1
        print(
            json.dumps(
                {
                    "split": split,
                    "config": _config_name(config),
                    "piece": int(piece_id),
                    "completed": totals["pieces"],
                }
            ),
            flush=True,
        )
    metrics = metric_row(totals["tp"], totals["fp"], totals["errors"])
    metrics.update(
        {
            "config": config,
            "seconds": totals["seconds"],
            "pieces": totals["pieces"],
            "window_hits": totals["window_hits"],
            "window_misses": totals["window_misses"],
        }
    )
    return metrics


def _calibration_configs(baseline_floor: float) -> list[dict]:
    floors = [
        max(0.0, baseline_floor + offset)
        for offset in (-0.06, -0.03, 0.0, 0.03)
    ]
    return [
        {"score_floor": floor, "edit_rate": rate}
        for floor in floors
        for rate in (0.0100, 0.0125, 0.0150)
    ]


def _load_models(args: argparse.Namespace, device) -> tuple:
    from run_frozen_union_candidate_context_verifier import load_any_model

    return (
        *load_any_model(args.threeclass_checkpoint, device),
        *load_any_model(args.binary_checkpoint, device),
        *load_any_model(args.forward_checkpoint, device),
        *load_any_model(args.backward_checkpoint, device),
    )


def _load_dataset(args: argparse.Namespace, split: str):
    from voice_aware_dataset import PieceConsistentVoiceDataset

    return PieceConsistentVoiceDataset(
        root=args.data_root,
        split="validation" if split == "calibration" else "test",
        voice_method="onset_matching",
        window_size=256,
        stride=128,
        error_rate=0.01,
        seed=args.seed,
        verbose=True,
    )


def _split_total_errors(
    arrays: dict[str, np.ndarray],
    dataset,
) -> int:
    return int(
        sum(
            dataset._pieces[int(piece_id)].is_error.sum()
            for piece_id in np.unique(arrays["file_ids"])
        )
    )


def main() -> None:
    import torch
    from joblib import load

    args = parse_args()
    cache_dir = Path(args.cache_dir)
    progress_dir = Path(args.progress_dir)
    verifier_model = load(args.verifier_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.phase == "test":
        selected = load_selected_calibration(progress_dir)
        test_arrays = _load_raw_split(cache_dir, "test")
        test_piece_ids = sorted(
            int(piece_id)
            for piece_id in np.unique(test_arrays["file_ids"]).tolist()
        )
        assigned = partition_piece_ids(
            test_piece_ids, args.worker_index, args.worker_count
        )
        _run_config(
            split="test",
            arrays=test_arrays,
            dataset=_load_dataset(args, "test"),
            models=_load_models(args, device),
            verifier_model=verifier_model,
            config=selected["config"],
            beam_width=args.beam_width,
            progress_dir=progress_dir,
            piece_ids=assigned,
        )
        return

    calibration_arrays = _load_raw_split(cache_dir, "calibration")
    calibration_dataset = _load_dataset(args, "calibration")
    calibration_piece_ids = sorted(
        int(piece_id)
        for piece_id in np.unique(calibration_arrays["file_ids"]).tolist()
    )
    calibration_total = _split_total_errors(
        calibration_arrays, calibration_dataset
    )
    baseline_calibration, baseline_floor = _baseline(
        calibration_arrays,
        verifier_model,
        calibration_total,
        args.calibration_precision,
    )
    configs = _calibration_configs(baseline_floor)

    if args.phase == "calibration":
        assigned = partition_piece_ids(
            calibration_piece_ids, args.worker_index, args.worker_count
        )
        models = _load_models(args, device)
        _run_piece_configs(
            piece_ids=assigned,
            configs=configs,
            arrays=calibration_arrays,
            dataset=calibration_dataset,
            models=models,
            verifier_model=verifier_model,
            beam_width=args.beam_width,
            progress_dir=progress_dir,
        )
        return

    calibration_rows, selected = finalize_calibration_grid(
        progress_dir=progress_dir,
        configs=configs,
        expected_piece_ids=calibration_piece_ids,
        required_precision=args.calibration_precision,
    )
    test_arrays = _load_raw_split(cache_dir, "test")
    test_piece_ids = sorted(
        int(piece_id)
        for piece_id in np.unique(test_arrays["file_ids"]).tolist()
    )
    try:
        test_row = aggregate_config_checkpoints(
            progress_dir=progress_dir,
            split="test",
            config=selected["config"],
            expected_piece_ids=test_piece_ids,
        )
    except RuntimeError as error:
        print(
            json.dumps(
                {
                    "status": "calibration_selected",
                    "selected_calibration": selected,
                    "test": str(error),
                },
                indent=2,
            ),
            flush=True,
        )
        return

    test_dataset = _load_dataset(args, "test")
    test_total = _split_total_errors(test_arrays, test_dataset)
    baseline_test = _baseline_at_threshold(
        test_arrays,
        verifier_model,
        test_total,
        baseline_floor,
    )
    result = {
        "protocol": "genuine_incremental_global_context",
        "beam_width": args.beam_width,
        "calibration_precision": args.calibration_precision,
        "baseline_calibration": baseline_calibration,
        "baseline_test": baseline_test,
        "calibration_grid": calibration_rows,
        "selected_calibration": selected,
        "selected_test": test_row,
    }
    output = Path(args.output_json)
    atomic_write_json(output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
