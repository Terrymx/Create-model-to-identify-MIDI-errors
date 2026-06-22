from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


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
    for piece_id in sorted(np.unique(arrays["file_ids"]).tolist()):
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
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
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


def main() -> None:
    import torch
    from joblib import load

    from run_frozen_union_candidate_context_verifier import load_any_model
    from voice_aware_dataset import PieceConsistentVoiceDataset

    args = parse_args()
    cache_dir = Path(args.cache_dir)
    progress_dir = Path(args.progress_dir)
    verifier_model = load(args.verifier_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = (
        *load_any_model(args.threeclass_checkpoint, device),
        *load_any_model(args.binary_checkpoint, device),
        *load_any_model(args.forward_checkpoint, device),
        *load_any_model(args.backward_checkpoint, device),
    )
    calibration_arrays = _load_raw_split(cache_dir, "calibration")
    calibration_dataset = PieceConsistentVoiceDataset(
        root=args.data_root,
        split="validation",
        voice_method="onset_matching",
        window_size=256,
        stride=128,
        error_rate=0.01,
        seed=args.seed,
        verbose=True,
    )
    calibration_total = int(
        sum(
            calibration_dataset._pieces[int(piece_id)].is_error.sum()
            for piece_id in np.unique(calibration_arrays["file_ids"])
        )
    )
    baseline_calibration, baseline_floor = _baseline(
        calibration_arrays,
        verifier_model,
        calibration_total,
        args.calibration_precision,
    )
    floors = [
        max(0.0, baseline_floor + offset)
        for offset in (-0.06, -0.03, 0.0, 0.03)
    ]
    configs = [
        {"score_floor": floor, "edit_rate": rate}
        for floor in floors
        for rate in (0.0100, 0.0125, 0.0150)
    ]
    calibration_rows = [
        _run_config(
            split="calibration",
            arrays=calibration_arrays,
            dataset=calibration_dataset,
            models=models,
            verifier_model=verifier_model,
            config=config,
            beam_width=args.beam_width,
            progress_dir=progress_dir,
        )
        for config in configs
    ]
    selected = select_calibration_config(
        calibration_rows,
        args.calibration_precision,
    )
    test_arrays = _load_raw_split(cache_dir, "test")
    test_dataset = PieceConsistentVoiceDataset(
        root=args.data_root,
        split="test",
        voice_method="onset_matching",
        window_size=256,
        stride=128,
        error_rate=0.01,
        seed=args.seed,
        verbose=True,
    )
    test_total = int(
        sum(piece.is_error.sum() for piece in test_dataset._pieces)
    )
    baseline_test = _baseline_at_threshold(
        test_arrays,
        verifier_model,
        test_total,
        baseline_floor,
    )
    test_row = _run_config(
        split="test",
        arrays=test_arrays,
        dataset=test_dataset,
        models=models,
        verifier_model=verifier_model,
        config=selected["config"],
        beam_width=args.beam_width,
        progress_dir=progress_dir,
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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
