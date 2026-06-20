from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from counterfactual_beam_search import EditCandidate, beam_search_edits


def canonical_candidate_indices(
    arrays: dict[str, np.ndarray],
    window_size: int,
) -> np.ndarray:
    best: dict[tuple[int, int], tuple[int, int, int]] = {}
    for row, (file_id, position, local_position, dataset_index) in enumerate(
        zip(
            arrays["file_ids"],
            arrays["positions"],
            arrays["local_positions"],
            arrays["dataset_indices"],
        )
    ):
        edge_margin = min(int(local_position), window_size - 1 - int(local_position))
        key = (int(file_id), int(position))
        rank = (edge_margin, -int(dataset_index), -row)
        if key not in best or rank > best[key][0:3]:
            best[key] = (*rank, row)
    return np.asarray(
        sorted((value[3] for value in best.values())),
        dtype=np.int64,
    )


def detection_metrics(
    selected: np.ndarray,
    labels: np.ndarray,
    total_errors: int,
) -> dict[str, float | int]:
    chosen = selected.astype(bool)
    positive = labels.astype(bool)
    tp = int((chosen & positive).sum())
    fp = int((chosen & ~positive).sum())
    fn = int(total_errors - tp)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_errors, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "selected": int(chosen.sum()),
    }


def select_global_predictions(
    arrays: dict[str, np.ndarray],
    scores: np.ndarray,
    *,
    score_floor: float,
    beam_width: int,
    max_edit_rate: float,
    conflict_distance: int,
    conflict_penalty: float,
) -> np.ndarray:
    if len(scores) != len(arrays["file_ids"]):
        raise ValueError("Scores must align with candidate rows.")
    selected = np.zeros(len(scores), dtype=bool)
    for file_id in np.unique(arrays["file_ids"]):
        rows = np.flatnonzero(arrays["file_ids"] == file_id)
        note_count = int(arrays["file_note_counts"][rows[0]])
        max_edits = max(1, int(math.ceil(note_count * max_edit_rate)))
        candidates = []
        for row in rows:
            proposal_values = (
                arrays["c_proposals"][row]
                if "c_proposals" in arrays
                else arrays["proposals"][row]
            )
            if "c_ranking" in arrays:
                best_slot = int(np.argmax(arrays["c_ranking"][row]))
            elif "b_ranking" in arrays:
                best_slot = int(np.argmax(arrays["b_ranking"][row]))
            else:
                best_slot = 0
            candidates.append(
                EditCandidate(
                    candidate_index=int(row),
                    position=int(arrays["positions"][row]),
                    proposed_pitch=int(proposal_values[best_slot]),
                    utility=float(scores[row] - score_floor),
                )
            )
        state = beam_search_edits(
            candidates,
            beam_width=beam_width,
            max_edits=max_edits,
            proposal_floor=0.0,
            conflict_distance=conflict_distance,
            conflict_penalty=conflict_penalty,
        )
        for edit in state.edits:
            selected[edit.candidate_index] = True
    return selected


def _canonicalize(
    arrays: dict[str, np.ndarray],
    window_size: int,
) -> dict[str, np.ndarray]:
    rows = canonical_candidate_indices(arrays, window_size)
    return {
        name: values[rows]
        for name, values in arrays.items()
    }


def _score_floor_grid(scores: np.ndarray) -> list[float]:
    quantiles = np.quantile(
        scores,
        [0.50, 0.60, 0.70, 0.78, 0.84, 0.89, 0.93, 0.96, 0.98],
    )
    return sorted({float(value) for value in quantiles})


def _evaluate_setting(
    arrays: dict[str, np.ndarray],
    scores: np.ndarray,
    total_errors: int,
    setting: dict,
) -> dict:
    selected = select_global_predictions(arrays, scores, **setting)
    return {
        **detection_metrics(selected, arrays["labels"], total_errors),
        "setting": setting,
    }


def _select_calibrated_setting(
    calibration: dict[str, np.ndarray],
    scores: np.ndarray,
    total_errors: int,
    target_precision: float,
    variant: str,
) -> dict:
    if variant == "D1":
        structures = [(4, rate, 0, 0.0) for rate in (0.010, 0.0125, 0.015)]
    elif variant == "D2":
        structures = [
            (8, rate, 2, penalty)
            for rate in (0.010, 0.0125, 0.015, 0.020)
            for penalty in (0.05, 0.15)
        ]
    elif variant == "D3":
        structures = [
            (16, rate, distance, penalty)
            for rate in (0.0125, 0.015, 0.020)
            for distance in (2, 4)
            for penalty in (0.10, 0.25)
        ]
    else:
        raise ValueError(f"Unknown global variant: {variant}")
    rows = []
    for floor in _score_floor_grid(scores):
        for beam_width, rate, distance, penalty in structures:
            setting = {
                "score_floor": floor,
                "beam_width": beam_width,
                "max_edit_rate": rate,
                "conflict_distance": distance,
                "conflict_penalty": penalty,
            }
            rows.append(
                _evaluate_setting(
                    calibration,
                    scores,
                    total_errors,
                    setting,
                )
            )
    feasible = [row for row in rows if row["precision"] >= target_precision]
    return max(
        feasible or rows,
        key=lambda row: (
            row["recall"] if row["precision"] >= target_precision else -1.0,
            row["precision"],
            -row["selected"],
        ),
    )


def _select_independent_threshold(
    arrays: dict[str, np.ndarray],
    scores: np.ndarray,
    total_errors: int,
    target_precision: float,
) -> dict:
    rows = []
    for floor in sorted(np.unique(scores), reverse=True):
        selected = scores >= floor
        rows.append(
            {
                **detection_metrics(selected, arrays["labels"], total_errors),
                "score_floor": float(floor),
            }
        )
    feasible = [row for row in rows if row["precision"] >= target_precision]
    return max(
        feasible or rows,
        key=lambda row: (
            row["recall"] if row["precision"] >= target_precision else -1.0,
            row["precision"],
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-precision", type=float, default=0.80)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def _load_split(cache_dir: Path, name: str, window_size: int):
    with np.load(cache_dir / f"{name}.npz", allow_pickle=False) as loaded:
        metadata = json.loads(str(loaded["__metadata__"]))
        arrays = {
            key: loaded[key].copy()
            for key in loaded.files
            if key != "__metadata__"
        }
    return _canonicalize(arrays, window_size), metadata


def _total_errors(metadata: dict) -> int:
    stats = metadata["stats"]
    return int(stats.get("unique_error_notes", stats["error_notes"]))


def _write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Whole-piece counterfactual search",
        "",
        "| System | Precision | Recall | F1 | Selected |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    baseline = result["matched_c1"]
    lines.append(
        f"| matched C1 | {baseline['test']['precision']:.4f} | "
        f"{baseline['test']['recall']:.4f} | {baseline['test']['f1']:.4f} | "
        f"{baseline['test']['selected']} |"
    )
    for name, block in result["systems"].items():
        row = block["test"]
        lines.append(
            f"| {name} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['selected']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    from joblib import dump

    from run_counterfactual_edit_verifier import (
        build_c_variant_features,
        make_small_leaf,
    )

    args = parse_args()
    cache_dir = Path(args.cache_dir)
    train, train_meta = _load_split(cache_dir, "train", args.window_size)
    calibration, calibration_meta = _load_split(
        cache_dir, "calibration", args.window_size
    )
    test, test_meta = _load_split(cache_dir, "test", args.window_size)

    train_x = build_c_variant_features(
        train["base_features"],
        train["b_features"],
        train["b_ranking"],
        train["c_features"],
        train["c_ranking"],
        "C1",
    )
    calibration_x = build_c_variant_features(
        calibration["base_features"],
        calibration["b_features"],
        calibration["b_ranking"],
        calibration["c_features"],
        calibration["c_ranking"],
        "C1",
    )
    test_x = build_c_variant_features(
        test["base_features"],
        test["b_features"],
        test["b_ranking"],
        test["c_features"],
        test["c_ranking"],
        "C1",
    )
    model = make_small_leaf(args.seed)
    model.fit(train_x, train["labels"].astype(np.int64))
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    dump(model, checkpoint)
    calibration_scores = model.predict_proba(calibration_x)[:, 1]
    test_scores = model.predict_proba(test_x)[:, 1]
    calibration_total = _total_errors(calibration_meta)
    test_total = _total_errors(test_meta)

    baseline_calibration = _select_independent_threshold(
        calibration,
        calibration_scores,
        calibration_total,
        args.target_precision + 0.01,
    )
    baseline_setting = {
        "score_floor": baseline_calibration["score_floor"],
        "beam_width": 1,
        "max_edit_rate": 1.0,
        "conflict_distance": 0,
        "conflict_penalty": 0.0,
    }
    matched_c1 = {
        "calibration": baseline_calibration,
        "test": _evaluate_setting(
            test,
            test_scores,
            test_total,
            baseline_setting,
        ),
    }

    systems = {}
    for variant in ("D1", "D2", "D3"):
        calibrated = _select_calibrated_setting(
            calibration,
            calibration_scores,
            calibration_total,
            args.target_precision + 0.01,
            variant,
        )
        systems[variant] = {
            "calibration": calibrated,
            "test": _evaluate_setting(
                test,
                test_scores,
                test_total,
                calibrated["setting"],
            ),
        }
    result = {
        "protocol": "piece_consistent_post_corruption_deduplicated_notes",
        "target_precision": args.target_precision,
        "train_candidates": len(train["labels"]),
        "calibration_candidates": len(calibration["labels"]),
        "test_candidates": len(test["labels"]),
        "matched_c1": matched_c1,
        "systems": systems,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_markdown(Path(args.output_md), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
