from __future__ import annotations

import numpy as np


def fit_score_probability_bins(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    bin_count: int = 30,
    smoothing: float = 1.0,
) -> dict[str, np.ndarray]:
    values = np.asarray(scores, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.float32)
    if values.ndim != 1 or targets.ndim != 1 or len(values) != len(targets):
        raise ValueError("scores and labels must be aligned one-dimensional arrays.")
    if len(values) == 0:
        raise ValueError("Cannot fit probability bins on empty scores.")
    count = max(1, min(int(bin_count), len(values)))
    order = np.argsort(values, kind="mergesort")
    sorted_scores = values[order]
    sorted_labels = targets[order]
    chunks = np.array_split(np.arange(len(values)), count)
    edges = []
    probabilities = []
    prior = float((targets.sum() + smoothing) / (len(targets) + 2.0 * smoothing))
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        local_labels = sorted_labels[chunk]
        probability = float(
            (local_labels.sum() + smoothing * prior)
            / (len(local_labels) + smoothing)
        )
        edges.append(float(sorted_scores[chunk[-1]]))
        probabilities.append(probability)
    probabilities_array = np.maximum.accumulate(
        np.asarray(probabilities, dtype=np.float32)
    )
    return {
        "edges": np.asarray(edges, dtype=np.float32),
        "probabilities": probabilities_array,
    }


def apply_score_probability_bins(
    calibrator: dict[str, np.ndarray],
    scores: np.ndarray,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    edges = np.asarray(calibrator["edges"], dtype=np.float32)
    probabilities = np.asarray(calibrator["probabilities"], dtype=np.float32)
    if len(edges) != len(probabilities) or len(edges) == 0:
        raise ValueError("Invalid score probability calibrator.")
    indices = np.searchsorted(edges, values, side="left")
    indices = np.clip(indices, 0, len(probabilities) - 1)
    return probabilities[indices].astype(np.float32)


def select_piecewise_fdr(
    error_probabilities: np.ndarray,
    file_ids: np.ndarray,
    *,
    max_fdr: float,
    min_probability: float = 0.0,
) -> np.ndarray:
    probabilities = np.asarray(error_probabilities, dtype=np.float32)
    files = np.asarray(file_ids, dtype=np.int64)
    if probabilities.ndim != 1 or files.ndim != 1 or len(probabilities) != len(files):
        raise ValueError("error_probabilities and file_ids must align.")
    selected = np.zeros(len(probabilities), dtype=bool)
    for file_id in np.unique(files):
        local_indices = np.flatnonzero(files == file_id)
        if len(local_indices) == 0:
            continue
        local_probabilities = probabilities[local_indices]
        order = np.argsort(-local_probabilities, kind="mergesort")
        sorted_probabilities = local_probabilities[order]
        false_discoveries = np.cumsum(1.0 - sorted_probabilities)
        counts = np.arange(1, len(sorted_probabilities) + 1, dtype=np.float32)
        estimated_fdr = false_discoveries / counts
        feasible = np.flatnonzero(
            (estimated_fdr <= max_fdr)
            & (sorted_probabilities >= min_probability)
        )
        if len(feasible) == 0:
            continue
        keep_count = int(feasible[-1] + 1)
        selected[local_indices[order[:keep_count]]] = True
    return selected


def row_from_prediction(
    prediction: np.ndarray,
    labels: np.ndarray,
    total_errors: int,
) -> dict:
    selected = np.asarray(prediction, dtype=bool)
    target = np.asarray(labels, dtype=bool)
    if len(selected) != len(target):
        raise ValueError("prediction and labels must align.")
    tp = int(np.logical_and(selected, target).sum())
    fp = int(np.logical_and(selected, ~target).sum())
    fn = int(total_errors - tp)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(int(total_errors), 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "selected": int(selected.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _select_fdr_from_calibration(
    probabilities: np.ndarray,
    labels: np.ndarray,
    file_ids: np.ndarray,
    total_errors: int,
    target_precision: float,
    q_grid: np.ndarray,
) -> dict:
    rows = []
    for q in q_grid:
        prediction = select_piecewise_fdr(
            probabilities,
            file_ids,
            max_fdr=float(q),
        )
        row = row_from_prediction(prediction, labels, total_errors)
        row["selected_fdr"] = float(q)
        rows.append(row)
    feasible = [row for row in rows if row["precision"] >= target_precision]
    if feasible:
        return max(feasible, key=lambda row: (row["recall"], row["precision"]))
    return max(rows, key=lambda row: (row["precision"], row["recall"]))


def evaluate_fdr_score_rows(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    calibration_labels: np.ndarray,
    test_labels: np.ndarray,
    calibration_file_ids: np.ndarray,
    test_file_ids: np.ndarray,
    calibration_total_errors: int,
    test_total_errors: int,
    target_precision: float,
    *,
    q_grid: np.ndarray | None = None,
    bin_count: int = 30,
) -> dict:
    if q_grid is None:
        q_grid = np.asarray(
            [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.30],
            dtype=np.float32,
        )
    calibrator = fit_score_probability_bins(
        calibration_scores,
        calibration_labels,
        bin_count=bin_count,
    )
    calibration_probabilities = apply_score_probability_bins(
        calibrator,
        calibration_scores,
    )
    test_probabilities = apply_score_probability_bins(calibrator, test_scores)
    rows = []
    for margin in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]:
        selected = _select_fdr_from_calibration(
            calibration_probabilities,
            calibration_labels,
            calibration_file_ids,
            calibration_total_errors,
            target_precision + margin,
            q_grid,
        )
        test_prediction = select_piecewise_fdr(
            test_probabilities,
            test_file_ids,
            max_fdr=selected["selected_fdr"],
        )
        test_row = row_from_prediction(test_prediction, test_labels, test_total_errors)
        test_row["selected_fdr"] = selected["selected_fdr"]
        rows.append(
            {
                "requested_precision": target_precision + margin,
                "selected_fdr": selected["selected_fdr"],
                "selected_calibration": selected,
                "selected_test": test_row,
            }
        )
    feasible = [
        row for row in rows if row["selected_test"]["precision"] >= target_precision
    ]
    return {
        "calibrator": {
            "edges": calibrator["edges"].tolist(),
            "probabilities": calibrator["probabilities"].tolist(),
        },
        "margins": rows,
        "best_feasible_test": max(
            feasible or rows,
            key=lambda row: (
                row["selected_test"]["recall"]
                if row["selected_test"]["precision"] >= target_precision
                else -1.0,
                row["selected_test"]["precision"],
            ),
        ),
    }
