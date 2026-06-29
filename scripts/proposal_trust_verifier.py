from __future__ import annotations

import numpy as np


def aligned_c_scores(arrays: dict[str, np.ndarray]) -> np.ndarray:
    proposals = arrays["proposals"]
    if "c_ranking" not in arrays:
        return np.zeros(proposals.shape, dtype=np.float32)
    if "c_proposal_slots" not in arrays:
        scores = arrays["c_ranking"].astype(np.float32)
        if scores.shape == proposals.shape:
            return scores
        aligned = np.zeros(proposals.shape, dtype=np.float32)
        shared = min(scores.shape[1], proposals.shape[1])
        aligned[:, :shared] = scores[:, :shared]
        return aligned
    out = np.zeros(proposals.shape, dtype=np.float32)
    slots = arrays["c_proposal_slots"].astype(np.int64)
    scores = arrays["c_ranking"].astype(np.float32)
    for row in range(len(out)):
        for score_index, slot in enumerate(slots[row].tolist()):
            if 0 <= slot < out.shape[1]:
                out[row, slot] = scores[row, score_index]
    return out


def _best_and_second(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores, axis=1, kind="mergesort")
    best_index = order[:, 0]
    best = scores[np.arange(len(scores)), best_index]
    second = scores[np.arange(len(scores)), order[:, 1]] if scores.shape[1] > 1 else best
    return best_index.astype(np.int64), best.astype(np.float32), second.astype(np.float32)


def _softmax_stats(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True).clip(min=1e-9)
    entropy = -(probs * np.log(probs.clip(min=1e-9))).sum(axis=1)
    best_prob = probs.max(axis=1)
    return entropy.astype(np.float32), best_prob.astype(np.float32)


def build_proposal_trust_features(
    arrays: dict[str, np.ndarray],
    c2_scores: np.ndarray,
    e1_proposal_scores: np.ndarray,
) -> np.ndarray:
    proposals = arrays["proposals"].astype(np.float32)
    observed = arrays["observed_pitch"].astype(np.float32)
    b_scores = arrays["b_ranking"].astype(np.float32)
    c_scores = aligned_c_scores(arrays)
    valid = np.isfinite(proposals)
    masked_e1 = np.where(valid, e1_proposal_scores.astype(np.float32), -1e9)
    masked_b = np.where(valid, b_scores, -1e9)
    masked_c = np.where(valid, c_scores, -1e9)

    e1_best_index, e1_best, e1_second = _best_and_second(masked_e1)
    b_best_index, b_best, _b_second = _best_and_second(masked_b)
    c_best_index, c_best, _c_second = _best_and_second(masked_c)
    row_index = np.arange(len(proposals))
    best_proposal = proposals[row_index, e1_best_index]
    delta = best_proposal - observed
    entropy, best_prob = _softmax_stats(masked_e1)

    return np.column_stack(
        [
            c2_scores.astype(np.float32),
            e1_best,
            e1_best - e1_second,
            b_scores[row_index, e1_best_index],
            c_scores[row_index, e1_best_index],
            (e1_best_index == b_best_index).astype(np.float32),
            (e1_best_index == c_best_index).astype(np.float32),
            (b_best_index == c_best_index).astype(np.float32),
            delta / 24.0,
            np.abs(delta) / 24.0,
            best_proposal / 127.0,
            observed / 127.0,
            valid.sum(axis=1).astype(np.float32),
            entropy,
            best_prob,
        ]
    ).astype(np.float32)


def build_high_confidence_slice_mask(
    arrays: dict[str, np.ndarray],
    c2_scores: np.ndarray,
    e1_proposal_scores: np.ndarray,
    *,
    c2_max: float,
    e1_margin_min: float,
    e1_best_prob_min: float,
    require_b_agreement: bool,
    require_c_agreement: bool,
    motif_match_index: int,
    motif_gain_index: int,
    motif_match_max: float,
    motif_gain_max: float,
) -> np.ndarray:
    proposals = arrays["proposals"].astype(np.float32)
    b_scores = arrays["b_ranking"].astype(np.float32)
    c_scores = aligned_c_scores(arrays)
    valid = np.isfinite(proposals)
    masked_e1 = np.where(valid, e1_proposal_scores.astype(np.float32), -1e9)
    masked_b = np.where(valid, b_scores, -1e9)
    masked_c = np.where(valid, c_scores, -1e9)
    e1_best_index, e1_best, e1_second = _best_and_second(masked_e1)
    b_best_index, _b_best, _b_second = _best_and_second(masked_b)
    c_best_index, _c_best, _c_second = _best_and_second(masked_c)
    _entropy, best_prob = _softmax_stats(masked_e1)
    motif = arrays.get("motif_features")
    if motif is None:
        motif_match = np.zeros(len(c2_scores), dtype=np.float32)
        motif_gain = np.zeros(len(c2_scores), dtype=np.float32)
    else:
        motif_match = motif[:, motif_match_index].astype(np.float32)
        motif_gain = motif[:, motif_gain_index].astype(np.float32)
    mask = (
        (c2_scores.astype(np.float32) <= float(c2_max))
        & ((e1_best - e1_second) >= float(e1_margin_min))
        & (best_prob >= float(e1_best_prob_min))
        & (motif_match <= float(motif_match_max))
        & (motif_gain <= float(motif_gain_max))
    )
    if require_b_agreement:
        mask &= e1_best_index == b_best_index
    if require_c_agreement:
        mask &= e1_best_index == c_best_index
    return mask.astype(bool)


def mask_rescue_scores(
    rescue_scores: np.ndarray,
    eligible_mask: np.ndarray,
) -> np.ndarray:
    return np.where(eligible_mask.astype(bool), rescue_scores, -np.inf).astype(np.float32)


def build_trust_rescue_scores(
    baseline_scores: np.ndarray,
    rescue_scores: np.ndarray,
    *,
    baseline_threshold: float,
    rescue_threshold: float,
) -> np.ndarray:
    baseline_selected = baseline_scores >= baseline_threshold
    rescue_selected = (~baseline_selected) & (rescue_scores >= rescue_threshold)
    return (baseline_selected | rescue_selected).astype(np.float32)


def _row_from_selected(
    selected: np.ndarray,
    labels: np.ndarray,
    total_errors: int,
    *,
    threshold: float,
    baseline_threshold: float,
) -> dict:
    target = labels.astype(bool)
    tp = int(np.logical_and(selected, target).sum())
    fp = int(np.logical_and(selected, ~target).sum())
    fn = int(total_errors - tp)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_errors, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "baseline_threshold": float(baseline_threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _threshold_grid(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return np.asarray([np.inf], dtype=np.float32)
    quantiles = np.quantile(scores, np.linspace(0.01, 0.995, 160))
    fixed = np.linspace(float(scores.min()), float(scores.max()), 80)
    return np.unique(np.concatenate([quantiles, fixed, [np.inf]])).astype(np.float32)


def _select_score_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    total_errors: int,
    target_precision: float,
) -> dict:
    rows = []
    for threshold in _threshold_grid(scores):
        selected = scores >= threshold
        rows.append(
            _row_from_selected(
                selected,
                labels,
                total_errors,
                threshold=float(threshold),
                baseline_threshold=float(threshold),
            )
        )
    feasible = [row for row in rows if row["precision"] >= target_precision]
    return max(
        feasible or rows,
        key=lambda row: (
            row["recall"] if row["precision"] >= target_precision else -1.0,
            row["precision"],
        ),
    )


def _correction_metrics(
    proposals: np.ndarray,
    proposal_scores: np.ndarray,
    detected: np.ndarray,
    labels: np.ndarray,
    error_kinds: np.ndarray,
    target_pitch: np.ndarray,
) -> dict:
    replace = detected & labels.astype(bool) & (error_kinds == 1)
    count = int(replace.sum())
    if count == 0:
        return {
            "detected_replace_errors": 0,
            "top1_correct": 0,
            "top3_correct": 0,
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
        }
    order = np.argsort(-proposal_scores[replace], axis=1, kind="mergesort")
    ranked = np.take_along_axis(proposals[replace], order, axis=1)
    targets = target_pitch[replace]
    top1 = int((ranked[:, 0] == targets).sum())
    top3 = int((ranked[:, :3] == targets[:, None]).any(axis=1).sum())
    return {
        "detected_replace_errors": count,
        "top1_correct": top1,
        "top3_correct": top3,
        "top1_accuracy": top1 / count,
        "top3_accuracy": top3 / count,
    }


def evaluate_rescue_union(
    *,
    calibration_baseline_scores: np.ndarray,
    test_baseline_scores: np.ndarray,
    calibration_rescue_scores: np.ndarray,
    test_rescue_scores: np.ndarray,
    calibration: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    calibration_total_errors: int,
    test_total_errors: int,
    target_precision: float,
    test_proposal_scores: np.ndarray | None = None,
) -> dict:
    rows = []
    proposal_scores = test_proposal_scores if test_proposal_scores is not None else test["b_ranking"]
    for margin in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]:
        baseline_row = _select_score_threshold(
            calibration_baseline_scores,
            calibration["labels"].astype(np.int64),
            calibration_total_errors,
            target_precision + margin,
        )
        baseline_threshold = float(baseline_row["threshold"])
        selected_calibration_base = calibration_baseline_scores >= baseline_threshold
        candidate_rows = ~selected_calibration_base
        candidate_scores = calibration_rescue_scores[candidate_rows]
        selected_calibration = None
        for threshold in _threshold_grid(candidate_scores):
            combined = build_trust_rescue_scores(
                calibration_baseline_scores,
                calibration_rescue_scores,
                baseline_threshold=baseline_threshold,
                rescue_threshold=float(threshold),
            ) >= 0.5
            row = _row_from_selected(
                combined,
                calibration["labels"].astype(np.int64),
                calibration_total_errors,
                threshold=float(threshold),
                baseline_threshold=baseline_threshold,
            )
            if row["precision"] >= target_precision + margin:
                if selected_calibration is None or (
                    row["recall"],
                    row["precision"],
                ) > (
                    selected_calibration["recall"],
                    selected_calibration["precision"],
                ):
                    selected_calibration = row
        if selected_calibration is None:
            selected_calibration = baseline_row
            selected_calibration["threshold"] = float(np.inf)
            selected_calibration["baseline_threshold"] = baseline_threshold
        test_selected = build_trust_rescue_scores(
            test_baseline_scores,
            test_rescue_scores,
            baseline_threshold=baseline_threshold,
            rescue_threshold=float(selected_calibration["threshold"]),
        ) >= 0.5
        test_row = _row_from_selected(
            test_selected,
            test["labels"].astype(np.int64),
            test_total_errors,
            threshold=float(selected_calibration["threshold"]),
            baseline_threshold=baseline_threshold,
        )
        test_row["rescued_rows"] = int(
            ((test_baseline_scores < baseline_threshold)
            & (test_rescue_scores >= float(selected_calibration["threshold"]))).sum()
        )
        test_row["correction"] = _correction_metrics(
            test["proposals"],
            proposal_scores,
            test_selected,
            test["labels"],
            test["error_kind"],
            test["target_pitch"],
        )
        rows.append(
            {
                "requested_precision": target_precision + margin,
                "selected_calibration": selected_calibration,
                "selected_test": test_row,
            }
        )
    feasible = [
        row for row in rows if row["selected_test"]["precision"] >= target_precision
    ]
    return {
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
