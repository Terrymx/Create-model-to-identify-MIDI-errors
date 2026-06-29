from __future__ import annotations

import numpy as np


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else 0.0


def target_rank_summary(
    proposals: np.ndarray,
    target_pitch: np.ndarray,
    proposal_scores: np.ndarray,
) -> dict:
    proposals = np.asarray(proposals)
    target_pitch = np.asarray(target_pitch)
    proposal_scores = np.asarray(proposal_scores)
    if len(proposals) == 0:
        return {
            "rows": 0,
            "target_present": 0,
            "top1": 0.0,
            "top2": 0.0,
            "top3": 0.0,
            "mean_rank_when_present": 0.0,
        }
    order = np.argsort(-proposal_scores, axis=1, kind="mergesort")
    ranked = np.take_along_axis(proposals, order, axis=1)
    ranks = np.full(len(proposals), -1, dtype=np.int64)
    for row in range(len(proposals)):
        matches = np.flatnonzero(ranked[row] == target_pitch[row])
        if len(matches):
            ranks[row] = int(matches[0] + 1)
    present = ranks > 0
    return {
        "rows": int(len(proposals)),
        "target_present": int(present.sum()),
        "top1": _mean((ranks[present] == 1).astype(np.float32)),
        "top2": _mean((ranks[present] <= 2).astype(np.float32)),
        "top3": _mean((ranks[present] <= 3).astype(np.float32)),
        "mean_rank_when_present": _mean(ranks[present].astype(np.float32)),
    }


def group_summary(
    name: str,
    mask: np.ndarray,
    arrays: dict[str, np.ndarray],
    baseline_scores: np.ndarray,
    rescue_scores: np.ndarray,
    proposal_scores: np.ndarray | None = None,
) -> dict:
    mask = np.asarray(mask).astype(bool)
    labels = arrays["labels"].astype(bool)
    proposals = arrays["proposals"]
    target = arrays["target_pitch"]
    target_present = (proposals == target[:, None]).any(axis=1)
    rows = np.flatnonzero(mask)
    summary = {
        "name": name,
        "count": int(len(rows)),
        "positives": int(labels[rows].sum()),
        "negatives": int((~labels[rows]).sum()),
        "positive_rate": _mean(labels[rows].astype(np.float32)),
        "target_present_fraction": _mean(target_present[rows].astype(np.float32)),
        "mean_baseline_score": _mean(baseline_scores[rows].astype(np.float32)),
        "mean_rescue_score": _mean(rescue_scores[rows].astype(np.float32)),
    }
    if proposal_scores is not None:
        summary["target_rank"] = target_rank_summary(
            proposals[rows],
            target[rows],
            proposal_scores[rows],
        )
    return summary


def topk_group_summaries(
    *,
    name: str,
    candidate_mask: np.ndarray,
    arrays: dict[str, np.ndarray],
    baseline_scores: np.ndarray,
    rescue_scores: np.ndarray,
    proposal_scores: np.ndarray,
    ks: tuple[int, ...] = (25, 50, 100, 200, 500),
) -> list[dict]:
    rows = np.flatnonzero(candidate_mask.astype(bool))
    if len(rows) == 0:
        return []
    ranked_rows = rows[np.argsort(-rescue_scores[rows], kind="mergesort")]
    out = []
    for k in ks:
        selected = np.zeros(len(candidate_mask), dtype=bool)
        selected[ranked_rows[: min(k, len(ranked_rows))]] = True
        summary = group_summary(
            f"{name}_top{k}",
            selected,
            arrays,
            baseline_scores,
            rescue_scores,
            proposal_scores,
        )
        summary["k"] = int(k)
        out.append(summary)
    return out


def feature_delta_table(
    feature_names: list[str],
    features: np.ndarray,
    positive_mask: np.ndarray,
    negative_mask: np.ndarray,
) -> list[dict]:
    positive_rows = np.flatnonzero(positive_mask.astype(bool))
    negative_rows = np.flatnonzero(negative_mask.astype(bool))
    rows = []
    for index, name in enumerate(feature_names):
        pos_mean = _mean(features[positive_rows, index].astype(np.float32))
        neg_mean = _mean(features[negative_rows, index].astype(np.float32))
        rows.append(
            {
                "feature": name,
                "positive_mean": pos_mean,
                "negative_mean": neg_mean,
                "delta": pos_mean - neg_mean,
            }
        )
    return sorted(rows, key=lambda row: abs(row["delta"]), reverse=True)
