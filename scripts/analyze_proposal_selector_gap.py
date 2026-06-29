from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from joblib import load

from build_counterfactual_candidate_cache import load_candidate_cache
from e1_edit_energy_verifier import (
    E1EditEnergyNet,
    E1Normalizer,
    build_e1_feature_tensors,
    predict_e1,
)
from motif_repetition_features import MOTIF_FEATURE_NAMES
from run_motif_repetition_verifier import append_motif_features, build_c2_motif_features
from voice_aware_dataset import PieceConsistentVoiceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--c2-checkpoint-dir", required=True)
    parser.add_argument("--e1-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--max-test-files", type=int, default=None)
    parser.add_argument("--motif-radius", type=int, default=4)
    parser.add_argument("--motif-min-similarity", type=float, default=0.84)
    parser.add_argument("--motif-exclude-radius", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def _fraction(mask: np.ndarray) -> float:
    return float(mask.mean()) if len(mask) else 0.0


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else 0.0


def _rank_of_target(scores: np.ndarray, proposals: np.ndarray, target_pitch: np.ndarray) -> np.ndarray:
    ranks = np.full(len(scores), -1, dtype=np.int64)
    order = np.argsort(-scores, axis=1, kind="mergesort")
    ranked_proposals = np.take_along_axis(proposals, order, axis=1)
    for row in range(len(scores)):
        matches = np.flatnonzero(ranked_proposals[row] == target_pitch[row])
        if len(matches):
            ranks[row] = int(matches[0] + 1)
    return ranks


def _selector_summary(
    name: str,
    scores: np.ndarray,
    proposals: np.ndarray,
    target_pitch: np.ndarray,
    rows: np.ndarray,
) -> dict:
    if len(rows) == 0:
        return {
            "name": name,
            "rows": 0,
            "target_present": 0,
            "top1": 0.0,
            "top2": 0.0,
            "top3": 0.0,
            "mean_rank_when_present": 0.0,
        }
    ranks = _rank_of_target(scores[rows], proposals[rows], target_pitch[rows])
    present = ranks > 0
    return {
        "name": name,
        "rows": int(len(rows)),
        "target_present": int(present.sum()),
        "top1": _fraction(ranks == 1),
        "top2": _fraction((ranks > 0) & (ranks <= 2)),
        "top3": _fraction((ranks > 0) & (ranks <= 3)),
        "mean_rank_when_present": _mean(ranks[present].astype(np.float32)),
    }


def _aligned_c_scores(arrays: dict[str, np.ndarray]) -> np.ndarray:
    proposals = arrays["proposals"]
    out = np.full(proposals.shape, -1e9, dtype=np.float32)
    slots = arrays["c_proposal_slots"].astype(np.int64)
    c_scores = arrays["c_ranking"].astype(np.float32)
    for row in range(len(out)):
        for c_index, slot in enumerate(slots[row].tolist()):
            if 0 <= slot < out.shape[1]:
                out[row, slot] = c_scores[row, c_index]
    return out


def _load_e1_scores(
    checkpoint_path: Path,
    arrays: dict[str, np.ndarray],
    candidate_x: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    raw = build_e1_feature_tensors(arrays, candidate_x)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    normalizer_payload = checkpoint["normalizer"]
    normalizer = E1Normalizer(
        candidate_mean=np.asarray(normalizer_payload["candidate_mean"], dtype=np.float32),
        candidate_std=np.asarray(normalizer_payload["candidate_std"], dtype=np.float32),
        proposal_mean=np.asarray(normalizer_payload["proposal_mean"], dtype=np.float32),
        proposal_std=np.asarray(normalizer_payload["proposal_std"], dtype=np.float32),
    )
    tensors = normalizer.transform(raw)
    model = E1EditEnergyNet(
        candidate_dim=int(checkpoint["candidate_dim"]),
        proposal_dim=int(checkpoint["proposal_dim"]),
        hidden_dim=int(checkpoint["args"]["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    _scores, _logits, proposal_scores = predict_e1(
        model,
        tensors,
        batch_size=batch_size,
        device=device,
    )
    return proposal_scores


def _group_summary(
    name: str,
    rows: np.ndarray,
    arrays: dict[str, np.ndarray],
    scores: np.ndarray,
    threshold: float,
) -> dict:
    motif = arrays["motif_features"][rows]
    proposals = arrays["proposals"][rows]
    target = arrays["target_pitch"][rows]
    target_present = (proposals == target[:, None]).any(axis=1)
    motif_match = motif[:, MOTIF_FEATURE_NAMES.index("motif_match_count")] > 0
    motif_gain = motif[:, MOTIF_FEATURE_NAMES.index("proposal_consensus_gain")]
    return {
        "name": name,
        "count": int(len(rows)),
        "target_present_fraction": _fraction(target_present),
        "motif_matched_fraction": _fraction(motif_match),
        "positive_motif_gain_fraction": _fraction(motif_gain > 0),
        "mean_score": _mean(scores[rows]),
        "near_threshold_fraction": _fraction(np.abs(scores[rows] - threshold) <= 0.05),
        "far_below_threshold_fraction": _fraction((scores[rows] - threshold) < -0.20),
    }


def _write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Proposal Selector Gap Analysis",
        "",
        "## Oracle Detection Gap",
        "",
        f"- total errors: `{result['oracle']['total_errors']}`",
        f"- current true positives: `{result['oracle']['current_tp']}`",
        f"- candidate FN rows: `{result['oracle']['candidate_fn_rows']}`",
        f"- candidate FN rows with target in proposals: `{result['oracle']['candidate_fn_target_present']}`",
        f"- ideal recall if all target-present candidate FN were rescued: `{result['oracle']['ideal_recall_rescue_target_present']:.4f}`",
        f"- ideal recall if all candidate FN rows were rescued: `{result['oracle']['ideal_recall_rescue_all_candidate_fn']:.4f}`",
        "",
        "## Candidate FN Groups",
        "",
        "| Group | Count | Target present | Motif matched | Positive motif gain | Near threshold | Far below | Mean score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in result["groups"]:
        lines.append(
            f"| {group['name']} | {group['count']} | {group['target_present_fraction']:.4f} | "
            f"{group['motif_matched_fraction']:.4f} | {group['positive_motif_gain_fraction']:.4f} | "
            f"{group['near_threshold_fraction']:.4f} | {group['far_below_threshold_fraction']:.4f} | "
            f"{group['mean_score']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Selector Target-Rank Accuracy",
            "",
            "| Rows | Selector | Target present | Top-1 | Top-2 | Top-3 | Mean rank |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row_group, summaries in result["selector_summaries"].items():
        for summary in summaries:
            lines.append(
                f"| {row_group} | {summary['name']} | {summary['target_present']} | "
                f"{summary['top1']:.4f} | {summary['top2']:.4f} | {summary['top3']:.4f} | "
                f"{summary['mean_rank_when_present']:.4f} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    result = json.loads(Path(args.result_json).read_text(encoding="utf-8"))
    test, test_meta = load_candidate_cache(Path(args.cache_dir) / "test.npz")
    dataset = PieceConsistentVoiceDataset(
        root=args.data_root,
        split="test",
        voice_method="onset_matching",
        window_size=args.window_size,
        stride=args.stride,
        error_rate=args.error_rate,
        seed=args.seed,
        max_files=args.max_test_files,
        verbose=True,
    )
    test = append_motif_features(
        test,
        dataset,
        radius=args.motif_radius,
        min_similarity=args.motif_min_similarity,
        exclude_radius=args.motif_exclude_radius,
    )
    test_x = build_c2_motif_features(test)
    model = load(Path(args.c2_checkpoint_dir) / "c2_motif_hgb.joblib")
    detection_scores = model.predict_proba(test_x)[:, 1]
    threshold = float(result["systems"]["C2_motif_hgb"]["best_feasible_test"]["selected_test"]["threshold"])
    labels = test["labels"].astype(bool)
    selected = detection_scores >= threshold
    positive = np.flatnonzero(labels)
    true_positive = np.flatnonzero(selected & labels)
    candidate_fn = np.flatnonzero((~selected) & labels)
    replace_fn = candidate_fn[test["error_kind"][candidate_fn] == 1]
    target_present_mask = (test["proposals"] == test["target_pitch"][:, None]).any(axis=1)
    target_present_fn = candidate_fn[target_present_mask[candidate_fn]]
    replace_target_present_fn = replace_fn[target_present_mask[replace_fn]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e1_scores = _load_e1_scores(
        Path(args.e1_checkpoint),
        test,
        test_x,
        batch_size=args.batch_size,
        device=device,
    )
    selectors = {
        "B_ranking": test["b_ranking"].astype(np.float32),
        "C_aligned_ranking": _aligned_c_scores(test),
        "E1_proposal_scores": e1_scores.astype(np.float32),
    }
    row_groups = {
        "all_candidate_positives": positive,
        "true_positives": true_positive,
        "candidate_fn": candidate_fn,
        "candidate_fn_target_present": target_present_fn,
        "replace_fn_target_present": replace_target_present_fn,
    }
    selector_summaries = {
        group_name: [
            _selector_summary(
                selector_name,
                selector_scores,
                test["proposals"],
                test["target_pitch"],
                rows,
            )
            for selector_name, selector_scores in selectors.items()
        ]
        for group_name, rows in row_groups.items()
    }
    total_errors = int(test_meta["stats"]["error_notes"])
    current_tp = int(len(true_positive))
    analysis = {
        "threshold": threshold,
        "oracle": {
            "total_errors": total_errors,
            "current_tp": current_tp,
            "current_recall": current_tp / total_errors,
            "candidate_fn_rows": int(len(candidate_fn)),
            "candidate_fn_target_present": int(len(target_present_fn)),
            "replace_fn_target_present": int(len(replace_target_present_fn)),
            "ideal_recall_rescue_target_present": (current_tp + len(target_present_fn)) / total_errors,
            "ideal_recall_rescue_replace_target_present": (current_tp + len(replace_target_present_fn)) / total_errors,
            "ideal_recall_rescue_all_candidate_fn": (current_tp + len(candidate_fn)) / total_errors,
        },
        "groups": [
            _group_summary("candidate_fn", candidate_fn, test, detection_scores, threshold),
            _group_summary("target_present_fn", target_present_fn, test, detection_scores, threshold),
            _group_summary("replace_target_present_fn", replace_target_present_fn, test, detection_scores, threshold),
        ],
        "selector_summaries": selector_summaries,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    _write_markdown(Path(args.output_md), analysis)
    print(json.dumps(analysis["oracle"], indent=2), flush=True)


if __name__ == "__main__":
    main()
