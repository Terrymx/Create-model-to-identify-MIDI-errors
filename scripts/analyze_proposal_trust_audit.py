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
from proposal_trust_audit import feature_delta_table, group_summary, topk_group_summaries
from proposal_trust_verifier import (
    build_high_confidence_slice_mask,
    build_proposal_trust_features,
    mask_rescue_scores,
)
from run_motif_repetition_verifier import append_motif_features, build_c2_motif_features
from voice_aware_dataset import PieceConsistentVoiceDataset


TRUST_FEATURE_NAMES = [
    "c2_score",
    "e1_best",
    "e1_margin",
    "b_at_e1_best",
    "c_at_e1_best",
    "e1_b_agree",
    "e1_c_agree",
    "b_c_agree",
    "best_delta",
    "best_abs_delta",
    "best_pitch",
    "observed_pitch",
    "valid_proposals",
    "e1_entropy",
    "e1_best_prob",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--c2-checkpoint-dir", required=True)
    parser.add_argument("--e1-checkpoint", required=True)
    parser.add_argument("--trust-checkpoint-dir", required=True)
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


def _load_e1_model(checkpoint_path: Path, device: torch.device) -> tuple[E1EditEnergyNet, E1Normalizer]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    normalizer_payload = checkpoint["normalizer"]
    normalizer = E1Normalizer(
        candidate_mean=np.asarray(normalizer_payload["candidate_mean"], dtype=np.float32),
        candidate_std=np.asarray(normalizer_payload["candidate_std"], dtype=np.float32),
        proposal_mean=np.asarray(normalizer_payload["proposal_mean"], dtype=np.float32),
        proposal_std=np.asarray(normalizer_payload["proposal_std"], dtype=np.float32),
    )
    model = E1EditEnergyNet(
        candidate_dim=int(checkpoint["candidate_dim"]),
        proposal_dim=int(checkpoint["proposal_dim"]),
        hidden_dim=int(checkpoint["args"]["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, normalizer


def _score_e1(
    model: E1EditEnergyNet,
    normalizer: E1Normalizer,
    arrays: dict[str, np.ndarray],
    candidate_x: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    raw = build_e1_feature_tensors(arrays, candidate_x)
    tensors = normalizer.transform(raw)
    _scores, _logits, proposal_scores = predict_e1(
        model,
        tensors,
        batch_size=batch_size,
        device=device,
    )
    return proposal_scores


def _system_selection(
    result: dict,
    system_name: str,
    c2_scores: np.ndarray,
    rescue_scores: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    row = result["systems"][system_name]["best_feasible_test"]["selected_test"]
    baseline_threshold = float(row.get("baseline_threshold", row["threshold"]))
    rescue_threshold = float(row.get("threshold", np.inf))
    selected = c2_scores >= baseline_threshold
    if rescue_scores is not None and np.isfinite(rescue_threshold):
        selected = selected | ((c2_scores < baseline_threshold) & (rescue_scores >= rescue_threshold))
    return selected, baseline_threshold, rescue_threshold


def _slice_mask_from_result(
    result: dict,
    system_name: str,
    arrays: dict[str, np.ndarray],
    c2_scores: np.ndarray,
    e1_proposal_scores: np.ndarray,
) -> np.ndarray:
    params = result["systems"][system_name].get("slice_params")
    if not params:
        return np.ones(len(c2_scores), dtype=bool)
    allowed = {
        key: params[key]
        for key in [
            "c2_max",
            "e1_margin_min",
            "e1_best_prob_min",
            "require_b_agreement",
            "require_c_agreement",
            "motif_match_max",
            "motif_gain_max",
        ]
    }
    return build_high_confidence_slice_mask(
        arrays,
        c2_scores,
        e1_proposal_scores,
        motif_match_index=MOTIF_FEATURE_NAMES.index("motif_match_count"),
        motif_gain_index=MOTIF_FEATURE_NAMES.index("proposal_consensus_gain"),
        **allowed,
    )


def _write_markdown(path: Path, analysis: dict) -> None:
    lines = [
        "# Proposal-Trust Rescue Audit",
        "",
        "## System Delta",
        "",
        "| System | Precision | Recall | TP delta | FP delta | Rescued |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in analysis["system_delta"]:
        lines.append(
            f"| {row['system']} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['tp_delta']} | {row['fp_delta']} | {row['rescued_rows']} |"
        )
    lines.extend(
        [
            "",
            "## Group Summary",
            "",
            "| Group | Count | Positive rate | Target present | Mean C2 | Mean rescue |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in analysis["groups"]:
        lines.append(
            f"| {row['name']} | {row['count']} | {row['positive_rate']:.4f} | "
            f"{row['target_present_fraction']:.4f} | {row['mean_baseline_score']:.4f} | "
            f"{row['mean_rescue_score']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Top-K Eligible by Rescue Score",
            "",
            "| Group | Count | Positive rate | Target present | E1 Top-1 | E1 Top-3 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in analysis["topk"]:
        rank = row.get("target_rank", {})
        lines.append(
            f"| {row['name']} | {row['count']} | {row['positive_rate']:.4f} | "
            f"{row['target_present_fraction']:.4f} | {rank.get('top1', 0.0):.4f} | "
            f"{rank.get('top3', 0.0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Largest Trust-Feature Deltas: Narrow Eligible TP vs FP",
            "",
            "| Feature | TP mean | FP mean | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in analysis["feature_deltas"][:12]:
        lines.append(
            f"| {row['feature']} | {row['positive_mean']:.4f} | "
            f"{row['negative_mean']:.4f} | {row['delta']:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    result = json.loads(Path(args.result_json).read_text(encoding="utf-8"))
    test, _test_meta = load_candidate_cache(Path(args.cache_dir) / "test.npz")
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
    c2_model = load(Path(args.c2_checkpoint_dir) / "c2_motif_hgb.joblib")
    c2_scores = c2_model.predict_proba(test_x)[:, 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e1_model, e1_normalizer = _load_e1_model(Path(args.e1_checkpoint), device)
    e1_proposal_scores = _score_e1(
        e1_model,
        e1_normalizer,
        test,
        test_x,
        batch_size=args.batch_size,
        device=device,
    )
    trust_features = build_proposal_trust_features(test, c2_scores, e1_proposal_scores)
    full_features = np.concatenate([test_x, trust_features], axis=1).astype(np.float32)
    trust_model = load(Path(args.trust_checkpoint_dir) / "proposal_trust_hgb.joblib")
    full_model = load(Path(args.trust_checkpoint_dir) / "c2_motif_proposal_trust_hgb.joblib")
    trust_scores = trust_model.predict_proba(trust_features)[:, 1]
    full_scores = full_model.predict_proba(full_features)[:, 1]
    labels = test["labels"].astype(bool)
    baseline_selected, baseline_threshold, _baseline_rescue_threshold = _system_selection(
        result,
        "C2_motif_hgb",
        c2_scores,
    )
    system_delta = []
    base_row = result["systems"]["C2_motif_hgb"]["best_feasible_test"]["selected_test"]
    for name in result["systems"]:
        row = result["systems"][name]["best_feasible_test"]["selected_test"]
        system_delta.append(
            {
                "system": name,
                "precision": row["precision"],
                "recall": row["recall"],
                "tp_delta": int(row["tp"] - base_row["tp"]),
                "fp_delta": int(row["fp"] - base_row["fp"]),
                "rescued_rows": int(row.get("rescued_rows", 0)),
            }
        )
    candidate_fn = labels & ~baseline_selected
    baseline_fp = (~labels) & baseline_selected
    narrow_trust_mask = _slice_mask_from_result(
        result,
        "C2_motif_narrow_trust_rescue",
        test,
        c2_scores,
        e1_proposal_scores,
    )
    narrow_full_mask = _slice_mask_from_result(
        result,
        "C2_motif_narrow_full_rescue",
        test,
        c2_scores,
        e1_proposal_scores,
    )
    narrow_trust_scores = mask_rescue_scores(trust_scores, narrow_trust_mask)
    narrow_full_scores = mask_rescue_scores(full_scores, narrow_full_mask)
    selected_narrow_trust, narrow_trust_base_threshold, narrow_trust_threshold = _system_selection(
        result,
        "C2_motif_narrow_trust_rescue",
        c2_scores,
        narrow_trust_scores,
    )
    selected_narrow_full, narrow_full_base_threshold, narrow_full_threshold = _system_selection(
        result,
        "C2_motif_narrow_full_rescue",
        c2_scores,
        narrow_full_scores,
    )
    rescued_trust = selected_narrow_trust & (c2_scores < narrow_trust_base_threshold)
    rescued_full = selected_narrow_full & (c2_scores < narrow_full_base_threshold)
    groups = [
        group_summary("baseline_candidate_fn", candidate_fn, test, c2_scores, trust_scores, e1_proposal_scores),
        group_summary("baseline_false_positive", baseline_fp, test, c2_scores, trust_scores, e1_proposal_scores),
        group_summary("narrow_trust_eligible", narrow_trust_mask, test, c2_scores, trust_scores, e1_proposal_scores),
        group_summary("narrow_trust_eligible_tp", narrow_trust_mask & labels, test, c2_scores, trust_scores, e1_proposal_scores),
        group_summary("narrow_trust_eligible_fp", narrow_trust_mask & ~labels, test, c2_scores, trust_scores, e1_proposal_scores),
        group_summary("narrow_trust_rescued", rescued_trust, test, c2_scores, trust_scores, e1_proposal_scores),
        group_summary("narrow_full_eligible", narrow_full_mask, test, c2_scores, full_scores, e1_proposal_scores),
        group_summary("narrow_full_rescued", rescued_full, test, c2_scores, full_scores, e1_proposal_scores),
    ]
    topk = []
    topk.extend(
        topk_group_summaries(
            name="narrow_trust_eligible_rejected",
            candidate_mask=narrow_trust_mask & (c2_scores < narrow_trust_base_threshold),
            arrays=test,
            baseline_scores=c2_scores,
            rescue_scores=trust_scores,
            proposal_scores=e1_proposal_scores,
        )
    )
    topk.extend(
        topk_group_summaries(
            name="narrow_full_eligible_rejected",
            candidate_mask=narrow_full_mask & (c2_scores < narrow_full_base_threshold),
            arrays=test,
            baseline_scores=c2_scores,
            rescue_scores=full_scores,
            proposal_scores=e1_proposal_scores,
        )
    )
    feature_deltas = feature_delta_table(
        TRUST_FEATURE_NAMES,
        trust_features,
        narrow_trust_mask & labels,
        narrow_trust_mask & ~labels,
    )
    analysis = {
        "thresholds": {
            "baseline": baseline_threshold,
            "narrow_trust_baseline": narrow_trust_base_threshold,
            "narrow_trust_rescue": narrow_trust_threshold,
            "narrow_full_baseline": narrow_full_base_threshold,
            "narrow_full_rescue": narrow_full_threshold,
        },
        "system_delta": system_delta,
        "groups": groups,
        "topk": topk,
        "feature_deltas": feature_deltas,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    _write_markdown(Path(args.output_md), analysis)
    print(json.dumps(analysis["thresholds"], indent=2), flush=True)


if __name__ == "__main__":
    main()
