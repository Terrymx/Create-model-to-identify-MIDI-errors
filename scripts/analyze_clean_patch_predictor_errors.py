from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from joblib import load

from build_counterfactual_candidate_cache import load_candidate_cache
from clean_patch_predictor import CleanPatchPredictor
from motif_repetition_features import MOTIF_FEATURE_NAMES
from run_clean_patch_predictor_verifier import (
    make_piece_dataset,
    score_candidate_patch_features,
)
from run_motif_repetition_verifier import (
    append_motif_features,
    build_c2_motif_features,
)


PATCH_FEATURE_NAMES = [
    "observed_energy",
    "best_edited_energy",
    "mean_edited_energy",
    "best_gain",
    "mean_gain",
    "proposal_margin",
    "valid_proposal_count",
    "any_improved",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--max-validation-files", type=int, default=None)
    parser.add_argument("--max-test-files", type=int, default=None)
    parser.add_argument("--motif-radius", type=int, default=4)
    parser.add_argument("--motif-min-similarity", type=float, default=0.84)
    parser.add_argument("--motif-exclude-radius", type=int, default=16)
    parser.add_argument("--patch-radius", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else 0.0


def _fraction(mask: np.ndarray) -> float:
    return float(mask.mean()) if len(mask) else 0.0


def _describe_group(
    name: str,
    rows: np.ndarray,
    arrays: dict[str, np.ndarray],
    patch_x: np.ndarray,
    baseline_scores: np.ndarray,
    patch_scores: np.ndarray,
) -> dict:
    features = arrays["piece_features_at_candidate"][rows]
    motif = arrays["motif_features"][rows]
    labels = arrays["labels"][rows].astype(bool)
    target = arrays["target_pitch"][rows]
    observed = arrays["observed_pitch"][rows]
    replace = arrays["error_kind"][rows] == 1
    in_scale = (features[:, 8] >= 0.5) | (features[:, 9] >= 0.5)
    chord_tone = features[:, 20] >= 0.5
    short_local = features[:, 33] <= 0.35
    duration = features[:, 2]
    motif_matches = motif[:, MOTIF_FEATURE_NAMES.index("motif_match_count")]
    motif_gain = motif[:, MOTIF_FEATURE_NAMES.index("proposal_consensus_gain")]
    patch_gain = patch_x[rows, PATCH_FEATURE_NAMES.index("best_gain")]
    patch_any = patch_x[rows, PATCH_FEATURE_NAMES.index("any_improved")] > 0.5
    pitch_distance = np.abs(target.astype(np.float32) - observed.astype(np.float32))
    return {
        "name": name,
        "count": int(len(rows)),
        "positive_count": int(labels.sum()),
        "negative_count": int((~labels).sum()),
        "replace_positive_count": int((labels & replace).sum()),
        "mean_baseline_score": _mean(baseline_scores[rows]),
        "mean_patch_hgb_score": _mean(patch_scores[rows]),
        "mean_patch_best_gain": _mean(patch_gain),
        "positive_patch_gain_fraction": _fraction(patch_gain > 0.0),
        "patch_any_improved_fraction": _fraction(patch_any),
        "mean_motif_gain": _mean(motif_gain),
        "motif_matched_fraction": _fraction(motif_matches > 0),
        "short_local_duration_fraction": _fraction(short_local),
        "in_scale_fraction": _fraction(in_scale),
        "chord_tone_fraction": _fraction(chord_tone),
        "mean_duration_feature": _mean(duration),
        "mean_pitch_distance": _mean(pitch_distance[labels]) if bool(labels.any()) else 0.0,
    }


def _examples(
    rows: np.ndarray,
    arrays: dict[str, np.ndarray],
    patch_x: np.ndarray,
    baseline_scores: np.ndarray,
    patch_scores: np.ndarray,
    limit: int = 12,
) -> list[dict]:
    if len(rows) == 0:
        return []
    patch_gain_index = PATCH_FEATURE_NAMES.index("best_gain")
    order = np.argsort(patch_x[rows, patch_gain_index], kind="mergesort")
    selected = rows[order[:limit]]
    examples = []
    for row in selected.tolist():
        examples.append(
            {
                "row": int(row),
                "file_id": int(arrays["file_ids"][row]),
                "position": int(arrays["positions"][row]),
                "label": int(arrays["labels"][row]),
                "error_kind": int(arrays["error_kind"][row]),
                "observed_pitch": int(arrays["observed_pitch"][row]),
                "target_pitch": int(arrays["target_pitch"][row]),
                "proposals": [int(value) for value in arrays["proposals"][row].tolist()],
                "baseline_score": float(baseline_scores[row]),
                "patch_hgb_score": float(patch_scores[row]),
                "patch_best_gain": float(patch_x[row, patch_gain_index]),
            }
        )
    return examples


def _write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Clean Patch Predictor Error Analysis",
        "",
        "## Selection Delta",
        "",
        "| Group | Count | Positives | Negatives | Mean patch gain | Short frac | In-scale frac | Motif matched |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in result["groups"]:
        lines.append(
            f"| {group['name']} | {group['count']} | {group['positive_count']} | "
            f"{group['negative_count']} | {group['mean_patch_best_gain']:.4f} | "
            f"{group['short_local_duration_fraction']:.4f} | "
            f"{group['in_scale_fraction']:.4f} | {group['motif_matched_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Net Effect",
            "",
            f"- lost true positives: `{result['transition_counts']['lost_tp']}`",
            f"- gained true positives: `{result['transition_counts']['gained_tp']}`",
            f"- removed false positives: `{result['transition_counts']['removed_fp']}`",
            f"- added false positives: `{result['transition_counts']['added_fp']}`",
            f"- net true-positive change: `{result['transition_counts']['net_tp_change']}`",
            f"- net false-positive change: `{result['transition_counts']['net_fp_change']}`",
            "",
            "## Lowest Patch-Gain Lost TP Examples",
            "",
            "| Row | File | Pos | Obs | Target | Base score | Patch score | Patch gain | Proposals |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for example in result["examples"]["lost_tp_lowest_patch_gain"]:
        lines.append(
            f"| {example['row']} | {example['file_id']} | {example['position']} | "
            f"{example['observed_pitch']} | {example['target_pitch']} | "
            f"{example['baseline_score']:.4f} | {example['patch_hgb_score']:.4f} | "
            f"{example['patch_best_gain']:.4f} | {example['proposals']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    result = json.loads(Path(args.result_json).read_text(encoding="utf-8"))
    test, test_meta = load_candidate_cache(Path(args.cache_dir) / "test.npz")
    dataset = make_piece_dataset(args, "test", error_rate=args.error_rate)
    motif_kwargs = {
        "radius": args.motif_radius,
        "min_similarity": args.motif_min_similarity,
        "exclude_radius": args.motif_exclude_radius,
    }
    test = append_motif_features(test, dataset, **motif_kwargs)
    piece_features = np.zeros((len(test["labels"]), dataset._pieces[0].features.shape[1]), dtype=np.float32)
    for file_id in np.unique(test["file_ids"].astype(np.int64)):
        row_mask = test["file_ids"].astype(np.int64) == file_id
        piece_features[row_mask] = dataset._pieces[int(file_id)].features[
            test["positions"][row_mask].astype(np.int64)
        ]
    test["piece_features_at_candidate"] = piece_features
    test_x = build_c2_motif_features(test)

    checkpoint = torch.load(Path(args.checkpoint_dir) / "clean_patch_predictor.pt", map_location="cpu")
    model_args = checkpoint["args"]
    patch_model = CleanPatchPredictor(
        patch_feature_dim=int(checkpoint["patch_feature_dim"]),
        hidden_dim=int(model_args["hidden_dim"]),
    )
    patch_model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_model.to(device)
    patch_x, _proposal_scores = score_candidate_patch_features(
        test,
        dataset,
        patch_model,
        radius=args.patch_radius,
        batch_size=args.batch_size,
        device=device,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    baseline_model = load(checkpoint_dir / "c2_motif_hgb.joblib")
    patch_hgb = load(checkpoint_dir / "c2_motif_clean_patch_hgb.joblib")
    baseline_scores = baseline_model.predict_proba(test_x)[:, 1]
    patch_scores = patch_hgb.predict_proba(np.concatenate([test_x, patch_x], axis=1))[:, 1]
    baseline_threshold = result["systems"]["C2_motif_hgb"]["best_feasible_test"]["selected_test"]["threshold"]
    patch_threshold = result["systems"]["C2_motif_clean_patch_hgb"]["best_feasible_test"]["selected_test"]["threshold"]
    baseline_detected = baseline_scores >= baseline_threshold
    patch_detected = patch_scores >= patch_threshold
    labels = test["labels"].astype(bool)

    common = np.flatnonzero(baseline_detected & patch_detected)
    baseline_only = np.flatnonzero(baseline_detected & ~patch_detected)
    patch_only = np.flatnonzero(~baseline_detected & patch_detected)
    neither = np.flatnonzero(~baseline_detected & ~patch_detected)
    lost_tp = baseline_only[labels[baseline_only]]
    removed_fp = baseline_only[~labels[baseline_only]]
    gained_tp = patch_only[labels[patch_only]]
    added_fp = patch_only[~labels[patch_only]]
    result_analysis = {
        "source_result": str(args.result_json),
        "test_stats": test_meta["stats"],
        "thresholds": {
            "baseline": float(baseline_threshold),
            "patch_hgb": float(patch_threshold),
        },
        "transition_counts": {
            "common_selected": int(len(common)),
            "baseline_only": int(len(baseline_only)),
            "patch_only": int(len(patch_only)),
            "neither": int(len(neither)),
            "lost_tp": int(len(lost_tp)),
            "removed_fp": int(len(removed_fp)),
            "gained_tp": int(len(gained_tp)),
            "added_fp": int(len(added_fp)),
            "net_tp_change": int(len(gained_tp) - len(lost_tp)),
            "net_fp_change": int(len(added_fp) - len(removed_fp)),
        },
        "groups": [
            _describe_group("common_selected", common, test, patch_x, baseline_scores, patch_scores),
            _describe_group("baseline_only", baseline_only, test, patch_x, baseline_scores, patch_scores),
            _describe_group("patch_only", patch_only, test, patch_x, baseline_scores, patch_scores),
            _describe_group("lost_tp", lost_tp, test, patch_x, baseline_scores, patch_scores),
            _describe_group("removed_fp", removed_fp, test, patch_x, baseline_scores, patch_scores),
            _describe_group("gained_tp", gained_tp, test, patch_x, baseline_scores, patch_scores),
            _describe_group("added_fp", added_fp, test, patch_x, baseline_scores, patch_scores),
        ],
        "examples": {
            "lost_tp_lowest_patch_gain": _examples(lost_tp, test, patch_x, baseline_scores, patch_scores),
            "removed_fp_lowest_patch_gain": _examples(removed_fp, test, patch_x, baseline_scores, patch_scores),
        },
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result_analysis, indent=2), encoding="utf-8")
    _write_markdown(Path(args.output_md), result_analysis)
    print(json.dumps(result_analysis["transition_counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
