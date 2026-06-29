from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import load

from build_counterfactual_candidate_cache import load_candidate_cache
from motif_repetition_features import MOTIF_FEATURE_NAMES
from run_motif_repetition_verifier import append_motif_features, build_c2_motif_features
from voice_aware_dataset import PieceConsistentVoiceDataset


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
    parser.add_argument("--max-test-files", type=int, default=None)
    parser.add_argument("--motif-radius", type=int, default=4)
    parser.add_argument("--motif-min-similarity", type=float, default=0.84)
    parser.add_argument("--motif-exclude-radius", type=int, default=16)
    return parser.parse_args()


def _fraction(mask: np.ndarray) -> float:
    return float(mask.mean()) if len(mask) else 0.0


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else 0.0


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    qs = np.quantile(values.astype(np.float32), [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "p10": float(qs[0]),
        "p25": float(qs[1]),
        "p50": float(qs[2]),
        "p75": float(qs[3]),
        "p90": float(qs[4]),
    }


def _piece_features_at_candidates(
    arrays: dict[str, np.ndarray],
    dataset: PieceConsistentVoiceDataset,
) -> np.ndarray:
    file_ids = arrays["file_ids"].astype(np.int64)
    positions = arrays["positions"].astype(np.int64)
    feature_dim = int(dataset._pieces[0].features.shape[1])
    out = np.zeros((len(file_ids), feature_dim), dtype=np.float32)
    for file_id in np.unique(file_ids):
        row_mask = file_ids == file_id
        out[row_mask] = dataset._pieces[int(file_id)].features[positions[row_mask]]
    return out


def _candidate_error_mask(arrays: dict[str, np.ndarray]) -> set[tuple[int, int]]:
    labels = arrays["labels"].astype(bool)
    file_ids = arrays["file_ids"].astype(np.int64)
    positions = arrays["positions"].astype(np.int64)
    return {
        (int(file_id), int(position))
        for file_id, position in zip(file_ids[labels], positions[labels])
    }


def _all_error_mask(dataset: PieceConsistentVoiceDataset) -> set[tuple[int, int]]:
    errors: set[tuple[int, int]] = set()
    for file_id, piece in enumerate(dataset._pieces):
        for position in np.flatnonzero(piece.is_error >= 0.5):
            errors.add((int(file_id), int(position)))
    return errors


def _summarize_rows(
    name: str,
    rows: np.ndarray,
    arrays: dict[str, np.ndarray],
    candidate_features: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    features = candidate_features[rows]
    motif = arrays["motif_features"][rows]
    proposals = arrays["proposals"][rows]
    target = arrays["target_pitch"][rows]
    observed = arrays["observed_pitch"][rows]
    labels = arrays["labels"][rows].astype(bool)
    replace = arrays["error_kind"][rows] == 1
    delete = arrays["error_kind"][rows] == 2
    in_scale = (features[:, 8] >= 0.5) | (features[:, 9] >= 0.5)
    chord_tone = features[:, 20] >= 0.5
    step_in = features[:, 26] >= 0.5
    step_out = features[:, 27] >= 0.5
    passing = features[:, 29] >= 0.5
    neighbor = features[:, 30] >= 0.5
    non_chord_resolution = features[:, 32] >= 0.5
    short_local = features[:, 33] <= 0.35
    motif_matches = motif[:, MOTIF_FEATURE_NAMES.index("motif_match_count")]
    motif_gain = motif[:, MOTIF_FEATURE_NAMES.index("proposal_consensus_gain")]
    proposal_contains_target = (proposals == target[:, None]).any(axis=1)
    abs_pitch_error = np.abs(target.astype(np.float32) - observed.astype(np.float32))
    score_margin = scores[rows] - float(threshold)
    near_threshold = np.abs(score_margin) <= 0.05
    return {
        "name": name,
        "count": int(len(rows)),
        "replace_fraction": _fraction(replace),
        "delete_fraction": _fraction(delete),
        "short_local_fraction": _fraction(short_local),
        "in_scale_fraction": _fraction(in_scale),
        "chord_tone_fraction": _fraction(chord_tone),
        "step_in_out_fraction": _fraction(step_in & step_out),
        "passing_fraction": _fraction(passing),
        "neighbor_fraction": _fraction(neighbor),
        "non_chord_resolution_fraction": _fraction(non_chord_resolution),
        "motif_matched_fraction": _fraction(motif_matches > 0),
        "positive_motif_gain_fraction": _fraction(motif_gain > 0),
        "proposal_contains_target_fraction": _fraction(proposal_contains_target),
        "near_threshold_fraction": _fraction(near_threshold),
        "mean_score": _mean(scores[rows]),
        "mean_score_margin": _mean(score_margin),
        "score_quantiles": _quantiles(scores[rows]),
        "mean_abs_pitch_error": _mean(abs_pitch_error[labels]) if bool(labels.any()) else 0.0,
        "mean_motif_gain": _mean(motif_gain),
    }


def _bucket_counts(
    rows: np.ndarray,
    arrays: dict[str, np.ndarray],
    candidate_features: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> list[dict]:
    features = candidate_features[rows]
    motif = arrays["motif_features"][rows]
    proposals = arrays["proposals"][rows]
    target = arrays["target_pitch"][rows]
    score_margin = scores[rows] - float(threshold)
    buckets = {
        "short_in_scale": (features[:, 33] <= 0.35) & ((features[:, 8] >= 0.5) | (features[:, 9] >= 0.5)),
        "short_chord_tone": (features[:, 33] <= 0.35) & (features[:, 20] >= 0.5),
        "motif_unmatched": motif[:, MOTIF_FEATURE_NAMES.index("motif_match_count")] <= 0,
        "motif_negative_or_zero_gain": motif[:, MOTIF_FEATURE_NAMES.index("proposal_consensus_gain")] <= 0,
        "proposal_missing_target": ~(proposals == target[:, None]).any(axis=1),
        "near_threshold": np.abs(score_margin) <= 0.05,
        "far_below_threshold": score_margin < -0.20,
    }
    return [
        {
            "bucket": name,
            "count": int(mask.sum()),
            "fraction": _fraction(mask),
        }
        for name, mask in buckets.items()
    ]


def _examples(
    rows: np.ndarray,
    arrays: dict[str, np.ndarray],
    scores: np.ndarray,
    threshold: float,
    limit: int = 16,
) -> list[dict]:
    if len(rows) == 0:
        return []
    order = np.argsort(np.abs(scores[rows] - float(threshold)), kind="mergesort")
    selected = rows[order[:limit]]
    examples = []
    for row in selected.tolist():
        examples.append(
            {
                "row": int(row),
                "file_id": int(arrays["file_ids"][row]),
                "position": int(arrays["positions"][row]),
                "score": float(scores[row]),
                "threshold_margin": float(scores[row] - float(threshold)),
                "error_kind": int(arrays["error_kind"][row]),
                "observed_pitch": int(arrays["observed_pitch"][row]),
                "target_pitch": int(arrays["target_pitch"][row]),
                "proposals": [int(value) for value in arrays["proposals"][row].tolist()],
            }
        )
    return examples


def _write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# C2 Motif False Negative Analysis",
        "",
        "## Recall Accounting",
        "",
        f"- total test errors: `{result['recall_accounting']['total_errors']}`",
        f"- candidate-covered errors: `{result['recall_accounting']['candidate_covered_errors']}`",
        f"- missed outside candidate pool: `{result['recall_accounting']['outside_candidate_errors']}`",
        f"- selected true positives: `{result['recall_accounting']['selected_true_positives']}`",
        f"- candidate false negatives: `{result['recall_accounting']['candidate_false_negatives']}`",
        f"- final false negatives: `{result['recall_accounting']['final_false_negatives']}`",
        "",
        "## Candidate Row Groups",
        "",
        "| Group | Count | Replace | Short | In-scale | Chord | Motif matched | Proposal has target | Near threshold | Mean score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in result["groups"]:
        lines.append(
            f"| {group['name']} | {group['count']} | {group['replace_fraction']:.4f} | "
            f"{group['short_local_fraction']:.4f} | {group['in_scale_fraction']:.4f} | "
            f"{group['chord_tone_fraction']:.4f} | {group['motif_matched_fraction']:.4f} | "
            f"{group['proposal_contains_target_fraction']:.4f} | {group['near_threshold_fraction']:.4f} | "
            f"{group['mean_score']:.4f} |"
        )
    lines.extend(["", "## Candidate FN Buckets", ""])
    for bucket in result["candidate_fn_buckets"]:
        lines.append(f"- {bucket['bucket']}: `{bucket['count']}` ({bucket['fraction']:.4f})")
    lines.extend(
        [
            "",
            "## Near-Threshold Candidate FN Examples",
            "",
            "| Row | File | Pos | Kind | Obs | Target | Score | Margin | Proposals |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for example in result["near_threshold_fn_examples"]:
        lines.append(
            f"| {example['row']} | {example['file_id']} | {example['position']} | "
            f"{example['error_kind']} | {example['observed_pitch']} | {example['target_pitch']} | "
            f"{example['score']:.4f} | {example['threshold_margin']:.4f} | {example['proposals']} |"
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
    motif_kwargs = {
        "radius": args.motif_radius,
        "min_similarity": args.motif_min_similarity,
        "exclude_radius": args.motif_exclude_radius,
    }
    test = append_motif_features(test, dataset, **motif_kwargs)
    candidate_features = _piece_features_at_candidates(test, dataset)
    test_x = build_c2_motif_features(test)
    model = load(Path(args.checkpoint_dir) / "c2_motif_hgb.joblib")
    scores = model.predict_proba(test_x)[:, 1]
    threshold = result["systems"]["C2_motif_hgb"]["best_feasible_test"]["selected_test"]["threshold"]
    selected = scores >= float(threshold)
    labels = test["labels"].astype(bool)
    positive_rows = np.flatnonzero(labels)
    tp_rows = np.flatnonzero(selected & labels)
    candidate_fn_rows = np.flatnonzero((~selected) & labels)
    fp_rows = np.flatnonzero(selected & (~labels))
    tn_rows = np.flatnonzero((~selected) & (~labels))
    all_errors = _all_error_mask(dataset)
    candidate_errors = _candidate_error_mask(test)
    outside_candidate = all_errors - candidate_errors
    accounting = {
        "total_errors": int(test_meta["stats"]["error_notes"]),
        "candidate_covered_errors": int(len(candidate_errors)),
        "outside_candidate_errors": int(len(outside_candidate)),
        "selected_true_positives": int(len(tp_rows)),
        "candidate_false_negatives": int(len(candidate_fn_rows)),
        "final_false_negatives": int(test_meta["stats"]["error_notes"] - len(tp_rows)),
        "candidate_recall_ceiling": float(test_meta["stats"]["candidate_recall_ceiling"]),
        "selected_recall": float(len(tp_rows) / int(test_meta["stats"]["error_notes"])),
    }
    analysis = {
        "threshold": float(threshold),
        "recall_accounting": accounting,
        "groups": [
            _summarize_rows("true_positives", tp_rows, test, candidate_features, scores, threshold),
            _summarize_rows("candidate_false_negatives", candidate_fn_rows, test, candidate_features, scores, threshold),
            _summarize_rows("false_positives", fp_rows, test, candidate_features, scores, threshold),
            _summarize_rows("true_negatives", tn_rows, test, candidate_features, scores, threshold),
            _summarize_rows("all_candidate_positives", positive_rows, test, candidate_features, scores, threshold),
        ],
        "candidate_fn_buckets": _bucket_counts(candidate_fn_rows, test, candidate_features, scores, threshold),
        "near_threshold_fn_examples": _examples(candidate_fn_rows, test, scores, threshold),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    _write_markdown(Path(args.output_md), analysis)
    print(json.dumps(accounting, indent=2), flush=True)


if __name__ == "__main__":
    main()
