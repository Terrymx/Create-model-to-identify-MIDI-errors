"""Run wrong-note detection and pitch correction on one MIDI file."""

from __future__ import annotations

import argparse
import json
import sys

import torch

from .data import extract_note_events, note_features
from .harmony import harmony_scores_for_pitches
from .model import build_wrong_note_model
from .train import build_explicit_correction_evidence, build_explicit_surprise

ACTION_NAMES = ["keep", "replace", "delete"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--midi", required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Detection threshold. Defaults to checkpoint valid_metrics.best_det_f0_5_threshold "
            "when available, then best_det_threshold, otherwise training det_threshold/0.3."
        ),
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of replacement pitch candidates to return")
    parser.add_argument(
        "--action-filter",
        choices=["any", "non-keep", "replace", "delete"],
        default="non-keep",
        help="Filter by predicted action after the detection threshold. Default is non-keep for precision-first use.",
    )
    parser.add_argument(
        "--min-action-prob",
        type=float,
        default=0.0,
        help="Minimum probability for the predicted action. Useful for stricter precision-first inference.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Keep only the highest-confidence N candidates after filtering.",
    )
    parser.add_argument(
        "--min-separation",
        type=float,
        default=0.0,
        help="Greedily suppress lower-confidence candidates within this many seconds of a kept candidate.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["time", "confidence"],
        default="time",
        help="Sort final output chronologically or by descending confidence.",
    )
    parser.add_argument(
        "--include-summary",
        action="store_true",
        help="Return an object with summary metadata and candidates instead of a raw candidate list.",
    )
    parser.add_argument(
        "--harmony-gain-weight",
        type=float,
        default=0.0,
        help="Add this multiplier times harmony_gain to adjusted_confidence_score. Default keeps model score unchanged.",
    )
    parser.add_argument(
        "--chord-tone-penalty",
        type=float,
        default=0.0,
        help="Subtract this amount when the input pitch is already harmony-consistent and the replacement does not improve it.",
    )
    parser.add_argument(
        "--harmony-current-threshold",
        type=float,
        default=0.65,
        help="Current-pitch harmony score that triggers chord-tone style protection for the penalty.",
    )
    parser.add_argument(
        "--min-harmony-gain",
        type=float,
        default=None,
        help="Drop replace candidates whose top replacement pitch is less harmonically plausible by more than this gain.",
    )
    parser.add_argument(
        "--harmony-onset-tolerance",
        type=float,
        default=0.08,
        help="Seconds around a note onset used to estimate local harmony for post-processing.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Number of notes per inference window. Defaults to checkpoint args.window_size, then 256.",
    )
    parser.add_argument(
        "--window-overlap",
        type=int,
        default=32,
        help="Overlap notes between inference windows. Predictions for overlapping notes keep the higher detection score.",
    )
    return parser.parse_args()


def _candidate_confidence(error_probability: float, keep_probability: float, action_probability: float) -> float:
    return error_probability * max(1.0 - keep_probability, action_probability)


def _passes_action_filter(action_name: str, action_filter: str) -> bool:
    if action_filter == "any":
        return True
    if action_filter == "non-keep":
        return action_name != "keep"
    return action_name == action_filter


def _suppress_close_candidates(candidates: list[dict], min_separation: float) -> list[dict]:
    if min_separation <= 0.0:
        return candidates

    kept: list[dict] = []
    for candidate in sorted(candidates, key=lambda item: item.get("adjusted_confidence_score", item["confidence_score"]), reverse=True):
        if all(abs(float(candidate["start"]) - float(existing["start"])) > min_separation for existing in kept):
            kept.append(candidate)
    return kept


def _iter_windows(note_count: int, window_size: int, overlap: int) -> list[tuple[int, int]]:
    if note_count <= 0:
        return []
    window_size = max(1, window_size)
    overlap = max(0, min(overlap, window_size - 1))
    stride = max(1, window_size - overlap)
    windows: list[tuple[int, int]] = []
    start = 0
    while start < note_count:
        end = min(note_count, start + window_size)
        windows.append((start, end))
        if end == note_count:
            break
        start += stride
    return windows


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_args = checkpoint.get("args", {})
    valid_metrics = checkpoint.get("valid_metrics", {})
    threshold = args.threshold
    if threshold is None:
        threshold = float(
            valid_metrics.get(
                "best_det_f0_5_threshold",
                valid_metrics.get("best_det_threshold", train_args.get("det_threshold", 0.3)),
            )
        )
        print(f"using detection threshold={threshold} from checkpoint/default", file=sys.stderr)
    state_dict = checkpoint["model_state_dict"]
    if "input_size" in train_args:
        input_size = int(train_args["input_size"])
    elif "encoder.weight_ih_l0" in state_dict:
        input_size = int(state_dict["encoder.weight_ih_l0"].shape[1])
    else:
        input_size = int(state_dict["input_projection.weight"].shape[1])
    model = build_wrong_note_model(
        model_type=str(train_args.get("model", "bigru")),
        input_size=input_size,
        hidden_size=int(train_args.get("hidden_size", 256)),
        num_layers=int(train_args.get("num_layers", 2)),
        transformer_d_model=int(train_args.get("transformer_d_model", 192)),
        transformer_heads=int(train_args.get("transformer_heads", 4)),
        transformer_ffn_dim=int(train_args.get("transformer_ffn_dim", 512)),
        dropout=float(train_args.get("dropout", 0.2)),
        explicit_surprise=bool(train_args.get("explicit_surprise", False)),
        explicit_correction_evidence=bool(train_args.get("explicit_correction_evidence", False)),
        surprise_embedding_dim=int(train_args.get("surprise_embedding_dim", 16)),
        correction_embedding_dim=int(train_args.get("correction_embedding_dim", 32)),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    notes = extract_note_events(args.midi)
    window_size = int(args.window_size or train_args.get("window_size", 256))
    windows = _iter_windows(len(notes), window_size, args.window_overlap)
    output_top_k = max(args.top_k, 1)
    probabilities = torch.zeros(len(notes), dtype=torch.float32)
    action_probabilities = torch.zeros((len(notes), len(ACTION_NAMES)), dtype=torch.float32)
    pitch_scores = torch.zeros((len(notes), output_top_k), dtype=torch.float32)
    pitch_indices = torch.zeros((len(notes), output_top_k), dtype=torch.long)
    reported_top_k = output_top_k
    with torch.no_grad():
        for window_start, window_end in windows:
            window_notes = notes[window_start:window_end]
            features = torch.from_numpy(note_features(window_notes)).unsqueeze(0).to(device)
            if features.shape[-1] > input_size:
                features = features[..., :input_size]
            elif features.shape[-1] < input_size:
                features = torch.nn.functional.pad(features, (0, input_size - features.shape[-1]))

            if bool(train_args.get("explicit_correction_evidence", False)):
                feature_mask = torch.ones(features.shape[:2], dtype=features.dtype, device=device)
                correction_evidence, _, _ = build_explicit_correction_evidence(
                    model,
                    features,
                    feature_mask,
                    groups=int(train_args.get("correction_evidence_groups", 4)),
                )
                outputs = model(features, correction_evidence=correction_evidence)
            elif bool(train_args.get("explicit_surprise", False)):
                feature_mask = torch.ones(features.shape[:2], dtype=features.dtype, device=device)
                surprise, surprise_available = build_explicit_surprise(
                    model,
                    features,
                    feature_mask,
                    training=False,
                    train_mask_rate=float(train_args.get("surprise_train_mask_rate", 0.25)),
                    eval_groups=int(train_args.get("surprise_eval_groups", 4)),
                )
                outputs = model(features, surprise=surprise, surprise_available=surprise_available)
            else:
                outputs = model(features)
            window_probabilities = torch.sigmoid(outputs["error_logits"])[0].cpu()
            window_action_probabilities = torch.softmax(outputs["kind_logits"], dim=-1)[0].cpu()
            top_k = min(output_top_k, outputs["pitch_logits"].shape[-1])
            reported_top_k = top_k
            window_pitch_scores, window_pitch_indices = torch.softmax(outputs["pitch_logits"], dim=-1)[0].topk(
                k=top_k,
                dim=-1,
            )
            window_pitch_scores = window_pitch_scores.cpu()
            window_pitch_indices = window_pitch_indices.cpu()

            for local_idx, global_idx in enumerate(range(window_start, window_end)):
                if window_probabilities[local_idx] >= probabilities[global_idx]:
                    probabilities[global_idx] = window_probabilities[local_idx]
                    action_probabilities[global_idx] = window_action_probabilities[local_idx]
                    pitch_scores[global_idx, :top_k] = window_pitch_scores[local_idx]
                    pitch_indices[global_idx, :top_k] = window_pitch_indices[local_idx]
    actions = action_probabilities.argmax(dim=-1)
    top_replacement_pitches = [int(pitch_indices[idx, 0]) for idx in range(len(notes))]
    harmony_current, harmony_candidate, harmony_gain = harmony_scores_for_pitches(
        notes,
        top_replacement_pitches,
        onset_tolerance=args.harmony_onset_tolerance,
    )

    results = []
    for idx, note in enumerate(notes):
        probability = float(probabilities[idx])
        action_id = int(actions[idx])
        action_name = ACTION_NAMES[action_id]
        action_probability = float(action_probabilities[idx, action_id])
        keep_probability = float(action_probabilities[idx, 0])
        if probability < threshold:
            continue
        if not _passes_action_filter(action_name, args.action_filter):
            continue
        if action_probability < args.min_action_prob:
            continue
        if action_name == "replace" and args.min_harmony_gain is not None and harmony_gain[idx] < args.min_harmony_gain:
            continue
        confidence_score = _candidate_confidence(probability, keep_probability, action_probability)
        adjusted_confidence_score = confidence_score + args.harmony_gain_weight * harmony_gain[idx]
        harmony_penalty_applied = False
        if (
            action_name == "replace"
            and args.chord_tone_penalty > 0.0
            and harmony_current[idx] >= args.harmony_current_threshold
            and harmony_gain[idx] <= 0.0
        ):
            adjusted_confidence_score -= args.chord_tone_penalty
            harmony_penalty_applied = True
        results.append(
            {
                "note_index": idx,
                "start": note.start,
                "end": note.end,
                "input_pitch": note.pitch,
                "predicted_action": action_name,
                "action_probabilities": {
                    ACTION_NAMES[action_idx]: float(action_probabilities[idx, action_idx])
                    for action_idx in range(len(ACTION_NAMES))
                },
                "top_pitches": [
                    {"pitch": int(pitch_indices[idx, rank]), "probability": float(pitch_scores[idx, rank])}
                    for rank in range(reported_top_k)
                ],
                "error_probability": probability,
                "confidence_score": confidence_score,
                "adjusted_confidence_score": adjusted_confidence_score,
                "harmony_current_score": harmony_current[idx],
                "harmony_candidate_score": harmony_candidate[idx],
                "harmony_gain": harmony_gain[idx],
                "harmony_penalty_applied": harmony_penalty_applied,
                "detection_threshold": threshold,
            }
        )

    results = _suppress_close_candidates(results, args.min_separation)
    results.sort(key=lambda item: item["adjusted_confidence_score"], reverse=True)
    if args.max_candidates is not None:
        results = results[: max(args.max_candidates, 0)]
    if args.sort_by == "time":
        results.sort(key=lambda item: (item["start"], item["note_index"]))

    if args.include_summary:
        output = {
            "midi": args.midi,
            "checkpoint": args.checkpoint,
            "note_count": len(notes),
            "candidate_count": len(results),
            "detection_threshold": threshold,
            "action_filter": args.action_filter,
            "min_action_prob": args.min_action_prob,
            "max_candidates": args.max_candidates,
            "min_separation": args.min_separation,
            "harmony_gain_weight": args.harmony_gain_weight,
            "chord_tone_penalty": args.chord_tone_penalty,
            "harmony_current_threshold": args.harmony_current_threshold,
            "min_harmony_gain": args.min_harmony_gain,
            "harmony_onset_tolerance": args.harmony_onset_tolerance,
            "window_size": window_size,
            "window_overlap": args.window_overlap,
            "window_count": len(windows),
            "candidates": results,
        }
    else:
        output = results
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
