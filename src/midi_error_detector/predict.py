"""Run wrong-note detection and pitch correction on one MIDI file."""

from __future__ import annotations

import argparse
import json
import sys

import torch

from .data import extract_note_events, note_features
from .model import build_wrong_note_model

ACTION_NAMES = ["keep", "replace", "delete"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--midi", required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Detection threshold. Defaults to checkpoint valid_metrics.best_det_threshold when available, otherwise training det_threshold/0.3.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of replacement pitch candidates to return")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_args = checkpoint.get("args", {})
    valid_metrics = checkpoint.get("valid_metrics", {})
    threshold = args.threshold
    if threshold is None:
        threshold = float(valid_metrics.get("best_det_threshold", train_args.get("det_threshold", 0.3)))
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
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    notes = extract_note_events(args.midi)
    features = torch.from_numpy(note_features(notes)).unsqueeze(0).to(device)
    if features.shape[-1] > input_size:
        features = features[..., :input_size]
    elif features.shape[-1] < input_size:
        features = torch.nn.functional.pad(features, (0, input_size - features.shape[-1]))
    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.sigmoid(outputs["error_logits"])[0].cpu()
        action_probabilities = torch.softmax(outputs["kind_logits"], dim=-1)[0].cpu()
        actions = action_probabilities.argmax(dim=-1)
        top_k = min(max(args.top_k, 1), outputs["pitch_logits"].shape[-1])
        pitch_scores, pitch_indices = torch.softmax(outputs["pitch_logits"], dim=-1)[0].topk(k=top_k, dim=-1)
        pitch_scores = pitch_scores.cpu()
        pitch_indices = pitch_indices.cpu()

    results = []
    for idx, note in enumerate(notes):
        probability = float(probabilities[idx])
        action_id = int(actions[idx])
        if probability < threshold and action_id == 0:
            continue
        results.append(
            {
                "note_index": idx,
                "start": note.start,
                "end": note.end,
                "input_pitch": note.pitch,
                "predicted_action": ACTION_NAMES[action_id],
                "action_probabilities": {
                    ACTION_NAMES[action_idx]: float(action_probabilities[idx, action_idx])
                    for action_idx in range(len(ACTION_NAMES))
                },
                "top_pitches": [
                    {"pitch": int(pitch_indices[idx, rank]), "probability": float(pitch_scores[idx, rank])}
                    for rank in range(top_k)
                ],
                "error_probability": probability,
                "detection_threshold": threshold,
            }
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
