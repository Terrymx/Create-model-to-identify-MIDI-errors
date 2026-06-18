from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from midi_error_detector.data import FEATURE_SIZE, MaestroWrongNoteDataset
from midi_error_detector.model import build_wrong_note_model
from midi_error_detector.train import build_explicit_surprise
from train_directional_likelihood import DIRECTIONAL_SAFE_FEATURE_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate leakage-safe directional likelihood signals.")
    parser.add_argument("--forward-checkpoint", required=True)
    parser.add_argument("--backward-checkpoint", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--eval-split", default="validation", choices=["validation", "test"])
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--candidate-threshold", type=float, default=0.55)
    parser.add_argument("--gate-increment", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--output-json", default="training_logs/directional_likelihood_gate.json")
    parser.add_argument("--output-md", default="training_logs/directional_likelihood_gate.md")
    return parser.parse_args()


def model_args(checkpoint: dict) -> SimpleNamespace:
    raw = checkpoint.get("args", {})
    return SimpleNamespace(
        model=raw.get("model", "transformer"),
        input_size=int(raw.get("input_size", FEATURE_SIZE)),
        hidden_size=int(raw.get("hidden_size", 256)),
        num_layers=int(raw.get("num_layers", 4)),
        d_model=int(raw.get("d_model", raw.get("transformer_d_model", 192))),
        heads=int(raw.get("heads", raw.get("transformer_heads", 4))),
        ffn_dim=int(raw.get("ffn_dim", raw.get("transformer_ffn_dim", 512))),
        dropout=float(raw.get("dropout", 0.15)),
        explicit_surprise=bool(raw.get("explicit_surprise", False)),
        surprise_embedding_dim=int(raw.get("surprise_embedding_dim", 16)),
        safe_feature_columns=list(raw.get("safe_feature_columns", DIRECTIONAL_SAFE_FEATURE_COLUMNS)),
        unified_correction=bool(raw.get("unified_correction", False)),
        delete_auxiliary_head=bool(raw.get("delete_auxiliary_head", False)),
    )


def load_model(
    path: str,
    device: torch.device,
    *,
    require_explicit_surprise: bool,
) -> tuple[torch.nn.Module, SimpleNamespace]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = model_args(checkpoint)
    if require_explicit_surprise and not args.explicit_surprise:
        raise ValueError(f"{path} is not an explicit-surprise detector")
    model = build_wrong_note_model(
        model_type=args.model,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        transformer_d_model=args.d_model,
        transformer_heads=args.heads,
        transformer_ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        explicit_surprise=require_explicit_surprise,
        surprise_embedding_dim=args.surprise_embedding_dim,
        unified_correction=args.unified_correction,
        delete_auxiliary_head=args.delete_auxiliary_head,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, args


def adapt_features(features: torch.Tensor, input_size: int) -> torch.Tensor:
    if features.shape[-1] > input_size:
        return features[..., :input_size]
    if features.shape[-1] < input_size:
        return torch.nn.functional.pad(features, (0, input_size - features.shape[-1]))
    return features


@torch.no_grad()
def directional_surprise(
    model: torch.nn.Module,
    raw_features: torch.Tensor,
    mask: torch.Tensor,
    direction: str,
    safe_columns: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    features = raw_features[:, :, safe_columns]
    observed_pitch = torch.round(raw_features[:, :, 0] * 127.0).long().clamp(0, 127)
    oriented = features
    oriented_pitch = observed_pitch
    oriented_mask = mask.bool()
    if direction == "backward":
        oriented = oriented.flip(1)
        oriented_pitch = oriented_pitch.flip(1)
        oriented_mask = oriented_mask.flip(1)

    shifted = torch.zeros_like(oriented)
    shifted[:, 1:] = oriented[:, :-1]
    available = oriented_mask.clone()
    available[:, 0] = False
    logits = model.predict_pitch(shifted, causal=True)
    log_probability = torch.log_softmax(logits, dim=-1).gather(
        dim=-1,
        index=oriented_pitch.unsqueeze(-1),
    ).squeeze(-1)
    surprise = -log_probability
    if direction == "backward":
        surprise = surprise.flip(1)
        available = available.flip(1)
    return surprise, available


def average_ranks(values: torch.Tensor) -> torch.Tensor:
    values = values.double()
    order = torch.argsort(values)
    sorted_values = values[order]
    ranks = torch.empty(len(values), dtype=torch.double)
    _, inverse, counts = torch.unique_consecutive(
        sorted_values,
        return_inverse=True,
        return_counts=True,
    )
    ends = counts.cumsum(0)
    starts = ends - counts
    group_ranks = (starts.double() + 1.0 + ends.double()) / 2.0
    ranks[order] = group_ranks[inverse]
    return ranks


def rank_auc(values: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.bool()
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return 0.5
    ranks = average_ranks(values)
    positive_rank_sum = float(ranks[labels].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double() - left.double().mean()
    right = right.double() - right.double().mean()
    denominator = torch.sqrt(left.square().sum() * right.square().sum()).clamp_min(1e-12)
    return float((left * right).sum() / denominator)


def js_divergence(clean: torch.Tensor, error: torch.Tensor, bins: int = 80) -> float:
    minimum = float(torch.minimum(clean.min(), error.min()))
    maximum = float(torch.maximum(clean.max(), error.max()))
    if maximum <= minimum:
        return 0.0
    clean_hist = torch.histc(clean.float(), bins=bins, min=minimum, max=maximum).double()
    error_hist = torch.histc(error.float(), bins=bins, min=minimum, max=maximum).double()
    clean_prob = (clean_hist + 1e-9) / (clean_hist.sum() + bins * 1e-9)
    error_prob = (error_hist + 1e-9) / (error_hist.sum() + bins * 1e-9)
    midpoint = 0.5 * (clean_prob + error_prob)
    return float(
        0.5 * (clean_prob * (clean_prob / midpoint).log()).sum()
        + 0.5 * (error_prob * (error_prob / midpoint).log()).sum()
    )


def summary(values: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    clean = values[~labels]
    error = values[labels]
    return {
        "auc": rank_auc(values, labels),
        "clean_mean": float(clean.mean()),
        "error_mean": float(error.mean()),
        "mean_gap": float(error.mean() - clean.mean()),
        "js_divergence": js_divergence(clean, error),
    }


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Directional Likelihood Signal Gate",
        "",
        f"- split: `{result['eval_split']}`",
        f"- error rate: `{result['error_rate']}`",
        f"- valid notes with both directions: `{result['valid_notes']}`",
        f"- error notes: `{result['error_notes']}`",
        f"- candidate threshold: `{result['candidates']['threshold']}`",
        f"- candidate precision: `{result['candidates']['precision']:.4f}`",
        f"- candidate recall ceiling: `{result['candidates']['recall_ceiling']:.4f}`",
        "",
        "| Population | Signal | AUC | Clean/FP mean | Error/TP mean | JS divergence |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for population, signals in (
        ("all notes", result["all_notes"]),
        ("candidates", result["candidates"]["signals"]),
    ):
        for name, row in signals.items():
            lines.append(
                f"| {population} | {name} | {row['auc']:.4f} | "
                f"{row['clean_mean']:.4f} | {row['error_mean']:.4f} | "
                f"{row['js_divergence']:.4f} |"
            )
    lines.extend(
        [
            "",
            f"- Step 2 candidate AUC: `{result['baseline_candidate_auc']:.4f}`",
            f"- best directional candidate signal: `{result['best_directional_signal']}`",
            f"- best directional candidate AUC: `{result['best_directional_candidate_auc']:.4f}`",
            f"- Gate target AUC: `{result['gate_target_auc']:.4f}`",
            f"- Gate passed: `{result['gate_passed']}`",
            f"- recommendation: {result['recommendation']}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    forward_model, forward_args = load_model(
        args.forward_checkpoint,
        device,
        require_explicit_surprise=False,
    )
    backward_model, backward_args = load_model(
        args.backward_checkpoint,
        device,
        require_explicit_surprise=False,
    )
    detector, detector_args = load_model(
        args.detector_checkpoint,
        device,
        require_explicit_surprise=True,
    )
    if forward_args.safe_feature_columns != backward_args.safe_feature_columns:
        raise ValueError("forward and backward checkpoints use different safe feature columns")

    dataset = MaestroWrongNoteDataset(
        root=args.data_root,
        split=args.eval_split,
        window_size=args.window_size,
        stride=args.stride,
        error_rate=args.error_rate,
        max_files=args.max_files,
        cache_notes=True,
        verbose=True,
    )
    dataset.set_epoch(0)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    parts: dict[str, list[torch.Tensor]] = {
        "forward": [],
        "backward": [],
        "step2_internal": [],
        "labels": [],
        "candidates": [],
    }
    for batch in tqdm(loader, desc="directional signal gate", unit="batch", dynamic_ncols=True):
        raw_features = batch["features"].to(device)
        mask = batch["mask"].to(device).bool()
        labels = batch["is_error"].to(device).bool()
        forward, forward_available = directional_surprise(
            forward_model,
            raw_features,
            mask,
            "forward",
            forward_args.safe_feature_columns,
        )
        backward, backward_available = directional_surprise(
            backward_model,
            raw_features,
            mask,
            "backward",
            backward_args.safe_feature_columns,
        )
        both_available = mask & forward_available & backward_available

        detector_features = adapt_features(raw_features, detector_args.input_size)
        internal, internal_available = build_explicit_surprise(
            detector,
            detector_features,
            mask.float(),
            training=False,
            train_mask_rate=0.0,
            eval_groups=args.groups,
        )
        outputs = detector(
            detector_features,
            surprise=internal,
            surprise_available=internal_available,
        )
        candidates = torch.sigmoid(outputs["error_logits"]) >= args.candidate_threshold
        parts["forward"].append(forward[both_available].cpu())
        parts["backward"].append(backward[both_available].cpu())
        parts["step2_internal"].append(internal[both_available].cpu())
        parts["labels"].append(labels[both_available].cpu())
        parts["candidates"].append(candidates[both_available].cpu())

    forward = torch.cat(parts["forward"])
    backward = torch.cat(parts["backward"])
    internal = torch.cat(parts["step2_internal"])
    labels = torch.cat(parts["labels"]).bool()
    candidates = torch.cat(parts["candidates"]).bool()
    signals = {
        "forward": forward,
        "backward": backward,
        "directional_mean": 0.5 * (forward + backward),
        "directional_min": torch.minimum(forward, backward),
        "directional_max": torch.maximum(forward, backward),
        "directional_disagreement": (forward - backward).abs(),
        "step2_internal": internal,
    }
    all_summaries = {name: summary(values, labels) for name, values in signals.items()}
    candidate_labels = labels[candidates]
    candidate_summaries = {
        name: summary(values[candidates], candidate_labels)
        for name, values in signals.items()
    }
    directional_names = [name for name in signals if name != "step2_internal"]
    best_name = max(directional_names, key=lambda name: candidate_summaries[name]["auc"])
    baseline_auc = candidate_summaries["step2_internal"]["auc"]
    best_auc = candidate_summaries[best_name]["auc"]
    target_auc = baseline_auc + args.gate_increment
    gate_passed = best_auc >= target_auc
    correlations = {
        name: {
            "pearson_with_step2": pearson(values, internal),
            "spearman_with_step2": pearson(average_ranks(values), average_ranks(internal)),
        }
        for name, values in signals.items()
        if name != "step2_internal"
    }
    if gate_passed:
        recommendation = (
            "The directional evidence passes the hard-candidate separation Gate. "
            "Proceed to a frozen-Step-2 fusion probe before full detector retraining."
        )
    else:
        recommendation = (
            "The directional evidence does not pass the hard-candidate separation Gate. "
            "Do not spend a full detector run on this signal without redesigning the likelihood input/objective."
        )
    result = {
        "forward_checkpoint": args.forward_checkpoint,
        "backward_checkpoint": args.backward_checkpoint,
        "detector_checkpoint": args.detector_checkpoint,
        "eval_split": args.eval_split,
        "error_rate": args.error_rate,
        "valid_notes": int(labels.numel()),
        "error_notes": int(labels.sum()),
        "all_notes": all_summaries,
        "candidates": {
            "threshold": args.candidate_threshold,
            "count": int(candidates.sum()),
            "precision": float(candidate_labels.float().mean()),
            "recall_ceiling": float(candidate_labels.sum() / labels.sum().clamp_min(1)),
            "signals": candidate_summaries,
        },
        "correlations": correlations,
        "baseline_candidate_auc": baseline_auc,
        "best_directional_signal": best_name,
        "best_directional_candidate_auc": best_auc,
        "gate_increment": args.gate_increment,
        "gate_target_auc": target_auc,
        "gate_passed": gate_passed,
        "recommendation": recommendation,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
