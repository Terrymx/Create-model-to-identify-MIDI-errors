"""Train a BiGRU model on MAESTRO MIDI with synthetic wrong-note corruption."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import FEATURE_SIZE, MaestroWrongNoteDataset
from .model import build_wrong_note_model, masked_bce_with_logits, masked_kind_loss, masked_pitch_loss

PITCH_CONTEXT_FEATURE_COLUMNS = [
    0,
    4,
    6,
    7,
    8,
    9,
    10,
    11,
    15,
    16,
    17,
    18,
    19,
    20,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
]


def load_compatible_state_dict(model: torch.nn.Module, checkpoint_state: dict[str, torch.Tensor]) -> tuple[list[str], list[str], list[str]]:
    """Load matching checkpoint weights and partially copy widened input projections."""

    model_state = model.state_dict()
    adapted_state = dict(model_state)
    loaded: list[str] = []
    partial: list[str] = []
    skipped: list[str] = []
    for name, checkpoint_value in checkpoint_state.items():
        if name not in model_state:
            skipped.append(name)
            continue
        model_value = model_state[name]
        if model_value.shape == checkpoint_value.shape:
            adapted_state[name] = checkpoint_value
            loaded.append(name)
            continue
        if name == "input_projection.weight" and model_value.ndim == 2 and checkpoint_value.ndim == 2:
            copied = model_value.clone()
            rows = min(model_value.shape[0], checkpoint_value.shape[0])
            cols = min(model_value.shape[1], checkpoint_value.shape[1])
            copied[:rows, :cols] = checkpoint_value[:rows, :cols]
            adapted_state[name] = copied
            partial.append(f"{name} old={tuple(checkpoint_value.shape)} new={tuple(model_value.shape)} copied_cols={cols}")
            continue
        skipped.append(f"{name} old={tuple(checkpoint_value.shape)} new={tuple(model_value.shape)}")
    model.load_state_dict(adapted_state)
    return loaded, partial, skipped


def f_beta(precision: float, recall: float, beta: float) -> float:
    beta_squared = beta * beta
    return (1.0 + beta_squared) * precision * recall / max(beta_squared * precision + recall, 1e-12)


def hard_pairwise_ranking_loss(
    logits: torch.Tensor,
    is_error: torch.Tensor,
    mask: torch.Tensor,
    margin: float,
    top_k: int,
) -> torch.Tensor:
    """Rank hard positive errors above hard negative clean notes within a batch."""

    valid = mask.bool()
    positive_logits = logits[(is_error >= 0.5) & valid]
    negative_logits = logits[(is_error < 0.5) & valid]
    if positive_logits.numel() == 0 or negative_logits.numel() == 0 or top_k <= 0:
        return logits.sum() * 0.0

    hard_positive_count = min(top_k, positive_logits.numel())
    hard_negative_count = min(top_k, negative_logits.numel())
    hard_positives = torch.topk(positive_logits, k=hard_positive_count, largest=False).values
    hard_negatives = torch.topk(negative_logits, k=hard_negative_count, largest=True).values
    pairwise_margin = margin - hard_positives.unsqueeze(1) + hard_negatives.unsqueeze(0)
    return torch.relu(pairwise_margin).mean()


def masked_pitch_reconstruction_loss(
    model: torch.nn.Module,
    features: torch.Tensor,
    target_pitch: torch.Tensor,
    mask: torch.Tensor,
    mask_rate: float,
    feature_columns: list[int],
) -> tuple[torch.Tensor, float]:
    """Hide pitch-derived features on sampled notes and predict their clean pitch."""

    if mask_rate <= 0.0:
        return features.sum() * 0.0, 0.0
    valid = mask.bool()
    sampled = (torch.rand(mask.shape, device=features.device) < mask_rate) & valid
    if sampled.sum() == 0:
        return features.sum() * 0.0, 0.0

    masked_features = features.clone()
    masked_features[:, :, feature_columns] = torch.where(
        sampled.unsqueeze(-1),
        torch.zeros_like(masked_features[:, :, feature_columns]),
        masked_features[:, :, feature_columns],
    )
    outputs = model(masked_features)
    loss = masked_pitch_loss(outputs["pitch_logits"], target_pitch, sampled.float())
    return loss, float(sampled.sum())


def parse_error_rate_stages(raw_stages: str | None) -> list[list[float]] | None:
    """Parse semicolon-separated curriculum stages, e.g. '0.08,0.12;0.02,0.05;0.005,0.01'."""

    if raw_stages is None:
        return None
    stages: list[list[float]] = []
    for raw_stage in raw_stages.split(";"):
        raw_stage = raw_stage.strip()
        if not raw_stage:
            continue
        rates = [float(value.strip()) for value in raw_stage.split(",") if value.strip()]
        if not rates:
            raise ValueError(f"Empty curriculum stage in {raw_stages!r}")
        stages.append(rates)
    if not stages:
        raise ValueError("--curriculum-error-rate-stages must contain at least one stage")
    return stages


def select_curriculum_error_rates(
    stages: list[list[float]] | None,
    default_rates: list[float],
    epoch: int,
    total_epochs: int,
) -> tuple[str, list[float], int]:
    """Return the active curriculum stage name, rates, and 1-based epoch within that stage."""

    if not stages:
        return "coverage", default_rates, epoch
    stage_count = len(stages)
    stage_idx = min((epoch - 1) * stage_count // max(total_epochs, 1), stage_count - 1)
    stage_start = (stage_idx * total_epochs) // stage_count + 1
    return f"curriculum_{stage_idx + 1}", stages[stage_idx], epoch - stage_start + 1


def _clone_replay_sample(
    batch: dict[str, torch.Tensor],
    item_idx: int,
    score: float,
    replay_weight: float,
    replay_kind: str,
) -> dict[str, torch.Tensor | float | str]:
    return {
        "score": float(score),
        "replay_kind": replay_kind,
        "features": batch["features"][item_idx].detach().cpu().clone(),
        "is_error": batch["is_error"][item_idx].detach().cpu().clone(),
        "target_pitch": batch["target_pitch"][item_idx].detach().cpu().clone(),
        "error_kind": batch["error_kind"][item_idx].detach().cpu().clone(),
        "det_weight": batch["det_weight"][item_idx].detach().cpu().clone() * float(replay_weight),
        "mask": batch["mask"][item_idx].detach().cpu().clone(),
    }


def append_hard_replay_samples(
    replay_buffer: list[dict[str, torch.Tensor | float]],
    batch: dict[str, torch.Tensor],
    logits: torch.Tensor,
    max_size: int,
) -> None:
    """Keep the hardest windows from the current epoch for next-epoch replay."""

    if max_size <= 0:
        return
    with torch.no_grad():
        mask = batch["mask"].to(logits.device)
        is_error = batch["is_error"].to(logits.device)
        probabilities = torch.sigmoid(logits.detach())
        valid = mask.bool()
        error_mask = (is_error >= 0.5) & valid
        clean_mask = (is_error < 0.5) & valid

        error_counts = error_mask.float().sum(dim=1)
        clean_counts = clean_mask.float().sum(dim=1)
        false_negative_pressure = ((1.0 - probabilities) * error_mask.float()).sum(dim=1) / error_counts.clamp_min(1.0)
        false_positive_pressure = (probabilities * clean_mask.float()).amax(dim=1)
        has_signal = (error_counts > 0) | (clean_counts > 0)
        hard_scores = false_negative_pressure + 0.5 * false_positive_pressure
        hard_scores = torch.where(has_signal, hard_scores, torch.zeros_like(hard_scores))

        for item_idx, score in enumerate(hard_scores.detach().cpu().tolist()):
            if score <= 0.0:
                continue
            replay_buffer.append(
                {
                    "score": float(score),
                    "features": batch["features"][item_idx].detach().cpu().clone(),
                    "is_error": batch["is_error"][item_idx].detach().cpu().clone(),
                    "target_pitch": batch["target_pitch"][item_idx].detach().cpu().clone(),
                    "error_kind": batch["error_kind"][item_idx].detach().cpu().clone(),
                    "det_weight": batch["det_weight"][item_idx].detach().cpu().clone(),
                    "mask": batch["mask"][item_idx].detach().cpu().clone(),
                }
            )
        if len(replay_buffer) > max_size * 2:
            replay_buffer.sort(key=lambda sample: float(sample["score"]), reverse=True)
            del replay_buffer[max_size:]


def append_asymmetric_hard_replay_samples(
    fn_replay_buffer: list[dict[str, torch.Tensor | float | str]],
    fp_replay_buffer: list[dict[str, torch.Tensor | float | str]],
    batch: dict[str, torch.Tensor],
    logits: torch.Tensor,
    max_size: int,
    fn_fraction: float,
    fn_weight: float,
    fp_weight: float,
    det_threshold: float,
) -> None:
    """Keep separate false-negative and false-positive replay windows."""

    if max_size <= 0:
        return
    fn_limit = max(1, int(round(max_size * fn_fraction)))
    fp_limit = max(1, max_size - fn_limit)
    with torch.no_grad():
        mask = batch["mask"].to(logits.device)
        is_error = batch["is_error"].to(logits.device)
        probabilities = torch.sigmoid(logits.detach())
        valid = mask.bool()
        error_mask = (is_error >= 0.5) & valid
        clean_mask = (is_error < 0.5) & valid

        fn_counts = (error_mask & (probabilities < det_threshold)).float().sum(dim=1)
        error_counts = error_mask.float().sum(dim=1)
        false_negative_pressure = ((det_threshold - probabilities).clamp_min(0.0) * error_mask.float()).sum(dim=1)
        false_negative_pressure = false_negative_pressure / error_counts.clamp_min(1.0)

        fp_pressure_per_note = (probabilities - det_threshold).clamp_min(0.0) * clean_mask.float()
        false_positive_pressure = fp_pressure_per_note.amax(dim=1)
        fp_counts = (clean_mask & (probabilities >= det_threshold)).float().sum(dim=1)

        for item_idx, score in enumerate(false_negative_pressure.detach().cpu().tolist()):
            if score <= 0.0 or float(fn_counts[item_idx].detach().cpu()) <= 0.0:
                continue
            fn_replay_buffer.append(_clone_replay_sample(batch, item_idx, score, fn_weight, "fn"))
        for item_idx, score in enumerate(false_positive_pressure.detach().cpu().tolist()):
            if score <= 0.0 or float(fp_counts[item_idx].detach().cpu()) <= 0.0:
                continue
            fp_replay_buffer.append(_clone_replay_sample(batch, item_idx, score, fp_weight, "fp"))

        if len(fn_replay_buffer) > fn_limit * 2:
            fn_replay_buffer.sort(key=lambda sample: float(sample["score"]), reverse=True)
            del fn_replay_buffer[fn_limit:]
        if len(fp_replay_buffer) > fp_limit * 2:
            fp_replay_buffer.sort(key=lambda sample: float(sample["score"]), reverse=True)
            del fp_replay_buffer[fp_limit:]


def select_asymmetric_replay_samples(
    fn_replay_buffer: list[dict[str, torch.Tensor | float | str]],
    fp_replay_buffer: list[dict[str, torch.Tensor | float | str]],
    max_size: int,
    fn_fraction: float,
) -> list[dict[str, torch.Tensor | float | str]]:
    if max_size <= 0:
        return []
    fn_limit = max(1, int(round(max_size * fn_fraction)))
    fp_limit = max(1, max_size - fn_limit)
    fn_replay_buffer.sort(key=lambda sample: float(sample["score"]), reverse=True)
    fp_replay_buffer.sort(key=lambda sample: float(sample["score"]), reverse=True)
    selected = fn_replay_buffer[:fn_limit] + fp_replay_buffer[:fp_limit]
    selected.sort(key=lambda sample: float(sample["score"]), reverse=True)
    return selected[:max_size]


def make_replay_batches(
    replay_buffer: list[dict[str, torch.Tensor | float | str]],
    batch_size: int,
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield mini-batches from the previous epoch's hardest windows."""

    tensor_keys = ["features", "is_error", "target_pitch", "error_kind", "det_weight", "mask"]
    for start in range(0, len(replay_buffer), batch_size):
        samples = replay_buffer[start : start + batch_size]
        yield {
            key: torch.stack([sample[key] for sample in samples if isinstance(sample[key], torch.Tensor)])
            for key in tensor_keys
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Extracted MAESTRO MIDI archive directory")
    parser.add_argument("--version", default="v3.0.0", help="MAESTRO version, e.g. v3.0.0")
    parser.add_argument("--output", default="checkpoints/bigru_wrong_note.pt")
    parser.add_argument("--init-checkpoint", default=None, help="Optional checkpoint to initialize/fine-tune from.")
    parser.add_argument("--eval-split", default="test", choices=["validation", "test"], help="Held-out split used for metrics")
    parser.add_argument("--epochs", type=int, default=20, help="Corrupted-data fine-tuning epochs")
    parser.add_argument(
        "--clean-epochs",
        type=int,
        default=1,
        help="Clean-data warm-up epochs before corrupted-data training; uses error_rate=0.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--error-rate", type=float, default=0.08, help="Held-out evaluation corruption rate")
    parser.add_argument(
        "--train-error-rate",
        type=float,
        default=0.15,
        help="Corrupted training-stage error rate; higher than eval by default to improve recall.",
    )
    parser.add_argument(
        "--train-error-rates",
        type=float,
        nargs="+",
        default=None,
        help="Optional per-epoch cycle of training corruption rates, useful for sparse-error fine-tuning.",
    )
    parser.add_argument(
        "--calibration-epochs",
        type=int,
        default=0,
        help="Use sparse calibration error rates for the last N corrupted epochs; 0 disables this phase.",
    )
    parser.add_argument(
        "--calibration-error-rates",
        type=float,
        nargs="+",
        default=None,
        help="Per-epoch cycle used during the final calibration phase, e.g. 0.005 0.01 0.02.",
    )
    parser.add_argument(
        "--curriculum-error-rate-stages",
        default=None,
        help=(
            "Semicolon-separated per-stage error-rate cycles, e.g. "
            "'0.08,0.12;0.02,0.05,0.08;0.005,0.01,0.02'. "
            "When set, it overrides train/calibration error-rate scheduling."
        ),
    )
    parser.add_argument(
        "--det-threshold",
        type=float,
        default=0.3,
        help="Detection probability threshold used for precision/recall/F1 metrics.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-factor", type=float, default=0.5, help="Factor for ReduceLROnPlateau when the saved metric plateaus.")
    parser.add_argument("--lr-patience", type=int, default=4, help="Plateau epochs before reducing LR; set 0 to disable scheduling.")
    parser.add_argument("--lr-threshold", type=float, default=0.002, help="Minimum absolute save-metric improvement counted by the LR scheduler.")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Lower bound for scheduled learning rate.")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many corrupt epochs without save-metric improvement; 0 disables early stopping.",
    )
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--model", choices=["bigru", "transformer"], default="bigru")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--transformer-d-model", type=int, default=192)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ffn-dim", type=int, default=512)
    parser.add_argument("--pitch-loss-weight", type=float, default=0.5)
    parser.add_argument("--kind-loss-weight", type=float, default=0.3)
    parser.add_argument(
        "--det-loss-weight",
        type=float,
        default=1.0,
        help="Global multiplier for detection BCE; clean masked auxiliary passes can set this to 0.",
    )
    parser.add_argument(
        "--masked-pitch-loss-weight",
        type=float,
        default=0.0,
        help="Weight for masked pitch reconstruction from clean/contextual MIDI features.",
    )
    parser.add_argument(
        "--masked-pitch-rate",
        type=float,
        default=0.15,
        help="Fraction of valid notes whose pitch-derived feature columns are hidden for masked reconstruction.",
    )
    parser.add_argument(
        "--clean-mask-batches-per-epoch",
        type=int,
        default=0,
        help="Run this many clean masked-learning batches before each corrupted epoch; 0 disables the interleaved phase.",
    )
    parser.add_argument(
        "--ranking-loss-weight",
        type=float,
        default=0.0,
        help="Weight for hard pairwise ranking loss between low-scoring errors and high-scoring clean notes.",
    )
    parser.add_argument(
        "--ranking-margin",
        type=float,
        default=1.0,
        help="Required logit margin for ranking true errors above hard clean negatives.",
    )
    parser.add_argument(
        "--ranking-top-k",
        type=int,
        default=64,
        help="Number of hardest positives and negatives per batch used by ranking loss.",
    )
    parser.add_argument(
        "--hard-replay-size",
        type=int,
        default=0,
        help="Keep this many hardest windows from each training epoch and replay them before the next epoch.",
    )
    parser.add_argument(
        "--hard-replay-epochs",
        type=int,
        default=1,
        help="Number of replay passes over the previous epoch's hard window buffer.",
    )
    parser.add_argument(
        "--asymmetric-hard-replay",
        action="store_true",
        help="Split hard replay into false-negative and false-positive buckets instead of one mixed buffer.",
    )
    parser.add_argument(
        "--fn-replay-fraction",
        type=float,
        default=0.75,
        help="Fraction of hard replay windows reserved for false negatives when asymmetric replay is enabled.",
    )
    parser.add_argument(
        "--fn-replay-weight",
        type=float,
        default=1.5,
        help="Detection-loss multiplier applied to false-negative replay windows.",
    )
    parser.add_argument(
        "--fp-replay-weight",
        type=float,
        default=0.4,
        help="Detection-loss multiplier applied to false-positive replay windows.",
    )
    parser.add_argument(
        "--det-pos-weight",
        type=float,
        default=3.0,
        help="Positive-class weight for wrong-note detection BCE; increase if recall is too low.",
    )
    parser.add_argument(
        "--clean-theory-weight",
        type=float,
        default=1.0,
        help="Extra detection-loss weight for theory-plausible clean notes.",
    )
    parser.add_argument(
        "--error-theory-weight",
        type=float,
        default=1.0,
        help="Extra detection-loss weight for theory-suspicious corrupted notes.",
    )
    parser.add_argument(
        "--kind-class-weights",
        type=float,
        nargs=3,
        default=[1.0, 6.0, 4.0],
        metavar=("KEEP", "REPLACE", "DELETE"),
        help="Cross-entropy weights for keep/replace/delete action classes.",
    )
    parser.add_argument(
        "--threshold-sweep",
        type=float,
        nargs="+",
        default=[0.2, 0.25, 0.3, 0.35, 0.4, 0.5],
        help="Detection thresholds to evaluate on the held-out split; best_det_* reports the best F1 among them.",
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.8,
        help="Precision floor used by precision_recall_score checkpoint selection.",
    )
    parser.add_argument(
        "--save-metric",
        default="task_score",
        choices=[
            "task_score",
            "precision_task_score",
            "precision_recall_score",
            "det_f1",
            "det_f0_5",
            "best_det_f1",
            "best_det_f0_5",
            "precision_constrained_recall",
            "replace_pitch_top3",
            "loss",
        ],
        help="Metric used to select the best checkpoint; task_score combines detection/action/top3 quality.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Debug on the first N files of each split")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--quiet", action="store_true", help="Disable dataset-loading progress bars")
    parser.add_argument(
        "--no-cache-notes",
        dest="cache_notes",
        action="store_false",
        help="Re-read MIDI files on demand instead of keeping parsed note events in memory.",
    )
    parser.set_defaults(cache_notes=True)
    return parser.parse_args()


def make_loader(args: argparse.Namespace, split: str, shuffle: bool, error_rate: float | None = None) -> DataLoader:
    effective_error_rate = args.error_rate if error_rate is None else error_rate
    print(
        f"building {split} loader: error_rate={effective_error_rate}, "
        f"max_files={args.max_files}, cache_notes={args.cache_notes}",
        flush=True,
    )
    dataset = MaestroWrongNoteDataset(
        root=args.data_root,
        split=split,
        version=args.version,
        window_size=args.window_size,
        stride=args.stride,
        error_rate=effective_error_rate,
        max_files=args.max_files,
        cache_notes=args.cache_notes,
        verbose=not args.quiet,
        clean_theory_weight=args.clean_theory_weight,
        error_theory_weight=args.error_theory_weight,
    )
    print(f"{split} loader ready: files={len(dataset.files)}, windows={len(dataset)}", flush=True)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: BiGRUWrongNoteModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    det_loss_weight: float,
    pitch_loss_weight: float,
    kind_loss_weight: float,
    desc: str,
    det_threshold: float,
    det_pos_weight: torch.Tensor | None,
    kind_class_weight: torch.Tensor | None,
    threshold_sweep: list[float] | None = None,
    target_precision: float = 0.8,
    ranking_loss_weight: float = 0.0,
    ranking_margin: float = 1.0,
    ranking_top_k: int = 64,
    hard_replay_buffer: list[dict[str, torch.Tensor | float | str]] | None = None,
    hard_replay_size: int = 0,
    fn_replay_buffer: list[dict[str, torch.Tensor | float | str]] | None = None,
    fp_replay_buffer: list[dict[str, torch.Tensor | float | str]] | None = None,
    fn_replay_fraction: float = 0.75,
    fn_replay_weight: float = 1.5,
    fp_replay_weight: float = 0.4,
    masked_pitch_loss_weight: float = 0.0,
    masked_pitch_rate: float = 0.15,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "det_loss": 0.0,
        "ranking_loss": 0.0,
        "masked_pitch_loss": 0.0,
        "masked_pitch_notes": 0.0,
        "pitch_loss": 0.0,
        "kind_loss": 0.0,
        "pitch_correct": 0.0,
        "replace_pitch_top1": 0.0,
        "replace_pitch_top3": 0.0,
        "kind_correct": 0.0,
        "replace_kind_correct": 0.0,
        "delete_kind_correct": 0.0,
        "tp": 0.0,
        "fp": 0.0,
        "tn": 0.0,
        "fn": 0.0,
        "error_notes": 0.0,
        "replace_notes": 0.0,
        "delete_notes": 0.0,
        "notes": 0.0,
    }
    sweep_thresholds = sorted(set(threshold_sweep or [det_threshold]))
    if det_threshold not in sweep_thresholds:
        sweep_thresholds.append(det_threshold)
        sweep_thresholds.sort()
    threshold_stats = {threshold: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for threshold in sweep_thresholds}

    for batch_idx, batch in enumerate(tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True, leave=True), start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        features = batch["features"].to(device)
        is_error = batch["is_error"].to(device)
        target_pitch = batch["target_pitch"].to(device)
        error_kind = batch["error_kind"].to(device)
        det_weight = batch["det_weight"].to(device)
        mask = batch["mask"].to(device)

        with torch.set_grad_enabled(training):
            outputs = model(features)
            det_loss = masked_bce_with_logits(
                outputs["error_logits"],
                is_error,
                mask,
                pos_weight=det_pos_weight,
                sample_weight=det_weight,
            )
            pitch_mask = mask * (error_kind != 2).float()
            pitch_loss = masked_pitch_loss(outputs["pitch_logits"], target_pitch, pitch_mask)
            kind_loss = masked_kind_loss(outputs["kind_logits"], error_kind, mask, class_weight=kind_class_weight)
            ranking_loss = hard_pairwise_ranking_loss(
                outputs["error_logits"],
                is_error,
                mask,
                margin=ranking_margin,
                top_k=ranking_top_k,
            )
            if masked_pitch_loss_weight > 0.0:
                masked_recon_loss, masked_pitch_notes = masked_pitch_reconstruction_loss(
                    model,
                    features,
                    target_pitch,
                    mask,
                    mask_rate=masked_pitch_rate,
                    feature_columns=PITCH_CONTEXT_FEATURE_COLUMNS,
                )
            else:
                masked_recon_loss = features.sum() * 0.0
                masked_pitch_notes = 0.0
            loss = (
                det_loss_weight * det_loss
                + pitch_loss_weight * pitch_loss
                + kind_loss_weight * kind_loss
                + ranking_loss_weight * ranking_loss
                + masked_pitch_loss_weight * masked_recon_loss
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if fn_replay_buffer is not None and fp_replay_buffer is not None:
                    append_asymmetric_hard_replay_samples(
                        fn_replay_buffer,
                        fp_replay_buffer,
                        batch,
                        outputs["error_logits"],
                        max_size=hard_replay_size,
                        fn_fraction=fn_replay_fraction,
                        fn_weight=fn_replay_weight,
                        fp_weight=fp_replay_weight,
                        det_threshold=det_threshold,
                    )
                elif hard_replay_buffer is not None:
                    append_hard_replay_samples(
                        hard_replay_buffer,
                        batch,
                        outputs["error_logits"],
                        max_size=hard_replay_size,
                    )

        valid_mask = mask.bool()
        error_targets = is_error.bool() & valid_mask
        clean_targets = (~is_error.bool()) & valid_mask
        error_probabilities = torch.sigmoid(outputs["error_logits"])
        error_predictions = (error_probabilities >= det_threshold) & valid_mask
        clean_predictions = (~error_predictions) & valid_mask
        replace_targets = (error_kind == 1) & valid_mask
        delete_targets = (error_kind == 2) & valid_mask
        kind_predictions = outputs["kind_logits"].argmax(dim=-1)
        pitch_predictions = outputs["pitch_logits"].argmax(dim=-1)
        pitch_top3 = outputs["pitch_logits"].topk(k=3, dim=-1).indices
        pitch_correct = pitch_predictions == target_pitch
        pitch_top3_correct = (pitch_top3 == target_pitch.unsqueeze(-1)).any(dim=-1)
        kind_correct = kind_predictions == error_kind

        totals["loss"] += float(loss.detach()) * float(mask.sum())
        totals["det_loss"] += float(det_loss.detach()) * float(mask.sum())
        totals["ranking_loss"] += float(ranking_loss.detach()) * float(mask.sum())
        totals["masked_pitch_loss"] += float(masked_recon_loss.detach()) * max(masked_pitch_notes, 1.0)
        totals["masked_pitch_notes"] += masked_pitch_notes
        totals["pitch_loss"] += float(pitch_loss.detach()) * float(pitch_mask.sum().clamp_min(1.0))
        totals["kind_loss"] += float(kind_loss.detach()) * float(mask.sum())
        totals["pitch_correct"] += float((pitch_correct & valid_mask & (error_kind != 2)).sum())
        totals["replace_pitch_top1"] += float((pitch_correct & replace_targets).sum())
        totals["replace_pitch_top3"] += float((pitch_top3_correct & replace_targets).sum())
        totals["kind_correct"] += float((kind_correct & valid_mask).sum())
        totals["replace_kind_correct"] += float((kind_correct & replace_targets).sum())
        totals["delete_kind_correct"] += float((kind_correct & delete_targets).sum())
        totals["tp"] += float((error_predictions & error_targets).sum())
        totals["fp"] += float((error_predictions & clean_targets).sum())
        totals["tn"] += float((clean_predictions & clean_targets).sum())
        totals["fn"] += float((clean_predictions & error_targets).sum())
        for threshold, stats in threshold_stats.items():
            threshold_predictions = (error_probabilities >= threshold) & valid_mask
            threshold_clean_predictions = (~threshold_predictions) & valid_mask
            stats["tp"] += float((threshold_predictions & error_targets).sum())
            stats["fp"] += float((threshold_predictions & clean_targets).sum())
            stats["fn"] += float((threshold_clean_predictions & error_targets).sum())
        totals["error_notes"] += float(error_targets.sum())
        totals["replace_notes"] += float(replace_targets.sum())
        totals["delete_notes"] += float(delete_targets.sum())
        totals["notes"] += float(mask.sum())

    notes = max(totals["notes"], 1.0)
    correction_notes = max(notes - totals["delete_notes"], 1.0)
    replace_notes = max(totals["replace_notes"], 1.0)
    delete_notes = max(totals["delete_notes"], 1.0)
    precision_denominator = max(totals["tp"] + totals["fp"], 1.0)
    recall_denominator = max(totals["tp"] + totals["fn"], 1.0)
    precision = totals["tp"] / precision_denominator
    recall = totals["tp"] / recall_denominator
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    f0_5 = f_beta(precision, recall, beta=0.5)
    best_threshold = det_threshold
    best_threshold_precision = precision
    best_threshold_recall = recall
    best_threshold_f1 = f1
    best_f0_5_threshold = det_threshold
    best_f0_5_precision = precision
    best_f0_5_recall = recall
    best_threshold_f0_5 = f0_5
    precision_constrained_threshold = det_threshold
    precision_constrained_precision = 0.0
    precision_constrained_recall = 0.0
    precision_constrained_f1 = 0.0
    for threshold, stats in threshold_stats.items():
        threshold_precision = stats["tp"] / max(stats["tp"] + stats["fp"], 1.0)
        threshold_recall = stats["tp"] / max(stats["tp"] + stats["fn"], 1.0)
        threshold_f1 = 2.0 * threshold_precision * threshold_recall / max(threshold_precision + threshold_recall, 1e-12)
        threshold_f0_5 = f_beta(threshold_precision, threshold_recall, beta=0.5)
        if threshold_f1 > best_threshold_f1:
            best_threshold = threshold
            best_threshold_precision = threshold_precision
            best_threshold_recall = threshold_recall
            best_threshold_f1 = threshold_f1
        if threshold_f0_5 > best_threshold_f0_5:
            best_f0_5_threshold = threshold
            best_f0_5_precision = threshold_precision
            best_f0_5_recall = threshold_recall
            best_threshold_f0_5 = threshold_f0_5
        if threshold_precision >= target_precision and (
            threshold_recall > precision_constrained_recall
            or (
                threshold_recall == precision_constrained_recall
                and threshold_f1 > precision_constrained_f1
            )
        ):
            precision_constrained_threshold = threshold
            precision_constrained_precision = threshold_precision
            precision_constrained_recall = threshold_recall
            precision_constrained_f1 = threshold_f1

    replace_kind_acc = totals["replace_kind_correct"] / replace_notes
    replace_pitch_top3 = totals["replace_pitch_top3"] / replace_notes
    task_score = 0.50 * best_threshold_f1 + 0.25 * replace_pitch_top3 + 0.25 * replace_kind_acc
    precision_task_score = 0.60 * best_threshold_f0_5 + 0.25 * replace_pitch_top3 + 0.15 * replace_kind_acc
    precision_shortfall = max(0.0, target_precision - best_f0_5_precision)
    precision_recall_score = (
        precision_constrained_recall
        + 0.20 * replace_pitch_top3
        + 0.10 * replace_kind_acc
        - 2.0 * precision_shortfall
    )
    return {
        "loss": totals["loss"] / notes,
        "det_loss": totals["det_loss"] / notes,
        "ranking_loss": totals["ranking_loss"] / notes,
        "masked_pitch_loss": totals["masked_pitch_loss"] / max(totals["masked_pitch_notes"], 1.0),
        "pitch_loss": totals["pitch_loss"] / correction_notes,
        "kind_loss": totals["kind_loss"] / notes,
        "pitch_acc": totals["pitch_correct"] / correction_notes,
        "replace_pitch_top1": totals["replace_pitch_top1"] / replace_notes,
        "replace_pitch_top3": replace_pitch_top3,
        "kind_acc": totals["kind_correct"] / notes,
        "replace_kind_acc": replace_kind_acc,
        "delete_kind_acc": totals["delete_kind_correct"] / delete_notes,
        "det_acc": (totals["tp"] + totals["tn"]) / notes,
        "det_precision": precision,
        "det_recall": recall,
        "det_f1": f1,
        "det_f0_5": f0_5,
        "best_det_threshold": best_threshold,
        "best_det_precision": best_threshold_precision,
        "best_det_recall": best_threshold_recall,
        "best_det_f1": best_threshold_f1,
        "best_det_f0_5_threshold": best_f0_5_threshold,
        "best_det_f0_5_precision": best_f0_5_precision,
        "best_det_f0_5_recall": best_f0_5_recall,
        "best_det_f0_5": best_threshold_f0_5,
        "precision_constrained_threshold": precision_constrained_threshold,
        "precision_constrained_precision": precision_constrained_precision,
        "precision_constrained_recall": precision_constrained_recall,
        "precision_constrained_f1": precision_constrained_f1,
        "task_score": task_score,
        "precision_task_score": precision_task_score,
        "precision_recall_score": precision_recall_score,
        "error_rate": totals["error_notes"] / notes,
        "replace_rate": totals["replace_notes"] / notes,
        "delete_rate": totals["delete_notes"] / notes,
        "det_threshold": det_threshold,
    }


def main() -> None:
    args = parse_args()
    curriculum_error_rate_stages = parse_error_rate_stages(args.curriculum_error_rate_stages)
    if not 0.0 <= args.fn_replay_fraction <= 1.0:
        raise ValueError("--fn-replay-fraction must be between 0 and 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device={device}", flush=True)
    initial_train_error_rate = 0.0 if args.clean_epochs > 0 else args.train_error_rate
    train_loader = make_loader(args, "train", shuffle=True, error_rate=initial_train_error_rate)
    # Held-out metrics are computed only on corrupted data, because the target
    # application is finding and correcting errors rather than scoring clean MIDI.
    test_loader = make_loader(args, args.eval_split, shuffle=False, error_rate=args.error_rate)

    model = build_wrong_note_model(
        model_type=args.model,
        input_size=FEATURE_SIZE,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        transformer_d_model=args.transformer_d_model,
        transformer_heads=args.transformer_heads,
        transformer_ffn_dim=args.transformer_ffn_dim,
        dropout=args.dropout,
    ).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        loaded_keys, partial_keys, skipped_keys = load_compatible_state_dict(model, checkpoint["model_state_dict"])
        print(
            f"loaded init checkpoint={args.init_checkpoint} "
            f"stage={checkpoint.get('stage')} epoch={checkpoint.get('epoch')}",
            flush=True,
        )
        if partial_keys or skipped_keys:
            print(
                f"checkpoint compatibility: loaded={len(loaded_keys)} partial={partial_keys} "
                f"skipped={skipped_keys}",
                flush=True,
            )
    args.input_size = FEATURE_SIZE
    print(f"model={args.model} parameters={sum(param.numel() for param in model.parameters()):,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler_mode = "min" if args.save_metric == "loss" else "max"
    scheduler = None
    if args.lr_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=args.lr_factor,
            patience=args.lr_patience,
            threshold=args.lr_threshold,
            threshold_mode="abs",
            min_lr=args.min_lr,
        )
    det_pos_weight = torch.tensor(args.det_pos_weight, device=device) if args.det_pos_weight > 0 else None
    kind_class_weight = torch.tensor(args.kind_class_weights, dtype=torch.float32, device=device)
    best_valid = float("inf") if args.save_metric == "loss" else -float("inf")
    print(
        f"loss weights: det_pos_weight={args.det_pos_weight}, "
        f"det_loss_weight={args.det_loss_weight}, pitch_loss_weight={args.pitch_loss_weight}, "
        f"kind_loss_weight={args.kind_loss_weight}, kind_class_weights={args.kind_class_weights}, "
        f"masked_pitch_loss_weight={args.masked_pitch_loss_weight}, masked_pitch_rate={args.masked_pitch_rate}, "
        f"clean_mask_batches_per_epoch={args.clean_mask_batches_per_epoch}, ranking_loss_weight={args.ranking_loss_weight}, "
        f"ranking_margin={args.ranking_margin}, ranking_top_k={args.ranking_top_k}, "
        f"hard_replay_size={args.hard_replay_size}, hard_replay_epochs={args.hard_replay_epochs}, "
        f"asymmetric_hard_replay={args.asymmetric_hard_replay}, fn_replay_fraction={args.fn_replay_fraction}, "
        f"fn_replay_weight={args.fn_replay_weight}, fp_replay_weight={args.fp_replay_weight}, "
        f"save_metric={args.save_metric}",
        flush=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save_if_best(stage: str, epoch: int, valid_metrics: dict[str, float]) -> bool:
        nonlocal best_valid
        current_score = valid_metrics[args.save_metric]
        improved = current_score < best_valid if args.save_metric == "loss" else current_score > best_valid
        if improved:
            best_valid = current_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "stage": stage,
                    "epoch": epoch,
                    "valid_metrics": valid_metrics,
                },
                output,
            )
            print(f"saved {output} ({args.save_metric}={current_score:.6f})", flush=True)
        return improved

    for epoch in range(1, args.clean_epochs + 1):
        train_loader.dataset.set_epoch(epoch)
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.det_loss_weight,
            args.pitch_loss_weight,
            args.kind_loss_weight,
            desc=f"clean {epoch}/{args.clean_epochs}",
            det_threshold=args.det_threshold,
            det_pos_weight=det_pos_weight,
            kind_class_weight=kind_class_weight,
            threshold_sweep=None,
            target_precision=args.target_precision,
            ranking_loss_weight=args.ranking_loss_weight,
            ranking_margin=args.ranking_margin,
            ranking_top_k=args.ranking_top_k,
            masked_pitch_loss_weight=args.masked_pitch_loss_weight,
            masked_pitch_rate=args.masked_pitch_rate,
        )
        print(f"stage=clean epoch={epoch}/{args.clean_epochs} train={train_metrics}", flush=True)

    train_error_rates = args.train_error_rates or [args.train_error_rate]
    calibration_error_rates = args.calibration_error_rates or train_error_rates
    first_phase_rates = curriculum_error_rate_stages[0] if curriculum_error_rate_stages else train_error_rates
    train_loader.dataset.error_rate = first_phase_rates[0]
    print(
        f"switching train loader to corrupted stage: train_error_rates={train_error_rates}, "
        f"calibration_epochs={args.calibration_epochs}, calibration_error_rates={calibration_error_rates}, "
        f"curriculum_error_rate_stages={curriculum_error_rate_stages}, "
        f"eval_error_rate={args.error_rate}, det_threshold={args.det_threshold}, target_precision={args.target_precision}",
        flush=True,
    )
    epochs_without_improvement = 0
    previous_hard_replay: list[dict[str, torch.Tensor | float | str]] = []
    for epoch in range(1, args.epochs + 1):
        if curriculum_error_rate_stages:
            phase, active_error_rates, active_epoch = select_curriculum_error_rates(
                curriculum_error_rate_stages,
                train_error_rates,
                epoch,
                args.epochs,
            )
        else:
            in_calibration = args.calibration_epochs > 0 and epoch > args.epochs - args.calibration_epochs
            active_error_rates = calibration_error_rates if in_calibration else train_error_rates
            active_epoch = epoch - (args.epochs - args.calibration_epochs) if in_calibration else epoch
            phase = "calibration" if in_calibration else "coverage"
        train_error_rate = active_error_rates[(active_epoch - 1) % len(active_error_rates)]
        train_loader.dataset.error_rate = train_error_rate
        print(
            f"corrupt epoch={epoch}/{args.epochs} phase={phase} active_rates={active_error_rates} "
            f"train_error_rate={train_error_rate}",
            flush=True,
        )
        if args.clean_mask_batches_per_epoch > 0 and args.masked_pitch_loss_weight > 0.0:
            train_loader.dataset.error_rate = 0.0
            train_loader.dataset.set_epoch(args.clean_epochs + args.epochs + epoch)
            clean_mask_metrics = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                0.0,
                0.0,
                0.0,
                desc=f"clean mask {epoch}/{args.epochs}",
                det_threshold=args.det_threshold,
                det_pos_weight=None,
                kind_class_weight=kind_class_weight,
                threshold_sweep=None,
                target_precision=args.target_precision,
                ranking_loss_weight=0.0,
                ranking_margin=args.ranking_margin,
                ranking_top_k=args.ranking_top_k,
                masked_pitch_loss_weight=args.masked_pitch_loss_weight,
                masked_pitch_rate=args.masked_pitch_rate,
                max_batches=args.clean_mask_batches_per_epoch,
            )
            print(
                f"stage=clean_mask epoch={epoch}/{args.epochs} batches={args.clean_mask_batches_per_epoch} "
                f"train={clean_mask_metrics}",
                flush=True,
            )
            train_loader.dataset.error_rate = train_error_rate
        if previous_hard_replay and args.hard_replay_epochs > 0:
            previous_hard_replay.sort(key=lambda sample: float(sample["score"]), reverse=True)
            replay_batches = list(make_replay_batches(previous_hard_replay, args.batch_size))
            for replay_epoch in range(1, args.hard_replay_epochs + 1):
                replay_metrics = run_epoch(
                    model,
                    replay_batches,
                    optimizer,
                    device,
                    args.det_loss_weight,
                    args.pitch_loss_weight,
                    args.kind_loss_weight,
                    desc=f"hard replay {epoch}/{args.epochs}.{replay_epoch}",
                    det_threshold=args.det_threshold,
                    det_pos_weight=det_pos_weight,
                    kind_class_weight=kind_class_weight,
                    threshold_sweep=None,
                    target_precision=args.target_precision,
                    ranking_loss_weight=args.ranking_loss_weight,
                    ranking_margin=args.ranking_margin,
                    ranking_top_k=args.ranking_top_k,
                    masked_pitch_loss_weight=0.0,
                )
                print(
                    f"stage=hard_replay epoch={epoch}/{args.epochs} replay_epoch={replay_epoch}/"
                    f"{args.hard_replay_epochs} samples={len(previous_hard_replay)} train={replay_metrics}",
                    flush=True,
                )
        train_loader.dataset.set_epoch(args.clean_epochs + epoch)
        current_hard_replay: list[dict[str, torch.Tensor | float | str]] = []
        current_fn_replay: list[dict[str, torch.Tensor | float | str]] = []
        current_fp_replay: list[dict[str, torch.Tensor | float | str]] = []
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.det_loss_weight,
            args.pitch_loss_weight,
            args.kind_loss_weight,
            desc=f"corrupt train {epoch}/{args.epochs}",
            det_threshold=args.det_threshold,
            det_pos_weight=det_pos_weight,
            kind_class_weight=kind_class_weight,
            threshold_sweep=None,
            target_precision=args.target_precision,
            ranking_loss_weight=args.ranking_loss_weight,
            ranking_margin=args.ranking_margin,
            ranking_top_k=args.ranking_top_k,
            hard_replay_buffer=current_hard_replay
            if args.hard_replay_size > 0 and not args.asymmetric_hard_replay
            else None,
            hard_replay_size=args.hard_replay_size,
            fn_replay_buffer=current_fn_replay
            if args.hard_replay_size > 0 and args.asymmetric_hard_replay
            else None,
            fp_replay_buffer=current_fp_replay
            if args.hard_replay_size > 0 and args.asymmetric_hard_replay
            else None,
            fn_replay_fraction=args.fn_replay_fraction,
            fn_replay_weight=args.fn_replay_weight,
            fp_replay_weight=args.fp_replay_weight,
            masked_pitch_loss_weight=0.0,
        )
        if args.asymmetric_hard_replay:
            previous_hard_replay = select_asymmetric_replay_samples(
                current_fn_replay,
                current_fp_replay,
                args.hard_replay_size,
                args.fn_replay_fraction,
            )
        else:
            current_hard_replay.sort(key=lambda sample: float(sample["score"]), reverse=True)
            previous_hard_replay = current_hard_replay[: args.hard_replay_size]
        if args.hard_replay_size > 0:
            if args.asymmetric_hard_replay:
                print(
                    f"collected hard replay windows={len(previous_hard_replay)} "
                    f"fn_candidates={len(current_fn_replay)} fp_candidates={len(current_fp_replay)} "
                    f"for next epoch",
                    flush=True,
                )
            else:
                print(f"collected hard replay windows={len(previous_hard_replay)} for next epoch", flush=True)
        test_loader.dataset.set_epoch(0)
        valid_metrics = run_epoch(
            model,
            test_loader,
            None,
            device,
            args.det_loss_weight,
            args.pitch_loss_weight,
            args.kind_loss_weight,
            desc=f"corrupt {args.eval_split} {epoch}/{args.epochs}",
            det_threshold=args.det_threshold,
            det_pos_weight=det_pos_weight,
            kind_class_weight=kind_class_weight,
            threshold_sweep=args.threshold_sweep,
            target_precision=args.target_precision,
            ranking_loss_weight=0.0,
            ranking_margin=args.ranking_margin,
            ranking_top_k=args.ranking_top_k,
            masked_pitch_loss_weight=0.0,
        )
        print(f"stage=corrupt epoch={epoch}/{args.epochs} train={train_metrics} {args.eval_split}={valid_metrics}", flush=True)
        improved = save_if_best("corrupt", epoch, valid_metrics)
        epochs_without_improvement = 0 if improved else epochs_without_improvement + 1
        if scheduler is not None:
            previous_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(valid_metrics[args.save_metric])
            current_lr = optimizer.param_groups[0]["lr"]
            if current_lr < previous_lr:
                print(
                    f"reduced learning rate: {previous_lr:.6g} -> {current_lr:.6g} "
                    f"after {args.lr_patience} plateau epochs on {args.save_metric}",
                    flush=True,
                )
        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"early stopping after {epochs_without_improvement} corrupt epochs without "
                f"{args.save_metric} improvement",
                flush=True,
            )
            break


if __name__ == "__main__":
    main()
