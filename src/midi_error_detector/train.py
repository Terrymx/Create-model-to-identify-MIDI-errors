"""Train a BiGRU model on MAESTRO MIDI with synthetic wrong-note corruption."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import FEATURE_SIZE, MaestroWrongNoteDataset
from .model import build_wrong_note_model, masked_bce_with_logits, masked_kind_loss, masked_pitch_loss


def f_beta(precision: float, recall: float, beta: float) -> float:
    beta_squared = beta * beta
    return (1.0 + beta_squared) * precision * recall / max(beta_squared * precision + recall, 1e-12)


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
        "--det-pos-weight",
        type=float,
        default=3.0,
        help="Positive-class weight for wrong-note detection BCE; increase if recall is too low.",
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
        "--save-metric",
        default="task_score",
        choices=[
            "task_score",
            "precision_task_score",
            "det_f1",
            "det_f0_5",
            "best_det_f1",
            "best_det_f0_5",
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
    pitch_loss_weight: float,
    kind_loss_weight: float,
    desc: str,
    det_threshold: float,
    det_pos_weight: torch.Tensor | None,
    kind_class_weight: torch.Tensor | None,
    threshold_sweep: list[float] | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "det_loss": 0.0,
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

    for batch in tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True, leave=True):
        features = batch["features"].to(device)
        is_error = batch["is_error"].to(device)
        target_pitch = batch["target_pitch"].to(device)
        error_kind = batch["error_kind"].to(device)
        mask = batch["mask"].to(device)

        with torch.set_grad_enabled(training):
            outputs = model(features)
            det_loss = masked_bce_with_logits(outputs["error_logits"], is_error, mask, pos_weight=det_pos_weight)
            pitch_mask = mask * (error_kind != 2).float()
            pitch_loss = masked_pitch_loss(outputs["pitch_logits"], target_pitch, pitch_mask)
            kind_loss = masked_kind_loss(outputs["kind_logits"], error_kind, mask, class_weight=kind_class_weight)
            loss = det_loss + pitch_loss_weight * pitch_loss + kind_loss_weight * kind_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

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

    replace_kind_acc = totals["replace_kind_correct"] / replace_notes
    replace_pitch_top3 = totals["replace_pitch_top3"] / replace_notes
    task_score = 0.50 * best_threshold_f1 + 0.25 * replace_pitch_top3 + 0.25 * replace_kind_acc
    precision_task_score = 0.60 * best_threshold_f0_5 + 0.25 * replace_pitch_top3 + 0.15 * replace_kind_acc
    return {
        "loss": totals["loss"] / notes,
        "det_loss": totals["det_loss"] / notes,
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
        "task_score": task_score,
        "precision_task_score": precision_task_score,
        "error_rate": totals["error_notes"] / notes,
        "replace_rate": totals["replace_notes"] / notes,
        "delete_rate": totals["delete_notes"] / notes,
        "det_threshold": det_threshold,
    }


def main() -> None:
    args = parse_args()
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
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"loaded init checkpoint={args.init_checkpoint} "
            f"stage={checkpoint.get('stage')} epoch={checkpoint.get('epoch')}",
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
        f"kind_class_weights={args.kind_class_weights}, save_metric={args.save_metric}",
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
            args.pitch_loss_weight,
            args.kind_loss_weight,
            desc=f"clean {epoch}/{args.clean_epochs}",
            det_threshold=args.det_threshold,
            det_pos_weight=det_pos_weight,
            kind_class_weight=kind_class_weight,
            threshold_sweep=None,
        )
        print(f"stage=clean epoch={epoch}/{args.clean_epochs} train={train_metrics}", flush=True)

    train_error_rates = args.train_error_rates or [args.train_error_rate]
    train_loader.dataset.error_rate = train_error_rates[0]
    print(
        f"switching train loader to corrupted stage: train_error_rates={train_error_rates}, "
        f"eval_error_rate={args.error_rate}, det_threshold={args.det_threshold}",
        flush=True,
    )
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_error_rate = train_error_rates[(epoch - 1) % len(train_error_rates)]
        train_loader.dataset.error_rate = train_error_rate
        print(f"corrupt epoch={epoch}/{args.epochs} train_error_rate={train_error_rate}", flush=True)
        train_loader.dataset.set_epoch(args.clean_epochs + epoch)
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.pitch_loss_weight,
            args.kind_loss_weight,
            desc=f"corrupt train {epoch}/{args.epochs}",
            det_threshold=args.det_threshold,
            det_pos_weight=det_pos_weight,
            kind_class_weight=kind_class_weight,
            threshold_sweep=None,
        )
        test_loader.dataset.set_epoch(0)
        valid_metrics = run_epoch(
            model,
            test_loader,
            None,
            device,
            args.pitch_loss_weight,
            args.kind_loss_weight,
            desc=f"corrupt {args.eval_split} {epoch}/{args.epochs}",
            det_threshold=args.det_threshold,
            det_pos_weight=det_pos_weight,
            kind_class_weight=kind_class_weight,
            threshold_sweep=args.threshold_sweep,
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
