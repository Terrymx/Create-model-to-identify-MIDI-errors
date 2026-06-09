from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from midi_error_detector.data import FEATURE_SIZE, MaestroWrongNoteDataset
from midi_error_detector.model import build_wrong_note_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one-sided clean-music pitch likelihood models.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--direction", required=True, choices=["forward", "backward"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def directional_batch(
    features: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    direction: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if direction == "backward":
        features = features.flip(1)
        targets = targets.flip(1)
        mask = mask.flip(1)
    shifted = torch.zeros_like(features)
    shifted[:, 1:] = features[:, :-1]
    prediction_mask = mask.clone()
    prediction_mask[:, 0] = False
    return shifted, targets, prediction_mask


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    direction: str,
    optimizer: torch.optim.Optimizer | None,
    description: str,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    notes = 0
    for batch in tqdm(loader, desc=description, unit="batch", dynamic_ncols=True):
        features = batch["features"].to(device)
        targets = batch["target_pitch"].to(device)
        mask = batch["mask"].to(device).bool()
        shifted, targets, prediction_mask = directional_batch(features, targets, mask, direction)
        with torch.set_grad_enabled(training):
            logits = model.predict_pitch(shifted, causal=True)
            per_note_loss = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2),
                targets,
                reduction="none",
            )
            loss = (per_note_loss * prediction_mask).sum() / prediction_mask.sum().clamp_min(1)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        count = int(prediction_mask.sum())
        loss_sum += float((per_note_loss * prediction_mask).sum())
        correct += int(((logits.argmax(-1) == targets) & prediction_mask).sum())
        notes += count
    return loss_sum / max(notes, 1), correct / max(notes, 1)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} direction={args.direction}", flush=True)
    train_dataset = MaestroWrongNoteDataset(
        root=args.data_root,
        split="train",
        window_size=args.window_size,
        stride=args.stride,
        error_rate=0.0,
        max_files=args.max_files,
        cache_notes=True,
        verbose=True,
    )
    validation_dataset = MaestroWrongNoteDataset(
        root=args.data_root,
        split="validation",
        window_size=args.window_size,
        stride=args.stride,
        error_rate=0.0,
        max_files=args.max_files,
        cache_notes=True,
        verbose=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = build_wrong_note_model(
        model_type="transformer",
        input_size=FEATURE_SIZE,
        num_layers=args.num_layers,
        transformer_d_model=args.d_model,
        transformer_heads=args.heads,
        transformer_ffn_dim=args.ffn_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_loss = float("inf")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            device,
            args.direction,
            optimizer,
            f"{args.direction} train {epoch}/{args.epochs}",
        )
        validation_dataset.set_epoch(0)
        validation_loss, validation_accuracy = run_epoch(
            model,
            validation_loader,
            device,
            args.direction,
            None,
            f"{args.direction} validation {epoch}/{args.epochs}",
        )
        print(
            f"epoch={epoch}/{args.epochs} train_loss={train_loss:.6f} "
            f"train_acc={train_accuracy:.6f} validation_loss={validation_loss:.6f} "
            f"validation_perplexity={torch.exp(torch.tensor(validation_loss)).item():.6f} "
            f"validation_acc={validation_accuracy:.6f}",
            flush=True,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args) | {"model": "transformer", "input_size": FEATURE_SIZE},
                    "stage": f"{args.direction}_likelihood",
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                },
                output,
            )
            print(f"saved {output} validation_loss={validation_loss:.6f}", flush=True)


if __name__ == "__main__":
    main()
