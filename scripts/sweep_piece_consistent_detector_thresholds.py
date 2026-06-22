from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def canonical_note_rows(
    file_ids: np.ndarray,
    positions: np.ndarray,
    local_positions: np.ndarray,
    window_size: int,
) -> np.ndarray:
    best: dict[tuple[int, int], tuple[int, int]] = {}
    for row, (file_id, position, local_position) in enumerate(
        zip(file_ids, positions, local_positions)
    ):
        margin = min(int(local_position), window_size - 1 - int(local_position))
        key = (int(file_id), int(position))
        rank = (margin, -row)
        if key not in best or rank > best[key]:
            best[key] = rank
    selected = []
    for file_id, position in sorted(best):
        matches = np.flatnonzero(
            (file_ids == file_id) & (positions == position)
        )
        selected.append(
            max(
                matches.tolist(),
                key=lambda row: (
                    min(
                        int(local_positions[row]),
                        window_size - 1 - int(local_positions[row]),
                    ),
                    -row,
                ),
            )
        )
    return np.asarray(selected, dtype=np.int64)


def union_candidate_row(
    labels: np.ndarray,
    three_scores: np.ndarray,
    binary_scores: np.ndarray,
    *,
    three_threshold: float,
    binary_threshold: float,
) -> dict[str, float | int]:
    candidates = (three_scores >= three_threshold) | (
        binary_scores >= binary_threshold
    )
    positive = labels.astype(bool)
    candidate_errors = int((candidates & positive).sum())
    candidate_count = int(candidates.sum())
    error_count = int(positive.sum())
    return {
        "three_threshold": float(three_threshold),
        "binary_threshold": float(binary_threshold),
        "candidates": candidate_count,
        "candidate_errors": candidate_errors,
        "candidate_precision": candidate_errors / max(candidate_count, 1),
        "candidate_recall": candidate_errors / max(error_count, 1),
        "candidate_rate": candidate_count / max(len(labels), 1),
    }


def _threshold_values() -> np.ndarray:
    return np.unique(
        np.concatenate(
            [
                np.linspace(0.05, 0.50, 10),
                np.linspace(0.55, 0.90, 15),
                np.asarray([0.95, 0.97, 0.99]),
            ]
        )
    )


def build_sweep(
    labels: np.ndarray,
    three_scores: np.ndarray,
    binary_scores: np.ndarray,
) -> dict:
    thresholds = _threshold_values()
    union_rows = [
        union_candidate_row(
            labels,
            three_scores,
            binary_scores,
            three_threshold=float(three_threshold),
            binary_threshold=float(binary_threshold),
        )
        for three_threshold in thresholds
        for binary_threshold in thresholds
    ]
    return {
        "notes": int(len(labels)),
        "errors": int(labels.astype(bool).sum()),
        "current": union_candidate_row(
            labels,
            three_scores,
            binary_scores,
            three_threshold=0.60,
            binary_threshold=0.50,
        ),
        "three_only": [
            union_candidate_row(
                labels,
                three_scores,
                np.full_like(binary_scores, -1.0),
                three_threshold=float(threshold),
                binary_threshold=1.1,
            )
            for threshold in thresholds
        ],
        "binary_only": [
            union_candidate_row(
                labels,
                np.full_like(three_scores, -1.0),
                binary_scores,
                three_threshold=1.1,
                binary_threshold=float(threshold),
            )
            for threshold in thresholds
        ],
        "union": union_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threeclass-checkpoint", required=True)
    parser.add_argument("--binary-checkpoint", required=True)
    parser.add_argument("--forward-checkpoint", required=True)
    parser.add_argument("--backward-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--surprise-eval-groups", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from run_frozen_union_candidate_context_verifier import (
        IndexedSubset,
        detector_signals,
        load_any_model,
    )
    from voice_aware_dataset import PieceConsistentVoiceDataset

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = (
        *load_any_model(args.threeclass_checkpoint, device),
        *load_any_model(args.binary_checkpoint, device),
        *load_any_model(args.forward_checkpoint, device),
        *load_any_model(args.backward_checkpoint, device),
    )
    (
        three_model,
        three_args,
        binary_model,
        binary_args,
        forward_model,
        forward_args,
        backward_model,
        backward_args,
    ) = models
    dataset = PieceConsistentVoiceDataset(
        root=args.data_root,
        split=args.split,
        voice_method="onset_matching",
        window_size=args.window_size,
        stride=args.stride,
        error_rate=args.error_rate,
        seed=args.seed,
        max_files=args.max_files,
        verbose=True,
    )
    loader = DataLoader(
        IndexedSubset(dataset, list(range(len(dataset)))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    rows: dict[str, list[np.ndarray]] = {
        "labels": [],
        "three_scores": [],
        "binary_scores": [],
        "file_ids": [],
        "positions": [],
        "local_positions": [],
    }
    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc=f"detector sweep {args.split}",
            unit="batch",
            dynamic_ncols=True,
        ):
            features = batch["features"].to(device)
            mask = batch["mask"].to(device).bool()
            three = detector_signals(
                three_model,
                three_args,
                forward_model,
                forward_args,
                backward_model,
                backward_args,
                features,
                mask,
                args.surprise_eval_groups,
            )
            binary = detector_signals(
                binary_model,
                binary_args,
                forward_model,
                forward_args,
                backward_model,
                backward_args,
                features,
                mask,
                args.surprise_eval_groups,
            )
            valid = mask & three["available"] & binary["available"]
            batch_size, length = valid.shape
            file_ids = batch["__file_id"].view(batch_size, 1).expand(
                batch_size, length
            )
            starts = batch["__window_start"].view(batch_size, 1).expand(
                batch_size, length
            )
            local = torch.arange(length).view(1, length).expand(
                batch_size, length
            )
            valid_cpu = valid.cpu()
            rows["labels"].append(batch["is_error"][valid_cpu].numpy())
            rows["three_scores"].append(
                three["probability"][valid].cpu().numpy()
            )
            rows["binary_scores"].append(
                binary["probability"][valid].cpu().numpy()
            )
            rows["file_ids"].append(file_ids[valid_cpu].numpy())
            rows["positions"].append((starts + local)[valid_cpu].numpy())
            rows["local_positions"].append(local[valid_cpu].numpy())
    arrays = {
        name: np.concatenate(parts)
        for name, parts in rows.items()
    }
    canonical = canonical_note_rows(
        arrays["file_ids"],
        arrays["positions"],
        arrays["local_positions"],
        args.window_size,
    )
    result = build_sweep(
        arrays["labels"][canonical],
        arrays["three_scores"][canonical],
        arrays["binary_scores"][canonical],
    )
    result["protocol"] = "piece_consistent_unique_note"
    result["split"] = args.split
    result["window_rows"] = int(len(arrays["labels"]))
    result["unique_rows"] = int(len(canonical))
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
