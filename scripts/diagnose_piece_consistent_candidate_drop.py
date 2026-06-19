from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from midi_error_detector.data import KIND_TO_ID, MaestroWrongNoteDataset, note_features
from run_frozen_union_candidate_context_verifier import (
    IndexedSubset,
    detector_signals,
    load_any_model,
)
from voice_aware_dataset import PieceConsistentVoiceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threeclass-checkpoint", required=True)
    parser.add_argument("--binary-checkpoint", required=True)
    parser.add_argument("--forward-checkpoint", required=True)
    parser.add_argument("--backward-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-test-files", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--surprise-eval-groups", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = (
        *load_any_model(args.threeclass_checkpoint, device),
        *load_any_model(args.binary_checkpoint, device),
        *load_any_model(args.forward_checkpoint, device),
        *load_any_model(args.backward_checkpoint, device),
    )
    three_model, three_args, binary_model, binary_args, forward_model, forward_args, backward_model, backward_args = models
    dataset = PieceConsistentVoiceDataset(
        root=args.data_root,
        split="test",
        voice_method="onset_matching",
        window_size=args.window_size,
        stride=args.stride,
        error_rate=0.01,
        seed=args.seed,
        max_files=args.max_test_files,
        verbose=True,
    )
    loader = DataLoader(
        IndexedSubset(dataset, list(range(len(dataset)))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    modes = {
        "piece_features": Counter(),
        "window_features": Counter(),
        "legacy_window_corruption": Counter(),
    }
    feature_abs_sum = np.zeros(36, dtype=np.float64)
    feature_count = 0
    kind_names = {value: key for key, value in KIND_TO_ID.items()}

    for batch in tqdm(loader, desc="candidate drop diagnostic", unit="batch"):
        mask = batch["mask"].to(device).bool()
        labels = batch["is_error"].to(device).bool()
        kinds = batch["error_kind"].to(device)
        piece_features = batch["features"].to(device)
        batch_size = piece_features.shape[0]
        window_rows = []
        for row in range(batch_size):
            file_id = int(batch["__file_id"][row])
            start = int(batch["__window_start"][row])
            length = int(mask[row].sum())
            notes = dataset._pieces[file_id].notes[start : start + length]
            values = note_features(notes)
            if length < args.window_size:
                values = np.pad(values, ((0, args.window_size - length), (0, 0)))
            window_rows.append(torch.from_numpy(values))
        window_features = torch.stack(window_rows).to(device)
        valid_cpu = mask.cpu().numpy()
        difference = (piece_features - window_features).abs().cpu().numpy()
        feature_abs_sum += (difference * valid_cpu[..., None]).sum(axis=(0, 1))
        feature_count += int(valid_cpu.sum())

        for mode_name, features in (
            ("piece_features", piece_features),
            ("window_features", window_features),
        ):
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
            candidates = valid & (
                (three["probability"] >= 0.60)
                | (binary["probability"] >= 0.50)
            )
            stats = modes[mode_name]
            stats["notes"] += int(valid.sum())
            stats["errors"] += int((labels & valid).sum())
            stats["candidate_errors"] += int((labels & candidates).sum())
            stats["candidates"] += int(candidates.sum())
            for kind_id, kind_name in kind_names.items():
                kind_mask = labels & valid & (kinds == kind_id)
                stats[f"errors_{kind_name}"] += int(kind_mask.sum())
                stats[f"candidate_errors_{kind_name}"] += int((kind_mask & candidates).sum())

    legacy_dataset = MaestroWrongNoteDataset(
        root=args.data_root,
        split="test",
        window_size=args.window_size,
        stride=args.stride,
        error_rate=0.01,
        seed=13,
        max_files=args.max_test_files,
        cache_notes=True,
        verbose=True,
    )
    legacy_loader = DataLoader(
        legacy_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    legacy_stats = modes["legacy_window_corruption"]
    for batch in tqdm(legacy_loader, desc="legacy window diagnostic", unit="batch"):
        features = batch["features"].to(device)
        mask = batch["mask"].to(device).bool()
        labels = batch["is_error"].to(device).bool()
        kinds = batch["error_kind"].to(device)
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
        candidates = valid & (
            (three["probability"] >= 0.60)
            | (binary["probability"] >= 0.50)
        )
        legacy_stats["notes"] += int(valid.sum())
        legacy_stats["errors"] += int((labels & valid).sum())
        legacy_stats["candidate_errors"] += int((labels & candidates).sum())
        legacy_stats["candidates"] += int(candidates.sum())
        for kind_id, kind_name in kind_names.items():
            kind_mask = labels & valid & (kinds == kind_id)
            legacy_stats[f"errors_{kind_name}"] += int(kind_mask.sum())
            legacy_stats[f"candidate_errors_{kind_name}"] += int((kind_mask & candidates).sum())

    result = {
        "modes": {},
        "mean_absolute_feature_drift": (feature_abs_sum / max(feature_count, 1)).tolist(),
    }
    for mode_name, stats in modes.items():
        row = dict(stats)
        row["candidate_recall"] = row["candidate_errors"] / max(row["errors"], 1)
        for kind_name in kind_names.values():
            row[f"candidate_recall_{kind_name}"] = row[f"candidate_errors_{kind_name}"] / max(
                row[f"errors_{kind_name}"],
                1,
            )
        result["modes"][mode_name] = row
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
