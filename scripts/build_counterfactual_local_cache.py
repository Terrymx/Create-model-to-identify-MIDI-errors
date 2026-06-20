from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from build_counterfactual_candidate_cache import (
    load_candidate_cache,
    save_candidate_cache,
)
from counterfactual_edit_features import (
    apply_observed_pitch_edit,
    directional_pitch_distribution,
    local_edit_impact_features,
)
from midi_error_detector.data import MaestroWrongNoteDataset
from run_frozen_union_candidate_context_verifier import load_any_model


def top_proposal_indices(scores: np.ndarray, count: int = 2) -> np.ndarray:
    if scores.ndim != 2:
        raise ValueError("Proposal scores must have shape [candidates, proposals].")
    if count <= 0 or count > scores.shape[1]:
        raise ValueError("Requested proposal count is out of range.")
    return np.argsort(-scores, axis=1, kind="mergesort")[:, :count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--forward-checkpoint", required=True)
    parser.add_argument("--backward-checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--max-validation-files", type=int, default=None)
    parser.add_argument("--max-test-files", type=int, default=None)
    return parser.parse_args()


def make_dataset(args: argparse.Namespace, split: str):
    dataset = MaestroWrongNoteDataset(
        root=args.data_root,
        split=split,
        window_size=args.window_size,
        stride=args.stride,
        error_rate=args.error_rate,
        seed=13,
        max_files=(
            args.max_validation_files
            if split == "validation"
            else args.max_test_files
        ),
        cache_notes=True,
        verbose=True,
    )
    dataset.set_epoch(0)
    return dataset


def observed_log_probability(
    probability: torch.Tensor,
    raw_features: torch.Tensor,
) -> torch.Tensor:
    pitch = torch.round(raw_features[..., 0] * 127.0).long().clamp(0, 127)
    return probability.gather(-1, pitch.unsqueeze(-1)).squeeze(-1).clamp_min(
        1e-9
    ).log()


@torch.no_grad()
def augment_split(
    input_path: Path,
    output_path: Path,
    dataset,
    forward_model,
    forward_args,
    backward_model,
    backward_args,
    device: torch.device,
    batch_size: int,
) -> None:
    arrays, metadata = load_candidate_cache(input_path)
    selected_slots = top_proposal_indices(arrays["b_ranking"], count=2)
    candidate_rows = np.repeat(np.arange(len(selected_slots)), 2)
    proposal_slots = selected_slots.reshape(-1)
    feature_rows = []
    for start in tqdm(
        range(0, len(candidate_rows), batch_size),
        desc=f"local C {metadata['split']}",
        unit="batch",
        dynamic_ncols=True,
    ):
        row_ids = candidate_rows[start : start + batch_size]
        slots = proposal_slots[start : start + batch_size]
        samples = [
            dataset[int(arrays["dataset_indices"][row_id])]
            for row_id in row_ids
        ]
        raw = torch.stack([sample["features"] for sample in samples]).to(device)
        mask = torch.stack([sample["mask"] for sample in samples]).to(device).bool()
        positions = torch.from_numpy(
            arrays["local_positions"][row_ids].astype(np.int64)
        ).to(device)
        proposed = torch.from_numpy(
            arrays["proposals"][row_ids, slots].astype(np.int64)
        ).to(device)
        edited = apply_observed_pitch_edit(raw, positions, proposed)
        original_forward, _ = directional_pitch_distribution(
            forward_model,
            raw,
            mask,
            "forward",
            forward_args.safe_feature_columns,
        )
        original_backward, _ = directional_pitch_distribution(
            backward_model,
            raw,
            mask,
            "backward",
            backward_args.safe_feature_columns,
        )
        edited_forward, _ = directional_pitch_distribution(
            forward_model,
            edited,
            mask,
            "forward",
            forward_args.safe_feature_columns,
        )
        edited_backward, _ = directional_pitch_distribution(
            backward_model,
            edited,
            mask,
            "backward",
            backward_args.safe_feature_columns,
        )
        feature_rows.append(
            local_edit_impact_features(
                observed_log_probability(original_forward, raw),
                observed_log_probability(original_backward, raw),
                observed_log_probability(edited_forward, edited),
                observed_log_probability(edited_backward, edited),
                positions,
                mask,
                radii=(4, 8, 16),
            ).cpu()
        )
    arrays["c_features"] = (
        torch.cat(feature_rows)
        .reshape(len(selected_slots), 2, -1)
        .numpy()
        .astype(np.float32)
    )
    arrays["c_proposal_slots"] = selected_slots.astype(np.int64)
    arrays["c_proposals"] = np.take_along_axis(
        arrays["proposals"],
        selected_slots,
        axis=1,
    )
    arrays["c_ranking"] = np.take_along_axis(
        arrays["b_ranking"],
        selected_slots,
        axis=1,
    )
    metadata["c_radii"] = [4, 8, 16]
    metadata["c_proposals_per_candidate"] = 2
    save_candidate_cache(output_path, arrays, metadata)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forward_model, forward_args = load_any_model(args.forward_checkpoint, device)
    backward_model, backward_args = load_any_model(args.backward_checkpoint, device)
    validation = make_dataset(args, "validation")
    test = make_dataset(args, "test")
    input_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, dataset in [
        ("train", validation),
        ("calibration", validation),
        ("test", test),
    ]:
        augment_split(
            input_dir / f"{split_name}.npz",
            output_dir / f"{split_name}.npz",
            dataset,
            forward_model,
            forward_args,
            backward_model,
            backward_args,
            device,
            args.batch_size,
        )


if __name__ == "__main__":
    main()
