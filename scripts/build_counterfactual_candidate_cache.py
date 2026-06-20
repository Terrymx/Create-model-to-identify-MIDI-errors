from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from counterfactual_edit_features import (
    build_replacement_proposals,
    counterfactual_target_features,
    directional_pitch_distribution,
)
from run_frozen_union_candidate_context_verifier import (
    detector_signals,
    load_any_model,
)
from run_union_candidate_verifier import split_indices_by_file
from run_verifier_improvement_suite import collect_block, make_dataset


def aggregate_proposal_features(
    proposal_features: np.ndarray,
    ranking_scores: np.ndarray,
) -> np.ndarray:
    if proposal_features.ndim != 3:
        raise ValueError("Proposal features must have shape [candidates, proposals, features].")
    if ranking_scores.shape != proposal_features.shape[:2]:
        raise ValueError("Ranking scores must align with candidate/proposal rows.")
    flattened = proposal_features.reshape(len(proposal_features), -1)
    maximum = proposal_features.max(axis=1)
    mean = proposal_features.mean(axis=1)
    best_index = ranking_scores.argmax(axis=1)
    best = proposal_features[np.arange(len(proposal_features)), best_index]
    return np.concatenate([flattened, maximum, mean, best], axis=1).astype(np.float32)


def save_candidate_cache(
    path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
        __metadata__=np.asarray(json.dumps(metadata), dtype=np.str_),
    )


def load_candidate_cache(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as loaded:
        metadata = json.loads(str(loaded["__metadata__"]))
        arrays = {
            name: loaded[name].copy()
            for name in loaded.files
            if name != "__metadata__"
        }
    return arrays, metadata


class CounterfactualIndexedSubset(Dataset):
    def __init__(self, dataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        dataset_index = self.indices[index]
        sample = self.dataset[dataset_index]
        file_id, window_start = self.dataset.index[dataset_index]
        sample["__dataset_index"] = torch.tensor(dataset_index, dtype=torch.long)
        sample["__file_id"] = torch.tensor(file_id, dtype=torch.long)
        sample["__window_start"] = torch.tensor(window_start, dtype=torch.long)
        sample["__file_note_count"] = torch.tensor(
            self.dataset._note_counts[file_id],
            dtype=torch.long,
        )
        sample["voice_features"] = torch.zeros(
            self.dataset.window_size,
            18,
            dtype=torch.float32,
        )
        return sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threeclass-checkpoint", required=True)
    parser.add_argument("--binary-checkpoint", required=True)
    parser.add_argument("--forward-checkpoint", required=True)
    parser.add_argument("--backward-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-validation-files", type=int, default=None)
    parser.add_argument("--max-test-files", type=int, default=None)
    return parser.parse_args()


def make_counterfactual_loader(
    dataset,
    indices: list[int],
    batch_size: int,
) -> DataLoader:
    return DataLoader(
        CounterfactualIndexedSubset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def collect_counterfactual_arrays(
    models: tuple,
    dataset,
    indices: list[int],
    device: torch.device,
    batch_size: int,
    description: str,
) -> dict[str, np.ndarray]:
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
    rows: dict[str, list[torch.Tensor]] = {
        "labels": [],
        "target_pitch": [],
        "error_kind": [],
        "observed_pitch": [],
        "proposals": [],
        "b_features": [],
        "b_ranking": [],
        "file_ids": [],
        "positions": [],
        "dataset_indices": [],
        "local_positions": [],
    }
    for batch in tqdm(
        make_counterfactual_loader(dataset, indices, batch_size),
        desc=description,
        unit="batch",
        dynamic_ncols=True,
    ):
        raw = batch["features"].to(device)
        mask = batch["mask"].to(device).bool()
        labels = batch["is_error"].to(device)
        target_pitch = batch["target_pitch"].to(device)
        error_kind = batch["error_kind"].to(device)
        three = detector_signals(
            three_model,
            three_args,
            forward_model,
            forward_args,
            backward_model,
            backward_args,
            raw,
            mask,
            4,
        )
        binary = detector_signals(
            binary_model,
            binary_args,
            forward_model,
            forward_args,
            backward_model,
            backward_args,
            raw,
            mask,
            4,
        )
        forward_probability, forward_available = directional_pitch_distribution(
            forward_model,
            raw,
            mask,
            "forward",
            forward_args.safe_feature_columns,
        )
        backward_probability, backward_available = directional_pitch_distribution(
            backward_model,
            raw,
            mask,
            "backward",
            backward_args.safe_feature_columns,
        )
        valid = (
            mask
            & three["available"]
            & binary["available"]
            & forward_available
            & backward_available
        )
        candidate_mask = valid & (
            (three["probability"] >= 0.60)
            | (binary["probability"] >= 0.50)
        )
        if not bool(candidate_mask.any()):
            continue
        observed_pitch = torch.round(raw[..., 0] * 127.0).long().clamp(0, 127)
        candidate_observed = observed_pitch[candidate_mask]
        candidate_detector = three["pitch_distribution"][candidate_mask]
        candidate_forward = forward_probability[candidate_mask]
        candidate_backward = backward_probability[candidate_mask]
        proposals = build_replacement_proposals(
            candidate_detector,
            candidate_forward,
            candidate_backward,
            candidate_observed,
            source_top_k=3,
            max_proposals=4,
        )
        proposal_count = proposals.shape[1]
        b_features = counterfactual_target_features(
            candidate_forward.repeat_interleave(proposal_count, dim=0),
            candidate_backward.repeat_interleave(proposal_count, dim=0),
            candidate_observed.repeat_interleave(proposal_count),
            proposals.reshape(-1),
        ).reshape(len(proposals), proposal_count, -1)
        b_ranking = b_features[..., 2]

        batch_size_value, length = candidate_mask.shape
        file_ids = batch["__file_id"].view(batch_size_value, 1).expand(
            batch_size_value, length
        )
        window_starts = batch["__window_start"].view(batch_size_value, 1).expand(
            batch_size_value, length
        )
        dataset_indices = batch["__dataset_index"].view(
            batch_size_value, 1
        ).expand(batch_size_value, length)
        local_positions = torch.arange(length).view(1, length).expand(
            batch_size_value, length
        )
        positions = window_starts + local_positions
        candidate_mask_cpu = candidate_mask.cpu()
        rows["labels"].append(labels[candidate_mask].cpu())
        rows["target_pitch"].append(target_pitch[candidate_mask].cpu())
        rows["error_kind"].append(error_kind[candidate_mask].cpu())
        rows["observed_pitch"].append(candidate_observed.cpu())
        rows["proposals"].append(proposals.cpu())
        rows["b_features"].append(b_features.cpu())
        rows["b_ranking"].append(b_ranking.cpu())
        rows["file_ids"].append(file_ids[candidate_mask_cpu].cpu())
        rows["positions"].append(positions[candidate_mask_cpu].cpu())
        rows["dataset_indices"].append(dataset_indices[candidate_mask_cpu].cpu())
        rows["local_positions"].append(local_positions[candidate_mask_cpu].cpu())
    if not rows["labels"]:
        raise RuntimeError(f"No candidates collected for {description}.")
    return {
        name: torch.cat(parts).numpy()
        for name, parts in rows.items()
    }


def build_and_save_split(
    models: tuple,
    dataset,
    indices: list[int],
    device: torch.device,
    args: argparse.Namespace,
    split_name: str,
) -> dict:
    block = collect_block(
        models,
        dataset,
        indices,
        device,
        args,
        f"collect base {split_name}",
    )
    arrays = collect_counterfactual_arrays(
        models,
        dataset,
        indices,
        device,
        args.batch_size,
        f"collect counterfactual {split_name}",
    )
    if len(block.labels) != len(arrays["labels"]):
        raise RuntimeError("Base and counterfactual candidate counts differ.")
    np.testing.assert_array_equal(block.labels.numpy(), arrays["labels"])
    np.testing.assert_array_equal(block.file_ids.numpy(), arrays["file_ids"])
    arrays["base_features"] = block.old.numpy()
    arrays["aggregated_b_features"] = aggregate_proposal_features(
        arrays["b_features"],
        arrays["b_ranking"],
    )
    metadata = {
        "split": split_name,
        "candidate_count": len(block.labels),
        "stats": block.stats,
        "threeclass_candidate_threshold": 0.60,
        "binary_candidate_threshold": 0.50,
        "pitch_source": "post_corruption_observed_pitch",
        "proposal_sources": ["threeclass_correction", "forward", "backward"],
    }
    save_candidate_cache(
        Path(args.output_dir) / f"{split_name}.npz",
        arrays,
        metadata,
    )
    return metadata


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = (
        *load_any_model(args.threeclass_checkpoint, device),
        *load_any_model(args.binary_checkpoint, device),
        *load_any_model(args.forward_checkpoint, device),
        *load_any_model(args.backward_checkpoint, device),
    )
    validation = make_dataset(args, "validation", args.max_validation_files)
    train_indices, calibration_indices, train_files, calibration_files = (
        split_indices_by_file(validation, 0.25, args.seed)
    )
    test = make_dataset(args, "test", args.max_test_files)
    results = {
        "train": build_and_save_split(
            models, validation, train_indices, device, args, "train"
        ),
        "calibration": build_and_save_split(
            models, validation, calibration_indices, device, args, "calibration"
        ),
        "test": build_and_save_split(
            models, test, list(range(len(test))), device, args, "test"
        ),
        "train_files": train_files,
        "calibration_files": calibration_files,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
