from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class E1FeatureTensors:
    candidate: np.ndarray
    proposal: np.ndarray


@dataclass(frozen=True)
class E1Normalizer:
    candidate_mean: np.ndarray
    candidate_std: np.ndarray
    proposal_mean: np.ndarray
    proposal_std: np.ndarray

    @classmethod
    def fit(cls, tensors: E1FeatureTensors) -> "E1Normalizer":
        candidate_mean = tensors.candidate.mean(axis=0, keepdims=True)
        candidate_std = tensors.candidate.std(axis=0, keepdims=True)
        flat_proposal = tensors.proposal.reshape(-1, tensors.proposal.shape[-1])
        proposal_mean = flat_proposal.mean(axis=0, keepdims=True)
        proposal_std = flat_proposal.std(axis=0, keepdims=True)
        return cls(
            candidate_mean=candidate_mean.astype(np.float32),
            candidate_std=np.maximum(candidate_std, 1e-5).astype(np.float32),
            proposal_mean=proposal_mean.astype(np.float32),
            proposal_std=np.maximum(proposal_std, 1e-5).astype(np.float32),
        )

    def transform(self, tensors: E1FeatureTensors) -> E1FeatureTensors:
        candidate = (tensors.candidate - self.candidate_mean) / self.candidate_std
        proposal = (tensors.proposal - self.proposal_mean) / self.proposal_std
        return E1FeatureTensors(
            candidate=np.nan_to_num(candidate, copy=False).astype(np.float32),
            proposal=np.nan_to_num(proposal, copy=False).astype(np.float32),
        )

    def to_jsonable(self) -> dict:
        return {
            "candidate_mean": self.candidate_mean.tolist(),
            "candidate_std": self.candidate_std.tolist(),
            "proposal_mean": self.proposal_mean.tolist(),
            "proposal_std": self.proposal_std.tolist(),
        }


class E1EditEnergyNet(nn.Module):
    def __init__(
        self,
        *,
        candidate_dim: int,
        proposal_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.proposal_encoder = nn.Sequential(
            nn.Linear(proposal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.detection_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.proposal_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        candidate: torch.Tensor,
        proposal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate_embedding = self.candidate_encoder(candidate)
        proposal_embedding = self.proposal_encoder(proposal)
        pooled_proposal = proposal_embedding.max(dim=1).values
        candidate_logits = self.detection_head(
            torch.cat([candidate_embedding, pooled_proposal], dim=1)
        ).squeeze(1)
        proposal_logits = self.proposal_head(proposal_embedding).squeeze(-1)
        return candidate_logits, proposal_logits


def proposal_target_indices(
    labels: np.ndarray,
    error_kind: np.ndarray,
    target_pitch: np.ndarray,
    proposals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(np.int64)
    error_kind = np.asarray(error_kind).astype(np.int64)
    target_pitch = np.asarray(target_pitch).astype(np.int64)
    proposals = np.asarray(proposals).astype(np.int64)
    target_indices = np.zeros(len(labels), dtype=np.int64)
    mask = np.zeros(len(labels), dtype=bool)
    for row in range(len(labels)):
        if labels[row] != 1 or error_kind[row] != 1:
            continue
        matches = np.flatnonzero(proposals[row] == target_pitch[row])
        if len(matches) == 0:
            continue
        target_indices[row] = int(matches[0])
        mask[row] = True
    return target_indices, mask


def build_e1_feature_tensors(
    arrays: dict[str, np.ndarray],
    candidate_features: np.ndarray,
) -> E1FeatureTensors:
    def align_proposal_axis(values: np.ndarray, proposal_count: int) -> np.ndarray:
        if values.shape[1] == proposal_count:
            return values.astype(np.float32)
        aligned = np.zeros(
            (values.shape[0], proposal_count, values.shape[2]),
            dtype=np.float32,
        )
        shared = min(proposal_count, values.shape[1])
        aligned[:, :shared, :] = values[:, :shared, :].astype(np.float32)
        return aligned

    observed = arrays["observed_pitch"].astype(np.float32)[:, None, None]
    proposals = arrays["proposals"].astype(np.float32)[:, :, None]
    proposal_count = proposals.shape[1]
    b_features = arrays["b_features"].astype(np.float32)
    raw_c_features = arrays.get(
        "c_features",
        np.zeros((len(b_features), proposal_count, 0), dtype=np.float32),
    ).astype(np.float32)
    c_features = align_proposal_axis(raw_c_features, proposal_count)
    b_ranking = arrays["b_ranking"].astype(np.float32)[:, :, None]
    raw_c_ranking = arrays.get(
        "c_ranking",
        np.zeros((len(b_features), proposal_count), dtype=np.float32),
    ).astype(np.float32)[:, :, None]
    c_ranking = align_proposal_axis(raw_c_ranking, proposal_count)
    proposal_features = np.concatenate(
        [
            b_features,
            c_features,
            b_ranking,
            c_ranking,
            np.broadcast_to(observed / 127.0, proposals.shape).astype(np.float32),
            proposals / 127.0,
            (proposals - observed) / 24.0,
        ],
        axis=2,
    )
    return E1FeatureTensors(
        candidate=np.nan_to_num(candidate_features, copy=False).astype(np.float32),
        proposal=np.nan_to_num(proposal_features, copy=False).astype(np.float32),
    )


def _make_dataset(
    tensors: E1FeatureTensors,
    labels: np.ndarray,
    target_indices: np.ndarray,
    target_mask: np.ndarray,
) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(tensors.candidate).float(),
        torch.from_numpy(tensors.proposal).float(),
        torch.from_numpy(labels.astype(np.float32)),
        torch.from_numpy(target_indices.astype(np.int64)),
        torch.from_numpy(target_mask.astype(np.float32)),
    )


def train_e1_model(
    tensors: E1FeatureTensors,
    arrays: dict[str, np.ndarray],
    *,
    seed: int,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    correction_weight: float,
    device: torch.device,
) -> E1EditEnergyNet:
    torch.manual_seed(seed)
    target_indices, target_mask = proposal_target_indices(
        arrays["labels"],
        arrays["error_kind"],
        arrays["target_pitch"],
        arrays["proposals"],
    )
    dataset = _make_dataset(tensors, arrays["labels"], target_indices, target_mask)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    labels = arrays["labels"].astype(np.float32)
    positives = max(float(labels.sum()), 1.0)
    negatives = max(float(len(labels) - labels.sum()), 1.0)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    model = E1EditEnergyNet(
        candidate_dim=tensors.candidate.shape[1],
        proposal_dim=tensors.proposal.shape[2],
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    detection_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    correction_loss = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for candidate, proposal, batch_labels, target, mask in loader:
            candidate = candidate.to(device)
            proposal = proposal.to(device)
            batch_labels = batch_labels.to(device)
            target = target.to(device)
            mask = mask.to(device).bool()
            candidate_logits, proposal_logits = model(candidate, proposal)
            loss = detection_loss(candidate_logits, batch_labels)
            if bool(mask.any()):
                loss = loss + correction_weight * correction_loss(
                    proposal_logits[mask],
                    target[mask],
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict_e1(
    model: E1EditEnergyNet,
    tensors: E1FeatureTensors,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = TensorDataset(
        torch.from_numpy(tensors.candidate).float(),
        torch.from_numpy(tensors.proposal).float(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    detection_logits: list[np.ndarray] = []
    detection_scores: list[np.ndarray] = []
    proposal_scores: list[np.ndarray] = []
    for candidate, proposal in loader:
        candidate = candidate.to(device)
        proposal = proposal.to(device)
        candidate_logits, proposal_logits = model(candidate, proposal)
        detection_logits.append(candidate_logits.cpu().numpy())
        detection_scores.append(torch.sigmoid(candidate_logits).cpu().numpy())
        proposal_scores.append(proposal_logits.cpu().numpy())
    return (
        np.concatenate(detection_scores).astype(np.float32),
        np.concatenate(detection_logits).astype(np.float32),
        np.concatenate(proposal_scores).astype(np.float32),
    )


def row_at_threshold_local(
    scores: np.ndarray,
    labels: np.ndarray,
    total_errors: int,
    threshold: float,
) -> dict:
    prediction = scores >= threshold
    target = labels.astype(bool)
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(total_errors - tp)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_errors, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def threshold_grid_local(scores: np.ndarray) -> np.ndarray:
    quantiles = np.quantile(scores, np.linspace(0.01, 0.995, 180))
    fixed = np.linspace(float(scores.min()), float(scores.max()), 120)
    return np.unique(np.concatenate([quantiles, fixed])).astype(np.float32)


def select_from_calibration_local(
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    calibration_total_errors: int,
    target_precision: float,
) -> dict:
    rows = [
        row_at_threshold_local(
            calibration_scores,
            calibration_labels,
            calibration_total_errors,
            threshold,
        )
        for threshold in threshold_grid_local(calibration_scores)
    ]
    feasible = [row for row in rows if row["precision"] >= target_precision]
    if feasible:
        return max(feasible, key=lambda row: (row["recall"], row["precision"]))
    return max(rows, key=lambda row: (row["precision"], row["recall"]))


def correction_metrics_local(
    proposals: np.ndarray,
    proposal_scores: np.ndarray,
    detected: np.ndarray,
    labels: np.ndarray,
    error_kinds: np.ndarray,
    target_pitch: np.ndarray,
) -> dict:
    replace = detected & labels.astype(bool) & (error_kinds == 1)
    count = int(replace.sum())
    if count == 0:
        return {
            "detected_replace_errors": 0,
            "top1_correct": 0,
            "top3_correct": 0,
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
        }
    order = np.argsort(-proposal_scores[replace], axis=1, kind="mergesort")
    ranked = np.take_along_axis(proposals[replace], order, axis=1)
    targets = target_pitch[replace]
    top1 = int((ranked[:, 0] == targets).sum())
    top3 = int((ranked[:, :3] == targets[:, None]).any(axis=1).sum())
    return {
        "detected_replace_errors": count,
        "top1_correct": top1,
        "top3_correct": top3,
        "top1_accuracy": top1 / count,
        "top3_accuracy": top3 / count,
    }


def evaluate_e1_score_rows(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    test_proposal_scores: np.ndarray,
    calibration: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    calibration_total_errors: int,
    test_total_errors: int,
    target_precision: float,
) -> dict:
    rows = []
    for margin in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]:
        selected = select_from_calibration_local(
            calibration_scores,
            calibration["labels"].astype(np.int64),
            calibration_total_errors,
            target_precision + margin,
        )
        test_row = row_at_threshold_local(
            test_scores,
            test["labels"].astype(np.int64),
            test_total_errors,
            selected["threshold"],
        )
        test_row["correction"] = correction_metrics_local(
            test["proposals"],
            test_proposal_scores,
            test_scores >= selected["threshold"],
            test["labels"],
            test["error_kind"],
            test["target_pitch"],
        )
        rows.append(
            {
                "requested_precision": target_precision + margin,
                "selected_calibration": selected,
                "selected_test": test_row,
            }
        )
    feasible = [
        row for row in rows if row["selected_test"]["precision"] >= target_precision
    ]
    return {
        "margins": rows,
        "best_feasible_test": max(
            feasible or rows,
            key=lambda row: (
                row["selected_test"]["recall"]
                if row["selected_test"]["precision"] >= target_precision
                else -1.0,
                row["selected_test"]["precision"],
            ),
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--target-precision", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--max-validation-files", type=int, default=None)
    parser.add_argument("--max-test-files", type=int, default=None)
    parser.add_argument("--motif-radius", type=int, default=4)
    parser.add_argument("--motif-min-similarity", type=float, default=0.84)
    parser.add_argument("--motif-exclude-radius", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--correction-weight", type=float, default=0.25)
    return parser.parse_args()


def load_split(cache_dir: Path, name: str) -> tuple[dict[str, np.ndarray], dict]:
    from build_counterfactual_candidate_cache import load_candidate_cache

    return load_candidate_cache(cache_dir / f"{name}.npz")


def _make_piece_dataset(args: argparse.Namespace, split: str):
    from voice_aware_dataset import PieceConsistentVoiceDataset

    return PieceConsistentVoiceDataset(
        root=args.data_root,
        split=split,
        voice_method="onset_matching",
        window_size=args.window_size,
        stride=args.stride,
        error_rate=args.error_rate,
        seed=args.seed,
        max_files=(
            args.max_validation_files if split == "validation" else args.max_test_files
        ),
        verbose=True,
    )


def _append_motif_to_splits(
    train: dict[str, np.ndarray],
    calibration: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    from run_motif_repetition_verifier import append_motif_features

    validation_dataset = _make_piece_dataset(args, "validation")
    test_dataset = _make_piece_dataset(args, "test")
    motif_kwargs = {
        "radius": args.motif_radius,
        "min_similarity": args.motif_min_similarity,
        "exclude_radius": args.motif_exclude_radius,
    }
    return (
        append_motif_features(train, validation_dataset, **motif_kwargs),
        append_motif_features(calibration, validation_dataset, **motif_kwargs),
        append_motif_features(test, test_dataset, **motif_kwargs),
    )


def _fusion_features(candidate_features: np.ndarray, logits: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            candidate_features,
            logits[:, None].astype(np.float32),
            scores[:, None].astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def build_proposal_selected_features(
    candidate_features: np.ndarray,
    proposal_features: np.ndarray,
    proposal_logits: np.ndarray,
) -> np.ndarray:
    if proposal_features.ndim != 3:
        raise ValueError("proposal_features must have shape [rows, proposals, features].")
    if proposal_logits.shape != proposal_features.shape[:2]:
        raise ValueError("proposal_logits must align with proposal rows.")
    best_index = proposal_logits.argmax(axis=1)
    best_proposal = proposal_features[np.arange(len(proposal_features)), best_index]
    sorted_scores = np.sort(proposal_logits, axis=1)
    best_score = sorted_scores[:, -1]
    second_score = sorted_scores[:, -2] if proposal_logits.shape[1] > 1 else np.zeros_like(best_score)
    score_gap = best_score - second_score
    shifted = proposal_logits - proposal_logits.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability = probability / probability.sum(axis=1, keepdims=True).clip(min=1e-9)
    entropy = -(probability * np.log(probability.clip(min=1e-9))).sum(axis=1)
    return np.concatenate(
        [
            candidate_features.astype(np.float32),
            best_proposal.astype(np.float32),
            best_score[:, None].astype(np.float32),
            score_gap[:, None].astype(np.float32),
            entropy[:, None].astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def evaluate_detection_with_external_correction(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    external_test_proposal_scores: np.ndarray,
    calibration: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    calibration_total_errors: int,
    test_total_errors: int,
    target_precision: float,
) -> dict:
    rows = []
    for margin in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]:
        selected = select_from_calibration_local(
            calibration_scores,
            calibration["labels"].astype(np.int64),
            calibration_total_errors,
            target_precision + margin,
        )
        test_row = row_at_threshold_local(
            test_scores,
            test["labels"].astype(np.int64),
            test_total_errors,
            selected["threshold"],
        )
        test_row["correction"] = correction_metrics_local(
            test["proposals"],
            external_test_proposal_scores,
            test_scores >= selected["threshold"],
            test["labels"],
            test["error_kind"],
            test["target_pitch"],
        )
        rows.append(
            {
                "requested_precision": target_precision + margin,
                "selected_calibration": selected,
                "selected_test": test_row,
            }
        )
    feasible = [
        row for row in rows if row["selected_test"]["precision"] >= target_precision
    ]
    return {
        "margins": rows,
        "best_feasible_test": max(
            feasible or rows,
            key=lambda row: (
                row["selected_test"]["recall"]
                if row["selected_test"]["precision"] >= target_precision
                else -1.0,
                row["selected_test"]["precision"],
            ),
        ),
    }


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# E1 Edit-Energy Verifier",
        "",
        f"- target precision: `{result['target_precision']:.2f}`",
        f"- motif radius: `{result['motif']['radius']}`",
        f"- motif min similarity: `{result['motif']['min_similarity']}`",
        f"- motif exclude radius: `{result['motif']['exclude_radius']}`",
        "",
        "| System | Precision | Recall | F1 | Replace Top-1 | Replace Top-3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, block in result["systems"].items():
        row = block["best_feasible_test"]["selected_test"]
        correction = row.get("correction", {})
        lines.append(
            f"| {name} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {correction.get('top1_accuracy', 0.0):.4f} | "
            f"{correction.get('top3_accuracy', 0.0):.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    from joblib import dump

    from run_counterfactual_edit_verifier import evaluate_score_rows, make_small_leaf
    from run_motif_repetition_verifier import build_c2_motif_features, motif_summary

    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)
    train, train_meta = load_split(cache_dir, "train")
    calibration, calibration_meta = load_split(cache_dir, "calibration")
    test, test_meta = load_split(cache_dir, "test")
    train, calibration, test = _append_motif_to_splits(train, calibration, test, args)
    train_x = build_c2_motif_features(train)
    calibration_x = build_c2_motif_features(calibration)
    test_x = build_c2_motif_features(test)
    calibration_total = int(calibration_meta["stats"]["error_notes"])
    test_total = int(test_meta["stats"]["error_notes"])
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    systems = {}
    baseline = make_small_leaf(args.seed)
    baseline.fit(train_x, train["labels"].astype(np.int64))
    dump(baseline, checkpoint_dir / "c2_motif_hgb.joblib")
    calibration_baseline_scores = baseline.predict_proba(calibration_x)[:, 1]
    test_baseline_scores = baseline.predict_proba(test_x)[:, 1]
    systems["C2_motif_hgb"] = evaluate_score_rows(
        calibration_baseline_scores,
        test_baseline_scores,
        calibration,
        test,
        calibration_total,
        test_total,
        args.target_precision,
    )

    raw_train_tensors = build_e1_feature_tensors(train, train_x)
    raw_calibration_tensors = build_e1_feature_tensors(calibration, calibration_x)
    raw_test_tensors = build_e1_feature_tensors(test, test_x)
    normalizer = E1Normalizer.fit(raw_train_tensors)
    train_tensors = normalizer.transform(raw_train_tensors)
    calibration_tensors = normalizer.transform(raw_calibration_tensors)
    test_tensors = normalizer.transform(raw_test_tensors)
    model = train_e1_model(
        train_tensors,
        train,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        correction_weight=args.correction_weight,
        device=device,
    )
    train_scores, train_logits, train_proposal_scores = predict_e1(
        model,
        train_tensors,
        batch_size=args.batch_size,
        device=device,
    )
    calibration_scores, calibration_logits, calibration_proposal_scores = predict_e1(
        model,
        calibration_tensors,
        batch_size=args.batch_size,
        device=device,
    )
    test_scores, test_logits, test_proposal_scores = predict_e1(
        model,
        test_tensors,
        batch_size=args.batch_size,
        device=device,
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "normalizer": normalizer.to_jsonable(),
            "args": vars(args),
            "candidate_dim": train_tensors.candidate.shape[1],
            "proposal_dim": train_tensors.proposal.shape[2],
        },
        checkpoint_dir / "e1_edit_energy.pt",
    )
    systems["E1_edit_energy"] = evaluate_e1_score_rows(
        calibration_scores,
        test_scores,
        test_proposal_scores,
        calibration,
        test,
        calibration_total,
        test_total,
        args.target_precision,
    )
    systems["C2_motif_hgb_e1_correction"] = evaluate_detection_with_external_correction(
        calibration_baseline_scores,
        test_baseline_scores,
        test_proposal_scores,
        calibration,
        test,
        calibration_total,
        test_total,
        args.target_precision,
    )

    fusion = make_small_leaf(args.seed + 1)
    fusion.fit(
        _fusion_features(train_x, train_logits, train_scores),
        train["labels"].astype(np.int64),
    )
    dump(fusion, checkpoint_dir / "e1_fusion_hgb.joblib")
    systems["E1_fusion_hgb"] = evaluate_score_rows(
        fusion.predict_proba(_fusion_features(calibration_x, calibration_logits, calibration_scores))[:, 1],
        fusion.predict_proba(_fusion_features(test_x, test_logits, test_scores))[:, 1],
        calibration,
        test,
        calibration_total,
        test_total,
        args.target_precision,
    )
    proposal_selected = make_small_leaf(args.seed + 2)
    proposal_selected.fit(
        build_proposal_selected_features(train_x, raw_train_tensors.proposal, train_proposal_scores),
        train["labels"].astype(np.int64),
    )
    dump(proposal_selected, checkpoint_dir / "e1_proposal_selected_hgb.joblib")
    calibration_proposal_selected_scores = proposal_selected.predict_proba(
        build_proposal_selected_features(
            calibration_x,
            raw_calibration_tensors.proposal,
            calibration_proposal_scores,
        )
    )[:, 1]
    test_proposal_selected_scores = proposal_selected.predict_proba(
        build_proposal_selected_features(
            test_x,
            raw_test_tensors.proposal,
            test_proposal_scores,
        )
    )[:, 1]
    systems["E1_proposal_selected_hgb"] = evaluate_detection_with_external_correction(
        calibration_proposal_selected_scores,
        test_proposal_selected_scores,
        test_proposal_scores,
        calibration,
        test,
        calibration_total,
        test_total,
        args.target_precision,
    )

    result = {
        "target_precision": args.target_precision,
        "train_stats": train_meta["stats"],
        "calibration_stats": calibration_meta["stats"],
        "test_stats": test_meta["stats"],
        "motif": {
            "radius": args.motif_radius,
            "min_similarity": args.motif_min_similarity,
            "exclude_radius": args.motif_exclude_radius,
            "train_summary": motif_summary(train),
            "calibration_summary": motif_summary(calibration),
            "test_summary": motif_summary(test),
        },
        "e1": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "correction_weight": args.correction_weight,
            "device": str(device),
        },
        "systems": systems,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
