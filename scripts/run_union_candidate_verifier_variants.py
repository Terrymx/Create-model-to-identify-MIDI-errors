from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from evaluate_directional_likelihood_gate import load_model
from run_union_candidate_verifier import (
    CandidateVerifier,
    collect_candidates,
    make_dataset,
    make_loader,
    metric_rows,
    select_operating_point,
    split_indices_by_file,
    standardize,
)


class LinearVerifier(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


class ShallowVerifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare union-candidate verifier variants.")
    parser.add_argument("--threeclass-checkpoint", required=True)
    parser.add_argument("--binary-checkpoint", required=True)
    parser.add_argument("--forward-checkpoint", required=True)
    parser.add_argument("--backward-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-json", default="training_logs/union_candidate_verifier_variants.json")
    parser.add_argument("--output-md", default="training_logs/union_candidate_verifier_variants.md")
    parser.add_argument("--checkpoint-dir", default="checkpoints/union_candidate_verifier_variants")
    parser.add_argument("--threeclass-candidate-threshold", type=float, default=0.45)
    parser.add_argument("--binary-candidate-threshold", type=float, default=0.45)
    parser.add_argument("--target-precision", type=float, default=0.80)
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--calibration-file-fraction", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--surprise-eval-groups", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--skip-sklearn", action="store_true")
    parser.add_argument("--svm-max-train", type=int, default=12000)
    parser.add_argument("--sklearn-max-iter", type=int, default=3000)
    parser.add_argument("--max-validation-files", type=int, default=None)
    parser.add_argument("--max-test-files", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def predict(model: nn.Module, features: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    return torch.cat(
        [
            torch.sigmoid(model(features[start : start + 8192].to(device))).cpu()
            for start in range(0, len(features), 8192)
        ]
    )


def selection_key(row: dict, target_precision: float) -> tuple[int, float, float]:
    feasible = row["precision"] >= target_precision
    return (
        1 if feasible else 0,
        row["recall"] if feasible else row["precision"],
        row["precision"] if feasible else row["recall"],
    )


def train_variant(
    name: str,
    model: nn.Module,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    calibration_features: torch.Tensor,
    calibration_labels: torch.Tensor,
    calibration_total_errors: int,
    args: argparse.Namespace,
    device: torch.device,
    *,
    pos_weight_scale: float,
    epochs: int,
    lr: float,
) -> tuple[nn.Module, int, dict]:
    positives = float(train_labels.sum())
    negatives = float(len(train_labels) - positives)
    pos_weight_value = (negatives / max(positives, 1.0)) * pos_weight_scale
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    best_epoch = 0
    best_row: dict | None = None
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train_labels), generator=generator)
        total_loss = 0.0
        model.train()
        for start in range(0, len(order), 4096):
            indices = order[start : start + 4096]
            features = train_features[indices].to(device)
            labels = train_labels[indices].to(device)
            loss = criterion(model(features), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)

        calibration_rows = metric_rows(
            predict(model, calibration_features, device),
            calibration_labels,
            calibration_total_errors,
        )
        selected = select_operating_point(calibration_rows, args.target_precision)
        if best_row is None or selection_key(selected, args.target_precision) > selection_key(best_row, args.target_precision):
            best_epoch = epoch
            best_row = selected
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"variant={name} epoch={epoch}/{epochs} loss={total_loss / len(train_labels):.6f} "
            f"cal_threshold={selected['threshold']:.2f} cal_precision={selected['precision']:.4f} "
            f"cal_recall={selected['recall']:.4f} best_epoch={best_epoch}",
            flush=True,
        )

    if best_state is None or best_row is None:
        raise RuntimeError(f"No checkpoint selected for {name}")
    model.load_state_dict(best_state)
    return model, best_epoch, best_row


def test_rows_at_selected(scores: torch.Tensor, labels: torch.Tensor, total_errors: int, threshold: float) -> dict:
    rows = metric_rows(scores, labels, total_errors)
    return next(row for row in rows if row["threshold"] == threshold)


def sklearn_probability(model, features: np.ndarray) -> torch.Tensor:
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(features)
        scores = 1.0 / (1.0 + np.exp(-np.clip(raw, -50.0, 50.0)))
    else:
        scores = model.predict(features).astype(np.float32)
    return torch.from_numpy(np.asarray(scores, dtype=np.float32))


def fit_sklearn_variant(
    name: str,
    model,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    calibration_features: torch.Tensor,
    calibration_labels: torch.Tensor,
    calibration_total_errors: int,
    args: argparse.Namespace,
    *,
    max_train: int | None = None,
) -> tuple[object, dict]:
    train_x_np = train_features.numpy()
    train_y_np = train_labels.numpy().astype(np.int64)
    if max_train is not None and len(train_y_np) > max_train:
        rng = np.random.default_rng(args.seed)
        positives = np.flatnonzero(train_y_np == 1)
        negatives = np.flatnonzero(train_y_np == 0)
        pos_take = min(len(positives), max(1, max_train // 3))
        neg_take = min(len(negatives), max_train - pos_take)
        sampled = np.concatenate(
            [
                rng.choice(positives, size=pos_take, replace=False),
                rng.choice(negatives, size=neg_take, replace=False),
            ]
        )
        rng.shuffle(sampled)
        train_x_np = train_x_np[sampled]
        train_y_np = train_y_np[sampled]
        print(f"variant={name} sklearn_subsample={len(sampled)}", flush=True)
    model.fit(train_x_np, train_y_np)
    calibration_scores = sklearn_probability(model, calibration_features.numpy())
    calibration_rows = metric_rows(
        calibration_scores,
        calibration_labels,
        calibration_total_errors,
    )
    selected = select_operating_point(calibration_rows, args.target_precision)
    print(
        f"variant={name} sklearn_done cal_threshold={selected['threshold']:.2f} "
        f"cal_precision={selected['precision']:.4f} cal_recall={selected['recall']:.4f}",
        flush=True,
    )
    return model, selected


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Union Candidate Verifier Variants",
        "",
        f"- target precision: `{result['target_precision']}`",
        f"- test candidate recall ceiling: `{result['test_stats']['candidate_recall_ceiling']:.4f}`",
        f"- test candidate precision: `{result['test_stats']['candidate_precision']:.4f}`",
        "",
        "| Variant | Best epoch | Cal threshold | Cal P | Cal R | Test P | Test R | Test frontier P | Test frontier R |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in result["variants"].items():
        cal = row["selected_calibration"]
        test = row["selected_test"]
        frontier = row["test_frontier"]
        best_epoch = row["best_epoch"] if row["best_epoch"] is not None else "-"
        lines.append(
            f"| {name} | {best_epoch} | {cal['threshold']:.2f} | "
            f"{cal['precision']:.4f} | {cal['recall']:.4f} | "
            f"{test['precision']:.4f} | {test['recall']:.4f} | "
            f"{frontier['precision']:.4f} | {frontier['recall']:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    three_model, three_args = load_model(args.threeclass_checkpoint, device, require_explicit_surprise=True)
    binary_model, binary_args = load_model(args.binary_checkpoint, device, require_explicit_surprise=True)
    forward_model, forward_args = load_model(args.forward_checkpoint, device, require_explicit_surprise=False)
    backward_model, backward_args = load_model(args.backward_checkpoint, device, require_explicit_surprise=False)

    validation = make_dataset(args, "validation", args.max_validation_files)
    train_indices, calibration_indices, train_files, calibration_files = split_indices_by_file(
        validation,
        args.calibration_file_fraction,
        args.seed,
    )
    test = make_dataset(args, "test", args.max_test_files)
    collection_args = (
        three_model,
        three_args,
        binary_model,
        binary_args,
        forward_model,
        forward_args,
        backward_model,
        backward_args,
    )
    train_x, train_y, _, train_stats = collect_candidates(
        *collection_args,
        make_loader(validation, train_indices, args.batch_size),
        device,
        args,
        "collect verifier train",
    )
    calibration_x, calibration_y, _, calibration_stats = collect_candidates(
        *collection_args,
        make_loader(validation, calibration_indices, args.batch_size),
        device,
        args,
        "collect verifier calibration",
    )
    test_x, test_y, _, test_stats = collect_candidates(
        *collection_args,
        make_loader(test, list(range(len(test))), args.batch_size),
        device,
        args,
        "collect verifier test",
    )
    print(f"train stats={train_stats}", flush=True)
    print(f"calibration stats={calibration_stats}", flush=True)
    print(f"test stats={test_stats}", flush=True)

    train_x, normalization, standardized = standardize(train_x, calibration_x, test_x)
    calibration_x, test_x = standardized
    input_size = train_x.shape[1]
    variant_specs = [
        ("linear_balanced", LinearVerifier(input_size), 1.0, args.epochs, args.lr),
        ("linear_precision", LinearVerifier(input_size), 0.50, args.epochs, args.lr),
        ("linear_high_precision", LinearVerifier(input_size), 0.25, args.epochs, args.lr),
        ("shallow_precision", ShallowVerifier(input_size, 64, 0.10), 0.50, args.epochs, args.lr),
        ("mlp_balanced", CandidateVerifier(input_size, args.hidden_size, args.dropout), 1.0, args.epochs, args.lr),
        ("mlp_precision", CandidateVerifier(input_size, args.hidden_size, args.dropout), 0.50, args.epochs, args.lr),
        ("mlp_high_precision", CandidateVerifier(input_size, args.hidden_size, args.dropout), 0.25, args.epochs, args.lr),
    ]

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    variants: dict[str, dict] = {}
    for name, model, pos_weight_scale, epochs, lr in variant_specs:
        model = model.to(device)
        trained, best_epoch, selected_calibration = train_variant(
            name,
            model,
            train_x,
            train_y,
            calibration_x,
            calibration_y,
            calibration_stats["error_notes"],
            args,
            device,
            pos_weight_scale=pos_weight_scale,
            epochs=epochs,
            lr=lr,
        )
        test_scores = predict(trained, test_x, device)
        test_thresholds = metric_rows(test_scores, test_y, test_stats["error_notes"])
        selected_test = next(
            row for row in test_thresholds
            if row["threshold"] == selected_calibration["threshold"]
        )
        test_frontier = select_operating_point(test_thresholds, args.target_precision)
        variants[name] = {
            "pos_weight_scale": pos_weight_scale,
            "best_epoch": best_epoch,
            "selected_calibration": selected_calibration,
            "selected_test": selected_test,
            "test_frontier": test_frontier,
        }
        torch.save(
            {
                "model_state_dict": trained.state_dict(),
                "normalization": normalization,
                "input_size": input_size,
                "variant": name,
                "args": vars(args),
                "result": variants[name],
            },
            checkpoint_dir / f"{name}.pt",
        )

    if not args.skip_sklearn:
        try:
            from joblib import dump
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import LinearSVC, SVC
        except ImportError as exc:
            print(f"sklearn variants skipped: {exc}", flush=True)
        else:
            sklearn_specs = [
                (
                    "sklearn_logreg_plain",
                    LogisticRegression(max_iter=args.sklearn_max_iter, random_state=args.seed),
                    None,
                ),
                (
                    "sklearn_logreg_balanced",
                    LogisticRegression(
                        max_iter=args.sklearn_max_iter,
                        class_weight="balanced",
                        random_state=args.seed,
                    ),
                    None,
                ),
                (
                    "sklearn_linear_svm_balanced",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(
                            class_weight="balanced",
                            random_state=args.seed,
                            max_iter=args.sklearn_max_iter,
                        ),
                        method="sigmoid",
                        cv=3,
                    ),
                    None,
                ),
                (
                    "sklearn_rbf_svm_balanced",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        probability=True,
                        random_state=args.seed,
                        max_iter=args.sklearn_max_iter,
                    ),
                    args.svm_max_train,
                ),
                (
                    "sklearn_random_forest_balanced",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=args.seed,
                        n_jobs=-1,
                    ),
                    None,
                ),
                (
                    "sklearn_hist_gradient_boosting",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.05,
                        l2_regularization=0.01,
                        class_weight="balanced",
                        random_state=args.seed,
                    ),
                    None,
                ),
            ]
            for name, model, max_train in sklearn_specs:
                trained, selected_calibration = fit_sklearn_variant(
                    name,
                    model,
                    train_x,
                    train_y,
                    calibration_x,
                    calibration_y,
                    calibration_stats["error_notes"],
                    args,
                    max_train=max_train,
                )
                test_scores = sklearn_probability(trained, test_x.numpy())
                test_thresholds = metric_rows(test_scores, test_y, test_stats["error_notes"])
                selected_test = next(
                    row for row in test_thresholds
                    if row["threshold"] == selected_calibration["threshold"]
                )
                test_frontier = select_operating_point(test_thresholds, args.target_precision)
                variants[name] = {
                    "pos_weight_scale": None,
                    "best_epoch": None,
                    "selected_calibration": selected_calibration,
                    "selected_test": selected_test,
                    "test_frontier": test_frontier,
                }
                dump(trained, checkpoint_dir / f"{name}.joblib")

    result = {
        "threeclass_checkpoint": args.threeclass_checkpoint,
        "binary_checkpoint": args.binary_checkpoint,
        "target_precision": args.target_precision,
        "train_file_ids": train_files,
        "calibration_file_ids": calibration_files,
        "train_stats": train_stats,
        "calibration_stats": calibration_stats,
        "test_stats": test_stats,
        "variants": variants,
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
