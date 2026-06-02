from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from midi_error_detector.data import FEATURE_SIZE, MaestroWrongNoteDataset
from midi_error_detector.model import build_wrong_note_model


class CandidateReranker(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a second-stage candidate verifier/reranker.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="checkpoints/candidate_reranker.pt")
    parser.add_argument("--output-json", default="training_logs/candidate_reranker_eval.json")
    parser.add_argument("--output-md", default="training_logs/candidate_reranker_eval.md")
    parser.add_argument("--train-split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--eval-split", default="test", choices=["validation", "test"])
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--candidate-threshold", type=float, default=0.85)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-eval-files", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--rerank-thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    return parser.parse_args()


def checkpoint_args(checkpoint: dict) -> SimpleNamespace:
    saved = dict(checkpoint.get("args", {}))
    return SimpleNamespace(
        model=saved.get("model", "bigru"),
        input_size=int(saved.get("input_size", FEATURE_SIZE)),
        hidden_size=saved.get("hidden_size", 256),
        num_layers=saved.get("num_layers", 2),
        transformer_d_model=saved.get("transformer_d_model", 192),
        transformer_heads=saved.get("transformer_heads", 4),
        transformer_ffn_dim=saved.get("transformer_ffn_dim", 512),
        dropout=saved.get("dropout", 0.2),
    )


def f_beta(precision: float, recall: float, beta: float) -> float:
    beta2 = beta * beta
    return (1.0 + beta2) * precision * recall / max(beta2 * precision + recall, 1e-12)


def make_base_model(path: str, device: torch.device) -> tuple[nn.Module, SimpleNamespace]:
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint_args(checkpoint)
    model = build_wrong_note_model(
        model_type=args.model,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        transformer_d_model=args.transformer_d_model,
        transformer_heads=args.transformer_heads,
        transformer_ffn_dim=args.transformer_ffn_dim,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, args


def make_loader(args: argparse.Namespace, split: str, max_files: int | None) -> DataLoader:
    dataset = MaestroWrongNoteDataset(
        root=args.data_root,
        split=split,
        window_size=args.window_size,
        stride=args.stride,
        error_rate=args.error_rate,
        max_files=max_files,
        cache_notes=True,
        verbose=True,
    )
    dataset.set_epoch(0)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def collect_candidates(
    model: nn.Module,
    model_args: SimpleNamespace,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    rows: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    stats = {"candidate_count": 0, "positive_count": 0, "error_notes": 0, "notes": 0}
    for batch in tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True):
        full_features = batch["features"].to(device)
        features = full_features
        if features.shape[-1] > model_args.input_size:
            features = features[..., : model_args.input_size]
        elif features.shape[-1] < model_args.input_size:
            features = torch.nn.functional.pad(features, (0, model_args.input_size - features.shape[-1]))

        outputs = model(features)
        error_prob = torch.sigmoid(outputs["error_logits"])
        action_prob = torch.softmax(outputs["kind_logits"], dim=-1)
        action_pred = action_prob.argmax(dim=-1)
        pitch_prob = torch.softmax(outputs["pitch_logits"], dim=-1)
        top2_prob, top2_pitch = pitch_prob.topk(k=2, dim=-1)
        mask = batch["mask"].to(device).bool()
        target = batch["is_error"].to(device).bool()
        candidate_mask = (error_prob >= threshold) & (action_pred != 0) & mask
        stats["notes"] += int(mask.sum().item())
        stats["error_notes"] += int((target & mask).sum().item())
        stats["candidate_count"] += int(candidate_mask.sum().item())
        stats["positive_count"] += int((candidate_mask & target).sum().item())
        if not bool(candidate_mask.any()):
            continue

        selected_features = full_features[candidate_mask]
        selected_error_prob = error_prob[candidate_mask].unsqueeze(1)
        selected_action_prob = action_prob[candidate_mask]
        selected_top1_prob = top2_prob[..., 0][candidate_mask].unsqueeze(1)
        selected_pitch_margin = (top2_prob[..., 0] - top2_prob[..., 1])[candidate_mask].unsqueeze(1)
        input_pitch = (full_features[..., 0] * 127.0).round()
        selected_pitch_shift = ((top2_pitch[..., 0].float() - input_pitch) / 24.0).clamp(-1.0, 1.0)[candidate_mask].unsqueeze(1)
        selected_action_is_replace = (action_pred == 1).float()[candidate_mask].unsqueeze(1)
        selected_action_is_delete = (action_pred == 2).float()[candidate_mask].unsqueeze(1)
        candidate_rows = torch.cat(
            [
                selected_error_prob,
                selected_action_prob,
                selected_top1_prob,
                selected_pitch_margin,
                selected_pitch_shift,
                selected_action_is_replace,
                selected_action_is_delete,
                selected_features,
            ],
            dim=1,
        )
        rows.append(candidate_rows.cpu())
        labels.append(target[candidate_mask].float().cpu())

    if not rows:
        raise RuntimeError(f"No candidates collected for {desc} at threshold={threshold}")
    return torch.cat(rows, dim=0), torch.cat(labels, dim=0), stats


def train_reranker(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    input_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> CandidateReranker:
    model = CandidateReranker(input_size=input_size, hidden_size=args.hidden_size).to(device)
    positive = float(train_y.sum().item())
    negative = float(len(train_y) - positive)
    pos_weight = torch.tensor([negative / max(positive, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(13)
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(train_y), generator=generator)
        total_loss = 0.0
        model.train()
        for start in range(0, len(order), 4096):
            batch_idx = order[start : start + 4096]
            x = train_x[batch_idx].to(device)
            y = train_y[batch_idx].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_idx)
        print(f"reranker epoch={epoch}/{args.epochs} loss={total_loss / len(train_y):.6f}", flush=True)
    return model


@torch.no_grad()
def evaluate_reranker(model: CandidateReranker, x: torch.Tensor, y: torch.Tensor, thresholds: list[float], device: torch.device) -> list[dict]:
    model.eval()
    scores = []
    for start in range(0, len(y), 8192):
        scores.append(torch.sigmoid(model(x[start : start + 8192].to(device))).cpu())
    score = torch.cat(scores)
    rows: list[dict] = []
    for threshold in thresholds:
        pred = score >= threshold
        target = y.bool()
        tp = int((pred & target).sum().item())
        fp = int((pred & (~target)).sum().item())
        fn = int(((~pred) & target).sum().item())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        rows.append(
            {
                "rerank_threshold": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
                "f0_5": f_beta(precision, recall, 0.5),
            }
        )
    return rows


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Candidate Reranker Evaluation",
        "",
        f"- checkpoint: `{result['base_checkpoint']}`",
        f"- candidate threshold: `{result['candidate_threshold']}`",
        f"- train candidates: `{result['train_stats']['candidate_count']}`",
        f"- eval candidates: `{result['eval_stats']['candidate_count']}`",
        "",
        "| Rerank Threshold | Precision | Recall | F1 | F0.5 | TP | FP | FN |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["thresholds"]:
        lines.append(
            "| "
            f"{row['rerank_threshold']:.2f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['f0_5']:.4f} | {row['tp']} | {row['fp']} | {row['fn']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, base_args = make_base_model(args.checkpoint, device)
    train_loader = make_loader(args, args.train_split, args.max_train_files)
    eval_loader = make_loader(args, args.eval_split, args.max_eval_files)
    train_x, train_y, train_stats = collect_candidates(
        base_model,
        base_args,
        train_loader,
        device,
        args.candidate_threshold,
        f"collect {args.train_split}",
    )
    eval_x, eval_y, eval_stats = collect_candidates(
        base_model,
        base_args,
        eval_loader,
        device,
        args.candidate_threshold,
        f"collect {args.eval_split}",
    )
    print(f"train candidates={len(train_y)} positives={int(train_y.sum().item())}", flush=True)
    print(f"eval candidates={len(eval_y)} positives={int(eval_y.sum().item())}", flush=True)
    reranker = train_reranker(train_x, train_y, train_x.shape[1], args, device)
    rows = evaluate_reranker(reranker, eval_x, eval_y, sorted(set(args.rerank_thresholds)), device)
    result = {
        "base_checkpoint": args.checkpoint,
        "candidate_threshold": args.candidate_threshold,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "train_stats": train_stats,
        "eval_stats": eval_stats,
        "thresholds": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": reranker.state_dict(),
            "input_size": train_x.shape[1],
            "args": vars(args),
            "eval_result": result,
        },
        output,
    )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), result)
    print(json.dumps(result, indent=2), flush=True)
    print(f"saved {output}", flush=True)
    print(f"wrote {args.output_json}", flush=True)
    print(f"wrote {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
