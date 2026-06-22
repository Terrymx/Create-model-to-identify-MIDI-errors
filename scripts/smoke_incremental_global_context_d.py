from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from joblib import load

from incremental_global_context import PieceEdit
from incremental_piece_scorer import IncrementalPieceScorer
from run_counterfactual_piece_b import load_canonical_split
from run_frozen_union_candidate_context_verifier import load_any_model
from voice_aware_dataset import PieceConsistentVoiceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--threeclass-checkpoint", required=True)
    parser.add_argument("--binary-checkpoint", required=True)
    parser.add_argument("--forward-checkpoint", required=True)
    parser.add_argument("--backward-checkpoint", required=True)
    parser.add_argument("--verifier-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arrays, _ = load_canonical_split(Path(args.cache_dir), args.split, 256)
    counts = {
        int(file_id): int((arrays["file_ids"] == file_id).sum())
        for file_id in np.unique(arrays["file_ids"])
    }
    piece_id = min(
        (file_id for file_id, count in counts.items() if count >= 2),
        key=lambda file_id: counts[file_id],
    )
    rows = np.flatnonzero(arrays["file_ids"] == piece_id)
    piece_arrays = {name: values[rows] for name, values in arrays.items()}
    piece_arrays["window_starts"] = (
        piece_arrays["positions"] - piece_arrays["local_positions"]
    ).astype(np.int64)
    dataset = PieceConsistentVoiceDataset(
        root=args.data_root,
        split=args.split,
        voice_method="onset_matching",
        window_size=256,
        stride=128,
        error_rate=0.01,
        seed=args.seed,
        max_files=args.max_files,
        verbose=True,
    )
    models = (
        *load_any_model(args.threeclass_checkpoint, device),
        *load_any_model(args.binary_checkpoint, device),
        *load_any_model(args.forward_checkpoint, device),
        *load_any_model(args.backward_checkpoint, device),
    )
    scorer = IncrementalPieceScorer(
        piece_id=piece_id,
        piece_features=dataset._pieces[piece_id].features,
        candidate_arrays=piece_arrays,
        models=models,
        verifier_model=load(args.verifier_checkpoint),
        device=device,
    )
    start = time.perf_counter()
    baseline = scorer.score_all(())
    baseline_seconds = time.perf_counter() - start
    row = int(np.argmax(baseline))
    best_slot = int(np.argmax(piece_arrays["c_ranking"][row]))
    edit = PieceEdit(
        int(piece_arrays["positions"][row]),
        int(piece_arrays["c_proposals"][row, best_slot]),
    )
    start = time.perf_counter()
    edited = scorer.score_all((edit,))
    edited_seconds = time.perf_counter() - start
    result = {
        "piece_id": piece_id,
        "candidate_count": len(rows),
        "edit": {
            "position": edit.position,
            "proposed_pitch": edit.proposed_pitch,
        },
        "baseline_seconds": baseline_seconds,
        "edited_seconds": edited_seconds,
        "window_cache_hits": scorer.window_cache.hits,
        "window_cache_misses": scorer.window_cache.misses,
        "changed_candidate_scores": int(
            (~np.isclose(baseline, edited, atol=1e-7)).sum()
        ),
        "maximum_score_change": float(np.max(np.abs(baseline - edited))),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
