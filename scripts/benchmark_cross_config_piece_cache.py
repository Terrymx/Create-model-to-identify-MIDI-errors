from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from run_incremental_global_context_formal import (
    _baseline,
    _calibration_configs,
    _load_dataset,
    _load_models,
    _load_raw_split,
    _prepare_piece_search,
    _search_prepared_piece,
    _split_total_errors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--threeclass-checkpoint", required=True)
    parser.add_argument("--binary-checkpoint", required=True)
    parser.add_argument("--forward-checkpoint", required=True)
    parser.add_argument("--backward-checkpoint", required=True)
    parser.add_argument("--verifier-checkpoint", required=True)
    parser.add_argument("--piece-id", type=int, required=True)
    parser.add_argument("--config-indices", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--calibration-precision", type=float, default=0.81)
    return parser.parse_args()


def comparable(row: dict) -> dict:
    return {
        "tp": row["tp"],
        "fp": row["fp"],
        "errors": row["errors"],
        "edits": row["edits"],
    }


def main() -> None:
    import torch
    from joblib import load

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verifier_model = load(args.verifier_checkpoint)
    models = _load_models(args, device)
    arrays = _load_raw_split(Path(args.cache_dir), "calibration")
    dataset = _load_dataset(args, "calibration")
    piece_ids = set(int(value) for value in np.unique(arrays["file_ids"]))
    if args.piece_id not in piece_ids:
        raise ValueError(f"piece {args.piece_id} is not in calibration cache")
    total_errors = _split_total_errors(arrays, dataset)
    _, baseline_floor = _baseline(
        arrays,
        verifier_model,
        total_errors,
        args.calibration_precision,
    )
    all_configs = _calibration_configs(baseline_floor)
    configs = [all_configs[index] for index in args.config_indices]

    independent_rows = []
    independent_start = time.perf_counter()
    for config in configs:
        prepared = _prepare_piece_search(
            piece_id=args.piece_id,
            arrays=arrays,
            dataset=dataset,
            models=models,
            verifier_model=verifier_model,
        )
        independent_rows.append(
            _search_prepared_piece(prepared, config, args.beam_width)
        )
    independent_seconds = time.perf_counter() - independent_start

    shared_start = time.perf_counter()
    shared_prepared = _prepare_piece_search(
        piece_id=args.piece_id,
        arrays=arrays,
        dataset=dataset,
        models=models,
        verifier_model=verifier_model,
    )
    shared_rows = [
        _search_prepared_piece(shared_prepared, config, args.beam_width)
        for config in configs
    ]
    shared_seconds = time.perf_counter() - shared_start

    for independent, shared in zip(independent_rows, shared_rows):
        if comparable(independent) != comparable(shared):
            raise AssertionError(
                json.dumps(
                    {
                        "independent": comparable(independent),
                        "shared": comparable(shared),
                    },
                    indent=2,
                )
            )

    result = {
        "piece_id": args.piece_id,
        "configs": configs,
        "identical": True,
        "independent_seconds": independent_seconds,
        "shared_seconds": shared_seconds,
        "speedup": independent_seconds / max(shared_seconds, 1e-9),
        "independent_window_misses": sum(
            row["window_misses"] for row in independent_rows
        ),
        "shared_window_misses": sum(
            row["window_misses"] for row in shared_rows
        ),
        "shared_score_states": len(shared_prepared["scorer"].score_cache),
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
