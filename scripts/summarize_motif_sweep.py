from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def iter_result_rows(paths: Iterable[Path]) -> Iterable[dict]:
    for path in paths:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        motif = result.get("motif", {})
        for system, block in result.get("systems", {}).items():
            row = block.get("best_feasible_test", {}).get("selected_test", {})
            if not row:
                continue
            yield {
                "source": Path(path).name,
                "system": system,
                "radius": motif.get("radius"),
                "min_similarity": motif.get("min_similarity"),
                "exclude_radius": motif.get("exclude_radius"),
                "precision": row.get("precision", 0.0),
                "recall": row.get("recall", 0.0),
                "f1": row.get("f1", 0.0),
                "tp": row.get("tp", 0),
                "fp": row.get("fp", 0),
                "fn": row.get("fn", 0),
                "threshold": row.get("threshold"),
                "selected_fdr": row.get("selected_fdr"),
            }


def best_rows(
    rows: Iterable[dict],
    *,
    min_precision: float = 0.80,
    system: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    feasible = [
        row
        for row in rows
        if float(row["precision"]) >= min_precision
        and (system is None or row["system"] == system)
    ]
    ranked = sorted(
        feasible,
        key=lambda row: (
            float(row["recall"]),
            float(row["precision"]),
            float(row["f1"]),
        ),
        reverse=True,
    )
    return ranked[:limit] if limit is not None else ranked


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return ""
    return str(value)


def write_markdown(rows: Iterable[dict], output: Path) -> None:
    columns = [
        "source",
        "system",
        "radius",
        "min_similarity",
        "exclude_radius",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
    ]
    lines = [
        "# Motif/Repetition Sweep Summary",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(col)) for col in columns) + " |")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_paths", nargs="+", type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--min-precision", type=float, default=0.80)
    parser.add_argument("--system", default=None)
    parser.add_argument("--limit", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = best_rows(
        iter_result_rows(args.json_paths),
        min_precision=args.min_precision,
        system=args.system,
        limit=args.limit,
    )
    write_markdown(rows, args.output_md)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
