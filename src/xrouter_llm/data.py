from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from xrouter_llm.types import BenchmarkRow


def coerce_benchmark_row(row: BenchmarkRow | Mapping[str, Any]) -> BenchmarkRow:
    if isinstance(row, BenchmarkRow):
        return row

    return BenchmarkRow(
        prompt_id=str(row["prompt_id"]),
        prompt=str(row["prompt"]),
        model_id=str(row["model_id"]),
        score=float(row["score"]),
        cost_usd=_optional_float(row.get("cost_usd")),
        latency_s=_optional_float(row.get("latency_s")),
        task=None if row.get("task") is None else str(row.get("task")),
    )


def coerce_benchmark_rows(rows: Iterable[BenchmarkRow | Mapping[str, Any]]) -> list[BenchmarkRow]:
    return [coerce_benchmark_row(row) for row in rows]


def load_jsonl(path: str | Path) -> list[BenchmarkRow]:
    output: list[BenchmarkRow] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                output.append(coerce_benchmark_row(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
    return output


def load_csv(path: str | Path) -> list[BenchmarkRow]:
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        return [coerce_benchmark_row(row) for row in csv.DictReader(file)]


def split_by_prompt(
    rows: Sequence[BenchmarkRow | Mapping[str, Any]],
    *,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> tuple[list[BenchmarkRow], list[BenchmarkRow]]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")

    normalized = coerce_benchmark_rows(rows)
    prompt_ids = sorted({row.prompt_id for row in normalized})
    if len(prompt_ids) < 2:
        raise ValueError("At least two prompt_id values are required for a train/test split")

    rng = np.random.default_rng(random_state)
    shuffled = list(prompt_ids)
    rng.shuffle(shuffled)

    test_count = max(1, int(round(len(shuffled) * test_size)))
    test_ids = set(shuffled[:test_count])

    train_rows = [row for row in normalized if row.prompt_id not in test_ids]
    test_rows = [row for row in normalized if row.prompt_id in test_ids]
    return train_rows, test_rows


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
