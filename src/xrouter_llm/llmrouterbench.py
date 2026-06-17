from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download

from xrouter_llm.profiles import ModelBenchmarkProfile
from xrouter_llm.types import BenchmarkRow


LLMROUTERBENCH_RECORD_KEYS = {
    "origin_query",
    "prompt",
    "prediction",
    "ground_truth",
    "score",
    "prompt_tokens",
    "completion_tokens",
    "cost",
}
LLMROUTERBENCH_DATASET_ID = "NPULH/LLMRouterBench"
LLMROUTERBENCH_ARCHIVE = "bench-release.tar.gz"


def download_llmrouterbench(*, output_dir: str | Path = "data/raw") -> Path:
    path = hf_hub_download(
        repo_id=LLMROUTERBENCH_DATASET_ID,
        filename=LLMROUTERBENCH_ARCHIVE,
        repo_type="dataset",
        local_dir=str(output_dir),
    )
    return Path(path)


def load_llmrouterbench(
    path: str | Path,
    *,
    max_prompts: int | None = None,
    random_state: int | None = 0,
) -> list[BenchmarkRow]:
    """Load LLMRouterBench result JSON/JSONL files into BenchmarkRow records."""

    root = Path(path)
    rows: list[BenchmarkRow] = []
    for source in _iter_record_sources(root):
        metadata = _infer_path_metadata(source.path, root)
        for index, record in enumerate(source.records):
            rows.append(
                llmrouterbench_record_to_row(
                    record,
                    fallback_model_id=metadata["model_id"],
                    fallback_task=metadata["task"],
                    fallback_split=metadata["split"],
                    source_path=source.path,
                    record_index=index,
                )
            )

    if not rows:
        raise ValueError(f"LLMRouterBench loader produced no rows from {root}")
    return _limit_rows_by_prompt(rows, max_prompts=max_prompts, random_state=random_state)


def extract_llmrouterbench_profiles(path: str | Path) -> list[ModelBenchmarkProfile]:
    """Build benchmark profiles from LLMRouterBench aggregate scores and cost rows."""

    root = Path(path)
    score_by_model_task: dict[str, dict[str, list[float]]] = {}
    cost_observations: dict[str, list[tuple[float, float, float]]] = {}

    for source in _iter_record_sources(root):
        metadata = _infer_path_metadata(source.path, root)
        for index, record in enumerate(source.records):
            row = llmrouterbench_record_to_row(
                record,
                fallback_model_id=metadata["model_id"],
                fallback_task=metadata["task"],
                fallback_split=metadata["split"],
                source_path=source.path,
                record_index=index,
            )
            task = row.task or metadata["task"] or "unknown"
            model_scores = score_by_model_task.setdefault(row.model_id, {})
            model_scores.setdefault(_benchmark_key(task), []).append(row.score)
            model_scores.setdefault("llmrouterbench_overall", []).append(row.score)

            prompt_tokens = _optional_float(_first_present(record, "prompt_tokens", "input_tokens"))
            completion_tokens = _optional_float(
                _first_present(record, "completion_tokens", "output_tokens")
            )
            cost = _optional_float(_first_present(record, "cost", "cost_usd", "total_cost"))
            if prompt_tokens is not None and completion_tokens is not None and cost is not None:
                cost_observations.setdefault(row.model_id, []).append(
                    (prompt_tokens / 1000.0, completion_tokens / 1000.0, cost)
                )

    profiles: list[ModelBenchmarkProfile] = []
    for model_id in sorted(score_by_model_task):
        input_cost, output_cost = _fit_token_costs(cost_observations.get(model_id, []))
        profiles.append(
            ModelBenchmarkProfile(
                model_id=model_id,
                benchmarks={
                    benchmark: float(np.mean(scores))
                    for benchmark, scores in sorted(score_by_model_task[model_id].items())
                },
                source_quality="dataset_aggregate",
                source_urls=("https://huggingface.co/datasets/NPULH/LLMRouterBench",),
                input_cost_per_1k=input_cost,
                output_cost_per_1k=output_cost,
            )
        )
    return profiles


def llmrouterbench_record_to_row(
    record: Mapping[str, Any],
    *,
    fallback_model_id: str,
    fallback_task: str | None,
    fallback_split: str | None,
    source_path: Path,
    record_index: int,
) -> BenchmarkRow:
    prompt = _first_present(record, "prompt", "origin_query", "query", "question", "input")
    if prompt is None:
        raise ValueError(f"Missing prompt/origin_query in {source_path} record {record_index}")

    model_id = _first_present(record, "model_id", "model", "llm", "generator") or fallback_model_id
    if not model_id:
        raise ValueError(f"Missing model id in {source_path} record {record_index}")

    score_value = _first_present(record, "score", "accuracy", "correct", "reward")
    if score_value is None:
        raise ValueError(f"Missing score in {source_path} record {record_index}")

    prompt_id = _first_present(record, "prompt_id", "sample_id", "question_id", "id")
    if prompt_id is None:
        prompt_id = _stable_prompt_id(prompt=str(prompt), task=fallback_task, split=fallback_split)

    task = _first_present(record, "dataset", "task", "benchmark", "eval_name") or fallback_task
    return BenchmarkRow(
        prompt_id=str(prompt_id),
        prompt=str(prompt),
        model_id=str(model_id),
        score=_coerce_score(score_value),
        cost_usd=_optional_float(_first_present(record, "cost", "cost_usd", "total_cost")),
        latency_s=_optional_float(_first_present(record, "latency", "latency_s", "latency_seconds")),
        task=None if task is None else str(task),
    )


class _RecordSource:
    def __init__(self, path: Path, records: list[Mapping[str, Any]]) -> None:
        self.path = path
        self.records = records


def _iter_record_sources(root: Path) -> list[_RecordSource]:
    if root.is_file() and tarfile.is_tarfile(root):
        return _iter_tar_record_sources(root)
    return [_RecordSource(path, _read_records(path)) for path in _iter_result_files(root)]


def _iter_tar_record_sources(root: Path) -> list[_RecordSource]:
    sources: list[_RecordSource] = []
    with tarfile.open(root, "r:*") as archive:
        for member in sorted(archive.getmembers(), key=lambda item: item.name):
            if not member.isfile():
                continue
            member_path = Path(member.name)
            if member_path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            file = archive.extractfile(member)
            if file is None:
                continue
            payload = file.read()
            sources.append(
                _RecordSource(member_path, _read_records_from_text(member_path, payload.decode("utf-8")))
            )
    return sources


def _iter_result_files(root: Path) -> list[Path]:
    if root.is_file():
        if root.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError(f"Unsupported LLMRouterBench file extension: {root}")
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
    ]
    return sorted(files)


def _read_records(path: Path) -> list[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return _read_records_from_text(path, file.read())


def _read_records_from_text(path: Path, text: str) -> list[Mapping[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(io.StringIO(text), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, Mapping):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(value)
        return records

    value = json.loads(text)
    return list(_records_from_json(value, source_path=path))


def _records_from_json(value: Any, *, source_path: Path) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, Mapping):
                raise ValueError(f"Expected JSON object in list from {source_path}")
            yield item
        return

    if isinstance(value, Mapping):
        if _looks_like_record(value):
            yield value
            return

        for key in ("data", "results", "records", "instances", "samples", "items"):
            nested = value.get(key)
            if nested is not None:
                yield from _records_from_json(nested, source_path=source_path)
                return

    raise ValueError(f"Could not find LLMRouterBench records in {source_path}")


def _looks_like_record(value: Mapping[str, Any]) -> bool:
    return "score" in value and bool(LLMROUTERBENCH_RECORD_KEYS.intersection(value))


def _infer_path_metadata(file_path: Path, root: Path) -> dict[str, str | None]:
    parts = list(file_path.parts)
    if "bench" in parts:
        index = len(parts) - 1 - parts[::-1].index("bench")
        return {
            "task": _part_after(parts, index, 1),
            "split": _part_after(parts, index, 2),
            "model_id": _part_after(parts, index, 3) or file_path.parent.name,
        }

    relative_parts = _relative_parts(file_path, root)
    task = relative_parts[0] if len(relative_parts) >= 4 else None
    split = relative_parts[1] if len(relative_parts) >= 4 else None
    model_id = relative_parts[2] if len(relative_parts) >= 4 else file_path.parent.name
    return {"task": task, "split": split, "model_id": model_id}


def _relative_parts(file_path: Path, root: Path) -> tuple[str, ...]:
    try:
        return file_path.relative_to(root if root.is_dir() else root.parent).parts
    except ValueError:
        return file_path.parts


def _part_after(parts: list[str], index: int, offset: int) -> str | None:
    target = index + offset
    return parts[target] if target < len(parts) - 1 else None


def _first_present(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def _stable_prompt_id(*, prompt: str, task: str | None, split: str | None) -> str:
    namespace = "\x1f".join(part or "" for part in (task, split, prompt))
    digest = hashlib.sha1(namespace.encode("utf-8")).hexdigest()[:16]
    return f"llmrouterbench:{digest}"


def _benchmark_key(task: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", task.lower()).strip("_")
    return f"llmrouterbench_{slug or 'unknown'}"


def _coerce_score(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, Mapping):
        nested = _first_present(value, "score", "accuracy", "correct", "reward")
        if nested is None:
            raise ValueError(f"Could not coerce score mapping: {value}")
        return _coerce_score(nested)
    return float(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _fit_token_costs(observations: list[tuple[float, float, float]]) -> tuple[float | None, float | None]:
    if len(observations) < 2:
        return None, None
    x = np.asarray([[prompt, completion] for prompt, completion, _ in observations], dtype=float)
    y = np.asarray([cost for _, _, cost in observations], dtype=float)
    if np.linalg.matrix_rank(x) < 2:
        return None, None
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    input_cost, output_cost = np.clip(coefficients, 0.0, None)
    return float(input_cost), float(output_cost)


def _limit_rows_by_prompt(
    rows: list[BenchmarkRow],
    *,
    max_prompts: int | None,
    random_state: int | None,
) -> list[BenchmarkRow]:
    if max_prompts is None:
        return rows
    if max_prompts < 1:
        raise ValueError("max_prompts must be at least 1")

    prompt_ids = sorted({row.prompt_id for row in rows})
    if len(prompt_ids) <= max_prompts:
        return rows

    rng = np.random.default_rng(random_state)
    selected = set(rng.choice(prompt_ids, size=max_prompts, replace=False).tolist())
    return [row for row in rows if row.prompt_id in selected]
