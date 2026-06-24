from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import hf_hub_download

from xrouter_llm.types import BenchmarkRow


DEFAULT_XAGENT_LABEL_REPO = "Xorbits/xagent-xrouter-labels"

XAGENT_LABEL_FILES = {
    "full": "xagent_openrouter_official_candidates_100_4models.public.jsonl",
    "train80": "xagent_openrouter_official_candidates_100_4models_train80.public.jsonl",
    "holdout20": "xagent_openrouter_official_candidates_100_4models_holdout20.public.jsonl",
    "sample": "xagent_openrouter_official_sample.public.jsonl",
}


def load_xagent_openrouter_labels(
    source: str | Path = DEFAULT_XAGENT_LABEL_REPO,
    *,
    split: str = "full",
) -> list[BenchmarkRow]:
    """Load scrubbed xagent routing labels from a local JSONL file or HF dataset.

    The public HF dataset currently contains plain JSONL files. Loading through
    ``huggingface_hub`` keeps the dependency surface small and avoids relying on
    a generated datasets schema.
    """

    path = Path(source).expanduser()
    if path.exists():
        return _load_xagent_jsonl(path)

    repo_id, resolved_split = _parse_hf_source(str(source), split=split)
    filename = XAGENT_LABEL_FILES.get(resolved_split, resolved_split)
    downloaded = hf_hub_download(repo_id, filename, repo_type="dataset")
    return _load_xagent_jsonl(Path(downloaded))


def _parse_hf_source(source: str, *, split: str) -> tuple[str, str]:
    if ":" not in source:
        return source, split

    repo_id, maybe_split = source.rsplit(":", 1)
    if "/" not in maybe_split:
        return repo_id, maybe_split
    return source, split


def _load_xagent_jsonl(path: Path) -> list[BenchmarkRow]:
    output: list[BenchmarkRow] = []
    for line_number, record in _iter_jsonl(path):
        score = _score(record, line_number=line_number)
        prompt = str(record.get("prompt") or "")
        model_id = str(record.get("candidate_model") or "")
        if not prompt:
            raise ValueError(f"Missing prompt in xagent label line {line_number}")
        if not model_id:
            raise ValueError(f"Missing candidate_model in xagent label line {line_number}")

        prompt_sha = record.get("prompt_sha256")
        prompt_id = f"xagent:{prompt_sha}" if prompt_sha else _fallback_prompt_id(record)
        output.append(
            BenchmarkRow(
                prompt_id=prompt_id,
                prompt=prompt,
                model_id=model_id,
                score=score,
                cost_usd=_usage_cost(record.get("candidate_usage")),
                task=f"xagent:{record.get('category') or 'unknown'}",
            )
        )
    if not output:
        raise ValueError(f"No xagent labels loaded from {path}")
    return output


load_xagent_labels = load_xagent_openrouter_labels


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid xagent JSONL at line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Invalid xagent JSONL at line {line_number}: expected object")
            yield line_number, value


def _score(record: dict[str, Any], *, line_number: int) -> float:
    judge = record.get("judge")
    if not isinstance(judge, dict) or judge.get("score") is None:
        raise ValueError(f"Missing judge.score in xagent label line {line_number}")
    score = float(judge["score"])
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"judge.score must be in [0, 1] at line {line_number}: {score}")
    return score


def _fallback_prompt_id(record: dict[str, Any]) -> str:
    event_id = record.get("event_id", "unknown")
    end_event_id = record.get("end_event_id", "unknown")
    category = record.get("category", "unknown")
    return f"xagent:{event_id}:{end_event_id}:{category}"


def _usage_cost(usage: object) -> float | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get("cost")
    if value is None:
        return None
    return float(value)
