"""Load agent-psychometrics per-(task, agent) outcome matrices as BenchmarkRows.

Source: https://github.com/dariakryvosheieva/agent-psychometrics (MIT). Each
`data/<dataset>/responses.jsonl` line is one subject (an LLM x scaffold combo)
with a `responses` map {task_id: 0/1}. Task text comes from `tasks.jsonl`
(`problem_statement`) when present, else from an external `task_text` mapping
(e.g. SWE-bench Verified instance problem statements).

These are agentic tasks (coding-agent, terminal/CLI) -- much closer to real
xagent prompts than the single-turn llmrouterbench tasks, so they enrich the
difficulty model. The subjects have no benchmark profiles, so they feed the
difficulty axis only (the capability axis stays on profiled training models).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Mapping

from xrouter_llm.types import BenchmarkRow


def _parse_responses(value: object) -> dict[str, int]:
    if isinstance(value, Mapping):
        return {str(k): int(v) for k, v in value.items()}
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        return {str(k): int(v) for k, v in parsed.items()}
    raise ValueError(f"Unexpected responses payload: {type(value).__name__}")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_agent_psychometrics(
    root: str | Path,
    dataset: str,
    *,
    task_text: Mapping[str, str] | None = None,
    namespace: str | None = None,
) -> list[BenchmarkRow]:
    """Load one agent-psychometrics dataset directory into BenchmarkRows.

    `root` is the repo root; `dataset` is e.g. "terminalbench",
    "swebench_verified". `task_text` overrides/provides task_id -> prompt text
    when the dataset has no local tasks.jsonl.
    """
    base = Path(root) / "data" / dataset
    responses_path = base / "responses.jsonl"
    if not responses_path.exists():
        raise FileNotFoundError(responses_path)

    text: dict[str, str] = dict(task_text or {})
    tasks_path = base / "tasks.jsonl"
    if not text and tasks_path.exists():
        for task in _read_jsonl(tasks_path):
            tid = str(task.get("task_id"))
            statement = task.get("problem_statement")
            if statement:
                text[tid] = str(statement)

    ns = namespace or f"agentpsych:{dataset}"
    rows: list[BenchmarkRow] = []
    skipped_no_text = 0
    for subject in _read_jsonl(responses_path):
        model_id = str(subject.get("subject_id"))
        for task_id, passed in _parse_responses(subject["responses"]).items():
            prompt = text.get(task_id)
            if prompt is None:
                skipped_no_text += 1
                continue
            rows.append(
                BenchmarkRow(
                    prompt_id=f"{ns}:{task_id}",
                    prompt=prompt,
                    model_id=model_id,
                    score=float(1 if passed else 0),
                    task=dataset,
                )
            )
    if not rows:
        raise ValueError(
            f"No rows for {dataset}: task text missing for all tasks "
            f"(skipped {skipped_no_text}); pass task_text=..."
        )
    return rows
