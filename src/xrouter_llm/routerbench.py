from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd
from huggingface_hub import hf_hub_download

from xrouter_llm.types import BenchmarkRow


ROUTERBENCH_DATASET_ID = "withmartian/routerbench"
ROUTERBENCH_FILES = {
    "0shot": "routerbench_0shot.pkl",
    "5shot": "routerbench_5shot.pkl",
    "raw": "routerbench_raw.pkl",
}
ROUTERBENCH_METADATA_COLUMNS = {
    "sample_id",
    "prompt",
    "eval_name",
    "oracle_model_to_route_to",
}


def download_routerbench(
    *,
    split: Literal["0shot", "5shot", "raw"] = "0shot",
    output_dir: str | Path = "data/raw",
) -> Path:
    if split not in ROUTERBENCH_FILES:
        raise ValueError(f"Unknown RouterBench split {split!r}; expected one of {sorted(ROUTERBENCH_FILES)}")

    path = hf_hub_download(
        repo_id=ROUTERBENCH_DATASET_ID,
        filename=ROUTERBENCH_FILES[split],
        repo_type="dataset",
        local_dir=str(output_dir),
    )
    return Path(path)


def load_routerbench_pickle(
    path: str | Path,
    *,
    max_prompts: int | None = None,
    model_ids: Sequence[str] | None = None,
    eval_names: Sequence[str] | None = None,
    random_state: int | None = 0,
) -> list[BenchmarkRow]:
    dataframe = pd.read_pickle(path)
    return routerbench_dataframe_to_rows(
        dataframe,
        max_prompts=max_prompts,
        model_ids=model_ids,
        eval_names=eval_names,
        random_state=random_state,
    )


def routerbench_dataframe_to_rows(
    dataframe: pd.DataFrame,
    *,
    max_prompts: int | None = None,
    model_ids: Sequence[str] | None = None,
    eval_names: Sequence[str] | None = None,
    random_state: int | None = 0,
) -> list[BenchmarkRow]:
    _validate_routerbench_dataframe(dataframe)
    df = dataframe.copy()

    if eval_names is not None:
        eval_name_set = set(eval_names)
        df = df[df["eval_name"].isin(eval_name_set)]

    if max_prompts is not None:
        if max_prompts < 1:
            raise ValueError("max_prompts must be at least 1")
        if len(df) > max_prompts:
            df = df.sample(n=max_prompts, random_state=random_state)

    selected_models = list(model_ids) if model_ids is not None else infer_routerbench_model_ids(df)
    missing_models = [model_id for model_id in selected_models if model_id not in df.columns]
    if missing_models:
        raise ValueError(f"RouterBench file is missing score columns for models: {missing_models}")

    rows: list[BenchmarkRow] = []
    for _, record in df.iterrows():
        prompt_id = str(record["sample_id"])
        prompt = normalize_routerbench_prompt(record["prompt"])
        task = str(record["eval_name"])

        for model_id in selected_models:
            score = record[model_id]
            if pd.isna(score):
                continue

            cost_column = f"{model_id}|total_cost"
            cost = record.get(cost_column)
            rows.append(
                BenchmarkRow(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    model_id=model_id,
                    score=float(score),
                    cost_usd=None if cost is None or pd.isna(cost) else float(cost),
                    task=task,
                )
            )

    if not rows:
        raise ValueError("RouterBench conversion produced no rows")
    return rows


def infer_routerbench_model_ids(dataframe: pd.DataFrame) -> list[str]:
    return [
        column
        for column in dataframe.columns
        if column not in ROUTERBENCH_METADATA_COLUMNS and "|" not in column
    ]


def normalize_routerbench_prompt(prompt: object) -> str:
    if isinstance(prompt, (list, tuple)):
        return "\n".join(str(part) for part in prompt)
    if not isinstance(prompt, str):
        return str(prompt)

    try:
        parsed = ast.literal_eval(prompt)
    except (ValueError, SyntaxError):
        return prompt

    if isinstance(parsed, (list, tuple)):
        return "\n".join(str(part) for part in parsed)
    return prompt


def write_jsonl(rows: Sequence[BenchmarkRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def _validate_routerbench_dataframe(dataframe: pd.DataFrame) -> None:
    required_columns = {"sample_id", "prompt", "eval_name"}
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(f"Not a RouterBench dataframe; missing columns: {sorted(missing)}")

    score_columns = infer_routerbench_model_ids(dataframe)
    if not score_columns:
        raise ValueError("Not a RouterBench dataframe; no model score columns found")
