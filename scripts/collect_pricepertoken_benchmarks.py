"""Fill known benchmark gaps from PricePerToken/Artificial Analysis payloads.

The PricePerToken benchmark pages are Nuxt pages with a devalue payload that
contains the full model table. This script extracts that payload and updates
``scripts/benchmarks_to_collect.csv`` conservatively:

* only exact slug/name matches plus explicit aliases are used;
* existing CSV values are never overwritten unless ``--overwrite`` is passed;
* for models with reasoning and non-reasoning rows, the row closest to already
  filled CSV values is selected.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.parse
import urllib.request
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_CSV = Path("scripts/benchmarks_to_collect.csv")
DEFAULT_PAGE = "https://pricepertoken.com/leaderboards/benchmark/hle"

CSV_TO_PPT = {
    "mmlu_pro": "benchmark_mmlu_pro",
    "hle": "benchmark_hle",
    "aime_2025": "benchmark_aime_25",
    "tau2_bench": "benchmark_tau2",
    "ifbench": "benchmark_ifbench",
    "simpleqa": "benchmark_simpleqa",
    "scicode": "benchmark_scicode",
    "aa_lcr": "benchmark_lcr",
    "terminal_bench": "benchmark_terminalbench",
    "humaneval": "benchmark_humaneval",
}

PRIORITY_FIELDS = (
    "mmlu_pro",
    "hle",
    "aime_2025",
    "tau2_bench",
    "ifbench",
    "scicode",
    "aa_lcr",
)

# Explicit aliases are deliberately narrow. Similar-but-not-identical models are
# left blank instead of being guessed.
MODEL_ALIASES = {
    "DeepSeek-R1-0528": "deepseek-deepseek-r1-0528",
    "Llama-3.1-8B-Instruct": "meta-llama-llama-3.1-8b-instruct",
    "NVIDIA-Nemotron-Nano-9B-v2": "nvidia-nemotron-nano-9b-v2",
    "Qwen2.5-Coder-7B-Instruct": "qwen-qwen2.5-coder-7b-instruct",
    "Qwen3-8B": "qwen-qwen3-8b",
    "claude-sonnet-4": "anthropic-claude-sonnet-4",
    "deepseek-v3-0324": "deepseek-deepseek-chat-v3-0324",
    "deepseek-r1-0528": "deepseek-deepseek-r1-0528",
    "deepseek-v3.1-terminus": "deepseek-deepseek-v3.1-terminus",
    "gemini-2.5-flash": "google-gemini-2.5-flash",
    "gemini-2.5-pro": "google-gemini-2.5-pro",
    "glm-4.6": "z-ai-glm-4.6",
    "gpt-4.1": "openai-gpt-4.1",
    "gpt-5": "openai-gpt-5",
    "gpt-5-chat": "openai-gpt-5-chat",
    "kimi-k2-0905": "moonshotai-kimi-k2-0905",
    "qwen3-235b-a22b-2507": "qwen-qwen3-235b-a22b-2507",
    "qwen3-235b-a22b-no-thinking": "qwen-qwen3-235b-a22b",
    "qwen3-235b-a22b-thinking": "qwen-qwen3-235b-a22b",
    "qwen3-235b-a22b-thinking-2507": "qwen-qwen3-235b-a22b-thinking-2507",
    "anthropic/claude-sonnet-4.6": "anthropic-claude-sonnet-4.6",
    "deepseek/deepseek-v4-flash": "deepseek-deepseek-v4-flash",
    "deepseek/deepseek-v4-pro": "deepseek-deepseek-v4-pro",
    "google/gemini-2.5-flash-lite": "google-gemini-2.5-flash-lite",
    "google/gemini-3-flash-preview": "google-gemini-3-flash-preview",
    "minimax/minimax-m3": "minimax-minimax-m3",
    "z-ai/glm-4.7": "z-ai-glm-4.7",
}

FORCED_MODE = {
    "qwen3-235b-a22b-no-thinking": "standard",
    "qwen3-235b-a22b-thinking": "reasoning",
    "qwen3-235b-a22b-thinking-2507": "reasoning",
    "anthropic/claude-sonnet-4.6": "standard",
    "deepseek/deepseek-v4-flash": "standard",
    "deepseek/deepseek-v4-pro": "standard",
    "minimax/minimax-m3": "standard",
}

OVERLAP_FIELDS = {
    "gpqa_diamond": "benchmark_gpqa",
    "livecodebench": "benchmark_livecodebench",
    "mmlu_pro": "benchmark_mmlu_pro",
    "hle": "benchmark_hle",
    "aime_2025": "benchmark_aime_25",
    "tau2_bench": "benchmark_tau2",
    "ifbench": "benchmark_ifbench",
    "simpleqa": "benchmark_simpleqa",
    "scicode": "benchmark_scicode",
    "aa_lcr": "benchmark_lcr",
    "terminal_bench": "benchmark_terminalbench",
    "humaneval": "benchmark_humaneval",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--page", default=DEFAULT_PAGE)
    parser.add_argument("--payload-file", type=Path)
    parser.add_argument("--fields", nargs="*", default=list(PRIORITY_FIELDS))
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    models = load_pricepertoken_models(args.page, payload_file=args.payload_file)
    rows = read_csv(args.csv)
    changes: list[tuple[str, str, str, str, str]] = []
    for row in rows:
        match = choose_match(row, models)
        if match is None:
            continue
        for csv_field in args.fields:
            if csv_field not in CSV_TO_PPT:
                raise ValueError(f"Unsupported field: {csv_field}")
            if row.get(csv_field) and not args.overwrite:
                continue
            value = match.get(CSV_TO_PPT[csv_field])
            if value is None:
                continue
            formatted = format_score(value)
            old_value = row.get(csv_field, "")
            row[csv_field] = formatted
            changes.append((row["model_id"], csv_field, old_value, formatted, match["slug"]))

    for model_id, field, old_value, new_value, source_slug in changes:
        before = old_value or "<blank>"
        print(f"{model_id}: {field} {before} -> {new_value} ({source_slug})")
    print(f"changes={len(changes)}")

    if args.write and changes:
        write_csv(args.csv, rows)


def load_pricepertoken_models(page_url: str, payload_file: Path | None = None) -> list[dict[str, Any]]:
    if payload_file is None:
        html = fetch_text(page_url)
        match = re.search(r'href="([^"]*_payload\.json[^"]*)"', html)
        if not match:
            raise RuntimeError(f"No payload link found in {page_url}")
        payload_url = urllib.parse.urljoin(page_url, match.group(1))
        payload = json.loads(fetch_text(payload_url))
    else:
        payload = json.loads(payload_file.read_text(encoding="utf-8"))
    root = resolve_devalue(payload, 3)
    return root["benchmarkModels"]


def resolve_devalue(payload: list[Any], start_index: int) -> Any:
    @lru_cache(maxsize=None)
    def resolve_index(index: int) -> Any:
        value = payload[index]
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            if value and value[0] in {"Reactive", "ShallowReactive", "Ref"} and isinstance(value[1], int):
                return resolve_index(value[1])
            return [resolve(item) for item in value]
        return value

    def resolve(value: Any) -> Any:
        if isinstance(value, int) and 0 <= value < len(payload):
            return resolve_index(value)
        return value

    return resolve_index(start_index)


def choose_match(row: Mapping[str, str], models: list[dict[str, Any]]) -> dict[str, Any] | None:
    wanted_slug = MODEL_ALIASES.get(row["model_id"])
    if wanted_slug is None:
        wanted_slug = MODEL_ALIASES.get(row["model_id"].split("/")[-1])
    if wanted_slug is None:
        return exact_match(row["model_id"], models)

    candidates = [model for model in models if model.get("slug") == wanted_slug]
    if not candidates:
        return None
    mode = FORCED_MODE.get(row["model_id"]) or FORCED_MODE.get(row["model_id"].split("/")[-1])
    if mode is not None:
        mode_candidates = [model for model in candidates if model.get("inference_mode") == mode]
        if mode_candidates:
            candidates = mode_candidates
    return min(candidates, key=lambda model: overlap_distance(row, model))


def exact_match(model_id: str, models: list[dict[str, Any]]) -> dict[str, Any] | None:
    keys = {normalize(model_id), normalize(model_id.split("/")[-1])}
    matches = []
    for model in models:
        values = {
            normalize(str(model.get("slug") or "")),
            normalize(str(model.get("model_name") or "")),
            normalize(str(model.get("name") or "")),
            normalize(str(model.get("aa_name") or "")),
            normalize(str(model.get("provider_id") or "")),
        }
        if keys & values:
            matches.append(model)
    if not matches:
        return None
    return min(matches, key=lambda model: overlap_distance({"model_id": model_id}, model))


def overlap_distance(row: Mapping[str, str], model: Mapping[str, Any]) -> float:
    diffs = []
    for csv_field, ppt_field in OVERLAP_FIELDS.items():
        row_value = row.get(csv_field)
        model_value = model.get(ppt_field)
        if not row_value or model_value is None:
            continue
        try:
            diffs.append(abs(float(row_value) - float(model_value)))
        except ValueError:
            continue
    if not diffs:
        return 0.0
    return sum(diffs) / len(diffs)


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def format_score(value: object) -> str:
    score = round(float(value), 1)
    return f"{score:.1f}"


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_error: Exception | None = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8")
        except Exception as exc:  # pragma: no cover - network retry path
            last_error = exc
    assert last_error is not None
    raise last_error


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
