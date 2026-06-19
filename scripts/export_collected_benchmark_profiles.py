"""Merge collected benchmark CSV values into a profile JSON file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_CSV = Path("scripts/benchmarks_to_collect.csv")
DEFAULT_INPUT = Path("artifacts/profiles/llmrouterbench_350k_profiles.json")
DEFAULT_OUTPUT = Path("artifacts/profiles/llmrouterbench_350k_profiles_collected.json")

NON_BENCHMARK_FIELDS = {"group", "model_id"}
SOURCE_URL = "https://pricepertoken.com/leaderboards/benchmark/hle"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--group", default="train")
    args = parser.parse_args()

    raw_profiles = json.loads(args.input.read_text(encoding="utf-8"))
    profiles = raw_profiles.get("models", raw_profiles)
    if not isinstance(profiles, list):
        raise ValueError(f"{args.input} must contain a profile list or {{'models': [...]}}")
    profiles_by_id = {str(profile["model_id"]): profile for profile in profiles}

    rows = [
        row
        for row in csv.DictReader(args.csv.open(newline="", encoding="utf-8"))
        if row.get("group") == args.group
    ]
    updated = 0
    missing_profiles: list[str] = []
    for row in rows:
        profile = profiles_by_id.get(row["model_id"])
        if profile is None:
            missing_profiles.append(row["model_id"])
            continue
        benchmarks = dict(profile.get("benchmarks", {}))
        for field, value in row.items():
            if field in NON_BENCHMARK_FIELDS or not value:
                continue
            old_value = benchmarks.get(field)
            new_value = float(value)
            if old_value != new_value:
                benchmarks[field] = new_value
                updated += 1
        profile["benchmarks"] = benchmarks
        urls = list(profile.get("source_urls", []))
        if SOURCE_URL not in urls:
            urls.append(SOURCE_URL)
        profile["source_urls"] = urls

    if missing_profiles:
        raise ValueError(f"CSV models missing from profile input: {missing_profiles}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload: Any = {"models": profiles} if isinstance(raw_profiles, dict) and "models" in raw_profiles else profiles
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"updated_benchmark_values={updated}")


if __name__ == "__main__":
    main()
