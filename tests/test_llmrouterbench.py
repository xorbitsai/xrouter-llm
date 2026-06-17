import json

from xrouter_llm import extract_llmrouterbench_profiles, load_llmrouterbench


def test_load_llmrouterbench_directory_format(tmp_path) -> None:
    records_dir = tmp_path / "results" / "bench" / "math" / "test" / "model-a"
    records_dir.mkdir(parents=True)
    (records_dir / "run.json").write_text(
        json.dumps(
            [
                {
                    "origin_query": "What is 1+1?",
                    "prompt": "What is 1+1?",
                    "prediction": "2",
                    "ground_truth": "2",
                    "score": 1,
                    "prompt_tokens": 1000,
                    "completion_tokens": 1000,
                    "cost": 0.003,
                },
                {
                    "origin_query": "What is 2+2?",
                    "prompt": "What is 2+2?",
                    "prediction": "4",
                    "ground_truth": "4",
                    "score": 0,
                    "prompt_tokens": 2000,
                    "completion_tokens": 1000,
                    "cost": 0.004,
                },
            ]
        ),
        encoding="utf-8",
    )

    rows = load_llmrouterbench(tmp_path)

    assert len(rows) == 2
    assert {row.model_id for row in rows} == {"model-a"}
    assert {row.task for row in rows} == {"math"}
    assert all(row.cost_usd is not None for row in rows)


def test_extract_llmrouterbench_profiles_aggregates_scores_and_costs(tmp_path) -> None:
    model_dir = tmp_path / "results" / "bench" / "math" / "test" / "model-a"
    model_dir.mkdir(parents=True)
    (model_dir / "run.jsonl").write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "prompt": "p1",
                    "score": 1.0,
                    "prompt_tokens": 1000,
                    "completion_tokens": 1000,
                    "cost": 0.003,
                },
                {
                    "prompt": "p2",
                    "score": 0.0,
                    "prompt_tokens": 2000,
                    "completion_tokens": 1000,
                    "cost": 0.004,
                },
                {
                    "prompt": "p3",
                    "score": 1.0,
                    "prompt_tokens": 1000,
                    "completion_tokens": 2000,
                    "cost": 0.005,
                },
            ]
        ),
        encoding="utf-8",
    )

    profiles = extract_llmrouterbench_profiles(tmp_path)

    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.model_id == "model-a"
    assert round(profile.benchmarks["llmrouterbench_math"], 3) == 0.667
    assert round(profile.benchmarks["llmrouterbench_overall"], 3) == 0.667
    assert round(profile.input_cost_per_1k or 0.0, 3) == 0.001
    assert round(profile.output_cost_per_1k or 0.0, 3) == 0.002
