from xrouter_llm import (
    BenchmarkProfileCatalog,
    ModelBenchmarkProfile,
    load_builtin_benchmark_profiles,
)


def test_builtin_profiles_cover_routerbench_models() -> None:
    catalog = load_builtin_benchmark_profiles()

    assert len(catalog) == 11
    claude = catalog.get("claude-v2")
    assert round(claude.normalized_benchmark("human_eval") or 0.0, 3) == 0.712
    assert catalog.get("GPT-4").model_id == "gpt-4-1106-preview"


def test_profile_catalog_merges_duplicate_model_profiles() -> None:
    catalog = BenchmarkProfileCatalog(
        [
            ModelBenchmarkProfile(model_id="model-a", benchmarks={"mmlu": 80.0}, aliases=("a",)),
            ModelBenchmarkProfile(
                model_id="model-a",
                benchmarks={"llmrouterbench_math": 0.7},
                source_quality="dataset_aggregate",
            ),
        ]
    )

    profile = catalog.get("model-a")

    assert profile.benchmarks["mmlu"] == 80.0
    assert profile.benchmarks["llmrouterbench_math"] == 0.7
    assert catalog.get("a").model_id == "model-a"
