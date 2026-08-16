from xrouter_llm import (
    BenchmarkProfileCatalog,
    ModelBenchmarkProfile,
    load_builtin_benchmark_profiles,
    normalize_modalities,
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
            ModelBenchmarkProfile(
                model_id="model-a",
                benchmarks={"mmlu": 80.0},
                aliases=("a",),
                input_modalities=("text", "image"),
            ),
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
    assert profile.input_modalities == ("text", "image")
    assert catalog.get("a").model_id == "model-a"


def test_normalize_modalities_handles_empty_scalar_and_mixed_values() -> None:
    assert normalize_modalities(None) == ()
    assert normalize_modalities(" Image ") == ("image",)
    assert normalize_modalities(7) == ("7",)
    assert normalize_modalities([" Image ", None, "", "IMAGE", "Audio"]) == (
        "image",
        "audio",
    )


def test_profile_mapping_accepts_null_input_modalities() -> None:
    profile = ModelBenchmarkProfile.from_mapping(
        {"model_id": "model-a", "input_modalities": None}
    )

    assert profile.input_modalities == ()
