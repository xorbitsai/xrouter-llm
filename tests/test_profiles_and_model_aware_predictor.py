from xrouter_llm import (
    BenchmarkProfileCatalog,
    BenchmarkRow,
    ModelAwareRouterPredictor,
    ModelBenchmarkProfile,
    load_builtin_benchmark_profiles,
)


def test_builtin_profiles_cover_routerbench_models() -> None:
    catalog = load_builtin_benchmark_profiles()

    assert len(catalog) == 11
    claude = catalog.get("claude-v2")
    assert round(claude.normalized_benchmark("human_eval") or 0.0, 3) == 0.712
    assert catalog.get("GPT-4").model_id == "gpt-4-1106-preview"


def test_model_aware_predictor_uses_profile_for_unseen_model() -> None:
    profiles = BenchmarkProfileCatalog(
        [
            ModelBenchmarkProfile(
                model_id="model-a",
                provider="provider-a",
                benchmarks={"mmlu": 50.0, "human_eval": 40.0},
            ),
            ModelBenchmarkProfile(
                model_id="model-b",
                provider="provider-b",
                benchmarks={"mmlu": 80.0, "human_eval": 70.0},
            ),
            ModelBenchmarkProfile(
                model_id="model-c",
                provider="provider-c",
                benchmarks={"mmlu": 90.0, "human_eval": 85.0},
            ),
        ]
    )
    rows = [
        BenchmarkRow("p1", "Answer a knowledge question", "model-a", 0.4),
        BenchmarkRow("p1", "Answer a knowledge question", "model-b", 0.8),
        BenchmarkRow("p2", "Write a Python function", "model-a", 0.5),
        BenchmarkRow("p2", "Write a Python function", "model-b", 0.9),
    ]

    predictor = ModelAwareRouterPredictor(
        benchmark_profiles=profiles,
        ensemble_size=3,
        random_state=1,
    ).fit(rows)
    predictions = predictor.predict("Write a Python parser", model_ids=["model-a", "model-c"])

    assert [prediction.model_id for prediction in predictions] == ["model-a", "model-c"]
    assert all(0.0 <= prediction.mu <= 1.0 for prediction in predictions)
    assert predictions[1].sigma >= predictor.min_sigma
    assert "model-c" not in predictor.trained_model_ids_


def test_predictor_can_register_new_profile_after_training() -> None:
    profiles = BenchmarkProfileCatalog(
        [
            ModelBenchmarkProfile(model_id="model-a", benchmarks={"mmlu": 50.0}),
            ModelBenchmarkProfile(model_id="model-b", benchmarks={"mmlu": 80.0}),
        ]
    )
    rows = [
        BenchmarkRow("p1", "Answer a knowledge question", "model-a", 0.4),
        BenchmarkRow("p1", "Answer a knowledge question", "model-b", 0.8),
        BenchmarkRow("p2", "Write a Python function", "model-a", 0.5),
        BenchmarkRow("p2", "Write a Python function", "model-b", 0.9),
    ]

    predictor = ModelAwareRouterPredictor(
        benchmark_profiles=profiles,
        ensemble_size=3,
        random_state=2,
    ).fit(rows)
    predictor.add_benchmark_profile(
        ModelBenchmarkProfile(model_id="new-model", benchmarks={"mmlu": 90.0})
    )

    prediction = predictor.predict("Answer this question", model_ids=["new-model"])[0]

    assert prediction.model_id == "new-model"
    assert 0.0 <= prediction.mu <= 1.0
    assert prediction.sigma >= predictor.min_sigma


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
