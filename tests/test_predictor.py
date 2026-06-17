from xrouter_llm import BenchmarkRow, ModelAwareRouterPredictor


def test_predictor_returns_bounded_quality_distribution() -> None:
    rows = [
        BenchmarkRow("p1", "Refactor Python code", "claude", 0.92),
        BenchmarkRow("p1", "Refactor Python code", "gpt", 0.86),
        BenchmarkRow("p2", "Solve algebra problem", "claude", 0.80),
        BenchmarkRow("p2", "Solve algebra problem", "gpt", 0.91),
        BenchmarkRow("p3", "Debug async JavaScript", "claude", 0.94),
        BenchmarkRow("p3", "Debug async JavaScript", "gpt", 0.87),
    ]

    predictor = ModelAwareRouterPredictor(
        ensemble_size=4,
        completion_score_threshold=0.9,
        random_state=3,
    ).fit(rows)
    predictions = predictor.predict("Refactor this Python class", model_ids=["claude", "gpt"])

    assert [prediction.model_id for prediction in predictions] == ["claude", "gpt"]
    for prediction in predictions:
        assert 0.0 <= prediction.mu <= 1.0
        assert 0.03 <= prediction.sigma <= 0.30
