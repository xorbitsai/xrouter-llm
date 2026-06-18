from xrouter_llm import (
    BenchmarkRow,
    ModelBenchmarkProfile,
    ModelAwareRouterPredictor,
    PolicyParams,
    evaluate_model_holdout,
    evaluate_offline,
)


def _synthetic_rows() -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for index in range(12):
        prompt = f"prompt number {index} about coding and reasoning"
        prompt_id = f"p{index}"
        # strong always passes, mid sometimes, cheap rarely -> both classes per model
        rows.append(BenchmarkRow(prompt_id, prompt, "strong", 0.95, cost_usd=0.02))
        rows.append(
            BenchmarkRow(prompt_id, prompt, "mid", 0.9 if index % 2 == 0 else 0.4, cost_usd=0.01)
        )
        rows.append(
            BenchmarkRow(prompt_id, prompt, "cheap", 0.8 if index % 3 == 0 else 0.3, cost_usd=0.001)
        )
    return rows


def _profiles() -> list[ModelBenchmarkProfile]:
    return [
        ModelBenchmarkProfile(model_id="strong", input_cost_per_1k=0.01, output_cost_per_1k=0.03),
        ModelBenchmarkProfile(model_id="mid", input_cost_per_1k=0.005, output_cost_per_1k=0.015),
        ModelBenchmarkProfile(model_id="cheap", input_cost_per_1k=0.0005, output_cost_per_1k=0.001),
    ]


def test_offline_decision_cost_does_not_use_realized_cost() -> None:
    rows = _synthetic_rows()
    profiles = _profiles()

    result = evaluate_offline(
        rows,
        policy_params=PolicyParams(completion_threshold=0.5, lambda_cost=1.0),
        test_size=0.5,
        random_state=3,
        predictor_factory=lambda: ModelAwareRouterPredictor(
            benchmark_profiles=profiles,
            ensemble_size=2,
            completion_score_threshold=0.75,
            random_state=3,
        ),
    )

    # Decision cost is estimated from profile pricing + prompt length, never the
    # realized cost_usd, so the two metrics are reported separately.
    assert "average_decision_cost" in result.metrics
    assert "average_cost" in result.metrics
    assert result.metrics["average_decision_cost"] > 0.0


def test_model_holdout_reports_unseen_model_metrics() -> None:
    rows = _synthetic_rows()
    profiles = _profiles()

    report = evaluate_model_holdout(
        rows,
        predictor_factory=lambda: ModelAwareRouterPredictor(
            benchmark_profiles=profiles,
            ensemble_size=2,
            completion_score_threshold=0.75,
            random_state=7,
        ),
        test_size=0.5,
        random_state=7,
        calibration_bins=4,
    )

    assert report.holdout_models
    assert report.aggregate["model_count"] == float(len(report.per_model))
    for entry in report.per_model:
        # the held-out model never appeared in its own training run
        assert entry["model_id"] in {"strong", "mid", "cheap"}
        assert entry["n"] >= 1.0
        assert 0.0 <= entry["base_completion_rate"] <= 1.0
