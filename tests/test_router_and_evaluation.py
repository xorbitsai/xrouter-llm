from xrouter_llm import (
    ModelProfile,
    ModelAwareRouterPredictor,
    PolicyParams,
    XRouter,
    build_fusion_prompt,
    evaluate_offline,
    evaluate_threshold_sweep,
    load_jsonl,
)


def test_router_returns_route_decision_with_costs() -> None:
    rows = load_jsonl("examples/benchmark.jsonl")
    predictor = ModelAwareRouterPredictor(
        ensemble_size=4,
        completion_score_threshold=0.9,
        random_state=5,
    ).fit(rows)
    router = XRouter(
        predictor,
        model_profiles=[
            ModelProfile("claude", input_cost_per_1k=0.003, output_cost_per_1k=0.015, base_latency_s=2.0),
            ModelProfile("gpt", input_cost_per_1k=0.002, output_cost_per_1k=0.010, base_latency_s=1.5),
            ModelProfile("deepseek", input_cost_per_1k=0.001, output_cost_per_1k=0.002, base_latency_s=1.0),
        ],
    )

    decision = router.route(
        "Find the bug in this Python retry loop.",
        policy_params=PolicyParams(max_k=2, allow_fusion=True),
    )

    assert 1 <= len(decision.selected_model_ids) <= 2
    assert decision.utility_breakdown.cost >= 0.0
    assert decision.utility_breakdown.latency >= 0.0


def test_offline_evaluation_reports_core_metrics() -> None:
    rows = load_jsonl("examples/benchmark.jsonl")

    result = evaluate_offline(
        rows,
        policy_params=PolicyParams(max_k=2, allow_fusion=True),
        test_size=0.33,
        random_state=11,
        predictor_factory=lambda: ModelAwareRouterPredictor(
            completion_score_threshold=0.9,
            random_state=11,
        ),
    )

    assert result.metrics["prompt_count"] >= 1.0
    assert "average_score" in result.metrics
    assert "completion_rate" in result.metrics
    assert "completion_probability_threshold" in result.metrics
    assert "completion_score_threshold" in result.metrics
    assert "oracle_cheapest_success_cost" in result.metrics
    assert "fusion_rate" in result.metrics
    assert result.route_distribution


def test_threshold_sweep_reports_cost_quality_tradeoff_and_calibration() -> None:
    rows = load_jsonl("examples/benchmark.jsonl")

    result = evaluate_threshold_sweep(
        rows,
        thresholds=[0.5, 0.7],
        test_size=0.33,
        random_state=17,
        calibration_bins=4,
        predictor_factory=lambda: ModelAwareRouterPredictor(
            completion_score_threshold=0.9,
            random_state=17,
        ),
    )

    assert result.completion_score_threshold == 0.9
    assert result.train_row_count > 0
    assert result.test_prompt_count >= 1
    assert [item["threshold"] for item in result.thresholds] == [0.5, 0.7]
    assert all("completion_rate" in item["metrics"] for item in result.thresholds)
    assert "expected_calibration_error" in result.calibration
    assert len(result.calibration["bins"]) == 4


def test_build_fusion_prompt_renders_candidate_answers() -> None:
    prompt = build_fusion_prompt(
        "Fix this bug",
        {
            "claude": "Use a lock.",
            "gpt": "Use an atomic update.",
        },
    )

    assert "[Model: claude]" in prompt
    assert "Use an atomic update." in prompt
    assert "Return the final answer only." in prompt
