from xrouter_llm import (
    BenchmarkRow,
    ModelPrediction,
    ModelProfile,
    PolicyParams,
    XRouter,
    build_fusion_prompt,
    evaluate_offline,
    evaluate_threshold_sweep,
    load_jsonl,
)


class _StubPredictor:
    """Deterministic stub used to exercise the router / evaluation machinery
    without training a real model (those are covered by the IRTRouter tests)."""

    completion_score_threshold = 0.9

    def __init__(self):
        self.model_ids_ = ()

    def fit(self, rows):
        self.model_ids_ = tuple(sorted({r.model_id for r in rows}))
        return self

    def normalize_score(self, score):
        return float(score)

    def predict(self, prompt, *, model_ids=None, costs=None, latencies=None, task=None):
        candidate_ids = tuple(model_ids) if model_ids is not None else self.model_ids_
        # stronger-looking ids get a slightly higher score; deterministic
        return [
            ModelPrediction(
                model_id=model_id,
                mu=0.95 if "claude" in model_id or "strong" in model_id else 0.85,
                sigma=0.05,
                cost=0.0 if costs is None else float(costs.get(model_id, 0.0)),
                latency=0.0 if latencies is None else float(latencies.get(model_id, 0.0)),
            )
            for model_id in candidate_ids
        ]


class _SparseCoveragePredictor(_StubPredictor):
    completion_score_threshold = 0.75
    model_ids_ = ("cheap", "strong")

    def fit(self, rows):
        return self

    def predict(self, prompt, *, model_ids=None, costs=None, latencies=None, task=None):
        candidate_ids = tuple(model_ids) if model_ids is not None else self.model_ids_
        return [
            ModelPrediction(
                model_id=model_id,
                mu=0.95 if model_id == "strong" else 0.80,
                sigma=0.03,
                cost=0.0 if costs is None else float(costs.get(model_id, 0.0)),
                latency=0.0 if latencies is None else float(latencies.get(model_id, 0.0)),
            )
            for model_id in candidate_ids
        ]


class _LegacyPredictor(_StubPredictor):
    def predict(self, prompt, *, model_ids=None, costs=None, latencies=None):
        candidate_ids = tuple(model_ids) if model_ids is not None else self.model_ids_
        return [
            ModelPrediction(
                model_id=model_id,
                mu=0.85,
                sigma=0.05,
                cost=0.0 if costs is None else float(costs.get(model_id, 0.0)),
                latency=0.0 if latencies is None else float(latencies.get(model_id, 0.0)),
            )
            for model_id in candidate_ids
        ]


def test_router_returns_route_decision_with_costs() -> None:
    rows = load_jsonl("examples/benchmark.jsonl")
    predictor = _StubPredictor().fit(rows)
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


def test_router_supports_predictors_without_task_parameter() -> None:
    predictor = _LegacyPredictor().fit(load_jsonl("examples/benchmark.jsonl"))
    router = XRouter(predictor)

    decision = router.route(
        "Find the bug in this Python retry loop.",
        candidate_models=["claude", "gpt"],
        task="coding",
    )

    assert decision.selected_model_ids


def test_offline_evaluation_reports_core_metrics() -> None:
    rows = load_jsonl("examples/benchmark.jsonl")

    result = evaluate_offline(
        rows,
        policy_params=PolicyParams(max_k=2, allow_fusion=True),
        test_size=0.33,
        random_state=11,
        predictor_factory=_StubPredictor,
    )

    assert result.metrics["prompt_count"] >= 1.0
    assert "average_score" in result.metrics
    assert "completion_rate" in result.metrics
    assert "average_cost" in result.metrics
    assert "fusion_rate" in result.metrics
    assert result.route_distribution


def test_offline_evaluation_supports_predictors_without_task_parameter() -> None:
    rows = load_jsonl("examples/benchmark.jsonl")

    result = evaluate_offline(
        rows,
        policy_params=PolicyParams(completion_threshold=0.75),
        test_size=0.5,
        random_state=0,
        predictor_factory=_LegacyPredictor,
    )

    assert result.metrics["completion_rate"] >= 0.0


def test_threshold_sweep_reports_cost_quality_tradeoff_and_calibration() -> None:
    rows = load_jsonl("examples/benchmark.jsonl")

    result = evaluate_threshold_sweep(
        rows,
        thresholds=[0.5, 0.7],
        test_size=0.33,
        random_state=17,
        calibration_bins=4,
        predictor_factory=_StubPredictor,
    )

    assert result.completion_score_threshold == 0.9
    assert result.train_row_count > 0
    assert result.test_prompt_count >= 1
    assert [item["threshold"] for item in result.thresholds] == [0.5, 0.7]
    assert all("completion_rate" in item["metrics"] for item in result.thresholds)
    assert "expected_calibration_error" in result.calibration
    assert len(result.calibration["bins"]) == 4


def test_threshold_sweep_restricts_offline_candidates_to_observed_prompt_models() -> None:
    rows = [
        BenchmarkRow(f"p{index}", f"prompt {index}", "cheap", 1.0, cost_usd=0.01)
        for index in range(6)
    ]

    result = evaluate_threshold_sweep(
        rows,
        thresholds=[0.5],
        test_size=0.5,
        random_state=1,
        predictor_factory=_SparseCoveragePredictor,
    )

    threshold_result = result.thresholds[0]
    assert threshold_result["metrics"]["prompt_count"] == float(result.test_prompt_count)
    assert threshold_result["route_distribution"] == {"cheap": result.test_prompt_count}


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
