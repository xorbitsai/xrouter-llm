from xrouter_llm import ModelPrediction, PolicyParams, RoutingPolicy


def test_policy_selects_capable_model_before_cost() -> None:
    predictions = [
        ModelPrediction("cheap", mu=0.70, sigma=0.03, cost=0.01, latency=1.0),
        ModelPrediction("strong", mu=0.90, sigma=0.03, cost=0.02, latency=1.0),
    ]

    decision = RoutingPolicy(PolicyParams(completion_threshold=0.75)).select(predictions)

    assert decision.selected_model_ids == ("strong",)
    assert not decision.is_fusion


def test_policy_selects_cheapest_capable_model() -> None:
    predictions = [
        ModelPrediction("cheap", mu=0.80, sigma=0.03, cost=0.01, latency=1.0),
        ModelPrediction("strong", mu=0.95, sigma=0.03, cost=0.20, latency=1.0),
    ]

    decision = RoutingPolicy(PolicyParams(completion_threshold=0.75)).select(predictions)

    assert decision.selected_model_ids == ("cheap",)
    assert not decision.is_fusion


def test_policy_adds_model_when_fusion_reaches_completion_threshold() -> None:
    predictions = [
        ModelPrediction("a", mu=0.70, sigma=0.12, cost=0.0, latency=1.0),
        ModelPrediction("b", mu=0.69, sigma=0.12, cost=0.0, latency=1.0),
    ]

    decision = RoutingPolicy(
        PolicyParams(max_k=2, allow_fusion=True, completion_threshold=0.75)
    ).select(predictions)

    assert decision.selected_model_ids == ("a", "b")
    assert decision.is_fusion
    assert decision.utility_breakdown.expected_quality >= 0.75
