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


def test_policy_uses_any_model_completion_for_multi_model_sets() -> None:
    predictions = [
        ModelPrediction("cheap-a", mu=0.55, sigma=0.0, cost=0.01, latency=1.0),
        ModelPrediction("cheap-b", mu=0.55, sigma=0.0, cost=0.01, latency=1.0),
        ModelPrediction("strong", mu=0.78, sigma=0.0, cost=0.20, latency=1.0),
    ]

    decision = RoutingPolicy(
        PolicyParams(
            max_k=2,
            allow_fusion=True,
            completion_threshold=0.79,
        )
    ).select(predictions)

    assert decision.selected_model_ids == ("cheap-a", "cheap-b")
    assert decision.is_fusion
    assert round(decision.utility_breakdown.expected_quality, 4) == 0.7975


def test_fallback_picks_cheapest_within_quality_margin() -> None:
    # No candidate clears the threshold: take the cheapest within the margin of
    # the best predicted completion, not the priciest highest-completion model.
    predictions = [
        ModelPrediction("opus", mu=0.58, sigma=0.0, cost=0.023, latency=0.0),
        ModelPrediction("glm", mu=0.56, sigma=0.0, cost=0.004, latency=0.0),
        ModelPrediction("flash", mu=0.50, sigma=0.0, cost=0.001, latency=0.0),
    ]

    decision = RoutingPolicy(
        PolicyParams(completion_threshold=0.7, fallback_quality_margin=0.05)
    ).select(predictions)
    assert decision.selected_model_ids == ("glm",)


def test_fallback_margin_zero_keeps_highest_completion() -> None:
    predictions = [
        ModelPrediction("opus", mu=0.58, sigma=0.0, cost=0.023, latency=0.0),
        ModelPrediction("glm", mu=0.56, sigma=0.0, cost=0.004, latency=0.0),
    ]

    decision = RoutingPolicy(
        PolicyParams(completion_threshold=0.7, fallback_quality_margin=0.0)
    ).select(predictions)
    assert decision.selected_model_ids == ("opus",)
