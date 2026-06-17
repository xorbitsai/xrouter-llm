from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from xrouter_llm.data import coerce_benchmark_rows, split_by_prompt
from xrouter_llm.model_aware_predictor import ModelAwareRouterPredictor
from xrouter_llm.policy import PolicyParams, RoutingPolicy
from xrouter_llm.router import XRouter
from xrouter_llm.types import BenchmarkRow, ModelProfile


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    route_distribution: dict[str, int]
    per_model_selection_rate: dict[str, float]
    decisions: list[dict[str, object]]


@dataclass(frozen=True)
class ThresholdSweepResult:
    completion_score_threshold: float
    train_row_count: int
    test_prompt_count: int
    model_count: int
    thresholds: list[dict[str, object]]
    calibration: dict[str, object]


def evaluate_offline(
    rows: Sequence[BenchmarkRow | Mapping[str, object]],
    *,
    policy_params: PolicyParams | None = None,
    model_profiles: Iterable[ModelProfile] | None = None,
    test_size: float = 0.2,
    random_state: int | None = None,
    predictor: object | None = None,
    predictor_factory: Callable[[], object] | None = None,
) -> EvaluationResult:
    all_rows = coerce_benchmark_rows(rows)
    train_rows, test_rows = split_by_prompt(
        all_rows,
        test_size=test_size,
        random_state=random_state,
    )

    fitted_predictor = predictor or (
        predictor_factory() if predictor_factory is not None else ModelAwareRouterPredictor(random_state=random_state)
    )
    fitted_predictor.fit(train_rows)
    router = XRouter(fitted_predictor, model_profiles=model_profiles)
    params = policy_params or PolicyParams()
    actual_completion_threshold = getattr(
        fitted_predictor,
        "completion_score_threshold",
        params.completion_threshold,
    )

    prompts = _group_test_rows(test_rows)
    route_distribution: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    decisions: list[dict[str, object]] = []
    actual_scores: list[float] = []
    completion_flags: list[bool] = []
    observed_costs: list[float] = []
    latencies: list[float] = []
    ks: list[int] = []
    oracle_scores: list[float] = []
    oracle_completion_flags: list[bool] = []
    oracle_cheapest_success_costs: list[float] = []
    selected_cost_over_oracle_success_costs: list[float] = []
    random_scores: list[float] = []
    rng = np.random.default_rng(random_state)

    for prompt_id, prompt_rows in prompts.items():
        prompt = prompt_rows[0].prompt
        actual_by_model = {
            row.model_id: fitted_predictor.normalize_score(row.score)
            for row in prompt_rows
        }
        actual_cost_by_model = {
            row.model_id: 0.0 if row.cost_usd is None else row.cost_usd
            for row in prompt_rows
        }

        if any(row.cost_usd is not None or row.latency_s is not None for row in prompt_rows):
            prompt_costs = {
                row.model_id: 0.0 if row.cost_usd is None else row.cost_usd
                for row in prompt_rows
            }
            prompt_latencies = {
                row.model_id: 0.0 if row.latency_s is None else row.latency_s
                for row in prompt_rows
            }
            predictions = fitted_predictor.predict(
                prompt,
                costs=prompt_costs,
                latencies=prompt_latencies,
            )
            decision = RoutingPolicy(params).select(predictions)
        else:
            decision = router.route(prompt, policy_params=params)
        selected = decision.selected_model_ids
        selected_actual_scores = [
            actual_by_model[model_id]
            for model_id in selected
            if model_id in actual_by_model
        ]
        if not selected_actual_scores:
            continue

        actual_quality = max(selected_actual_scores)
        completed = actual_quality >= actual_completion_threshold
        actual_scores.append(actual_quality)
        completion_flags.append(completed)
        observed_costs.append(decision.utility_breakdown.cost)
        latencies.append(decision.utility_breakdown.latency)
        ks.append(len(selected))
        oracle_quality = max(actual_by_model.values())
        oracle_scores.append(oracle_quality)
        successful_models = [
            model_id
            for model_id, actual_score in actual_by_model.items()
            if actual_score >= actual_completion_threshold
        ]
        oracle_completion_flags.append(bool(successful_models))
        if successful_models:
            oracle_cheapest_cost = min(actual_cost_by_model[model_id] for model_id in successful_models)
            oracle_cheapest_success_costs.append(oracle_cheapest_cost)
            if completed and oracle_cheapest_cost > 0:
                selected_cost_over_oracle_success_costs.append(
                    decision.utility_breakdown.cost / oracle_cheapest_cost
                )
        random_model = rng.choice(list(actual_by_model))
        random_scores.append(actual_by_model[str(random_model)])

        route_key = "+".join(selected)
        route_distribution[route_key] += 1
        model_counts.update(selected)
        decisions.append(
            {
                "prompt_id": prompt_id,
                "selected_model_ids": list(selected),
                "actual_quality_best_of_k_upper_bound": actual_quality,
                "decision": decision.to_dict(),
            }
        )

    if not actual_scores:
        raise ValueError("No evaluable test prompts; check candidate model coverage")

    best_fixed_score = _best_fixed_model_score(test_rows, fitted_predictor)
    best_fixed_completion_rate = _best_fixed_completion_rate(
        test_rows,
        fitted_predictor,
        threshold=actual_completion_threshold,
    )
    cheapest_success_fixed_cost = _cheapest_fixed_success_cost(
        test_rows,
        fitted_predictor,
        threshold=actual_completion_threshold,
    )
    cheapest_score = _cheapest_model_score(test_rows, fitted_predictor)
    average_score = float(np.mean(actual_scores))
    oracle_score = float(np.mean(oracle_scores))
    prompt_count = len(actual_scores)

    metrics = {
        "average_score": average_score,
        "completion_score_threshold": actual_completion_threshold,
        "completion_probability_threshold": params.completion_threshold,
        "completion_rate": float(np.mean(completion_flags)),
        "regret_vs_oracle_top1": oracle_score - average_score,
        "average_cost": float(np.mean(observed_costs)),
        "average_cost_when_completed": _nanmean([
            cost
            for cost, completed in zip(observed_costs, completion_flags)
            if completed
        ]),
        "average_latency": float(np.mean(latencies)),
        "fusion_rate": float(np.mean([k > 1 for k in ks])),
        "average_k": float(np.mean(ks)),
        "best_fixed_model_score": best_fixed_score,
        "best_fixed_completion_rate": best_fixed_completion_rate,
        "cheapest_fixed_success_cost": cheapest_success_fixed_cost,
        "cheapest_model_score": cheapest_score,
        "random_model_score": float(np.mean(random_scores)),
        "oracle_top1_score": oracle_score,
        "oracle_best_of_k_upper_bound_score": oracle_score,
        "oracle_completion_rate": float(np.mean(oracle_completion_flags)),
        "oracle_cheapest_success_cost": _nanmean(oracle_cheapest_success_costs),
        "selected_cost_over_oracle_success_cost": _nanmean(selected_cost_over_oracle_success_costs),
        "prompt_count": float(prompt_count),
    }

    per_model_selection_rate = {
        model_id: count / prompt_count
        for model_id, count in sorted(model_counts.items())
    }

    return EvaluationResult(
        metrics=metrics,
        route_distribution=dict(route_distribution),
        per_model_selection_rate=per_model_selection_rate,
        decisions=decisions,
    )


def evaluate_threshold_sweep(
    rows: Sequence[BenchmarkRow | Mapping[str, object]],
    *,
    thresholds: Sequence[float],
    predictor_factory: Callable[[], object] | None = None,
    model_profiles: Iterable[ModelProfile] | None = None,
    test_size: float = 0.2,
    random_state: int | None = None,
    calibration_bins: int = 10,
) -> ThresholdSweepResult:
    if not thresholds:
        raise ValueError("thresholds must not be empty")
    if calibration_bins < 1:
        raise ValueError("calibration_bins must be at least 1")
    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("all thresholds must be in [0, 1]")

    all_rows = coerce_benchmark_rows(rows)
    train_rows, test_rows = split_by_prompt(
        all_rows,
        test_size=test_size,
        random_state=random_state,
    )
    predictor = predictor_factory() if predictor_factory is not None else ModelAwareRouterPredictor(random_state=random_state)
    predictor.fit(train_rows)
    actual_completion_threshold = getattr(predictor, "completion_score_threshold", 0.75)
    prompt_evaluations = _collect_prompt_evaluations(
        predictor,
        test_rows,
        model_profiles=model_profiles,
        random_state=random_state,
    )

    threshold_results = [
        _evaluate_prompt_predictions_at_threshold(
            prompt_evaluations,
            completion_probability_threshold=threshold,
            actual_completion_threshold=actual_completion_threshold,
            random_state=random_state,
        )
        for threshold in thresholds
    ]

    calibration = _calibration_report(
        prompt_evaluations,
        actual_completion_threshold=actual_completion_threshold,
        bin_count=calibration_bins,
    )

    return ThresholdSweepResult(
        completion_score_threshold=actual_completion_threshold,
        train_row_count=len(train_rows),
        test_prompt_count=len(prompt_evaluations),
        model_count=len(getattr(predictor, "model_ids_", ())),
        thresholds=threshold_results,
        calibration=calibration,
    )


def _group_test_rows(rows: Sequence[BenchmarkRow]) -> dict[str, list[BenchmarkRow]]:
    grouped: dict[str, list[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        grouped[row.prompt_id].append(row)
    return dict(grouped)


def _collect_prompt_evaluations(
    predictor: object,
    rows: Sequence[BenchmarkRow],
    *,
    model_profiles: Iterable[ModelProfile] | None,
    random_state: int | None,
) -> list[dict[str, object]]:
    del random_state
    router = XRouter(predictor, model_profiles=model_profiles)
    prompt_evaluations: list[dict[str, object]] = []
    for prompt_id, prompt_rows in _group_test_rows(rows).items():
        prompt = prompt_rows[0].prompt
        actual_by_model = {
            row.model_id: predictor.normalize_score(row.score)
            for row in prompt_rows
        }
        actual_cost_by_model = {
            row.model_id: 0.0 if row.cost_usd is None else row.cost_usd
            for row in prompt_rows
        }

        if any(row.cost_usd is not None or row.latency_s is not None for row in prompt_rows):
            prompt_costs = {
                row.model_id: 0.0 if row.cost_usd is None else row.cost_usd
                for row in prompt_rows
            }
            prompt_latencies = {
                row.model_id: 0.0 if row.latency_s is None else row.latency_s
                for row in prompt_rows
            }
            predictions = predictor.predict(
                prompt,
                costs=prompt_costs,
                latencies=prompt_latencies,
            )
        else:
            candidate_ids = tuple(actual_by_model)
            costs = {
                model_id: router.catalog.estimate_cost(prompt, model_id)
                for model_id in candidate_ids
            }
            latencies = {
                model_id: router.catalog.estimate_latency(prompt, model_id)
                for model_id in candidate_ids
            }
            predictions = predictor.predict(
                prompt,
                model_ids=candidate_ids,
                costs=costs,
                latencies=latencies,
            )

        prompt_evaluations.append(
            {
                "prompt_id": prompt_id,
                "predictions": tuple(predictions),
                "actual_by_model": actual_by_model,
                "actual_cost_by_model": actual_cost_by_model,
            }
        )
    return prompt_evaluations


def _evaluate_prompt_predictions_at_threshold(
    prompt_evaluations: Sequence[Mapping[str, object]],
    *,
    completion_probability_threshold: float,
    actual_completion_threshold: float,
    random_state: int | None,
) -> dict[str, object]:
    params = PolicyParams(completion_threshold=completion_probability_threshold)
    route_distribution: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    actual_scores: list[float] = []
    completion_flags: list[bool] = []
    observed_costs: list[float] = []
    oracle_completion_flags: list[bool] = []
    oracle_cheapest_success_costs: list[float] = []
    selected_cost_over_oracle_success_costs: list[float] = []
    random_scores: list[float] = []
    rng = np.random.default_rng(random_state)

    for item in prompt_evaluations:
        predictions = item["predictions"]
        actual_by_model = item["actual_by_model"]
        actual_cost_by_model = item["actual_cost_by_model"]
        decision = RoutingPolicy(params).select(predictions)
        selected_scores = [
            actual_by_model[model_id]
            for model_id in decision.selected_model_ids
            if model_id in actual_by_model
        ]
        if not selected_scores:
            continue
        actual_quality = max(selected_scores)
        completed = actual_quality >= actual_completion_threshold
        actual_scores.append(actual_quality)
        completion_flags.append(completed)
        observed_costs.append(decision.utility_breakdown.cost)

        successful_models = [
            model_id
            for model_id, actual_score in actual_by_model.items()
            if actual_score >= actual_completion_threshold
        ]
        oracle_completion_flags.append(bool(successful_models))
        if successful_models:
            oracle_cheapest_cost = min(actual_cost_by_model[model_id] for model_id in successful_models)
            oracle_cheapest_success_costs.append(oracle_cheapest_cost)
            if completed and oracle_cheapest_cost > 0:
                selected_cost_over_oracle_success_costs.append(
                    decision.utility_breakdown.cost / oracle_cheapest_cost
                )

        random_model = rng.choice(list(actual_by_model))
        random_scores.append(actual_by_model[str(random_model)])

        route_key = "+".join(decision.selected_model_ids)
        route_distribution[route_key] += 1
        model_counts.update(decision.selected_model_ids)

    prompt_count = len(actual_scores)
    if prompt_count == 0:
        raise ValueError("No evaluable prompt predictions")

    metrics = {
        "completion_probability_threshold": completion_probability_threshold,
        "completion_score_threshold": actual_completion_threshold,
        "completion_rate": float(np.mean(completion_flags)),
        "average_cost": float(np.mean(observed_costs)),
        "average_cost_when_completed": _nanmean([
            cost
            for cost, completed in zip(observed_costs, completion_flags)
            if completed
        ]),
        "average_score": float(np.mean(actual_scores)),
        "oracle_completion_rate": float(np.mean(oracle_completion_flags)),
        "oracle_cheapest_success_cost": _nanmean(oracle_cheapest_success_costs),
        "selected_cost_over_oracle_success_cost": _nanmean(selected_cost_over_oracle_success_costs),
        "random_model_score": float(np.mean(random_scores)),
        "prompt_count": float(prompt_count),
    }
    return {
        "threshold": completion_probability_threshold,
        "metrics": metrics,
        "route_distribution": dict(route_distribution),
        "per_model_selection_rate": {
            model_id: count / prompt_count
            for model_id, count in sorted(model_counts.items())
        },
    }


def _calibration_report(
    prompt_evaluations: Sequence[Mapping[str, object]],
    *,
    actual_completion_threshold: float,
    bin_count: int,
) -> dict[str, object]:
    predictions: list[float] = []
    labels: list[float] = []
    for item in prompt_evaluations:
        actual_by_model = item["actual_by_model"]
        for prediction in item["predictions"]:
            if prediction.model_id not in actual_by_model:
                continue
            predictions.append(float(prediction.mu))
            labels.append(float(actual_by_model[prediction.model_id] >= actual_completion_threshold))

    if not predictions:
        raise ValueError("No predictions available for calibration")

    pred_array = np.asarray(predictions, dtype=float)
    label_array = np.asarray(labels, dtype=float)
    bins: list[dict[str, float]] = []
    ece = 0.0
    for bin_index in range(bin_count):
        lower = bin_index / bin_count
        upper = (bin_index + 1) / bin_count
        if bin_index == bin_count - 1:
            mask = (pred_array >= lower) & (pred_array <= upper)
        else:
            mask = (pred_array >= lower) & (pred_array < upper)
        count = int(mask.sum())
        if count:
            predicted_mean = float(pred_array[mask].mean())
            actual_rate = float(label_array[mask].mean())
            abs_error = abs(predicted_mean - actual_rate)
            ece += (count / len(pred_array)) * abs_error
        else:
            predicted_mean = float("nan")
            actual_rate = float("nan")
            abs_error = float("nan")
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "predicted_completion_mean": predicted_mean,
                "actual_completion_rate": actual_rate,
                "absolute_error": abs_error,
            }
        )

    return {
        "bin_count": bin_count,
        "prediction_count": int(len(pred_array)),
        "base_completion_rate": float(label_array.mean()),
        "mean_predicted_completion": float(pred_array.mean()),
        "expected_calibration_error": float(ece),
        "bins": bins,
    }


def _best_fixed_model_score(rows: Sequence[BenchmarkRow], predictor: object) -> float:
    by_model: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_model[row.model_id].append(predictor.normalize_score(row.score))
    return float(max(np.mean(scores) for scores in by_model.values()))


def _best_fixed_completion_rate(
    rows: Sequence[BenchmarkRow],
    predictor: object,
    *,
    threshold: float,
) -> float:
    by_model: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_model[row.model_id].append(predictor.normalize_score(row.score))
    return float(max(np.mean([score >= threshold for score in scores]) for scores in by_model.values()))


def _cheapest_fixed_success_cost(
    rows: Sequence[BenchmarkRow],
    predictor: object,
    *,
    threshold: float,
) -> float:
    scores: dict[str, list[float]] = defaultdict(list)
    costs: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        scores[row.model_id].append(predictor.normalize_score(row.score))
        if row.cost_usd is not None:
            costs[row.model_id].append(row.cost_usd)

    successful_fixed_models = [
        model_id
        for model_id, model_scores in scores.items()
        if all(score >= threshold for score in model_scores)
    ]
    if not successful_fixed_models:
        return float("nan")
    return float(
        min(
            np.mean(costs[model_id])
            for model_id in successful_fixed_models
            if model_id in costs
        )
    )


def _cheapest_model_score(rows: Sequence[BenchmarkRow], predictor: object) -> float:
    costs: dict[str, list[float]] = defaultdict(list)
    scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.cost_usd is not None:
            costs[row.model_id].append(row.cost_usd)
        scores[row.model_id].append(predictor.normalize_score(row.score))

    if not costs:
        return float("nan")

    cheapest_model = min(costs, key=lambda model_id: float(np.mean(costs[model_id])))
    return float(np.mean(scores[cheapest_model]))


def _nanmean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))
