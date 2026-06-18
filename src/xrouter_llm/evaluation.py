from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from xrouter_llm.catalog import estimate_tokens
from xrouter_llm.data import coerce_benchmark_rows, split_by_prompt
from xrouter_llm.irt_router import IRTRouter
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


@dataclass(frozen=True)
class ModelHoldoutResult:
    """Leave-one-model-out generalization of the completion predictor.

    Each held-out model is removed from training entirely, so the predictor can
    only reach it through its published profile. This directly tests the core
    invariant -- "generalize to a new model from its profile" -- which the
    prompt-only train/test split never exercises.
    """

    completion_score_threshold: float
    train_prompt_count: int
    test_prompt_count: int
    holdout_models: list[str]
    per_model: list[dict[str, object]]
    aggregate: dict[str, float]


def evaluate_offline(
    rows: Sequence[BenchmarkRow | Mapping[str, object]],
    *,
    policy_params: PolicyParams | None = None,
    model_profiles: Iterable[ModelProfile] | None = None,
    test_size: float = 0.2,
    random_state: int | None = None,
    predictor: object | None = None,
    predictor_factory: Callable[[], object] | None = None,
    expected_output_tokens: int = 512,
) -> EvaluationResult:
    all_rows = coerce_benchmark_rows(rows)
    train_rows, test_rows = split_by_prompt(
        all_rows,
        test_size=test_size,
        random_state=random_state,
    )

    fitted_predictor = predictor or (
        predictor_factory() if predictor_factory is not None else IRTRouter(random_state=random_state)
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
    decision_costs: list[float] = []
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
        actual_latency_by_model = {
            row.model_id: 0.0 if row.latency_s is None else row.latency_s
            for row in prompt_rows
        }

        candidate_ids = tuple(actual_by_model)
        estimated_costs, estimated_latencies = _decision_cost_and_latency(
            fitted_predictor,
            router,
            prompt,
            candidate_ids,
            expected_output_tokens=expected_output_tokens,
        )
        predictions = fitted_predictor.predict(
            prompt,
            model_ids=candidate_ids,
            costs=estimated_costs,
            latencies=estimated_latencies,
        )
        decision = RoutingPolicy(params).select(predictions)
        selected = decision.selected_model_ids
        selected_actual_scores = [
            actual_by_model[model_id]
            for model_id in selected
            if model_id in actual_by_model
        ]
        if not selected_actual_scores:
            continue

        # Decision used estimated cost (no leak); report the realized cost of
        # the models that were actually selected.
        realized_cost = sum(actual_cost_by_model.get(model_id, 0.0) for model_id in selected)
        realized_latency = max(
            (actual_latency_by_model.get(model_id, 0.0) for model_id in selected),
            default=0.0,
        )

        actual_quality = max(selected_actual_scores)
        completed = actual_quality >= actual_completion_threshold
        actual_scores.append(actual_quality)
        completion_flags.append(completed)
        observed_costs.append(realized_cost)
        decision_costs.append(decision.utility_breakdown.cost)
        latencies.append(realized_latency)
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
                    realized_cost / oracle_cheapest_cost
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
        "average_decision_cost": float(np.mean(decision_costs)),
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
    predictor = predictor_factory() if predictor_factory is not None else IRTRouter(random_state=random_state)
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


def evaluate_model_holdout(
    rows: Sequence[BenchmarkRow | Mapping[str, object]],
    *,
    holdout_models: Sequence[str] | None = None,
    predictor_factory: Callable[[], object] | None = None,
    test_size: float = 0.2,
    random_state: int | None = None,
    calibration_bins: int = 10,
) -> ModelHoldoutResult:
    """Leave-one-model-out evaluation of the completion predictor.

    For each held-out model the predictor is retrained on every *other* model's
    training rows, then asked to predict the held-out model's completions on the
    held-out prompts. Both the model and the prompts are unseen, so the only
    signal the predictor has for the held-out model is its published profile.

    A predictor that secretly relies on per-model memorization (e.g. model-id
    one-hot features) will look strong on the prompt-split metrics yet collapse
    here -- that gap is exactly what this report surfaces.
    """
    if calibration_bins < 1:
        raise ValueError("calibration_bins must be at least 1")

    all_rows = coerce_benchmark_rows(rows)
    train_rows, test_rows = split_by_prompt(
        all_rows,
        test_size=test_size,
        random_state=random_state,
    )

    all_models = sorted({row.model_id for row in all_rows})
    if holdout_models is None:
        targets = all_models
    else:
        targets = list(holdout_models)
        unknown = sorted(set(targets) - set(all_models))
        if unknown:
            raise ValueError(f"Unknown holdout models: {', '.join(unknown)}")

    completion_score_threshold = 0.75
    test_by_model = defaultdict(list)
    for row in test_rows:
        test_by_model[row.model_id].append(row)

    per_model: list[dict[str, object]] = []
    for model_id in targets:
        eval_rows = test_by_model.get(model_id, [])
        if not eval_rows:
            continue
        fit_rows = [row for row in train_rows if row.model_id != model_id]
        if not fit_rows:
            continue

        predictor = (
            predictor_factory()
            if predictor_factory is not None
            else IRTRouter(random_state=random_state)
        )
        try:
            predictor.fit(fit_rows)
        except ValueError:
            # fit requires both completion classes among the remaining models.
            continue
        completion_score_threshold = getattr(
            predictor,
            "completion_score_threshold",
            completion_score_threshold,
        )

        uses_task = getattr(predictor, "include_task_features", False)
        predicted_by_prompt: dict[str, float] = {}
        predictions: list[float] = []
        labels: list[float] = []
        for row in eval_rows:
            if row.prompt_id not in predicted_by_prompt:
                predict_kwargs = {"model_ids": [model_id]}
                if uses_task:
                    predict_kwargs["task"] = row.task
                prediction = predictor.predict(row.prompt, **predict_kwargs)[0]
                predicted_by_prompt[row.prompt_id] = float(prediction.mu)
            mu = predicted_by_prompt[row.prompt_id]
            label = float(
                predictor.normalize_score(row.score) >= completion_score_threshold
            )
            predictions.append(mu)
            labels.append(label)

        per_model.append(
            {
                "model_id": model_id,
                **_binary_holdout_metrics(predictions, labels, bin_count=calibration_bins),
            }
        )

    if not per_model:
        raise ValueError("No held-out model had both training peers and test rows")

    aggregate = _aggregate_holdout_metrics(per_model)

    return ModelHoldoutResult(
        completion_score_threshold=completion_score_threshold,
        train_prompt_count=len({row.prompt_id for row in train_rows}),
        test_prompt_count=len({row.prompt_id for row in test_rows}),
        holdout_models=[entry["model_id"] for entry in per_model],
        per_model=per_model,
        aggregate=aggregate,
    )


def _group_test_rows(rows: Sequence[BenchmarkRow]) -> dict[str, list[BenchmarkRow]]:
    grouped: dict[str, list[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        grouped[row.prompt_id].append(row)
    return dict(grouped)


def _decision_cost_and_latency(
    predictor: object,
    router: XRouter,
    prompt: str,
    candidate_ids: Sequence[str],
    *,
    expected_output_tokens: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Decision-time cost/latency estimates.

    Routing must commit to a model before its response (and therefore its
    realized cost) exists, so the policy may only see *predicted* cost: profile
    per-1k pricing applied to the prompt length plus an assumed output budget.
    Feeding the realized ``cost_usd`` back into the decision would leak the
    answer and make the reported cost frontier optimistic. The realized cost is
    used only for reporting, by the callers.
    """
    profile_catalog = getattr(predictor, "profile_catalog", None)
    input_tokens = estimate_tokens(prompt)
    costs: dict[str, float] = {}
    latencies: dict[str, float] = {}
    for model_id in candidate_ids:
        profile = profile_catalog.get(model_id) if profile_catalog is not None else None
        input_cost = getattr(profile, "input_cost_per_1k", None) if profile is not None else None
        output_cost = getattr(profile, "output_cost_per_1k", None) if profile is not None else None
        if input_cost is None and output_cost is None:
            costs[model_id] = router.catalog.estimate_cost(
                prompt,
                model_id,
                expected_output_tokens=expected_output_tokens,
            )
        else:
            costs[model_id] = (
                (input_tokens / 1000.0) * (input_cost or 0.0)
                + (expected_output_tokens / 1000.0) * (output_cost or 0.0)
            )
        latencies[model_id] = router.catalog.estimate_latency(
            prompt,
            model_id,
            expected_output_tokens=expected_output_tokens,
        )
    return costs, latencies


def _collect_prompt_evaluations(
    predictor: object,
    rows: Sequence[BenchmarkRow],
    *,
    model_profiles: Iterable[ModelProfile] | None,
    random_state: int | None,
    expected_output_tokens: int = 512,
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

        candidate_ids = tuple(actual_by_model)
        estimated_costs, estimated_latencies = _decision_cost_and_latency(
            predictor,
            router,
            prompt,
            candidate_ids,
            expected_output_tokens=expected_output_tokens,
        )
        predictions = predictor.predict(
            prompt,
            model_ids=candidate_ids,
            costs=estimated_costs,
            latencies=estimated_latencies,
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
    decision_costs: list[float] = []
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
        # Routing committed on estimated cost; report the realized cost.
        realized_cost = sum(
            actual_cost_by_model.get(model_id, 0.0)
            for model_id in decision.selected_model_ids
        )
        observed_costs.append(realized_cost)
        decision_costs.append(decision.utility_breakdown.cost)

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
                    realized_cost / oracle_cheapest_cost
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
        "average_decision_cost": float(np.mean(decision_costs)),
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


def _binary_holdout_metrics(
    predictions: Sequence[float],
    labels: Sequence[float],
    *,
    bin_count: int,
) -> dict[str, float]:
    pred_array = np.asarray(predictions, dtype=float)
    label_array = np.asarray(labels, dtype=float)
    base_rate = float(label_array.mean())
    predicted_mean = float(pred_array.mean())

    accuracy = float(np.mean((pred_array >= 0.5).astype(float) == label_array))
    brier = float(np.mean((pred_array - label_array) ** 2))

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
            ece += (count / len(pred_array)) * abs(
                float(pred_array[mask].mean()) - float(label_array[mask].mean())
            )

    return {
        "n": float(len(pred_array)),
        "base_completion_rate": base_rate,
        "predicted_completion_mean": predicted_mean,
        "calibration_gap": abs(predicted_mean - base_rate),
        "accuracy": accuracy,
        "auc": _safe_auc(label_array, pred_array),
        "brier": brier,
        "expected_calibration_error": float(ece),
    }


def _safe_auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, predictions))


def _aggregate_holdout_metrics(per_model: Sequence[Mapping[str, object]]) -> dict[str, float]:
    keys = [
        "base_completion_rate",
        "predicted_completion_mean",
        "calibration_gap",
        "accuracy",
        "auc",
        "brier",
        "expected_calibration_error",
    ]
    total_n = float(sum(float(entry["n"]) for entry in per_model))
    aggregate: dict[str, float] = {
        "model_count": float(len(per_model)),
        "total_n": total_n,
    }
    for key in keys:
        values = [float(entry[key]) for entry in per_model if np.isfinite(float(entry[key]))]
        weights = [
            float(entry["n"]) for entry in per_model if np.isfinite(float(entry[key]))
        ]
        aggregate[f"macro_{key}"] = float(np.mean(values)) if values else float("nan")
        aggregate[f"weighted_{key}"] = (
            float(np.average(values, weights=weights)) if values else float("nan")
        )
    return aggregate


def _nanmean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))
