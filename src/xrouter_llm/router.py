from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

from xrouter_llm.catalog import ModelCatalog
from xrouter_llm.policy import PolicyParams, RoutingPolicy
from xrouter_llm.predictor_utils import predict_with_optional_task
from xrouter_llm.types import ModelProfile, RouteDecision


class PredictorLike(Protocol):
    model_ids_: tuple[str, ...]

    def predict(
        self,
        prompt: str,
        *,
        model_ids: Sequence[str] | None = None,
        costs: dict[str, float] | None = None,
        latencies: dict[str, float] | None = None,
        task: str | None = None,
    ) -> object:
        ...


class XRouter:
    def __init__(
        self,
        predictor: PredictorLike,
        model_profiles: Iterable[ModelProfile] | ModelCatalog | None = None,
    ) -> None:
        self.predictor = predictor
        self.catalog = (
            model_profiles
            if isinstance(model_profiles, ModelCatalog)
            else ModelCatalog(model_profiles)
        )

    def route(
        self,
        prompt: str,
        *,
        candidate_models: Sequence[str] | None = None,
        policy_params: PolicyParams | None = None,
        expected_output_tokens: int = 512,
        task: str | None = None,
    ) -> RouteDecision:
        model_ids = tuple(candidate_models) if candidate_models is not None else None
        predicted_ids = model_ids if model_ids is not None else self.predictor.model_ids_
        costs = {
            model_id: self.catalog.estimate_cost(
                prompt,
                model_id,
                expected_output_tokens=expected_output_tokens,
            )
            for model_id in predicted_ids
        }
        latencies = {
            model_id: self.catalog.estimate_latency(
                prompt,
                model_id,
                expected_output_tokens=expected_output_tokens,
            )
            for model_id in predicted_ids
        }
        predictions = predict_with_optional_task(
            self.predictor,
            prompt,
            model_ids=model_ids,
            costs=costs,
            latencies=latencies,
            task=task,
        )
        return RoutingPolicy(policy_params).select(predictions)
