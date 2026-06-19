from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np

from xrouter_llm.types import ModelPrediction, RouteDecision, UtilityBreakdown


@dataclass(frozen=True)
class PolicyParams:
    max_k: int = 1
    allow_fusion: bool = False
    completion_threshold: float = 0.5
    lambda_cost: float = 1.0
    lambda_latency: float = 0.0
    min_fusion_gain: float = 0.0
    judge_cost: float = 0.0
    fusion_overhead: float = 0.0
    quality_samples: int = 4096
    random_state: int = 0
    clamp_quality_samples: bool = True

    def __post_init__(self) -> None:
        if self.max_k < 1:
            raise ValueError("max_k must be at least 1")
        if not 0.0 <= self.completion_threshold <= 1.0:
            raise ValueError("completion_threshold must be in [0, 1]")
        if self.quality_samples < 128:
            raise ValueError("quality_samples must be at least 128")


class RoutingPolicy:
    def __init__(self, params: PolicyParams | None = None) -> None:
        self.params = params or PolicyParams()

    def select(self, predictions: Sequence[ModelPrediction]) -> RouteDecision:
        candidates = tuple(predictions)
        if not candidates:
            raise ValueError("RoutingPolicy.select requires at least one prediction")

        candidate_sets = self._candidate_sets(candidates)
        breakdowns = [(selected, self.utility(selected)) for selected in candidate_sets]
        capable = [
            (selected, breakdown)
            for selected, breakdown in breakdowns
            if breakdown.expected_quality >= self.params.completion_threshold
        ]

        if capable:
            selected, selected_breakdown = min(
                capable,
                key=lambda item: (
                    item[1].cost,
                    item[1].latency,
                    len(item[0]),
                    -item[1].expected_quality,
                ),
            )
        else:
            selected, selected_breakdown = max(
                breakdowns,
                key=lambda item: (
                    item[1].expected_quality,
                    -item[1].cost,
                    -item[1].latency,
                    -len(item[0]),
                ),
            )

        return RouteDecision(
            selected_model_ids=tuple(prediction.model_id for prediction in selected),
            selected_predictions=tuple(selected),
            candidate_predictions=candidates,
            utility_breakdown=selected_breakdown,
        )

    def _candidate_sets(self, candidates: tuple[ModelPrediction, ...]) -> list[tuple[ModelPrediction, ...]]:
        if not self.params.allow_fusion or self.params.max_k == 1:
            return [(candidate,) for candidate in candidates]

        max_k = min(self.params.max_k, len(candidates))
        output: list[tuple[ModelPrediction, ...]] = []
        for size in range(1, max_k + 1):
            output.extend(combinations(candidates, size))
        return output

    def utility(self, selected: Sequence[ModelPrediction]) -> UtilityBreakdown:
        if not selected:
            raise ValueError("Cannot compute utility for an empty selection")

        expected_quality = self.expected_quality(selected)
        cost = sum(prediction.cost for prediction in selected)
        latency = max(prediction.latency for prediction in selected)
        judge_cost = self.params.judge_cost if len(selected) > 1 else 0.0
        fusion_overhead = self.params.fusion_overhead if len(selected) > 1 else 0.0
        utility = (
            expected_quality
            - self.params.lambda_cost * cost
            - self.params.lambda_latency * latency
            - judge_cost
            - fusion_overhead
        )

        return UtilityBreakdown(
            expected_quality=expected_quality,
            cost=cost,
            latency=latency,
            judge_cost=judge_cost,
            fusion_overhead=fusion_overhead,
            utility=utility,
        )

    def expected_quality(self, selected: Sequence[ModelPrediction]) -> float:
        if len(selected) == 1:
            return float(selected[0].mu)

        mus = np.asarray([prediction.mu for prediction in selected], dtype=float)
        sigmas = np.asarray([prediction.sigma for prediction in selected], dtype=float)
        rng = np.random.default_rng(self.params.random_state + len(selected))
        z = rng.standard_normal((self.params.quality_samples, len(selected)))
        samples = mus + sigmas * z

        if self.params.clamp_quality_samples:
            samples = np.clip(samples, 0.0, 1.0)

        return float(samples.max(axis=1).mean())
