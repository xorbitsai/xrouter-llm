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
    # When no candidate clears completion_threshold, the completion objective has
    # already failed — so don't pay for the highest predicted completion (usually
    # the strongest, priciest model). Consider every candidate within this margin
    # of the best predicted completion and take the cheapest one.
    fallback_quality_margin: float = 0.05
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
        if not 0.0 <= self.fallback_quality_margin <= 1.0:
            raise ValueError("fallback_quality_margin must be in [0, 1]")
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
            # Nothing clears the threshold. Rather than always paying for the
            # highest predicted completion, take the cheapest among candidates
            # within `fallback_quality_margin` of the best predicted completion.
            top_quality = max(breakdown.expected_quality for _, breakdown in breakdowns)
            cutoff = top_quality - self.params.fallback_quality_margin
            near_best = [
                item for item in breakdowns if item[1].expected_quality >= cutoff
            ]
            selected, selected_breakdown = min(
                near_best,
                key=lambda item: (
                    item[1].cost,
                    item[1].latency,
                    len(item[0]),
                    -item[1].expected_quality,
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

        # For model-set routing, `mu` is a completion probability, not a score to
        # fuse. The set completes if at least one selected model completes.
        failure_probability = 1.0
        for prediction in selected:
            failure_probability *= 1.0 - float(np.clip(prediction.mu, 0.0, 1.0))
        return float(1.0 - failure_probability)
