from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkRow:
    prompt_id: str
    prompt: str
    model_id: str
    score: float
    cost_usd: float | None = None
    latency_s: float | None = None
    task: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0
    base_latency_s: float = 0.0
    latency_per_1k_tokens_s: float = 0.0
    fixed_cost_usd: float = 0.0


@dataclass(frozen=True)
class ModelPrediction:
    model_id: str
    mu: float
    sigma: float
    cost: float = 0.0
    latency: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UtilityBreakdown:
    expected_quality: float
    cost: float
    latency: float
    judge_cost: float
    fusion_overhead: float
    utility: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteDecision:
    selected_model_ids: tuple[str, ...]
    selected_predictions: tuple[ModelPrediction, ...]
    candidate_predictions: tuple[ModelPrediction, ...]
    utility_breakdown: UtilityBreakdown

    @property
    def is_fusion(self) -> bool:
        return len(self.selected_model_ids) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_model_ids": list(self.selected_model_ids),
            "selected_predictions": [p.to_dict() for p in self.selected_predictions],
            "candidate_predictions": [p.to_dict() for p in self.candidate_predictions],
            "utility_breakdown": self.utility_breakdown.to_dict(),
            "is_fusion": self.is_fusion,
        }
