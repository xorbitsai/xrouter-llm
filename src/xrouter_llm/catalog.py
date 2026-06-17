from __future__ import annotations

import math
from collections.abc import Iterable

from xrouter_llm.types import ModelProfile


class ModelCatalog:
    def __init__(self, profiles: Iterable[ModelProfile] | None = None) -> None:
        self._profiles = {profile.model_id: profile for profile in profiles or []}

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles))

    def get(self, model_id: str) -> ModelProfile:
        return self._profiles.get(model_id, ModelProfile(model_id=model_id))

    def estimate_cost(
        self,
        prompt: str,
        model_id: str,
        *,
        expected_output_tokens: int = 512,
    ) -> float:
        profile = self.get(model_id)
        input_tokens = estimate_tokens(prompt)
        return float(
            profile.fixed_cost_usd
            + (input_tokens / 1000.0) * profile.input_cost_per_1k
            + (expected_output_tokens / 1000.0) * profile.output_cost_per_1k
        )

    def estimate_latency(
        self,
        prompt: str,
        model_id: str,
        *,
        expected_output_tokens: int = 512,
    ) -> float:
        profile = self.get(model_id)
        total_tokens = estimate_tokens(prompt) + expected_output_tokens
        return float(profile.base_latency_s + (total_tokens / 1000.0) * profile.latency_per_1k_tokens_s)


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))
