"""Routing service: named router configs + a decision/record pipeline.

A "router config" (the user's *auto config*) names a candidate model set (one
or many) and the policy knobs. A request references a config by name; the
service predicts completion for each candidate, applies the policy, records the
decision, and returns it. It does NOT call the underlying LLMs -- it answers
"which model should serve this prompt".
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from xrouter_llm.catalog import estimate_tokens
from xrouter_llm.policy import PolicyParams, RoutingPolicy
from xrouter_llm.profiles import BenchmarkProfileCatalog, load_benchmark_profiles
from xrouter_llm.store import CallStore


@dataclass(frozen=True)
class RouterConfig:
    name: str
    models: tuple[str, ...]
    completion_threshold: float = 0.7
    lambda_cost: float = 1.0
    lambda_latency: float = 0.0
    max_k: int = 1
    fallback_quality_margin: float = 0.05
    description: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RouterConfig":
        models = tuple(str(m) for m in data.get("models", ()))
        if not models:
            raise ValueError(f"Router config {data.get('name')!r} lists no models")
        return cls(
            name=str(data["name"]),
            models=models,
            completion_threshold=float(data.get("completion_threshold", 0.7)),
            lambda_cost=float(data.get("lambda_cost", 1.0)),
            lambda_latency=float(data.get("lambda_latency", 0.0)),
            max_k=int(data.get("max_k", 1)),
            fallback_quality_margin=float(data.get("fallback_quality_margin", 0.05)),
            description=str(data.get("description", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "models": list(self.models),
            "completion_threshold": self.completion_threshold,
            "lambda_cost": self.lambda_cost,
            "lambda_latency": self.lambda_latency,
            "max_k": self.max_k,
            "fallback_quality_margin": self.fallback_quality_margin,
            "description": self.description,
        }


def load_router_configs(path: str | Path) -> dict[str, RouterConfig]:
    """Load router configs from a directory (one per file) or a single file."""
    target = Path(path)
    files: list[Path]
    if target.is_dir():
        files = sorted(
            entry
            for entry in target.iterdir()
            if entry.is_file() and entry.suffix.lower() in {".yaml", ".yml", ".json"}
        )
    else:
        files = [target]

    configs: dict[str, RouterConfig] = {}
    for file in files:
        for mapping in _read_config_mappings(file):
            config = RouterConfig.from_mapping(mapping)
            configs[config.name] = config
    if not configs:
        raise ValueError(f"No router configs found at {path}")
    return configs


def _read_config_mappings(file_path: Path) -> list[Mapping[str, Any]]:
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if isinstance(data, Mapping):
        if "routers" in data:
            return list(data["routers"])
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"Router config file {file_path} must be a mapping or list")


class RoutingService:
    def __init__(
        self,
        predictor: Any,
        *,
        profiles: BenchmarkProfileCatalog,
        configs: Mapping[str, RouterConfig],
        store: CallStore,
        expected_output_tokens: int = 512,
    ) -> None:
        self.predictor = predictor
        self.profiles = profiles
        self.configs = dict(configs)
        self.store = store
        self.expected_output_tokens = expected_output_tokens
        # Make sure candidate models carry their published profile features.
        adder = getattr(predictor, "add_benchmark_profile", None)
        if callable(adder):
            for profile in profiles.profiles():
                adder(profile)

    def estimate_costs(self, prompt: str, models: tuple[str, ...]) -> dict[str, float]:
        input_tokens = estimate_tokens(prompt)
        costs: dict[str, float] = {}
        for model_id in models:
            profile = self.profiles.get(model_id)
            input_cost = profile.input_cost_per_1k or 0.0
            output_cost = profile.output_cost_per_1k or 0.0
            costs[model_id] = (
                (input_tokens / 1000.0) * input_cost
                + (self.expected_output_tokens / 1000.0) * output_cost
            )
        return costs

    def route(self, prompt: str, *, config_name: str, task: str | None = None) -> dict[str, Any]:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if config_name not in self.configs:
            raise KeyError(f"Unknown router config {config_name!r}")
        config = self.configs[config_name]

        costs = self.estimate_costs(prompt, config.models)
        latencies = {model_id: 0.0 for model_id in config.models}
        predictions = self.predictor.predict(
            prompt, model_ids=list(config.models), costs=costs, latencies=latencies
        )

        policy = RoutingPolicy(
            PolicyParams(
                completion_threshold=config.completion_threshold,
                lambda_cost=config.lambda_cost,
                lambda_latency=config.lambda_latency,
                max_k=config.max_k,
                fallback_quality_margin=config.fallback_quality_margin,
                allow_fusion=config.max_k > 1,
            )
        )
        decision = policy.select(predictions)

        candidates = [
            {
                "model_id": prediction.model_id,
                "mu": round(float(prediction.mu), 4),
                "sigma": round(float(prediction.sigma), 4),
                "cost": float(prediction.cost),
            }
            for prediction in decision.candidate_predictions
        ]
        selected = list(decision.selected_model_ids)
        breakdown = decision.utility_breakdown
        ts = time.time()
        call_id = self.store.record(
            ts=ts,
            config=config.name,
            prompt=prompt,
            task=task,
            selected=selected,
            candidates=candidates,
            expected_quality=float(breakdown.expected_quality),
            cost=float(breakdown.cost),
            latency=float(breakdown.latency),
        )
        return {
            "id": call_id,
            "ts": ts,
            "config": config.name,
            "prompt": prompt,
            "task": task,
            "selected": selected,
            "candidates": candidates,
            "expected_quality": round(float(breakdown.expected_quality), 4),
            "cost": float(breakdown.cost),
            "latency": float(breakdown.latency),
        }
