from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler


BENCHMARK_SCORE_SCALES = {
    "mt_bench": 10.0,
}

SOURCE_QUALITY_LEVELS = {
    "missing": 0.0,
    "third_party": 0.35,
    "self_eval": 0.45,
    "proxy_official": 0.65,
    "paper": 0.85,
    "model_card": 0.90,
    "official": 1.0,
}


@dataclass(frozen=True)
class ModelBenchmarkProfile:
    model_id: str
    benchmarks: Mapping[str, float | None] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    provider: str | None = None
    source_quality: str = "missing"
    source_urls: tuple[str, ...] = ()
    release_date: str | None = None
    context_length: int | None = None
    max_output_tokens: int | None = None
    parameters_b: float | None = None
    active_parameters_b: float | None = None
    input_cost_per_1k: float | None = None
    output_cost_per_1k: float | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModelBenchmarkProfile":
        return cls(
            model_id=str(data["model_id"]),
            aliases=tuple(str(value) for value in data.get("aliases", ())),
            provider=_optional_str(data.get("provider")),
            source_quality=str(data.get("source_quality", "missing")),
            source_urls=tuple(str(value) for value in data.get("source_urls", ())),
            release_date=_optional_str(data.get("release_date")),
            context_length=_optional_int(data.get("context_length")),
            max_output_tokens=_optional_int(data.get("max_output_tokens")),
            parameters_b=_optional_float(data.get("parameters_b")),
            active_parameters_b=_optional_float(data.get("active_parameters_b")),
            input_cost_per_1k=_optional_float(data.get("input_cost_per_1k")),
            output_cost_per_1k=_optional_float(data.get("output_cost_per_1k")),
            benchmarks={
                str(key): None if value is None else float(value)
                for key, value in data.get("benchmarks", {}).items()
            },
        )

    @classmethod
    def blank(cls, model_id: str) -> "ModelBenchmarkProfile":
        return cls(model_id=model_id)

    def normalized_benchmark(self, benchmark_name: str) -> float | None:
        value = self.benchmarks.get(benchmark_name)
        if value is None:
            return None
        scale = BENCHMARK_SCORE_SCALES.get(benchmark_name, 100.0 if value > 1.0 else 1.0)
        return float(np.clip(value / scale, 0.0, 1.0))

    @property
    def source_quality_score(self) -> float:
        return SOURCE_QUALITY_LEVELS.get(self.source_quality, SOURCE_QUALITY_LEVELS["third_party"])


class BenchmarkProfileCatalog:
    def __init__(self, profiles: Sequence[ModelBenchmarkProfile] | None = None) -> None:
        self._profiles: dict[str, ModelBenchmarkProfile] = {}
        self._aliases: dict[str, str] = {}
        for profile in profiles or ():
            self.add(profile)

    def add(self, profile: ModelBenchmarkProfile) -> None:
        self._profiles[profile.model_id] = profile
        for alias in profile.aliases:
            self._aliases[alias] = profile.model_id

    def get(self, model_id: str) -> ModelBenchmarkProfile:
        canonical_id = self._aliases.get(model_id, model_id)
        return self._profiles.get(canonical_id, ModelBenchmarkProfile.blank(model_id))

    def known_model_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles))

    def profiles(self) -> tuple[ModelBenchmarkProfile, ...]:
        return tuple(self._profiles.values())

    def __len__(self) -> int:
        return len(self._profiles)


class BenchmarkProfileFeaturizer:
    def __init__(self) -> None:
        self.benchmark_names_: tuple[str, ...] = ()
        self.providers_: tuple[str, ...] = ()
        self.numeric_means_: dict[str, float] = {}
        self.scaler_: StandardScaler | None = None

    def fit(self, profiles: Sequence[ModelBenchmarkProfile]) -> "BenchmarkProfileFeaturizer":
        self.benchmark_names_ = tuple(
            sorted({name for profile in profiles for name in profile.benchmarks})
        )
        self.providers_ = tuple(
            sorted({profile.provider for profile in profiles if profile.provider})
        )
        self.numeric_means_ = {}
        for benchmark in self.benchmark_names_:
            present_values = [
                value
                for profile in profiles
                if (value := profile.normalized_benchmark(benchmark)) is not None
            ]
            self.numeric_means_[benchmark] = float(np.mean(present_values)) if present_values else 0.5

        raw_numeric = np.asarray([self._raw_numeric(profile) for profile in profiles], dtype=float)
        self.scaler_ = StandardScaler().fit(raw_numeric)
        return self

    def transform(self, profiles: Sequence[ModelBenchmarkProfile]) -> np.ndarray:
        if self.scaler_ is None:
            raise RuntimeError("BenchmarkProfileFeaturizer is not fitted")
        numeric = self.scaler_.transform(
            np.asarray([self._raw_numeric(profile) for profile in profiles], dtype=float)
        )
        providers = np.asarray([self._provider_features(profile) for profile in profiles], dtype=float)
        return np.hstack([numeric, providers])

    def fit_transform(self, profiles: Sequence[ModelBenchmarkProfile]) -> np.ndarray:
        return self.fit(profiles).transform(profiles)

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for benchmark in self.benchmark_names_:
            names.append(f"benchmark:{benchmark}")
            names.append(f"benchmark_present:{benchmark}")
        names.extend(
            [
                "profile:benchmark_coverage",
                "profile:source_quality",
                "profile:log_context_length",
                "profile:log_max_output_tokens",
                "profile:log_parameters_b",
                "profile:log_active_parameters_b",
                "profile:log_input_cost_per_1k",
                "profile:log_output_cost_per_1k",
            ]
        )
        names.extend(f"provider:{provider}" for provider in self.providers_)
        return names

    def _raw_numeric(self, profile: ModelBenchmarkProfile) -> list[float]:
        values: list[float] = []
        present_count = 0
        for benchmark in self.benchmark_names_:
            value = profile.normalized_benchmark(benchmark)
            if value is None:
                values.append(self.numeric_means_.get(benchmark, 0.5))
                values.append(0.0)
            else:
                values.append(value)
                values.append(1.0)
                present_count += 1

        coverage = present_count / max(1, len(self.benchmark_names_))
        values.extend(
            [
                coverage,
                profile.source_quality_score,
                _log_feature(profile.context_length),
                _log_feature(profile.max_output_tokens),
                _log_feature(profile.parameters_b),
                _log_feature(profile.active_parameters_b),
                _log_feature(profile.input_cost_per_1k),
                _log_feature(profile.output_cost_per_1k),
            ]
        )
        return values

    def _provider_features(self, profile: ModelBenchmarkProfile) -> list[float]:
        return [1.0 if profile.provider == provider else 0.0 for provider in self.providers_]


def load_builtin_benchmark_profiles() -> BenchmarkProfileCatalog:
    with resources.files("xrouter_llm.resources").joinpath("routerbench_public_benchmarks.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)
    return BenchmarkProfileCatalog([ModelBenchmarkProfile.from_mapping(item) for item in data])


def load_benchmark_profiles(path: str | Path) -> BenchmarkProfileCatalog:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, Mapping):
        data = data.get("models", [])
    return BenchmarkProfileCatalog([ModelBenchmarkProfile.from_mapping(item) for item in data])


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _log_feature(value: float | int | None) -> float:
    if value is None or value <= 0:
        return 0.0
    return float(np.log1p(value))
