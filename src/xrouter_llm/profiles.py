from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np


BENCHMARK_SCORE_SCALES = {
    "mt_bench": 10.0,
}

SOURCE_QUALITY_LEVELS = {
    "missing": 0.0,
    "third_party": 0.35,
    "self_eval": 0.45,
    "proxy_official": 0.65,
    "dataset_aggregate": 0.75,
    "paper": 0.85,
    "model_card": 0.90,
    "official": 1.0,
}


@dataclass(frozen=True)
class UtcPriceOverride:
    start_minute: int
    end_minute: int
    input_cost_per_1k: float | None = None
    output_cost_per_1k: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.start_minute < 24 * 60:
            raise ValueError("UTC price override start must be within a day")
        if not 0 <= self.end_minute < 24 * 60:
            raise ValueError("UTC price override end must be within a day")
        if self.start_minute == self.end_minute:
            raise ValueError("UTC price override window must not be empty")
        costs = (self.input_cost_per_1k, self.output_cost_per_1k)
        if all(cost is None for cost in costs):
            raise ValueError("UTC price override must set an input or output cost")
        if any(cost is not None and cost < 0 for cost in costs):
            raise ValueError("UTC price override costs must be non-negative")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "UtcPriceOverride":
        return cls(
            start_minute=_parse_utc_clock(data["utc_start"]),
            end_minute=_parse_utc_clock(data["utc_end"]),
            input_cost_per_1k=_optional_float(data.get("input_cost_per_1k")),
            output_cost_per_1k=_optional_float(data.get("output_cost_per_1k")),
        )

    def applies_at(self, at: datetime) -> bool:
        if at.tzinfo is None or at.utcoffset() is None:
            raise ValueError("price lookup datetime must be timezone-aware")
        utc_at = at.astimezone(timezone.utc)
        minute = utc_at.hour * 60 + utc_at.minute
        if self.start_minute < self.end_minute:
            return self.start_minute <= minute < self.end_minute
        return minute >= self.start_minute or minute < self.end_minute


@dataclass(frozen=True)
class ModelBenchmarkProfile:
    model_id: str
    benchmarks: Mapping[str, float | None] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    provider: str | None = None
    input_modalities: tuple[str, ...] = ()
    source_quality: str = "missing"
    source_urls: tuple[str, ...] = ()
    release_date: str | None = None
    context_length: int | None = None
    max_output_tokens: int | None = None
    parameters_b: float | None = None
    active_parameters_b: float | None = None
    input_cost_per_1k: float | None = None
    output_cost_per_1k: float | None = None
    utc_price_overrides: tuple[UtcPriceOverride, ...] = ()

    def __post_init__(self) -> None:
        _validate_utc_price_overrides(self.utc_price_overrides)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModelBenchmarkProfile":
        return cls(
            model_id=str(data["model_id"]),
            aliases=tuple(str(value) for value in data.get("aliases", ())),
            provider=_optional_str(data.get("provider")),
            input_modalities=normalize_modalities(data.get("input_modalities", ())),
            source_quality=str(data.get("source_quality", "missing")),
            source_urls=tuple(str(value) for value in data.get("source_urls", ())),
            release_date=_optional_str(data.get("release_date")),
            context_length=_optional_int(data.get("context_length")),
            max_output_tokens=_optional_int(data.get("max_output_tokens")),
            parameters_b=_optional_float(data.get("parameters_b")),
            active_parameters_b=_optional_float(data.get("active_parameters_b")),
            input_cost_per_1k=_optional_float(data.get("input_cost_per_1k")),
            output_cost_per_1k=_optional_float(data.get("output_cost_per_1k")),
            utc_price_overrides=tuple(
                UtcPriceOverride.from_mapping(item)
                for item in (data.get("utc_price_overrides") or ())
            ),
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

    def supports_input_modalities(self, modalities: Sequence[str]) -> bool:
        required = set(normalize_modalities(modalities))
        return required.issubset(self.input_modalities)

    def costs_per_1k_at(
        self,
        at: datetime | None = None,
    ) -> tuple[float | None, float | None]:
        input_cost = self.input_cost_per_1k
        output_cost = self.output_cost_per_1k
        # Old pickles without this instance field inherit the immutable class
        # default, so direct attribute access remains backward compatible.
        if not self.utc_price_overrides:
            return input_cost, output_cost

        effective_at = at or datetime.now(timezone.utc)
        for override in self.utc_price_overrides:
            if override.applies_at(effective_at):
                return (
                    override.input_cost_per_1k
                    if override.input_cost_per_1k is not None
                    else input_cost,
                    override.output_cost_per_1k
                    if override.output_cost_per_1k is not None
                    else output_cost,
                )
        return input_cost, output_cost


class BenchmarkProfileCatalog:
    def __init__(self, profiles: Sequence[ModelBenchmarkProfile] | None = None) -> None:
        self._profiles: dict[str, ModelBenchmarkProfile] = {}
        self._aliases: dict[str, str] = {}
        for profile in profiles or ():
            self.add(profile)

    def add(self, profile: ModelBenchmarkProfile) -> None:
        existing = self._profiles.get(profile.model_id)
        if existing is not None:
            profile = merge_model_profiles(existing, profile)
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


def merge_model_profiles(
    base: ModelBenchmarkProfile,
    override: ModelBenchmarkProfile,
) -> ModelBenchmarkProfile:
    if base.model_id != override.model_id:
        raise ValueError("Can only merge profiles for the same model_id")

    source_quality = _higher_source_quality(base.source_quality, override.source_quality)
    override_defines_pricing = any(
        value is not None
        for value in (
            override.input_cost_per_1k,
            override.output_cost_per_1k,
        )
    ) or bool(override.utc_price_overrides)
    if override_defines_pricing:
        input_cost_per_1k = (
            override.input_cost_per_1k
            if override.input_cost_per_1k is not None
            else base.input_cost_per_1k
        )
        output_cost_per_1k = (
            override.output_cost_per_1k
            if override.output_cost_per_1k is not None
            else base.output_cost_per_1k
        )
        utc_price_overrides = override.utc_price_overrides
    else:
        input_cost_per_1k = base.input_cost_per_1k
        output_cost_per_1k = base.output_cost_per_1k
        utc_price_overrides = base.utc_price_overrides

    return ModelBenchmarkProfile(
        model_id=base.model_id,
        benchmarks={**base.benchmarks, **override.benchmarks},
        aliases=tuple(dict.fromkeys((*base.aliases, *override.aliases))),
        provider=override.provider or base.provider,
        input_modalities=override.input_modalities or base.input_modalities,
        source_quality=source_quality,
        source_urls=tuple(dict.fromkeys((*base.source_urls, *override.source_urls))),
        release_date=override.release_date or base.release_date,
        context_length=override.context_length or base.context_length,
        max_output_tokens=override.max_output_tokens or base.max_output_tokens,
        parameters_b=override.parameters_b or base.parameters_b,
        active_parameters_b=override.active_parameters_b or base.active_parameters_b,
        input_cost_per_1k=input_cost_per_1k,
        output_cost_per_1k=output_cost_per_1k,
        utc_price_overrides=utc_price_overrides,
    )


def load_builtin_benchmark_profiles() -> BenchmarkProfileCatalog:
    with resources.files("xrouter_llm.resources").joinpath("routerbench_public_benchmarks.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)
    return BenchmarkProfileCatalog([ModelBenchmarkProfile.from_mapping(item) for item in data])


def load_benchmark_profiles(path: str | Path) -> BenchmarkProfileCatalog:
    """Load model profiles from a JSON/YAML file or a directory of them.

    Accepts a single file (a list, a ``{"models": [...]}`` mapping, or one
    bare model mapping) or a directory, in which case every ``*.yaml``,
    ``*.yml`` and ``*.json`` inside is loaded -- one model per file is the
    intended layout for a model registry.
    """
    target = Path(path)
    if target.is_dir():
        files = sorted(
            entry
            for entry in target.iterdir()
            if entry.is_file() and entry.suffix.lower() in {".yaml", ".yml", ".json"}
        )
        mappings = [mapping for file in files for mapping in _read_profile_mappings(file)]
    else:
        mappings = _read_profile_mappings(target)
    return BenchmarkProfileCatalog([ModelBenchmarkProfile.from_mapping(item) for item in mappings])


def _read_profile_mappings(file_path: Path) -> list[Mapping[str, Any]]:
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if isinstance(data, Mapping):
        if "models" in data:
            return list(data["models"])
        return [data]  # a single bare model mapping (one model per file)
    if isinstance(data, list):
        return data
    raise ValueError(f"Profile file {file_path} must contain a mapping or list")


def combine_benchmark_profile_catalogs(
    catalogs: Sequence[BenchmarkProfileCatalog],
) -> BenchmarkProfileCatalog:
    combined = BenchmarkProfileCatalog()
    for catalog in catalogs:
        for profile in catalog.profiles():
            combined.add(profile)
    return combined


def _higher_source_quality(left: str, right: str) -> str:
    left_score = SOURCE_QUALITY_LEVELS.get(left, SOURCE_QUALITY_LEVELS["third_party"])
    right_score = SOURCE_QUALITY_LEVELS.get(right, SOURCE_QUALITY_LEVELS["third_party"])
    return right if right_score >= left_score else left


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


def _parse_utc_clock(value: Any) -> int:
    text = str(value).strip()
    try:
        parsed = datetime.strptime(text, "%H:%M")
    except ValueError as exc:
        raise ValueError(
            f"UTC price override time must use H:MM or HH:MM, got {value!r}"
        ) from exc
    return parsed.hour * 60 + parsed.minute


def _validate_utc_price_overrides(
    overrides: Sequence[UtcPriceOverride],
) -> None:
    segments: list[tuple[int, int, int]] = []
    for index, override in enumerate(overrides):
        if override.start_minute < override.end_minute:
            current_segments = ((override.start_minute, override.end_minute),)
        else:
            current_segments = (
                (override.start_minute, 24 * 60),
                (0, override.end_minute),
            )
        for start, end in current_segments:
            for existing_start, existing_end, existing_index in segments:
                if max(start, existing_start) < min(end, existing_end):
                    raise ValueError(
                        "UTC price override windows must not overlap "
                        f"(entries {existing_index + 1} and {index + 1})"
                    )
            segments.append((start, end, index))


def normalize_modalities(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    try:
        iter(values)
    except TypeError:
        values = (values,)
    return tuple(
        dict.fromkeys(
            str(value).strip().lower()
            for value in values
            if value is not None and str(value).strip()
        )
    )
