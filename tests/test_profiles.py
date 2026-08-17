from datetime import datetime, timedelta, timezone

import pytest

from xrouter_llm import (
    BenchmarkProfileCatalog,
    ModelBenchmarkProfile,
    load_builtin_benchmark_profiles,
    normalize_modalities,
)


def test_builtin_profiles_cover_routerbench_models() -> None:
    catalog = load_builtin_benchmark_profiles()

    assert len(catalog) == 11
    claude = catalog.get("claude-v2")
    assert round(claude.normalized_benchmark("human_eval") or 0.0, 3) == 0.712
    assert catalog.get("GPT-4").model_id == "gpt-4-1106-preview"


def test_profile_catalog_merges_duplicate_model_profiles() -> None:
    catalog = BenchmarkProfileCatalog(
        [
            ModelBenchmarkProfile(
                model_id="model-a",
                benchmarks={"mmlu": 80.0},
                aliases=("a",),
                input_modalities=("text", "image"),
            ),
            ModelBenchmarkProfile(
                model_id="model-a",
                benchmarks={"llmrouterbench_math": 0.7},
                source_quality="dataset_aggregate",
            ),
        ]
    )

    profile = catalog.get("model-a")

    assert profile.benchmarks["mmlu"] == 80.0
    assert profile.benchmarks["llmrouterbench_math"] == 0.7
    assert profile.input_modalities == ("text", "image")
    assert catalog.get("a").model_id == "model-a"


def test_normalize_modalities_handles_empty_scalar_and_mixed_values() -> None:
    assert normalize_modalities(None) == ()
    assert normalize_modalities(" Image ") == ("image",)
    assert normalize_modalities(7) == ("7",)
    assert normalize_modalities([" Image ", None, "", "IMAGE", "Audio"]) == (
        "image",
        "audio",
    )


def test_profile_mapping_accepts_null_input_modalities() -> None:
    profile = ModelBenchmarkProfile.from_mapping(
        {"model_id": "model-a", "input_modalities": None}
    )

    assert profile.input_modalities == ()


def test_profile_resolves_utc_price_overrides_and_boundaries() -> None:
    profile = ModelBenchmarkProfile.from_mapping(
        {
            "model_id": "scheduled",
            "input_cost_per_1k": 0.001,
            "output_cost_per_1k": 0.002,
            "utc_price_overrides": [
                {
                    "utc_start": "01:00",
                    "utc_end": "04:00",
                    "input_cost_per_1k": 0.003,
                    "output_cost_per_1k": 0.004,
                }
            ],
        }
    )

    utc_plus_8 = timezone(timedelta(hours=8))
    assert profile.costs_per_1k_at(
        datetime(2026, 8, 17, 9, 30, tzinfo=utc_plus_8)
    ) == (0.003, 0.004)
    assert profile.costs_per_1k_at(
        datetime(2026, 8, 17, 4, 0, tzinfo=timezone.utc)
    ) == (0.001, 0.002)
    with pytest.raises(ValueError, match="timezone-aware"):
        profile.costs_per_1k_at(datetime(2026, 8, 17, 2, 0))


def test_profile_supports_wrapping_utc_price_window() -> None:
    profile = ModelBenchmarkProfile.from_mapping(
        {
            "model_id": "scheduled",
            "input_cost_per_1k": 0.001,
            "utc_price_overrides": [
                {
                    "utc_start": "22:00",
                    "utc_end": "02:00",
                    "input_cost_per_1k": 0.003,
                }
            ],
        }
    )

    assert profile.costs_per_1k_at(
        datetime(2026, 8, 17, 23, 0, tzinfo=timezone.utc)
    ) == (0.003, None)
    assert profile.costs_per_1k_at(
        datetime(2026, 8, 18, 1, 59, tzinfo=timezone.utc)
    ) == (0.003, None)
    assert profile.costs_per_1k_at(
        datetime(2026, 8, 18, 2, 0, tzinfo=timezone.utc)
    ) == (0.001, None)


def test_profile_accepts_unpadded_utc_price_window() -> None:
    profile = ModelBenchmarkProfile.from_mapping(
        {
            "model_id": "scheduled",
            "input_cost_per_1k": 0.001,
            "utc_price_overrides": [
                {
                    "utc_start": "1:00",
                    "utc_end": "4:00",
                    "input_cost_per_1k": 0.003,
                }
            ],
        }
    )

    assert profile.costs_per_1k_at(
        datetime(2026, 8, 17, 2, 0, tzinfo=timezone.utc)
    ) == (0.003, None)


def test_profile_without_schedule_field_from_legacy_artifact_uses_base_costs() -> None:
    profile = ModelBenchmarkProfile(
        model_id="legacy",
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.002,
    )
    object.__delattr__(profile, "utc_price_overrides")

    assert profile.costs_per_1k_at(
        datetime(2026, 8, 17, 2, 0, tzinfo=timezone.utc)
    ) == (0.001, 0.002)
