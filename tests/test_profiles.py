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


def test_profile_catalog_replaces_scalar_and_scheduled_pricing_as_a_group() -> None:
    scheduled = ModelBenchmarkProfile.from_mapping(
        {
            "model_id": "model-a",
            "input_cost_per_1k": 0.00022,
            "output_cost_per_1k": 0.00066,
            "utc_price_overrides": [
                {
                    "utc_start": "01:00",
                    "utc_end": "04:00",
                    "input_cost_per_1k": 0.00044,
                    "output_cost_per_1k": 0.00132,
                }
            ],
        }
    )
    scalar = ModelBenchmarkProfile(
        model_id="model-a",
        input_cost_per_1k=0.000435,
        output_cost_per_1k=0.00087,
    )

    profile = BenchmarkProfileCatalog([scheduled, scalar]).get("model-a")

    assert profile.utc_price_overrides == ()
    assert profile.costs_per_1k_at(
        datetime(2026, 8, 17, 2, 0, tzinfo=timezone.utc)
    ) == (0.000435, 0.00087)


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
                    "utc_days": [
                        "monday",
                        "tuesday",
                        "wednesday",
                        "thursday",
                        "friday",
                    ],
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
    assert profile.costs_per_1k_at(
        datetime(2026, 8, 22, 2, 0, tzinfo=timezone.utc)
    ) == (0.001, 0.002)
    with pytest.raises(ValueError, match="timezone-aware"):
        profile.costs_per_1k_at(datetime(2026, 8, 17, 2, 0))


def test_profile_rejects_naive_price_lookup_without_a_schedule() -> None:
    profile = ModelBenchmarkProfile(
        model_id="unscheduled",
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.002,
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        profile.costs_per_1k_at(datetime(2026, 8, 17, 2, 0))
    assert profile.costs_per_1k_at(
        datetime(2026, 8, 17, 2, 0, tzinfo=timezone.utc)
    ) == (0.001, 0.002)
    assert profile.costs_per_1k_at() == (0.001, 0.002)


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


def test_profile_rejects_overlapping_utc_price_windows() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        ModelBenchmarkProfile.from_mapping(
            {
                "model_id": "scheduled",
                "utc_price_overrides": [
                    {
                        "utc_start": "22:00",
                        "utc_end": "02:00",
                        "input_cost_per_1k": 0.003,
                    },
                    {
                        "utc_start": "01:00",
                        "utc_end": "04:00",
                        "input_cost_per_1k": 0.004,
                    },
                ],
            }
        )


def test_profile_accepts_overlapping_windows_on_disjoint_utc_days() -> None:
    profile = ModelBenchmarkProfile.from_mapping(
        {
            "model_id": "scheduled",
            "input_cost_per_1k": 0.001,
            "utc_price_overrides": [
                {
                    "utc_start": "01:00",
                    "utc_end": "04:00",
                    "utc_days": ["monday"],
                    "input_cost_per_1k": 0.003,
                },
                {
                    "utc_start": "01:00",
                    "utc_end": "04:00",
                    "utc_days": ["saturday", "sunday"],
                    "input_cost_per_1k": 0.004,
                },
            ],
        }
    )

    assert profile.costs_per_1k_at(
        datetime(2026, 8, 17, 2, 0, tzinfo=timezone.utc)
    ) == (0.003, None)
    assert profile.costs_per_1k_at(
        datetime(2026, 8, 22, 2, 0, tzinfo=timezone.utc)
    ) == (0.004, None)


def test_profile_rejects_overlapping_windows_on_intersecting_utc_days() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        ModelBenchmarkProfile.from_mapping(
            {
                "model_id": "scheduled",
                "utc_price_overrides": [
                    {
                        "utc_start": "01:00",
                        "utc_end": "04:00",
                        "utc_days": ["monday", "tuesday"],
                        "input_cost_per_1k": 0.003,
                    },
                    {
                        "utc_start": "03:00",
                        "utc_end": "05:00",
                        "utc_days": ["tuesday", "wednesday"],
                        "input_cost_per_1k": 0.004,
                    },
                ],
            }
        )


@pytest.mark.parametrize("utc_days", [[], ["funday"], 1])
def test_profile_rejects_invalid_utc_price_override_days(utc_days: object) -> None:
    with pytest.raises(ValueError, match="UTC price override day"):
        ModelBenchmarkProfile.from_mapping(
            {
                "model_id": "scheduled",
                "utc_price_overrides": [
                    {
                        "utc_start": "01:00",
                        "utc_end": "04:00",
                        "utc_days": utc_days,
                        "input_cost_per_1k": 0.003,
                    }
                ],
            }
        )


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


def test_profile_accepts_yaml_sexagesimal_minutes_for_utc_price_window() -> None:
    profile = ModelBenchmarkProfile.from_mapping(
        {
            "model_id": "scheduled",
            "input_cost_per_1k": 0.001,
            "utc_price_overrides": [
                {
                    "utc_start": 60,
                    "utc_end": 240,
                    "input_cost_per_1k": 0.003,
                }
            ],
        }
    )

    assert profile.costs_per_1k_at(
        datetime(2026, 8, 17, 2, 0, tzinfo=timezone.utc)
    ) == (0.003, None)


def test_profile_rejects_misspelled_utc_price_override_key() -> None:
    with pytest.raises(
        ValueError,
        match=r"Unknown model profile pricing field.*utc_price_override",
    ):
        ModelBenchmarkProfile.from_mapping(
            {
                "model_id": "scheduled",
                "utc_price_override": [
                    {
                        "utc_start": "01:00",
                        "utc_end": "04:00",
                        "input_cost_per_1k": 0.003,
                    }
                ],
            }
        )


def test_profile_rejects_unknown_field_inside_utc_price_override() -> None:
    with pytest.raises(ValueError, match=r"Unknown UTC price override field.*utc_stop"):
        ModelBenchmarkProfile.from_mapping(
            {
                "model_id": "scheduled",
                "utc_price_overrides": [
                    {
                        "utc_start": "01:00",
                        "utc_end": "04:00",
                        "utc_stop": "05:00",
                        "input_cost_per_1k": 0.003,
                    }
                ],
            }
        )


def test_profile_without_schedule_field_from_legacy_artifact_uses_base_costs() -> None:
    profile = ModelBenchmarkProfile(
        model_id="legacy",
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.002,
    )
    object.__delattr__(profile, "utc_price_overrides")

    assert "utc_price_overrides" not in vars(profile)
    assert profile.utc_price_overrides == ()
    assert profile.costs_per_1k_at(
        datetime(2026, 8, 17, 2, 0, tzinfo=timezone.utc)
    ) == (0.001, 0.002)
