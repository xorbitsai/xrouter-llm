import textwrap
from datetime import datetime, timezone

from xrouter_llm.profiles import SOURCE_QUALITY_LEVELS, load_benchmark_profiles


def test_load_yaml_benchmark_profiles(tmp_path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        textwrap.dedent(
            """
            models:
              - model_id: demo-model
                provider: demo
                aliases: ["demo/demo-model"]
                input_modalities: [text, image, file]
                context_length: 1000000
                input_cost_per_1k: 0.001
                output_cost_per_1k: 0.002
                utc_price_overrides:
                  - utc_start: "01:00"
                    utc_end: "04:00"
                    utc_days: [monday, tuesday, wednesday, thursday, friday]
                    input_cost_per_1k: 0.003
                    output_cost_per_1k: 0.004
                benchmarks:
                  gpqa_diamond: 90.0
                  mmlu: 0.85
            """
        ),
        encoding="utf-8",
    )

    catalog = load_benchmark_profiles(path)
    assert len(catalog) == 1
    profile = catalog.get("demo-model")
    assert profile.provider == "demo"
    assert profile.input_cost_per_1k == 0.001
    assert profile.costs_per_1k_at(
        datetime(2026, 8, 17, 2, 0, tzinfo=timezone.utc)
    ) == (0.003, 0.004)
    assert profile.costs_per_1k_at(
        datetime(2026, 8, 22, 2, 0, tzinfo=timezone.utc)
    ) == (0.001, 0.002)
    assert profile.input_modalities == ("text", "image", "file")
    assert profile.supports_input_modalities(["image"])
    assert not profile.supports_input_modalities(["audio"])
    # published-percentage and 0-1 both normalize into [0, 1]
    assert profile.normalized_benchmark("gpqa_diamond") == 0.90
    assert profile.normalized_benchmark("mmlu") == 0.85
    # alias resolves to the same profile
    assert catalog.get("demo/demo-model").model_id == "demo-model"


def test_load_single_model_per_file_directory(tmp_path) -> None:
    (tmp_path / "a.yaml").write_text("model_id: a\nprovider: x\n", encoding="utf-8")
    (tmp_path / "b.yml").write_text("model_id: b\nprovider: y\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("not a profile", encoding="utf-8")

    catalog = load_benchmark_profiles(tmp_path)
    assert len(catalog) == 2
    assert {p.model_id for p in catalog.profiles()} == {"a", "b"}


def test_load_yaml_accepts_unquoted_sexagesimal_utc_clocks(tmp_path) -> None:
    path = tmp_path / "scheduled.yaml"
    path.write_text(
        textwrap.dedent(
            """
            model_id: scheduled
            input_cost_per_1k: 0.001
            utc_price_overrides:
              - utc_start: 1:00
                utc_end: 4:00
                input_cost_per_1k: 0.003
            """
        ),
        encoding="utf-8",
    )

    profile = load_benchmark_profiles(path).get("scheduled")

    assert profile.costs_per_1k_at(
        datetime(2026, 8, 17, 2, 0, tzinfo=timezone.utc)
    ) == (0.003, None)


def test_shipped_models_registry_loads() -> None:
    from xrouter_llm.paths import default_models_dir

    catalog = load_benchmark_profiles(default_models_dir())
    assert len(catalog) == 41
    assert {profile.source_quality for profile in catalog.profiles()} <= set(
        SOURCE_QUALITY_LEVELS
    )
    # model_id is the canonical OpenRouter slug; the bare id stays as an alias.
    opus = catalog.get("anthropic/claude-opus-5")
    assert opus.provider == "anthropic"
    assert opus.model_id == "anthropic/claude-opus-5"
    assert opus.supports_input_modalities(["image", "file"])
    assert catalog.get("claude-opus-5").model_id == "anthropic/claude-opus-5"
    # ids containing "/" survive the per-file layout
    glm_47 = catalog.get("z-ai/glm-4.7")
    assert glm_47.input_cost_per_1k == 0.0006
    assert glm_47.output_cost_per_1k == 0.0022
    glm_53_flash = catalog.get("glm-5.3-flash")
    assert glm_53_flash.model_id == "z-ai/glm-5.3-flash"
    assert glm_53_flash.supports_input_modalities(["image", "video", "file"])
    assert glm_53_flash.context_length == 1048576
    assert glm_53_flash.max_output_tokens == 131072
    assert glm_53_flash.parameters_b == 320
    assert glm_53_flash.active_parameters_b == 18
    assert glm_53_flash.input_cost_per_1k == 0.00015
    assert glm_53_flash.output_cost_per_1k == 0.00050
    assert glm_53_flash.benchmarks["gpqa_diamond"] == 91.2
    assert "livecodebench" not in glm_53_flash.benchmarks
    assert catalog.get("deepseek/deepseek-v4-flash").provider is None
    assert catalog.get("deepseek/deepseek-v4-pro").provider is None
    luna = catalog.get("gpt-5.6-luna")
    assert luna.model_id == "openai/gpt-5.6-luna"
    assert luna.input_cost_per_1k == 0.0002
    assert luna.output_cost_per_1k == 0.0012
    assert luna.benchmarks["gpqa_diamond"] == 91.1
    assert "livecodebench" not in luna.benchmarks
    terra = catalog.get("openai/gpt-5.6-terra")
    assert terra.input_cost_per_1k == 0.002
    assert terra.output_cost_per_1k == 0.012
    assert terra.benchmarks["livecodebench"] == 85.9
    sol = catalog.get("gpt-5.6")
    assert sol.model_id == "openai/gpt-5.6-sol"
    assert sol.input_cost_per_1k == 0.004
    assert sol.output_cost_per_1k == 0.020
    assert sol.benchmarks["gpqa_diamond"] == 94.1
    # 2026-07 additions: latest Gemini, Claude, and Kimi models
    sonnet_5 = catalog.get("claude-sonnet-5")
    assert sonnet_5.model_id == "anthropic/claude-sonnet-5"
    assert sonnet_5.input_cost_per_1k == 0.002
    assert sonnet_5.output_cost_per_1k == 0.010
    opus_5 = catalog.get("claude-opus-5")
    assert opus_5.model_id == "anthropic/claude-opus-5"
    assert opus_5.source_quality == "third_party"
    assert opus_5.supports_input_modalities(["image", "file"])
    assert opus_5.benchmarks["gpqa_diamond"] == 93.2
    assert opus_5.benchmarks["livecodebench"] == 89.0
    grok = catalog.get("grok-4.6")
    assert grok.model_id == "x-ai/grok-4.6"
    assert grok.context_length == 500000
    assert grok.benchmarks["gpqa_diamond"] == 94.7
    assert grok.benchmarks["livecodebench"] == 88.2
    assert catalog.get("google/gemini-3.1-pro-preview").provider == "google"
    assert catalog.get("google/gemini-3.1-pro-preview").benchmarks["livecodebench"] == 88.5
    assert catalog.get("google/gemini-3.1-flash-lite").benchmarks["livecodebench"] == 72.0
    kimi_k27 = catalog.get("moonshotai/kimi-k2.7-code")
    assert kimi_k27.provider == "moonshotai"
    assert kimi_k27.input_cost_per_1k == 0.00095
    assert kimi_k27.output_cost_per_1k == 0.004
    kimi_k3 = catalog.get("kimi-k3")
    assert kimi_k3.model_id == "moonshotai/kimi-k3"
    assert kimi_k3.parameters_b == 2800
    assert kimi_k3.benchmarks["gpqa_diamond"] == 93.5
    assert kimi_k3.benchmarks["livecodebench"] == 87.2
    qwen_plus = catalog.get("qwen3.7-plus")
    assert qwen_plus.model_id == "qwen/qwen3.7-plus"
    assert qwen_plus.source_quality == "third_party"
    assert qwen_plus.supports_input_modalities(["image", "video"])
    assert qwen_plus.max_output_tokens == 131072
    assert qwen_plus.benchmarks["gpqa_diamond"] == 90.0
    assert "livecodebench" not in qwen_plus.benchmarks
    qwen_flash = catalog.get("qwen3.8-flash")
    assert qwen_flash.model_id == "qwen/qwen3.8-flash"
    assert qwen_flash.source_quality == "self_eval"
    assert qwen_flash.supports_input_modalities(["image", "video"])
    assert qwen_flash.context_length == 1000000
    assert qwen_flash.max_output_tokens == 131072
    assert qwen_flash.parameters_b == 125
    assert qwen_flash.active_parameters_b == 6
    assert qwen_flash.input_cost_per_1k == 0.00015
    assert qwen_flash.output_cost_per_1k == 0.00047
    assert qwen_flash.benchmarks["gpqa_diamond"] == 91.7
    assert qwen_flash.benchmarks["livecodebench"] == 91.9
    # superseded models are removed from the registry
    removed = {
        "google/gemini-2.5-flash-lite",
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-opus-4.8",
        "openai/gpt-5.5",
        "deepseek/deepseek-v4-flash-0731",
        "deepseek/deepseek-v4-pro-0813",
    }
    assert removed.isdisjoint({p.model_id for p in catalog.profiles()})


def test_recent_models_are_in_bundled_multi_model_routers() -> None:
    from xrouter_llm.paths import default_routers_dir
    from xrouter_llm.serving import load_router_configs

    configs = load_router_configs(default_routers_dir())
    for config_name in ("auto", "quality-pair"):
        assert "deepseek/deepseek-v4.1-flash" not in configs[config_name].models
        assert "stepfun/step-5-preview" in configs[config_name].models
        assert "deepseek/deepseek-v4-pro-0813" not in configs[config_name].models
        assert "deepseek/deepseek-v4-flash-0731" not in configs[config_name].models
        assert "google/gemini-3.8-flash" in configs[config_name].models
        assert "google/gemini-3.7-flash" not in configs[config_name].models
        assert "google/gemini-3.5-flash" not in configs[config_name].models
        assert "google/gemini-3.1-pro-preview" in configs[config_name].models
        assert "google/gemini-3.1-flash-lite" in configs[config_name].models
        assert "z-ai/glm-5.3-flash" in configs[config_name].models
        assert "qwen/qwen3.8-flash" not in configs[config_name].models
    assert "z-ai/glm-5.3-flash" in configs["cheap-pair"].models
    assert "deepseek/deepseek-v4.1-flash" not in configs["cheap-pair"].models


def test_bundled_candidates_have_measured_capability() -> None:
    from xrouter_llm.paths import default_models_dir, default_routers_dir
    from xrouter_llm.serving import load_router_configs

    catalog = load_benchmark_profiles(default_models_dir())
    configs = load_router_configs(default_routers_dir())
    additions = {
        "openai/gpt-6-astra",
        "google/gemini-3.8-flash",
        "z-ai/glm-5.3",
        "qwen/qwen3.8-max-0902",
        "qwen/qwen3.8-omni-flash",
    }
    for name in ("auto", "quality-pair"):
        assert additions <= set(configs[name].models)
    for config in configs.values():
        assert len(config.models) == len(set(config.models))
        for model_id in config.models:
            profile = catalog.get(model_id)
            assert profile.provider is not None, model_id
            assert profile.source_urls, model_id
            assert any(
                profile.normalized_benchmark(key) is not None
                for key in ("gpqa_diamond", "livecodebench")
            ), model_id
            assert profile.model_id != "anthropic/claude-fable-5.1"

    pending = {
        "moonshotai/kimi-k3.1-pro",
        "minimax/minimax-m3.1-flash",
        "deepseek/deepseek-v4.1-flash-1017",
        "google/gemini-4-argon",
        "anthropic/claude-haiku-5.5",
        "mistralai/mistral-large-4-0",
        "anthropic/claude-sonnet-5.5",
        "anthropic/claude-opus-5.5",
        "openai/gpt-6.1-sol",
        "openai/gpt-6-luna",
        "x-ai/grok-4.7",
        "z-ai/glm-5.3-flashx",
        "xiaomi/mimo-v2.6-pro",
        "xiaomi/mimo-v2.6-flash",
    }
    for model_id in pending:
        profile = catalog.get(model_id)
        assert profile.model_id == model_id
        assert profile.provider is not None
        assert profile.benchmarks == {}
        assert profile.source_quality == "missing"
        assert profile.input_cost_per_1k > 0
        assert profile.output_cost_per_1k > 0
        assert catalog.get(model_id.split("/")[1]).model_id == model_id

    # Exact revisions must not acquire another checkpoint's capability scores.
    assert "livecodebench" not in catalog.get("qwen/qwen3.8-max-0902").benchmarks
    assert catalog.get("gpt-5.6").model_id == "openai/gpt-5.6-sol"
    assert catalog.get("claude-opus-5").model_id == "anthropic/claude-opus-5"


def test_deepseek_latest_alias_does_not_inherit_old_capability() -> None:
    from xrouter_llm.paths import default_models_dir

    catalog = load_benchmark_profiles(default_models_dir())
    latest = catalog.get("deepseek-flash")
    assert latest.model_id == "deepseek/deepseek-v4.1-flash-1017"
    assert latest.benchmarks == {}
    assert catalog.get("deepseek-v4.1-flash").benchmarks["gpqa_diamond"] == 90.9
    assert latest.costs_per_1k_at(
        datetime(2026, 10, 26, 2, 0, tzinfo=timezone.utc)
    ) == (0.00030, 0.00120)
    assert latest.costs_per_1k_at(
        datetime(2026, 10, 25, 2, 0, tzinfo=timezone.utc)
    ) == (0.00015, 0.00060)


def test_persisted_legacy_auto_bindings_remain_canonical() -> None:
    from xrouter_llm.paths import default_models_dir, default_routers_dir
    from xrouter_llm.serving import load_router_configs

    catalog = load_benchmark_profiles(default_models_dir())
    configs = load_router_configs(default_routers_dir())
    legacy_modalities = {
        "deepseek/deepseek-v4.1-flash": {"text", "image"},
        "google/gemini-3.5-flash": {"text", "image", "audio", "video", "file"},
        "google/gemini-3.7-flash": {"text", "image", "audio", "video", "file"},
        "z-ai/glm-5.2": {"text"},
        "openai/gpt-5.4-mini": {"text", "image", "file"},
    }
    # XAgent validates canonical membership, not alias resolution, for every
    # saved candidate before routing (including single-candidate Auto configs).
    for model_id, modalities in legacy_modalities.items():
        assert model_id in catalog.known_model_ids()
        profile = catalog.get(model_id)
        assert profile.model_id == model_id
        assert set(profile.input_modalities) == modalities
        assert profile.benchmarks
        for config in configs.values():
            assert model_id not in config.models
