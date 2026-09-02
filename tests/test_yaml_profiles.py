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
    assert len(catalog) == 21
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
    glm_52 = catalog.get("z-ai/glm-5.2")
    assert glm_52.provider == "z-ai"
    assert not glm_52.supports_input_modalities(["image"])
    assert glm_52.context_length == 1048576
    assert glm_52.input_cost_per_1k == 0.0014
    assert glm_52.output_cost_per_1k == 0.0044
    assert glm_52.benchmarks["livecodebench"] == 69.5
    glm_53_flash = catalog.get("glm-5.3-flash")
    assert glm_53_flash.model_id == "z-ai/glm-5.3-flash"
    assert glm_53_flash.supports_input_modalities(["image", "video", "file"])
    assert glm_53_flash.context_length == 1048576
    assert glm_53_flash.max_output_tokens == 131072
    assert glm_53_flash.parameters_b == 320
    assert glm_53_flash.active_parameters_b == 18
    assert glm_53_flash.input_cost_per_1k == 0.000075
    assert glm_53_flash.output_cost_per_1k == 0.00025
    assert glm_53_flash.benchmarks["gpqa_diamond"] == 91.2
    assert "livecodebench" not in glm_53_flash.benchmarks
    flash = catalog.get("deepseek/deepseek-v4-flash")
    assert flash.model_id == "deepseek/deepseek-v4-flash-0731"
    assert flash.input_cost_per_1k == 0.00022
    assert flash.output_cost_per_1k == 0.00066
    assert flash.costs_per_1k_at(
        datetime(2026, 8, 17, 2, 0, tzinfo=timezone.utc)
    ) == (0.00044, 0.00132)
    assert flash.costs_per_1k_at(
        datetime(2026, 8, 17, 4, 0, tzinfo=timezone.utc)
    ) == (0.00022, 0.00066)
    assert flash.costs_per_1k_at(
        datetime(2026, 8, 22, 2, 0, tzinfo=timezone.utc)
    ) == (0.00022, 0.00066)
    assert flash.benchmarks["gpqa_diamond"] == 90.8
    assert flash.benchmarks["livecodebench"] == 87.3
    pro = catalog.get("deepseek-v4-pro")
    assert pro.model_id == "deepseek/deepseek-v4-pro-0813"
    assert pro.input_cost_per_1k == 0.00066
    assert pro.output_cost_per_1k == 0.00198
    assert pro.costs_per_1k_at(
        datetime(2026, 8, 17, 2, 0, tzinfo=timezone.utc)
    ) == (0.00132, 0.00396)
    assert pro.costs_per_1k_at(
        datetime(2026, 8, 17, 4, 0, tzinfo=timezone.utc)
    ) == (0.00066, 0.00198)
    assert pro.costs_per_1k_at(
        datetime(2026, 8, 22, 2, 0, tzinfo=timezone.utc)
    ) == (0.00066, 0.00198)
    assert pro.benchmarks["gpqa_diamond"] == 92.8
    assert pro.benchmarks["livecodebench"] == 87.5
    luna = catalog.get("gpt-5.6-luna")
    assert luna.model_id == "openai/gpt-5.6-luna"
    assert luna.input_cost_per_1k == 0.0001
    assert luna.benchmarks["gpqa_diamond"] == 91.1
    assert "livecodebench" not in luna.benchmarks
    terra = catalog.get("openai/gpt-5.6-terra")
    assert terra.input_cost_per_1k == 0.001
    assert terra.benchmarks["livecodebench"] == 85.9
    sol = catalog.get("gpt-5.6")
    assert sol.model_id == "openai/gpt-5.6-sol"
    assert sol.input_cost_per_1k == 0.002
    assert sol.output_cost_per_1k == 0.010
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
    assert catalog.get("google/gemini-3.5-flash").benchmarks["gpqa_diamond"] == 92.2
    assert catalog.get("google/gemini-3.5-flash").benchmarks["livecodebench"] == 87.6
    gemini_37 = catalog.get("gemini-3.7-flash")
    assert gemini_37.model_id == "google/gemini-3.7-flash"
    assert gemini_37.supports_input_modalities(["image", "video", "audio"])
    assert gemini_37.input_cost_per_1k == 0.000375
    assert gemini_37.benchmarks["terminal_bench"] == 77.5
    assert gemini_37.benchmarks["gpqa_diamond"] == 94.5
    assert gemini_37.benchmarks["livecodebench"] == 88.7
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
    }
    assert removed.isdisjoint({p.model_id for p in catalog.profiles()})


def test_recent_models_are_in_bundled_multi_model_routers() -> None:
    from xrouter_llm.paths import default_routers_dir
    from xrouter_llm.serving import load_router_configs

    configs = load_router_configs(default_routers_dir())
    for config_name in ("auto", "quality-pair"):
        assert "google/gemini-3.7-flash" in configs[config_name].models
        assert "z-ai/glm-5.3-flash" in configs[config_name].models
    assert "qwen/qwen3.8-flash" not in configs["auto"].models
    assert "qwen/qwen3.8-flash" in configs["quality-pair"].models
