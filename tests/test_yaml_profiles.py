import textwrap

from xrouter_llm.profiles import load_benchmark_profiles


def test_load_yaml_benchmark_profiles(tmp_path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        textwrap.dedent(
            """
            models:
              - model_id: demo-model
                provider: demo
                aliases: ["demo/demo-model"]
                context_length: 1000000
                input_cost_per_1k: 0.001
                output_cost_per_1k: 0.002
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


def test_shipped_models_registry_loads() -> None:
    from xrouter_llm.paths import default_models_dir

    catalog = load_benchmark_profiles(default_models_dir())
    assert len(catalog) == 15
    # model_id is the canonical OpenRouter slug; the bare id stays as an alias.
    opus = catalog.get("anthropic/claude-opus-4.8")
    assert opus.provider == "anthropic"
    assert opus.model_id == "anthropic/claude-opus-4.8"
    assert catalog.get("claude-opus-4-8").model_id == "anthropic/claude-opus-4.8"
    # ids containing "/" survive the per-file layout
    assert catalog.get("z-ai/glm-5.2").provider == "z-ai"
    assert catalog.get("z-ai/glm-5.2").benchmarks["livecodebench"] == 69.5
    assert "livecodebench" not in catalog.get("deepseek/deepseek-v4-flash").benchmarks
    assert catalog.get("openai/gpt-5.5").provider == "openai"
    # 2026-07 additions: latest Gemini, Claude, and Kimi models
    assert catalog.get("claude-sonnet-5").model_id == "anthropic/claude-sonnet-5"
    opus_5 = catalog.get("claude-opus-5")
    assert opus_5.model_id == "anthropic/claude-opus-5"
    assert opus_5.source_quality == "third_party"
    assert opus_5.benchmarks["gpqa_diamond"] == 93.2
    assert opus_5.benchmarks["livecodebench"] == 89.0
    assert catalog.get("google/gemini-3.5-flash").benchmarks["gpqa_diamond"] == 92.2
    assert catalog.get("google/gemini-3.5-flash").benchmarks["livecodebench"] == 87.6
    assert catalog.get("google/gemini-3.1-pro-preview").provider == "google"
    assert catalog.get("google/gemini-3.1-pro-preview").benchmarks["livecodebench"] == 88.5
    assert catalog.get("google/gemini-3.1-flash-lite").benchmarks["livecodebench"] == 72.0
    assert catalog.get("moonshotai/kimi-k2.7-code").provider == "moonshotai"
    kimi_k3 = catalog.get("kimi-k3")
    assert kimi_k3.model_id == "moonshotai/kimi-k3"
    assert kimi_k3.parameters_b == 2800
    assert kimi_k3.benchmarks["gpqa_diamond"] == 93.5
    assert kimi_k3.benchmarks["livecodebench"] == 87.2
    # superseded models are removed from the registry
    removed = {"google/gemini-2.5-flash-lite", "anthropic/claude-sonnet-4.6"}
    assert removed.isdisjoint({p.model_id for p in catalog.profiles()})
