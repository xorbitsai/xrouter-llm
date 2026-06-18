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
    catalog = load_benchmark_profiles("config/models")
    assert len(catalog) == 8
    assert catalog.get("claude-opus-4-8").provider == "anthropic"
    # ids containing "/" and ":" survive the per-file layout
    assert catalog.get("nvidia/nemotron-3-ultra-550b-a55b:free").parameters_b == 550.0
