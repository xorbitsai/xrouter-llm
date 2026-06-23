import hashlib

import numpy as np

from xrouter_llm import (
    BenchmarkRow,
    IRTRouter,
    ModelBenchmarkProfile,
    PolicyParams,
    evaluate_agentic_holdout,
    evaluate_model_holdout,
    evaluate_offline,
)


class _StubBackend:
    name = "stub:holdout"

    def encode(self, texts):
        out = []
        for t in texts:
            seed = int(hashlib.sha1(t.encode("utf-8")).hexdigest()[:8], 16)
            out.append(np.random.default_rng(seed).standard_normal(16))
        return np.asarray(out, dtype=float)


def _synthetic_rows() -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for index in range(12):
        prompt = f"prompt number {index} about coding and reasoning"
        prompt_id = f"p{index}"
        rows.append(BenchmarkRow(prompt_id, prompt, "strong", 0.95, cost_usd=0.02))
        rows.append(
            BenchmarkRow(prompt_id, prompt, "mid", 0.9 if index % 2 == 0 else 0.4, cost_usd=0.01)
        )
        rows.append(
            BenchmarkRow(prompt_id, prompt, "cheap", 0.8 if index % 3 == 0 else 0.3, cost_usd=0.001)
        )
    return rows


def _agentic_rows() -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for index in range(12):
        prompt = f"agentic prompt {index} editing a repository"
        prompt_id = f"agentic-p{index}"
        rows.append(
            BenchmarkRow(prompt_id, prompt, "strong", 0.95, cost_usd=0.02, task="agentic/coding")
        )
        rows.append(
            BenchmarkRow(
                prompt_id,
                prompt,
                "mid",
                0.9 if index % 2 == 0 else 0.4,
                cost_usd=0.01,
                task="agentic/coding",
            )
        )
        rows.append(
            BenchmarkRow(
                prompt_id,
                prompt,
                "cheap",
                0.8 if index % 3 == 0 else 0.3,
                cost_usd=0.001,
                task="agentic/coding",
            )
        )
    return rows


def _profiles() -> list[ModelBenchmarkProfile]:
    return [
        ModelBenchmarkProfile("strong", benchmarks={"gpqa_diamond": 90.0}, input_cost_per_1k=0.01, output_cost_per_1k=0.03),
        ModelBenchmarkProfile("mid", benchmarks={"gpqa_diamond": 60.0}, input_cost_per_1k=0.005, output_cost_per_1k=0.015),
        ModelBenchmarkProfile("cheap", benchmarks={"gpqa_diamond": 30.0}, input_cost_per_1k=0.0005, output_cost_per_1k=0.001),
    ]


def _factory(tmp_path):
    return lambda: IRTRouter(
        benchmark_profiles=_profiles(),
        embedding_backend=_StubBackend(),
        embedding_cache_dir=str(tmp_path),
        min_models_per_prompt=2,
        completion_score_threshold=0.75,
        random_state=3,
    )


def test_offline_decision_cost_does_not_use_realized_cost(tmp_path) -> None:
    result = evaluate_offline(
        _synthetic_rows(),
        policy_params=PolicyParams(completion_threshold=0.5, lambda_cost=1.0),
        test_size=0.5,
        random_state=3,
        predictor_factory=_factory(tmp_path),
    )
    # Decision cost is estimated from profile pricing, not the realized cost_usd.
    assert "average_decision_cost" in result.metrics
    assert "average_cost" in result.metrics
    assert result.metrics["average_decision_cost"] > 0.0


def test_model_holdout_reports_unseen_model_metrics(tmp_path) -> None:
    report = evaluate_model_holdout(
        _synthetic_rows(),
        predictor_factory=_factory(tmp_path),
        test_size=0.5,
        random_state=7,
        calibration_bins=4,
    )

    assert report.holdout_models
    assert report.aggregate["model_count"] == float(len(report.per_model))
    for entry in report.per_model:
        assert entry["model_id"] in {"strong", "mid", "cheap"}
        assert entry["n"] >= 1.0
        assert 0.0 <= entry["base_completion_rate"] <= 1.0


def test_agentic_holdout_reports_subject_selection_metrics(tmp_path) -> None:
    report = evaluate_agentic_holdout(
        _agentic_rows(),
        thresholds=[0.6, 0.8],
        agentic_tasks=["agentic/coding"],
        predictor_factory=_factory(tmp_path),
        test_size=0.5,
        random_state=7,
        calibration_bins=4,
    )

    assert report.agentic_tasks == ["agentic/coding"]
    assert report.subject_count == 3
    assert report.test_prompt_count > 0
    assert len(report.thresholds) == 2
    assert report.top_subject["prompt_count"] == float(report.test_prompt_count)
    assert "agentic/coding" in report.per_task
    assert 0.0 <= report.top_subject["top1_completion_rate"] <= 1.0
