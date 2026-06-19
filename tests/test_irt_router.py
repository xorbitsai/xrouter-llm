import hashlib

import numpy as np

from xrouter_llm import BenchmarkRow, IRTRouter, ModelBenchmarkProfile, PromptConditionedIRTRouter


class _StubBackend:
    name = "stub:irt"

    def encode(self, texts):
        out = []
        for t in texts:
            seed = int(hashlib.sha1(t.encode("utf-8")).hexdigest()[:8], 16)
            out.append(np.random.default_rng(seed).standard_normal(16))
        return np.asarray(out, dtype=float)


def _profiles():
    return [
        ModelBenchmarkProfile("strong", benchmarks={"gpqa_diamond": 92.0, "livecodebench": 90.0}),
        ModelBenchmarkProfile("mid", benchmarks={"gpqa_diamond": 60.0, "livecodebench": 55.0}),
        ModelBenchmarkProfile("weak", benchmarks={"gpqa_diamond": 30.0, "livecodebench": 20.0}),
    ]


def _rows():
    cap = {"strong": 0.91, "mid": 0.575, "weak": 0.25}
    rows = []
    for i in range(24):
        prompt = f"task number {i} of varying difficulty"
        pid = f"p{i}"
        difficulty = (i % 6) / 6.0 + 0.1  # 0.1 .. ~1.0
        for m, c in cap.items():
            rows.append(BenchmarkRow(pid, prompt, m, 1.0 if c >= difficulty else 0.0))
    return rows


def test_irt_router_ranks_by_capability(tmp_path):
    router = IRTRouter(
        benchmark_profiles=_profiles(),
        embedding_backend=_StubBackend(),
        embedding_cache_dir=str(tmp_path),
        min_models_per_prompt=3,
        random_state=0,
    ).fit(_rows())

    # capability coefficient should be positive (benchmark predicts completion)
    assert router.combine_model_.coef_[0][0] > 0

    preds = {p.model_id: p.mu for p in router.predict("a new task", model_ids=["strong", "mid", "weak"])}
    assert all(0.0 <= v <= 1.0 for v in preds.values())
    # within one prompt, stronger benchmark -> higher predicted completion
    assert preds["strong"] > preds["mid"] > preds["weak"]


def test_irt_router_unseen_model_uses_its_benchmark(tmp_path):
    router = IRTRouter(
        benchmark_profiles=_profiles(),
        embedding_backend=_StubBackend(),
        embedding_cache_dir=str(tmp_path),
        min_models_per_prompt=3,
        random_state=0,
    ).fit(_rows())

    # a brand-new model, only known via its benchmark profile
    router.add_benchmark_profile(
        ModelBenchmarkProfile("newcomer", benchmarks={"gpqa_diamond": 95.0, "livecodebench": 93.0})
    )
    preds = {p.model_id: p.mu for p in router.predict("another task", model_ids=["newcomer", "weak"])}
    assert preds["newcomer"] > preds["weak"]


def test_prompt_conditioned_irt_outputs_valid_demand_and_predictions(tmp_path):
    router = PromptConditionedIRTRouter(
        benchmark_profiles=_profiles(),
        embedding_backend=_StubBackend(),
        embedding_cache_dir=str(tmp_path),
        min_models_per_prompt=3,
        random_state=0,
    ).fit(_rows())

    demand = router.estimate_demand("implement a parser")
    assert demand.shape == (2,)
    assert np.isclose(float(demand.sum()), 1.0)
    assert np.all(demand >= 0.0)

    preds = {p.model_id: p.mu for p in router.predict("implement a parser", model_ids=["strong", "mid", "weak"])}
    assert all(0.0 <= v <= 1.0 for v in preds.values())
    assert preds["strong"] > preds["weak"]
