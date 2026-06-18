import hashlib

import numpy as np

from xrouter_llm import (
    BenchmarkRow,
    EmbeddingEncoder,
    ModelAwareRouterPredictor,
    TfidfSvdEncoder,
)


class _StubBackend:
    """Deterministic offline embedding backend (hash -> seeded vector)."""

    name = "stub:test"

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        self.calls: list[int] = []

    def encode(self, texts):
        self.calls.append(len(list(texts)))
        out = []
        for text in texts:
            seed = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)
            out.append(np.random.default_rng(seed).standard_normal(self.dim))
        return np.asarray(out, dtype=float)


def test_tfidf_svd_encoder_matches_dense_shape() -> None:
    prompts = [f"prompt about topic {i}" for i in range(10)]
    encoder = TfidfSvdEncoder(n_components=4, random_state=0)
    dense = encoder.fit_transform(prompts)
    assert dense.shape[0] == 10
    assert dense.shape[1] <= 4


def test_embedding_encoder_caches_per_prompt(tmp_path) -> None:
    backend = _StubBackend(dim=16)
    prompts = [f"unique prompt {i}" for i in range(8)]
    encoder = EmbeddingEncoder(
        backend,
        n_components=4,
        random_state=0,
        cache_dir=tmp_path,
    )
    dense = encoder.fit_transform(prompts)
    assert dense.shape[0] == 8

    # A second encoder instance over the same prompts should hit the disk cache
    # and never call the backend.
    fresh_backend = _StubBackend(dim=16)
    reused = EmbeddingEncoder(fresh_backend, n_components=4, random_state=0, cache_dir=tmp_path)
    reused.fit_transform(prompts)
    assert fresh_backend.calls == []


def test_predictor_runs_with_injected_embedding_backend(tmp_path) -> None:
    rows: list[BenchmarkRow] = []
    for index in range(10):
        prompt = f"question {index} about reasoning"
        rows.append(BenchmarkRow(f"p{index}", prompt, "strong", 0.95))
        rows.append(BenchmarkRow(f"p{index}", prompt, "cheap", 0.9 if index % 2 else 0.3))

    predictor = ModelAwareRouterPredictor(
        ensemble_size=2,
        prompt_encoder="embedding",
        embedding_backend=_StubBackend(dim=24),
        embedding_cache_dir=str(tmp_path),
        completion_score_threshold=0.75,
        random_state=0,
    ).fit(rows)

    predictions = predictor.predict("a brand new reasoning question", model_ids=["strong", "cheap"])
    assert {p.model_id for p in predictions} == {"strong", "cheap"}
    assert all(0.0 <= p.mu <= 1.0 for p in predictions)
