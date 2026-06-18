import hashlib

import numpy as np

from xrouter_llm import EmbeddingEncoder, TfidfSvdEncoder


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
