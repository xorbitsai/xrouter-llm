"""Pluggable prompt encoders.

The router only ever needs a dense prompt vector; how that vector is produced
is swappable. ``TfidfSvdEncoder`` reproduces the original bag-of-words pipeline,
``EmbeddingEncoder`` turns prompts into semantic embeddings via a pluggable
``EmbeddingBackend`` (sentence-transformers today, a Xinference-backed remote
service later -- both just need an ``encode(texts) -> matrix`` method).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from xrouter_llm.features import PromptFeaturizer, prompt_numeric_features


@runtime_checkable
class PromptEncoder(Protocol):
    """Encodes prompts into a dense float matrix (n_prompts, n_features)."""

    def fit(self, prompts: Sequence[str]) -> "PromptEncoder": ...

    def transform(self, prompts: Sequence[str]) -> np.ndarray: ...

    def fit_transform(self, prompts: Sequence[str]) -> np.ndarray: ...


class TfidfSvdEncoder:
    """Original encoder: TF-IDF (+ numeric features) reduced with TruncatedSVD."""

    def __init__(
        self,
        *,
        max_tfidf_features: int = 20_000,
        n_components: int = 64,
        random_state: int | None = None,
    ) -> None:
        self.max_tfidf_features = max_tfidf_features
        self.n_components = n_components
        self.random_state = random_state
        self.featurizer_: PromptFeaturizer | None = None
        self.svd_: TruncatedSVD | None = None

    def fit(self, prompts: Sequence[str]) -> "TfidfSvdEncoder":
        prompts = list(prompts)
        self.featurizer_ = PromptFeaturizer(max_tfidf_features=self.max_tfidf_features)
        sparse_features = self.featurizer_.fit_transform(prompts)
        components = min(
            self.n_components,
            max(1, sparse_features.shape[0] - 1),
            max(1, sparse_features.shape[1] - 1),
        )
        self.svd_ = TruncatedSVD(n_components=components, random_state=self.random_state)
        self.svd_.fit(sparse_features)
        return self

    def transform(self, prompts: Sequence[str]) -> np.ndarray:
        if self.featurizer_ is None or self.svd_ is None:
            raise RuntimeError("TfidfSvdEncoder is not fitted")
        return self.svd_.transform(self.featurizer_.transform(list(prompts)))

    def fit_transform(self, prompts: Sequence[str]) -> np.ndarray:
        prompts = list(prompts)
        return self.fit(prompts).transform(prompts)


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Turns raw texts into an embedding matrix. The ``name`` keys the cache."""

    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerBackend:
    """sentence-transformers backend, defaulting to BAAI/bge-base-en-v1.5."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        *,
        device: str | None = None,
        normalize: bool = True,
        batch_size: int = 64,
        max_seq_length: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.batch_size = batch_size
        # Cap sequence length so a single very long prompt cannot blow up the
        # O(n^2) attention buffer (bge-m3 defaults to 8192).
        self.max_seq_length = max_seq_length
        self._model = None

    @property
    def name(self) -> str:
        return f"st:{self.model_name}"

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            if self.max_seq_length is not None:
                self._model.max_seq_length = self.max_seq_length
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure_model()
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=float)

    def __getstate__(self) -> dict:
        # Never pickle the loaded model into the joblib artifact; reload lazily.
        state = self.__dict__.copy()
        state["_model"] = None
        return state


class EmbeddingEncoder:
    """Semantic prompt encoder: backend embeddings -> scale -> SVD -> +numeric.

    Raw embeddings are cached on disk per prompt (one ``.npy`` file keyed by a
    hash of backend name + prompt text), so repeated runs over the same prompt
    set -- e.g. the 38x leave-one-model-out retraining -- encode each prompt
    only once. Per-prompt files keep concurrent workers race-safe.
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        *,
        n_components: int = 64,
        random_state: int | None = None,
        cache_dir: str | Path | None = None,
        include_numeric: bool = True,
    ) -> None:
        self.backend = backend
        self.n_components = n_components
        self.random_state = random_state
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.include_numeric = include_numeric
        self.embedding_scaler_: StandardScaler | None = None
        self.numeric_scaler_: StandardScaler | None = None
        self.svd_: TruncatedSVD | None = None
        self._mem_cache: dict[str, np.ndarray] = {}

    def fit(self, prompts: Sequence[str]) -> "EmbeddingEncoder":
        prompts = list(prompts)
        embeddings = self._encode_cached(prompts)
        self.embedding_scaler_ = StandardScaler().fit(embeddings)
        scaled = self.embedding_scaler_.transform(embeddings)
        components = min(
            self.n_components,
            max(1, scaled.shape[0] - 1),
            scaled.shape[1],
        )
        if components < scaled.shape[1]:
            self.svd_ = TruncatedSVD(n_components=components, random_state=self.random_state)
            self.svd_.fit(scaled)
        else:
            self.svd_ = None
        if self.include_numeric:
            self.numeric_scaler_ = StandardScaler().fit(prompt_numeric_features(prompts))
        return self

    def transform(self, prompts: Sequence[str]) -> np.ndarray:
        if self.embedding_scaler_ is None:
            raise RuntimeError("EmbeddingEncoder is not fitted")
        prompts = list(prompts)
        scaled = self.embedding_scaler_.transform(self._encode_cached(prompts))
        dense = self.svd_.transform(scaled) if self.svd_ is not None else scaled
        if self.include_numeric and self.numeric_scaler_ is not None:
            numeric = self.numeric_scaler_.transform(prompt_numeric_features(prompts))
            dense = np.hstack([dense, numeric])
        return dense

    def fit_transform(self, prompts: Sequence[str]) -> np.ndarray:
        prompts = list(prompts)
        return self.fit(prompts).transform(prompts)

    def _encode_cached(self, prompts: Sequence[str]) -> np.ndarray:
        prompts = list(prompts)
        vectors: list[np.ndarray | None] = [None] * len(prompts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for index, prompt in enumerate(prompts):
            key = self._cache_key(prompt)
            cached = self._mem_cache.get(key)
            if cached is None:
                cached = self._read_disk(key)
            if cached is None:
                missing_indices.append(index)
                missing_texts.append(prompt)
            else:
                self._mem_cache[key] = cached
                vectors[index] = cached

        if missing_texts:
            encoded = self.backend.encode(missing_texts)
            for offset, index in enumerate(missing_indices):
                vector = np.asarray(encoded[offset], dtype=float)
                key = self._cache_key(prompts[index])
                self._mem_cache[key] = vector
                self._write_disk(key, vector)
                vectors[index] = vector

        return np.vstack([np.asarray(vector, dtype=float) for vector in vectors])

    def _cache_key(self, prompt: str) -> str:
        digest = hashlib.sha1(f"{self.backend.name}\x00{prompt}".encode("utf-8"))
        return digest.hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        safe_backend = self.backend.name.replace("/", "_").replace(":", "_")
        return self.cache_dir / safe_backend / f"{key}.npy"

    def _read_disk(self, key: str) -> np.ndarray | None:
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        try:
            return np.load(path)
        except (OSError, ValueError):
            return None

    def _write_disk(self, key: str, vector: np.ndarray) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vector)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_mem_cache"] = {}
        return state


def build_prompt_encoder(
    mode: str,
    *,
    max_tfidf_features: int = 20_000,
    n_components: int = 64,
    random_state: int | None = None,
    embedding_backend: EmbeddingBackend | None = None,
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    embedding_cache_dir: str | Path | None = None,
) -> PromptEncoder:
    if mode == "tfidf_svd":
        return TfidfSvdEncoder(
            max_tfidf_features=max_tfidf_features,
            n_components=n_components,
            random_state=random_state,
        )
    if mode == "embedding":
        backend = embedding_backend or SentenceTransformerBackend(embedding_model)
        return EmbeddingEncoder(
            backend,
            n_components=n_components,
            random_state=random_state,
            cache_dir=embedding_cache_dir,
        )
    raise ValueError(f"Unknown prompt encoder mode {mode!r}")
