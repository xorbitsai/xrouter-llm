import argparse

import pytest

from xrouter_llm import EmbeddingEncoder, XinferenceEmbeddingBackend
from xrouter_llm.cli import _maybe_override_embedding_backend


class _OfflineBackend:
    name = "stub:cli"

    def encode(self, texts):
        raise AssertionError("encode must not run during override")


class _RankerPredictor:
    def __init__(self, encoder) -> None:
        self.prompt_encoder_ = encoder


def _args(**overrides) -> argparse.Namespace:
    values = {
        "override_embedding_backend": True,
        "embedding_backend": "xinference",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "xinference_base_url": "http://embeddings.internal:9997/v1",
        "xinference_api_key": None,
        "embedding_batch_size": 64,
        "embedding_timeout": 60.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_override_rebinds_ranker_prompt_encoder() -> None:
    encoder = EmbeddingEncoder(_OfflineBackend())

    _maybe_override_embedding_backend(_RankerPredictor(encoder), _args())

    assert isinstance(encoder.backend, XinferenceEmbeddingBackend)
    assert encoder.backend.base_url == "http://embeddings.internal:9997/v1"


def test_override_is_skipped_when_flag_is_off() -> None:
    original = _OfflineBackend()
    encoder = EmbeddingEncoder(original)

    _maybe_override_embedding_backend(_RankerPredictor(encoder), _args(override_embedding_backend=False))

    assert encoder.backend is original


def test_override_reports_missing_encoder_in_cli_terms() -> None:
    with pytest.raises(TypeError, match="--override-embedding-backend requires a fitted EmbeddingEncoder"):
        _maybe_override_embedding_backend(object(), _args())
