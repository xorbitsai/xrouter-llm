import hashlib
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from xrouter_llm import EmbeddingEncoder, SentenceTransformerBackend, TfidfSvdEncoder


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


class _FakeSentenceTransformerModel:
    max_seq_length = None

    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 2))


def _install_sentence_transformer(monkeypatch, constructor) -> None:
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=constructor),
    )


def test_sentence_transformer_backend_single_flights_concurrent_initialization(monkeypatch) -> None:
    constructor_started = threading.Event()
    second_constructor_started = threading.Event()
    allow_constructor = threading.Event()
    calls_lock = threading.Lock()
    calls = 0
    start_barrier = threading.Barrier(9)

    model = _FakeSentenceTransformerModel()

    def construct(model_name, *, device):
        nonlocal calls
        with calls_lock:
            calls += 1
            if calls >= 2:
                second_constructor_started.set()
        constructor_started.set()
        assert allow_constructor.wait(timeout=5)
        return model

    _install_sentence_transformer(monkeypatch, construct)
    backend = SentenceTransformerBackend("test/model", device="cpu", max_seq_length=512)

    def encode():
        start_barrier.wait(timeout=5)
        return backend.encode(["prompt"])

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(encode) for _ in range(8)]
        start_barrier.wait(timeout=5)
        assert constructor_started.wait(timeout=5)
        raced = second_constructor_started.wait(timeout=1)
        allow_constructor.set()
        vectors = [future.result(timeout=5) for future in futures]

    assert not raced
    assert calls == 1
    assert all(vector.shape == (1, 2) for vector in vectors)
    assert model.max_seq_length == 512


def test_sentence_transformer_backend_does_not_publish_partially_configured_model(monkeypatch) -> None:
    configuration_started = threading.Event()
    allow_configuration = threading.Event()
    second_call_returned = threading.Event()

    class Model:
        def __init__(self) -> None:
            self.configured_length = None

        @property
        def max_seq_length(self):
            return self.configured_length

        @max_seq_length.setter
        def max_seq_length(self, value):
            configuration_started.set()
            assert allow_configuration.wait(timeout=5)
            self.configured_length = value

    model = Model()
    _install_sentence_transformer(monkeypatch, lambda *args, **kwargs: model)
    backend = SentenceTransformerBackend(max_seq_length=256)

    def load_and_signal():
        loaded = backend._ensure_model()
        second_call_returned.set()
        return loaded

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(backend._ensure_model)
        assert configuration_started.wait(timeout=5)
        second = executor.submit(load_and_signal)
        returned_before_configuration = second_call_returned.wait(timeout=1)
        allow_configuration.set()
        assert first.result(timeout=5) is model
        assert second.result(timeout=5) is model

    assert not returned_before_configuration
    assert model.configured_length == 256


def test_sentence_transformer_backend_shares_concurrent_failure_and_allows_retry(monkeypatch) -> None:
    constructor_started = threading.Event()
    second_constructor_started = threading.Event()
    allow_failure = threading.Event()
    calls_lock = threading.Lock()
    calls = 0
    start_barrier = threading.Barrier(5)

    class LoadError(BaseException):
        pass

    def fail_load(*args, **kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
            attempt = calls
            if calls >= 2:
                second_constructor_started.set()
        constructor_started.set()
        assert allow_failure.wait(timeout=5)
        raise LoadError(f"load attempt {attempt} failed")

    _install_sentence_transformer(monkeypatch, fail_load)
    backend = SentenceTransformerBackend()

    def load():
        start_barrier.wait(timeout=5)
        return backend._ensure_model()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(load) for _ in range(4)]
        start_barrier.wait(timeout=5)
        assert constructor_started.wait(timeout=5)
        raced = second_constructor_started.wait(timeout=1)
        allow_failure.set()
        errors = []
        for future in futures:
            try:
                future.result(timeout=5)
            except LoadError as error:
                errors.append(error)

    assert not raced
    assert calls == 1
    assert len(errors) == 4
    assert all(str(error) == "load attempt 1 failed" for error in errors)

    recovered_model = _FakeSentenceTransformerModel()
    _install_sentence_transformer(monkeypatch, lambda *args, **kwargs: recovered_model)
    assert backend._ensure_model() is recovered_model


def test_sentence_transformer_backend_unblocks_waiters_when_publish_is_interrupted(
    monkeypatch,
) -> None:
    import xrouter_llm.encoders as encoders

    constructor_started = threading.Event()
    allow_constructor = threading.Event()
    waiter_started = threading.Event()
    created_futures = []

    class PublishInterrupted(BaseException):
        pass

    class InterruptingFuture(Future):
        def __init__(self) -> None:
            super().__init__()
            created_futures.append(self)

        def result(self, timeout=None):
            waiter_started.set()
            return super().result(timeout=timeout)

        def set_result(self, result) -> None:
            raise PublishInterrupted("interrupted before publishing the result")

    model = _FakeSentenceTransformerModel()

    def construct(*args, **kwargs):
        constructor_started.set()
        assert allow_constructor.wait(timeout=5)
        return model

    monkeypatch.setattr(encoders, "Future", InterruptingFuture)
    _install_sentence_transformer(monkeypatch, construct)
    backend = SentenceTransformerBackend()

    follower_error = None
    follower_timed_out = False
    with ThreadPoolExecutor(max_workers=2) as executor:
        leader = executor.submit(backend._ensure_model)
        assert constructor_started.wait(timeout=5)
        follower = executor.submit(backend._ensure_model)
        assert waiter_started.wait(timeout=5)
        allow_constructor.set()

        with pytest.raises(PublishInterrupted) as leader_error:
            leader.result(timeout=5)
        try:
            follower.result(timeout=1)
        except PublishInterrupted as error:
            follower_error = error
        except TimeoutError:
            follower_timed_out = True
        finally:
            if not created_futures[0].done():
                created_futures[0].set_exception(PublishInterrupted("test cleanup"))

    assert not follower_timed_out
    assert str(leader_error.value) == "interrupted before publishing the result"
    assert str(follower_error) == "interrupted before publishing the result"
    assert backend._model is None

    monkeypatch.setattr(encoders, "Future", Future)
    recovered_model = _FakeSentenceTransformerModel()
    _install_sentence_transformer(monkeypatch, lambda *args, **kwargs: recovered_model)
    assert backend._ensure_model() is recovered_model


def test_sentence_transformer_backend_restores_legacy_state(monkeypatch) -> None:
    backend = SentenceTransformerBackend()
    legacy_state = backend.__dict__.copy()
    legacy_state.pop("_model_init_lock")
    legacy_state.pop("_model_init_future")

    revived = SentenceTransformerBackend.__new__(SentenceTransformerBackend)
    revived.__setstate__(legacy_state)

    model = _FakeSentenceTransformerModel()
    _install_sentence_transformer(monkeypatch, lambda *args, **kwargs: model)

    assert revived.encode(["prompt"]).shape == (1, 2)


def test_sentence_transformer_backend_joblib_roundtrip_reloads_model(
    monkeypatch, tmp_path
) -> None:
    backend = SentenceTransformerBackend()
    backend._model = threading.Lock()
    artifact = tmp_path / "backend.joblib"

    joblib.dump(backend, artifact)
    restored = joblib.load(artifact)

    assert restored._model is None

    model = _FakeSentenceTransformerModel()
    _install_sentence_transformer(monkeypatch, lambda *args, **kwargs: model)

    assert restored.encode(["prompt"]).shape == (1, 2)


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


def test_prompt_embedding_view_keeps_mid_prompt_user_task():
    from xrouter_llm.encoders import prompt_embedding_view

    short = "translate this sentence"
    assert prompt_embedding_view(short, head_chars=600, tail_chars=600, focus_chars=600) == short

    # Templated agent prompt: template head, the user's request mid-prompt
    # after a <user> marker, more template filling the tail.
    prompt = (
        "<system> skill selection rules " + "rule " * 800
        + "</system> <user> ## User Task 西班牙和沙特比分多少 context "
        + "output format spec " * 800 + "</user>"
    )
    view = prompt_embedding_view(prompt, head_chars=600, tail_chars=600, focus_chars=600)
    assert len(view) < len(prompt)
    assert view.startswith("<system>")
    assert "西班牙和沙特比分多少" in view
    assert view.endswith("</user>")

    # CJK-heavy text shrinks below the token budget so every slice survives
    # the backend's 512-token truncation.
    cjk = "系统提示" * 1500 + "<user> 用户说：修复这个缺陷 " + "输出格式" * 1500
    cjk_view = prompt_embedding_view(cjk, head_chars=600, tail_chars=600, focus_chars=600)
    ascii_count = len(cjk_view.encode("ascii", "ignore"))
    assert ascii_count / 3.5 + (len(cjk_view) - ascii_count) <= 480
    assert "用户说：修复这个缺陷" in cjk_view


def test_embedding_encoder_view_defaults_off_and_numeric_uses_original_text():
    import numpy as np
    from xrouter_llm.encoders import EmbeddingEncoder

    class _Stub:
        name = "stub:view"

        def __init__(self):
            self.seen = []

        def encode(self, texts):
            self.seen.extend(texts)
            return np.asarray([[float(len(t)), 1.0] for t in texts])

    long_text = "head " * 500 + "<user> real question " + "tail " * 500

    stub_off = _Stub()
    EmbeddingEncoder(stub_off, n_components=2, cache_dir=None).fit([long_text])
    assert stub_off.seen == [long_text]

    stub_on = _Stub()
    encoder = EmbeddingEncoder(
        stub_on,
        n_components=2,
        cache_dir=None,
        include_numeric=True,
        view_head_chars=100,
        view_tail_chars=100,
        view_focus_chars=100,
    )
    encoder.fit([long_text, "short"])
    assert stub_on.seen[0] != long_text
    assert "real question" in stub_on.seen[0]
    assert stub_on.seen[1] == "short"
    # Numeric features are computed on the original text, not the view.
    from xrouter_llm.features import prompt_numeric_features
    expected = prompt_numeric_features([long_text, "short"])
    assert np.allclose(encoder.numeric_scaler_.mean_, expected.mean(axis=0))


def test_embedding_encoder_unpickles_pre_view_state():
    # Downstream predictor artifacts pickle EmbeddingEncoder instances; ones
    # serialized before the view attrs existed must keep working.
    import numpy as np
    from xrouter_llm.encoders import EmbeddingEncoder

    class _Stub:
        name = "stub:oldpickle"

        def encode(self, texts):
            return np.asarray([[float(len(t)), 1.0] for t in texts])

    encoder = EmbeddingEncoder(_Stub(), n_components=2, cache_dir=None)
    encoder.fit(["alpha", "beta gamma"])
    state = encoder.__dict__.copy()
    for key in ("view_head_chars", "view_tail_chars", "view_focus_chars", "view_focus_markers"):
        state.pop(key)

    revived = EmbeddingEncoder.__new__(EmbeddingEncoder)
    revived.__setstate__(state)

    assert revived.view_head_chars == 0
    assert revived._view("x" * 10_000) == "x" * 10_000
    assert revived.transform(["alpha"]).shape[0] == 1


def test_prompt_embedding_view_focus_merged_with_tail_keeps_user_request():
    from xrouter_llm.encoders import prompt_embedding_view

    # CJK-heavy so the token budget forces shrinking; the user request sits
    # right after a late <user> marker whose focus slice merges with the tail.
    text = "系统" * 3000 + "<user> 用户说：修复这个缺陷 " + "尾" * 500
    view = prompt_embedding_view(text, head_chars=600, tail_chars=600, focus_chars=600)
    assert "用户说：修复这个缺陷" in view


def test_prompt_embedding_view_no_slices_returns_text():
    from xrouter_llm.encoders import prompt_embedding_view

    text = "no markers here " * 100
    assert prompt_embedding_view(text, head_chars=0, tail_chars=0, focus_chars=50) == text


def test_prompt_embedding_view_short_chars_but_over_token_budget():
    from xrouter_llm.encoders import _estimated_tokens, prompt_embedding_view

    # Short by character count (< head+focus+tail) but CJK-heavy, so it
    # exceeds the backend token window; the view must still shrink it and
    # keep the user's request instead of letting the tokenizer truncate.
    text = "系" * 900 + "<user> 用户说：修复这个缺陷"
    assert len(text) < 1800
    assert _estimated_tokens(text) > 460
    view = prompt_embedding_view(text, head_chars=600, tail_chars=600, focus_chars=600)
    assert _estimated_tokens(view) <= 480
    assert "用户说：修复这个缺陷" in view

    # No marker and shorter than tail_chars: budget still respected, and the
    # clamped tail range must not wrap around via negative indexing.
    no_marker = "汉" * 500
    view2 = prompt_embedding_view(no_marker, head_chars=600, tail_chars=600, focus_chars=600)
    assert _estimated_tokens(view2) <= 480

    # Genuinely short text still passes through unchanged.
    short = "translate this sentence"
    assert prompt_embedding_view(short, head_chars=600, tail_chars=600, focus_chars=600) == short
