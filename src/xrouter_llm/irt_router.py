"""Factorized router: P(complete) = sigmoid(a*capability + b*difficulty + c).

Verified on the 350k sample: controlling for prompt difficulty, a model's
benchmark capability strongly predicts completion (logistic coef ~ +3.9 for
capability, ~ -1.3 for difficulty; within a prompt the passing models are the
higher-benchmark ones 80% of the time). The earlier joint classifier missed
this because the marginal benchmark<->completion correlation is washed out by
easy prompts -- the signal only appears once difficulty is controlled.

Two axes, each learned where it is sound:
- difficulty(prompt): Ridge on a multilingual embedding (Qwen3-Embedding-0.6B),
  trained on each prompt's empirical pass-rate. Multilingual (Chinese transfers
  from English data via cross-lingual embeddings). Chosen over bge-m3 by a
  controlled probe: higher held-out Pearson (0.60 vs 0.55) and it stops pinning
  trivial prompts to max difficulty (e.g. "1+1=?" 3.89 -> 0.36) while correctly
  ranking genuinely hard prompts highest. A frozen generative LM's hidden states
  (Qwen3.5-0.8B, mean-pooled, no fine-tuning) were worse -- not a clean axis.
- capability(model): the mean of the model's published `gpqa_diamond` and
  `livecodebench`, used directly -- so a brand-new model's benchmarks drive its
  ranking. No reliance on the dataset's (confounded) per-model pass-rate. Both
  are full-coverage on the training side (37/37). Evaluated on the *routing
  objective* (completion_rate/cost on the prompt split) with the fuller
  collected benchmark profiles, gpqa+lcb is at least as good as gpqa-only (the
  two are within noise and the sign flips with the prompt sample) with no
  coverage downside, so gpqa+lcb is the production default. Going wider (hle,
  tau2, mmlu_pro, aime, ...) does NOT help: as a flat mean it dilutes, and a
  learned weighting overfits at n=37 profiled models (cross-model AUC 0.71
  in-sample but 0.68 leave-one-model-out). Revisit multi-benchmark capability
  when the profiled-model count is far larger than 37.

A small logistic combines them. Same predict() contract as before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from xrouter_llm.data import coerce_benchmark_rows
from xrouter_llm.encoders import EmbeddingEncoder, SentenceTransformerBackend
from xrouter_llm.profiles import (
    BenchmarkProfileCatalog,
    ModelBenchmarkProfile,
    load_builtin_benchmark_profiles,
)
from xrouter_llm.score import ScoreNormalizer
from xrouter_llm.types import BenchmarkRow, ModelPrediction


def _logit(p: float, floor: float = 1e-3) -> float:
    p = min(max(p, floor), 1.0 - floor)
    return float(np.log(p / (1.0 - p)))


class IRTRouter:
    def __init__(
        self,
        *,
        benchmark_profiles: BenchmarkProfileCatalog | Sequence[ModelBenchmarkProfile] | None = None,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        embedding_backend: object | None = None,
        embedding_cache_dir: str | None = "artifacts/cache/embeddings",
        embedding_max_seq_length: int = 512,
        embedding_head_chars: int = 600,
        embedding_tail_chars: int = 600,
        embedding_focus_chars: int = 600,
        xagent_weight: float = 8.0,
        capability_benchmarks: Sequence[str] = ("gpqa_diamond", "livecodebench"),
        completion_score_threshold: float = 0.75,
        ridge_alpha: float = 1.0,
        min_models_per_prompt: int = 3,
        passrate_floor: float = 0.02,
        sigma: float = 0.12,
        random_state: int | None = None,
    ) -> None:
        self.profile_catalog = _coerce_catalog(benchmark_profiles)
        self.embedding_model = embedding_model
        self.embedding_backend = embedding_backend
        self.embedding_cache_dir = embedding_cache_dir
        self.embedding_max_seq_length = embedding_max_seq_length
        # Long templated agent prompts otherwise contribute only their template
        # head to the difficulty embedding: the tokenizer truncates at
        # max_seq_length while the user's actual request often sits mid-prompt
        # (after a <user> marker) with more template filling the tail.
        self.embedding_head_chars = embedding_head_chars
        self.embedding_tail_chars = embedding_tail_chars
        self.embedding_focus_chars = embedding_focus_chars
        if xagent_weight <= 0:
            raise ValueError("xagent_weight must be positive")
        # xagent-labeled prompts are a ~1% minority next to benchmark corpora;
        # unweighted they cannot move the difficulty regressor.
        self.xagent_weight = xagent_weight
        self.capability_benchmarks = tuple(capability_benchmarks)
        self.completion_score_threshold = completion_score_threshold
        self.ridge_alpha = ridge_alpha
        self.min_models_per_prompt = min_models_per_prompt
        # Clip the pass-rate before -logit so that "no model completed it"
        # prompts (often a grading artifact, not a truly impossible task) do not
        # define the max difficulty and pull every long/OOD prompt toward it.
        self.passrate_floor = passrate_floor
        self.sigma = sigma
        self.random_state = random_state

        self.normalizer_ = ScoreNormalizer()
        self.encoder_: EmbeddingEncoder | None = None
        self.difficulty_model_: Ridge | None = None
        self.difficulty_min_: float = -4.0
        self.difficulty_max_: float = 4.0
        self.combine_model_: LogisticRegression | None = None
        self.capability_means_: dict[str, float] = {}
        self.capability_mean_: float = 0.5
        self.model_ids_: tuple[str, ...] = ()

    # ---- capability from benchmark ------------------------------------------
    def _capability(self, profile: ModelBenchmarkProfile) -> float:
        # Average only the benchmarks the model actually published; do not
        # penalize a model for not reporting one (imputing a missing benchmark
        # with the training mean would wrongly drag down e.g. a strong model
        # that only published gpqa).
        present = [
            float(v)
            for b in self.capability_benchmarks
            if (v := profile.normalized_benchmark(b)) is not None
        ]
        return float(np.mean(present)) if present else self.capability_mean_

    # ---- fit ----------------------------------------------------------------
    def fit(self, rows: Sequence[BenchmarkRow | Mapping[str, object]]) -> "IRTRouter":
        normalized_rows = coerce_benchmark_rows(rows)
        if not normalized_rows:
            raise ValueError("IRTRouter.fit requires at least one row")
        self.normalizer_.fit([row.score for row in normalized_rows])
        self.model_ids_ = tuple(sorted({row.model_id for row in normalized_rows}))

        prompt_text: dict[str, str] = {}
        completed: dict[str, list[float]] = {}
        xagent_prompts: set[str] = set()
        for row in normalized_rows:
            label = 1.0 if self.normalizer_.transform(row.score) >= self.completion_score_threshold else 0.0
            prompt_text.setdefault(row.prompt_id, row.prompt)
            completed.setdefault(row.prompt_id, []).append(label)
            if row.task and row.task.startswith("xagent:"):
                xagent_prompts.add(row.prompt_id)

        prompt_ids = [p for p, labels in completed.items() if len(labels) >= self.min_models_per_prompt]
        if not prompt_ids:
            raise ValueError("No prompt has enough models for a pass-rate estimate")
        b_label = {
            p: -_logit(float(np.mean(completed[p])), self.passrate_floor) for p in prompt_ids
        }
        _bs = np.array(list(b_label.values()))
        self.difficulty_min_ = float(_bs.min())
        self.difficulty_max_ = float(_bs.max())

        # 1) difficulty regressor on multilingual embeddings
        backend = self.embedding_backend or SentenceTransformerBackend(
            self.embedding_model, max_seq_length=self.embedding_max_seq_length
        )
        self.encoder_ = EmbeddingEncoder(
            backend,
            n_components=4096,  # >= embedding dim -> no SVD reduction
            include_numeric=False,
            cache_dir=self.embedding_cache_dir,
            random_state=self.random_state,
            view_head_chars=self.embedding_head_chars,
            view_tail_chars=self.embedding_tail_chars,
            view_focus_chars=self.embedding_focus_chars,
        )
        x_prompt = self.encoder_.fit_transform([prompt_text[p] for p in prompt_ids])
        prompt_weights = np.asarray(
            [self.xagent_weight if p in xagent_prompts else 1.0 for p in prompt_ids]
        )
        self.difficulty_model_ = Ridge(alpha=self.ridge_alpha).fit(
            x_prompt,
            np.array([b_label[p] for p in prompt_ids]),
            sample_weight=prompt_weights,
        )

        # 2) capability per model = benchmark composite (training means for imputation)
        present: dict[str, list[float]] = {b: [] for b in self.capability_benchmarks}
        for model_id in self.model_ids_:
            profile = self.profile_catalog.get(model_id)
            for b in self.capability_benchmarks:
                v = profile.normalized_benchmark(b)
                if v is not None:
                    present[b].append(float(v))
        self.capability_means_ = {b: (float(np.mean(v)) if v else 0.5) for b, v in present.items()}
        all_caps = [m for vals in present.values() for m in vals]
        self.capability_mean_ = float(np.mean(all_caps)) if all_caps else 0.5

        # 3) combine: logistic  pass ~ [capability, difficulty].
        # Only fit on models that have a REAL capability benchmark (so agentic
        # subjects with no profile feed the difficulty axis above but not this
        # one); the capability<->completion link was validated on profiled
        # models only.
        capable = {
            m
            for m in self.model_ids_
            if any(
                self.profile_catalog.get(m).normalized_benchmark(b) is not None
                for b in self.capability_benchmarks
            )
        }
        cap_by_model = {m: self._capability(self.profile_catalog.get(m)) for m in capable}
        feats: list[list[float]] = []
        labels: list[float] = []
        for row in normalized_rows:
            if row.prompt_id not in b_label or row.model_id not in capable:
                continue
            label = 1.0 if self.normalizer_.transform(row.score) >= self.completion_score_threshold else 0.0
            feats.append([cap_by_model[row.model_id], b_label[row.prompt_id]])
            labels.append(label)
        if not feats:
            raise ValueError("No rows with a capability benchmark to fit the combine model")
        self.combine_model_ = LogisticRegression(max_iter=1000).fit(np.asarray(feats), np.asarray(labels))
        return self

    # ---- predict ------------------------------------------------------------
    def predict(
        self,
        prompt: str,
        *,
        model_ids: Sequence[str] | None = None,
        costs: Mapping[str, float] | None = None,
        latencies: Mapping[str, float] | None = None,
        task: str | None = None,
    ) -> list[ModelPrediction]:
        self._check_fitted()
        assert self.combine_model_ is not None
        candidate_ids = tuple(model_ids) if model_ids is not None else self.model_ids_
        if not candidate_ids:
            raise ValueError("No candidate model ids were provided")

        del task
        difficulty = self.estimate_difficulty(prompt)
        caps = np.array([[self._capability(self.profile_catalog.get(m)), difficulty] for m in candidate_ids])
        probs = self.combine_model_.predict_proba(caps)[:, 1]
        output: list[ModelPrediction] = []
        for model_id, mu in zip(candidate_ids, probs):
            output.append(
                ModelPrediction(
                    model_id=model_id,
                    mu=float(np.clip(mu, 0.0, 1.0)),
                    sigma=self.sigma,
                    cost=0.0 if costs is None else float(costs.get(model_id, 0.0)),
                    latency=0.0 if latencies is None else float(latencies.get(model_id, 0.0)),
                )
            )
        return output

    def add_benchmark_profile(self, profile: ModelBenchmarkProfile) -> None:
        self.profile_catalog.add(profile)

    def normalize_score(self, score: float) -> float:
        return self.normalizer_.transform(score)

    def estimate_difficulty(self, prompt: str) -> float:
        self._check_fitted()
        assert self.encoder_ is not None and self.difficulty_model_ is not None
        raw = float(self.difficulty_model_.predict(self.encoder_.transform([prompt]))[0])
        # never extrapolate beyond the difficulty range seen in training
        return float(np.clip(raw, self.difficulty_min_, self.difficulty_max_))

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "IRTRouter":
        predictor = joblib.load(path)
        if not isinstance(predictor, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(predictor).__name__}")
        predictor._check_fitted()
        return predictor

    def _check_fitted(self) -> None:
        if self.encoder_ is None or self.difficulty_model_ is None or self.combine_model_ is None:
            raise RuntimeError("IRTRouter is not fitted")


def _coerce_catalog(
    benchmark_profiles: BenchmarkProfileCatalog | Sequence[ModelBenchmarkProfile] | None,
) -> BenchmarkProfileCatalog:
    if benchmark_profiles is None:
        return load_builtin_benchmark_profiles()
    if isinstance(benchmark_profiles, BenchmarkProfileCatalog):
        return benchmark_profiles
    return BenchmarkProfileCatalog(benchmark_profiles)
