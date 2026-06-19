"""Experimental prompt-conditioned capability router.

This keeps the production invariant from :mod:`xrouter_llm.irt_router`, but
replaces one fixed model capability scalar with:

    capability(prompt, model) = demand(prompt) dot benchmark_vector(model)

``demand(prompt)`` is learned from prompt embeddings. For every training prompt
we infer which benchmark axes separated completing models from failing models,
then train a small ridge regressor from prompt embedding to that benchmark-weight
simplex. It is not a hard task classifier; the output is a continuous weighting
over benchmark dimensions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from xrouter_llm.data import coerce_benchmark_rows
from xrouter_llm.encoders import EmbeddingEncoder, SentenceTransformerBackend
from xrouter_llm.irt_router import _coerce_catalog, _logit
from xrouter_llm.profiles import BenchmarkProfileCatalog, ModelBenchmarkProfile
from xrouter_llm.score import ScoreNormalizer
from xrouter_llm.types import BenchmarkRow, ModelPrediction


class PromptConditionedIRTRouter:
    """Factorized router with prompt-conditioned benchmark weighting."""

    def __init__(
        self,
        *,
        benchmark_profiles: BenchmarkProfileCatalog | Sequence[ModelBenchmarkProfile] | None = None,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        embedding_backend: object | None = None,
        embedding_cache_dir: str | None = "artifacts/cache/embeddings",
        embedding_max_seq_length: int = 512,
        capability_benchmarks: Sequence[str] = ("gpqa_diamond", "livecodebench"),
        completion_score_threshold: float = 0.75,
        ridge_alpha: float = 1.0,
        demand_ridge_alpha: float = 10.0,
        min_models_per_prompt: int = 3,
        passrate_floor: float = 0.02,
        demand_floor: float = 0.0,
        sigma: float = 0.12,
        random_state: int | None = None,
    ) -> None:
        self.profile_catalog = _coerce_catalog(benchmark_profiles)
        self.embedding_model = embedding_model
        self.embedding_backend = embedding_backend
        self.embedding_cache_dir = embedding_cache_dir
        self.embedding_max_seq_length = embedding_max_seq_length
        self.capability_benchmarks = tuple(capability_benchmarks)
        self.completion_score_threshold = completion_score_threshold
        self.ridge_alpha = ridge_alpha
        self.demand_ridge_alpha = demand_ridge_alpha
        self.min_models_per_prompt = min_models_per_prompt
        self.passrate_floor = passrate_floor
        self.demand_floor = demand_floor
        self.sigma = sigma
        self.random_state = random_state

        self.normalizer_ = ScoreNormalizer()
        self.encoder_: EmbeddingEncoder | None = None
        self.difficulty_model_: Ridge | None = None
        self.demand_model_: Ridge | None = None
        self.combine_model_: LogisticRegression | None = None
        self.difficulty_min_: float = -4.0
        self.difficulty_max_: float = 4.0
        self.capability_means_: np.ndarray | None = None
        self.prior_demand_: np.ndarray | None = None
        self.model_ids_: tuple[str, ...] = ()

    def fit(
        self,
        rows: Sequence[BenchmarkRow | Mapping[str, object]],
    ) -> "PromptConditionedIRTRouter":
        normalized_rows = coerce_benchmark_rows(rows)
        if not normalized_rows:
            raise ValueError("PromptConditionedIRTRouter.fit requires at least one row")
        self.normalizer_.fit([row.score for row in normalized_rows])
        self.model_ids_ = tuple(sorted({row.model_id for row in normalized_rows}))

        prompt_text: dict[str, str] = {}
        completed: dict[str, list[float]] = {}
        labels_by_prompt: dict[str, dict[str, float]] = {}
        for row in normalized_rows:
            label = 1.0 if self.normalizer_.transform(row.score) >= self.completion_score_threshold else 0.0
            prompt_text.setdefault(row.prompt_id, row.prompt)
            completed.setdefault(row.prompt_id, []).append(label)
            labels_by_prompt.setdefault(row.prompt_id, {})[row.model_id] = label

        prompt_ids = [p for p, labels in completed.items() if len(labels) >= self.min_models_per_prompt]
        if not prompt_ids:
            raise ValueError("No prompt has enough models for a pass-rate estimate")
        b_label = {
            p: -_logit(float(np.mean(completed[p])), self.passrate_floor) for p in prompt_ids
        }
        difficulty_values = np.asarray(list(b_label.values()))
        self.difficulty_min_ = float(difficulty_values.min())
        self.difficulty_max_ = float(difficulty_values.max())

        backend = self.embedding_backend or SentenceTransformerBackend(
            self.embedding_model,
            max_seq_length=self.embedding_max_seq_length,
        )
        self.encoder_ = EmbeddingEncoder(
            backend,
            n_components=4096,
            include_numeric=False,
            cache_dir=self.embedding_cache_dir,
            random_state=self.random_state,
        )
        x_prompt = self.encoder_.fit_transform([prompt_text[p] for p in prompt_ids])
        self.difficulty_model_ = Ridge(alpha=self.ridge_alpha).fit(
            x_prompt,
            np.asarray([b_label[p] for p in prompt_ids]),
        )

        cap_by_model = self._fit_capability_vectors()
        capable = set(cap_by_model)
        if not capable:
            raise ValueError("No training model has any configured benchmark capability")

        self.prior_demand_ = self._global_prior_demand(normalized_rows, cap_by_model)
        demand_targets = np.vstack(
            [
                self._prompt_demand_target(labels_by_prompt[p], cap_by_model)
                for p in prompt_ids
            ]
        )
        self.demand_model_ = Ridge(alpha=self.demand_ridge_alpha).fit(x_prompt, demand_targets)

        predicted_demands = self._normalize_weight_matrix(self.demand_model_.predict(x_prompt))
        demand_by_prompt = {p: predicted_demands[i] for i, p in enumerate(prompt_ids)}
        feats: list[list[float]] = []
        labels: list[float] = []
        for row in normalized_rows:
            if row.prompt_id not in demand_by_prompt or row.model_id not in capable:
                continue
            label = 1.0 if self.normalizer_.transform(row.score) >= self.completion_score_threshold else 0.0
            capability = float(np.dot(demand_by_prompt[row.prompt_id], cap_by_model[row.model_id]))
            feats.append([capability, b_label[row.prompt_id]])
            labels.append(label)
        if not feats:
            raise ValueError("No rows with benchmark capability to fit the combine model")
        self.combine_model_ = LogisticRegression(max_iter=1000).fit(
            np.asarray(feats),
            np.asarray(labels),
        )
        return self

    def predict(
        self,
        prompt: str,
        *,
        model_ids: Sequence[str] | None = None,
        costs: Mapping[str, float] | None = None,
        latencies: Mapping[str, float] | None = None,
        task: str | None = None,
    ) -> list[ModelPrediction]:
        del task
        self._check_fitted()
        assert self.combine_model_ is not None

        candidate_ids = tuple(model_ids) if model_ids is not None else self.model_ids_
        if not candidate_ids:
            raise ValueError("No candidate model ids were provided")

        difficulty = self.estimate_difficulty(prompt)
        demand = self.estimate_demand(prompt)
        feats = np.asarray(
            [
                [float(np.dot(demand, self._capability_vector(self.profile_catalog.get(m)))), difficulty]
                for m in candidate_ids
            ]
        )
        probs = self.combine_model_.predict_proba(feats)[:, 1]
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
        return float(np.clip(raw, self.difficulty_min_, self.difficulty_max_))

    def estimate_demand(self, prompt: str) -> np.ndarray:
        self._check_fitted()
        assert self.encoder_ is not None and self.demand_model_ is not None
        return self._normalize_weights(self.demand_model_.predict(self.encoder_.transform([prompt]))[0])

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "PromptConditionedIRTRouter":
        predictor = joblib.load(path)
        if not isinstance(predictor, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(predictor).__name__}")
        predictor._check_fitted()
        return predictor

    def _fit_capability_vectors(self) -> dict[str, np.ndarray]:
        present: list[list[float]] = [[] for _ in self.capability_benchmarks]
        raw_by_model: dict[str, list[float | None]] = {}
        for model_id in self.model_ids_:
            profile = self.profile_catalog.get(model_id)
            values = [profile.normalized_benchmark(b) for b in self.capability_benchmarks]
            raw_by_model[model_id] = values
            for index, value in enumerate(values):
                if value is not None:
                    present[index].append(float(value))

        self.capability_means_ = np.asarray(
            [float(np.mean(values)) if values else 0.5 for values in present],
            dtype=float,
        )
        cap_by_model: dict[str, np.ndarray] = {}
        for model_id, values in raw_by_model.items():
            if any(value is not None for value in values):
                cap_by_model[model_id] = self._impute_capability_values(values)
        return cap_by_model

    def _capability_vector(self, profile: ModelBenchmarkProfile) -> np.ndarray:
        return self._impute_capability_values(
            [profile.normalized_benchmark(b) for b in self.capability_benchmarks]
        )

    def _impute_capability_values(self, values: Sequence[float | None]) -> np.ndarray:
        if self.capability_means_ is None:
            raise RuntimeError("Capability means are not fitted")
        output = np.asarray(
            [
                self.capability_means_[i] if value is None else float(value)
                for i, value in enumerate(values)
            ],
            dtype=float,
        )
        return np.clip(output, 0.0, 1.0)

    def _global_prior_demand(
        self,
        rows: Sequence[BenchmarkRow],
        cap_by_model: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        positive: list[np.ndarray] = []
        negative: list[np.ndarray] = []
        for row in rows:
            capability = cap_by_model.get(row.model_id)
            if capability is None:
                continue
            label = self.normalizer_.transform(row.score) >= self.completion_score_threshold
            (positive if label else negative).append(capability)
        if positive and negative:
            raw = np.mean(positive, axis=0) - np.mean(negative, axis=0)
            return self._normalize_weights(raw)
        return self._normalize_weights(np.ones(len(self.capability_benchmarks)))

    def _prompt_demand_target(
        self,
        labels_by_model: Mapping[str, float],
        cap_by_model: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        positive: list[np.ndarray] = []
        negative: list[np.ndarray] = []
        for model_id, label in labels_by_model.items():
            capability = cap_by_model.get(model_id)
            if capability is None:
                continue
            (positive if label >= 0.5 else negative).append(capability)
        if positive and negative:
            raw = np.mean(positive, axis=0) - np.mean(negative, axis=0)
            weights = self._normalize_weights(raw)
            if np.isfinite(weights).all():
                return weights
        assert self.prior_demand_ is not None
        return self.prior_demand_

    def _normalize_weight_matrix(self, values: np.ndarray) -> np.ndarray:
        return np.vstack([self._normalize_weights(row) for row in np.asarray(values)])

    def _normalize_weights(self, values: Sequence[float]) -> np.ndarray:
        raw = np.asarray(values, dtype=float)
        raw = np.maximum(raw, 0.0)
        if self.prior_demand_ is not None and raw.sum() <= 0:
            raw = np.asarray(self.prior_demand_, dtype=float)
        if raw.sum() <= 0:
            raw = np.ones(len(self.capability_benchmarks), dtype=float)
        raw = raw + self.demand_floor
        return raw / raw.sum()

    def _check_fitted(self) -> None:
        if (
            self.encoder_ is None
            or self.difficulty_model_ is None
            or self.demand_model_ is None
            or self.combine_model_ is None
            or self.capability_means_ is None
            or self.prior_demand_ is None
        ):
            raise RuntimeError("PromptConditionedIRTRouter is not fitted")
