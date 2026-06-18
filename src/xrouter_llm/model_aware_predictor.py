from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier

from xrouter_llm.data import coerce_benchmark_rows
from xrouter_llm.encoders import PromptEncoder, build_prompt_encoder
from xrouter_llm.profiles import (
    BenchmarkProfileCatalog,
    BenchmarkProfileFeaturizer,
    ModelBenchmarkProfile,
    load_builtin_benchmark_profiles,
)
from xrouter_llm.score import ScoreNormalizer, safe_std
from xrouter_llm.types import BenchmarkRow, ModelPrediction


@dataclass
class _CompletionEnsemble:
    classifiers: list[SGDClassifier]
    residual_std: float
    train_size: int
    positive_rate: float


class ModelAwareRouterPredictor:
    """Completion predictor over prompt features plus published model profiles."""

    def __init__(
        self,
        *,
        benchmark_profiles: BenchmarkProfileCatalog | Sequence[ModelBenchmarkProfile] | None = None,
        ensemble_size: int = 16,
        alpha: float = 0.0001,
        min_sigma: float = 0.03,
        max_sigma: float = 0.30,
        unseen_model_sigma_penalty: float = 0.08,
        missing_profile_sigma_penalty: float = 0.06,
        max_tfidf_features: int = 20_000,
        prompt_svd_components: int = 64,
        prompt_encoder: str | PromptEncoder = "tfidf_svd",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        embedding_backend: object | None = None,
        embedding_cache_dir: str | None = "artifacts/cache/embeddings",
        completion_score_threshold: float = 0.75,
        completion_epochs: int = 8,
        balance_classes: bool = True,
        include_model_id_features: bool = True,
        include_task_features: bool = False,
        include_coverage_feature: bool = True,
        batch_size: int = 4096,
        random_state: int | None = None,
    ) -> None:
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be at least 1")
        if min_sigma <= 0 or max_sigma < min_sigma:
            raise ValueError("Require 0 < min_sigma <= max_sigma")
        if prompt_svd_components < 1:
            raise ValueError("prompt_svd_components must be at least 1")
        if not 0.0 <= completion_score_threshold <= 1.0:
            raise ValueError("completion_score_threshold must be in [0, 1]")
        if completion_epochs < 1:
            raise ValueError("completion_epochs must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.profile_catalog = _coerce_catalog(benchmark_profiles)
        self.ensemble_size = ensemble_size
        self.alpha = alpha
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.unseen_model_sigma_penalty = unseen_model_sigma_penalty
        self.missing_profile_sigma_penalty = missing_profile_sigma_penalty
        self.max_tfidf_features = max_tfidf_features
        self.prompt_svd_components = prompt_svd_components
        self.prompt_encoder = prompt_encoder
        self.embedding_model = embedding_model
        self.embedding_backend = embedding_backend
        self.embedding_cache_dir = embedding_cache_dir
        self.completion_score_threshold = completion_score_threshold
        self.completion_epochs = completion_epochs
        self.balance_classes = balance_classes
        self.include_model_id_features = include_model_id_features
        self.include_task_features = include_task_features
        self.include_coverage_feature = include_coverage_feature
        self.batch_size = batch_size
        self.random_state = random_state

        self.prompt_encoder_: PromptEncoder | None = None
        self.task_vocabulary_: tuple[str, ...] = ()
        self.profile_featurizer_: BenchmarkProfileFeaturizer | None = None
        self.normalizer_ = ScoreNormalizer()
        self.ensemble_: _CompletionEnsemble | None = None
        self.model_ids_: tuple[str, ...] = ()
        self.trained_model_ids_: frozenset[str] = frozenset()

    def _encode_prompt(self, prompt: str) -> np.ndarray:
        encoder = getattr(self, "prompt_encoder_", None)
        if encoder is not None:
            return encoder.transform([prompt])[0]
        # Legacy artifact trained before pluggable encoders: TF-IDF + SVD.
        return self.prompt_svd_.transform(self.featurizer_.transform([prompt]))[0]

    def _task_vector(self, task: str | None) -> np.ndarray:
        vector = np.zeros(len(self.task_vocabulary_), dtype=float)
        if task is not None:
            try:
                vector[self.task_vocabulary_.index(task)] = 1.0
            except ValueError:
                pass
        return vector

    def _build_prompt_encoder(self) -> PromptEncoder:
        if not isinstance(self.prompt_encoder, str):
            return self.prompt_encoder
        return build_prompt_encoder(
            self.prompt_encoder,
            max_tfidf_features=self.max_tfidf_features,
            n_components=self.prompt_svd_components,
            random_state=self.random_state,
            embedding_backend=self.embedding_backend,
            embedding_model=self.embedding_model,
            embedding_cache_dir=self.embedding_cache_dir,
        )

    def fit(self, rows: Sequence[BenchmarkRow | Mapping[str, object]]) -> "ModelAwareRouterPredictor":
        normalized_rows = coerce_benchmark_rows(rows)
        if not normalized_rows:
            raise ValueError("ModelAwareRouterPredictor.fit requires at least one row")

        self.normalizer_.fit([row.score for row in normalized_rows])
        self.model_ids_ = tuple(sorted({row.model_id for row in normalized_rows}))
        self.trained_model_ids_ = frozenset(self.model_ids_)

        prompt_keys: list[tuple[str, str]] = []
        prompt_index: dict[tuple[str, str], int] = {}
        row_prompt_indices: list[int] = []
        for row in normalized_rows:
            key = (row.prompt_id, row.prompt)
            if key not in prompt_index:
                prompt_index[key] = len(prompt_keys)
                prompt_keys.append(key)
            row_prompt_indices.append(prompt_index[key])

        prompts = [prompt for _, prompt in prompt_keys]
        self.prompt_encoder_ = self._build_prompt_encoder()
        x_prompt_dense = self.prompt_encoder_.fit_transform(prompts)

        if self.include_task_features:
            self.task_vocabulary_ = tuple(
                sorted({row.task for row in normalized_rows if row.task})
            )
            key_to_task: dict[tuple[str, str], str | None] = {}
            for row in normalized_rows:
                key_to_task.setdefault((row.prompt_id, row.prompt), row.task)
            task_features = np.asarray(
                [self._task_vector(key_to_task.get(key)) for key in prompt_keys],
                dtype=float,
            )
            x_prompt_dense = np.hstack([x_prompt_dense, task_features])

        profile_fit_ids = sorted(set(self.model_ids_) | set(self.profile_catalog.known_model_ids()))
        fit_profiles = [self.profile_catalog.get(model_id) for model_id in profile_fit_ids]
        self.profile_featurizer_ = BenchmarkProfileFeaturizer(
            include_model_id_features=self.include_model_id_features,
            include_coverage_feature=self.include_coverage_feature,
        ).fit(fit_profiles)
        model_profile_features = {
            model_id: self.profile_featurizer_.transform([self.profile_catalog.get(model_id)])[0]
            for model_id in profile_fit_ids
        }

        row_prompt_indices_array = np.asarray(row_prompt_indices, dtype=int)
        row_model_ids = np.asarray([row.model_id for row in normalized_rows], dtype=object)
        y = np.asarray(
            [
                self.normalizer_.transform(row.score) >= self.completion_score_threshold
                for row in normalized_rows
            ],
            dtype=int,
        )
        if len(set(y.tolist())) < 2:
            raise ValueError("Completion training requires both successful and failed examples")

        positive_rate = float(np.mean(y))
        if self.balance_classes:
            sample_weight = np.where(
                y == 1,
                0.5 / max(positive_rate, 1e-6),
                0.5 / max(1.0 - positive_rate, 1e-6),
            )
        else:
            sample_weight = np.ones_like(y, dtype=float)

        rng = np.random.default_rng(self.random_state)
        classifiers: list[SGDClassifier] = []
        for _ in range(self.ensemble_size):
            classifier = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=self.alpha,
                learning_rate="invscaling",
                eta0=0.01,
                max_iter=1,
                tol=None,
                average=True,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            _fit_completion_classifier(
                classifier,
                prompt_indices=row_prompt_indices_array,
                model_ids=row_model_ids,
                labels=y,
                sample_weight=sample_weight,
                x_prompt_dense=x_prompt_dense,
                model_profile_features=model_profile_features,
                rng=rng,
                epochs=self.completion_epochs,
                batch_size=self.batch_size,
            )
            classifiers.append(classifier)

        residual_indices = rng.choice(
            np.arange(y.size),
            size=min(y.size, 20_000),
            replace=False,
        )
        residual_features = _make_completion_features(
            row_prompt_indices_array[residual_indices],
            row_model_ids[residual_indices],
            x_prompt_dense=x_prompt_dense,
            model_profile_features=model_profile_features,
        )
        ensemble_predictions = np.column_stack(
            [classifier.predict_proba(residual_features)[:, 1] for classifier in classifiers]
        ).mean(axis=1)
        residual_std = float(
            np.clip(
                safe_std(y[residual_indices] - ensemble_predictions),
                self.min_sigma,
                self.max_sigma,
            )
        )
        self.ensemble_ = _CompletionEnsemble(
            classifiers=classifiers,
            residual_std=residual_std,
            train_size=len(normalized_rows),
            positive_rate=positive_rate,
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
        self._check_fitted()
        assert self.profile_featurizer_ is not None
        assert self.ensemble_ is not None

        candidate_ids = tuple(model_ids) if model_ids is not None else self.model_ids_
        if not candidate_ids:
            raise ValueError("No candidate model ids were provided")

        prompt_dense = self._encode_prompt(prompt)
        if getattr(self, "include_task_features", False):
            prompt_dense = np.concatenate([prompt_dense, self._task_vector(task)])
        profiles = [self.profile_catalog.get(model_id) for model_id in candidate_ids]
        model_profile_features = {
            model_id: self.profile_featurizer_.transform([profile])[0]
            for model_id, profile in zip(candidate_ids, profiles)
        }
        features = _make_completion_features(
            np.zeros(len(candidate_ids), dtype=int),
            np.asarray(candidate_ids, dtype=object),
            x_prompt_dense=prompt_dense[None, :],
            model_profile_features=model_profile_features,
        )
        probability_matrix = np.column_stack(
            [classifier.predict_proba(features)[:, 1] for classifier in self.ensemble_.classifiers]
        )
        mean_probabilities = probability_matrix.mean(axis=1)
        std_probabilities = probability_matrix.std(axis=1)

        output: list[ModelPrediction] = []
        for row_index, model_id in enumerate(candidate_ids):
            profile = profiles[row_index]
            mu = float(np.clip(mean_probabilities[row_index], 0.0, 1.0))
            sigma = std_probabilities[row_index] + self.ensemble_.residual_std
            if model_id not in self.trained_model_ids_:
                sigma += self.unseen_model_sigma_penalty
            if not profile.benchmarks:
                sigma += self.missing_profile_sigma_penalty
            sigma = float(np.clip(sigma, self.min_sigma, self.max_sigma))
            output.append(
                ModelPrediction(
                    model_id=model_id,
                    mu=mu,
                    sigma=sigma,
                    cost=0.0 if costs is None else float(costs.get(model_id, 0.0)),
                    latency=0.0 if latencies is None else float(latencies.get(model_id, 0.0)),
                )
            )

        return output

    def add_benchmark_profile(self, profile: ModelBenchmarkProfile) -> None:
        self.profile_catalog.add(profile)

    def normalize_score(self, score: float) -> float:
        return self.normalizer_.transform(score)

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "ModelAwareRouterPredictor":
        predictor = joblib.load(path)
        if not isinstance(predictor, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(predictor).__name__}")
        predictor._check_fitted()
        return predictor

    def _check_fitted(self) -> None:
        has_encoder = getattr(self, "prompt_encoder_", None) is not None
        has_legacy_encoder = (
            getattr(self, "featurizer_", None) is not None
            and getattr(self, "prompt_svd_", None) is not None
        )
        if (
            (not has_encoder and not has_legacy_encoder)
            or self.profile_featurizer_ is None
            or self.ensemble_ is None
        ):
            raise RuntimeError("ModelAwareRouterPredictor is not fitted")


def _coerce_catalog(
    benchmark_profiles: BenchmarkProfileCatalog | Sequence[ModelBenchmarkProfile] | None,
) -> BenchmarkProfileCatalog:
    if benchmark_profiles is None:
        return load_builtin_benchmark_profiles()
    if isinstance(benchmark_profiles, BenchmarkProfileCatalog):
        return benchmark_profiles
    return BenchmarkProfileCatalog(benchmark_profiles)


def _fit_completion_classifier(
    classifier: SGDClassifier,
    *,
    prompt_indices: np.ndarray,
    model_ids: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    x_prompt_dense: np.ndarray,
    model_profile_features: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    epochs: int,
    batch_size: int,
) -> None:
    row_count = labels.size
    first_batch = True
    for _ in range(epochs):
        indices = rng.permutation(row_count)
        for start in range(0, row_count, batch_size):
            batch_indices = indices[start : start + batch_size]
            features = _make_completion_features(
                prompt_indices[batch_indices],
                model_ids[batch_indices],
                x_prompt_dense=x_prompt_dense,
                model_profile_features=model_profile_features,
            )
            if first_batch:
                classifier.partial_fit(
                    features,
                    labels[batch_indices],
                    classes=np.asarray([0, 1], dtype=int),
                    sample_weight=sample_weight[batch_indices],
                )
                first_batch = False
            else:
                classifier.partial_fit(
                    features,
                    labels[batch_indices],
                    sample_weight=sample_weight[batch_indices],
                )


def _make_completion_features(
    prompt_indices: np.ndarray,
    model_ids: np.ndarray,
    *,
    x_prompt_dense: np.ndarray,
    model_profile_features: Mapping[str, np.ndarray],
) -> np.ndarray:
    prompt_features = x_prompt_dense[prompt_indices]
    profile_features = np.vstack([model_profile_features[str(model_id)] for model_id in model_ids])
    interactions = (prompt_features[:, :, None] * profile_features[:, None, :]).reshape(len(prompt_indices), -1)
    return np.hstack([prompt_features, profile_features, interactions])
