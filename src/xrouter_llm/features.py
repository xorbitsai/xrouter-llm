from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


_WORD_PATTERN = re.compile(r"\w+")


class PromptFeaturizer:
    def __init__(
        self,
        *,
        max_tfidf_features: int = 20_000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=ngram_range,
            min_df=min_df,
            lowercase=True,
            strip_accents="unicode",
        )
        self.scaler = StandardScaler()

    def fit(self, prompts: Sequence[str]) -> "PromptFeaturizer":
        prompts = list(prompts)
        if not prompts:
            raise ValueError("Cannot fit PromptFeaturizer with no prompts")

        self.vectorizer.fit(prompts)
        self.scaler.fit(prompt_numeric_features(prompts))
        return self

    def transform(self, prompts: Sequence[str]) -> sparse.csr_matrix:
        prompts = list(prompts)
        text_features = self.vectorizer.transform(prompts)
        numeric_features = sparse.csr_matrix(
            self.scaler.transform(prompt_numeric_features(prompts))
        )
        return sparse.hstack([text_features, numeric_features], format="csr")

    def fit_transform(self, prompts: Sequence[str]) -> sparse.csr_matrix:
        return self.fit(prompts).transform(prompts)


def prompt_numeric_features(prompts: Sequence[str]) -> np.ndarray:
    rows = [_features_for_prompt(prompt) for prompt in prompts]
    return np.asarray(rows, dtype=float)


def _features_for_prompt(prompt: str) -> list[float]:
    words = _WORD_PATTERN.findall(prompt)
    word_count = len(words)
    char_count = len(prompt)
    line_count = prompt.count("\n") + 1 if prompt else 0
    digit_count = sum(char.isdigit() for char in prompt)
    upper_count = sum(char.isupper() for char in prompt)
    code_fence_count = prompt.count("```")
    question_count = prompt.count("?") + prompt.count("？")
    avg_word_len = sum(len(word) for word in words) / max(1, word_count)

    return [
        np.log1p(char_count),
        np.log1p(word_count),
        np.log1p(line_count),
        np.log1p(code_fence_count),
        np.log1p(question_count),
        digit_count / max(1, char_count),
        upper_count / max(1, char_count),
        avg_word_len,
    ]
