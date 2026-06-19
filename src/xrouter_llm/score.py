from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class ScoreNormalizer:
    def __init__(self) -> None:
        self.min_score_: float = 0.0
        self.max_score_: float = 1.0
        self.pass_through_: bool = True

    def fit(self, scores: Sequence[float]) -> "ScoreNormalizer":
        if not scores:
            raise ValueError("Cannot normalize an empty score list")
        self.min_score_ = float(min(scores))
        self.max_score_ = float(max(scores))
        self.pass_through_ = self.min_score_ >= 0.0 and self.max_score_ <= 1.0
        return self

    def transform(self, score: float) -> float:
        if self.pass_through_:
            return float(np.clip(score, 0.0, 1.0))
        if self.max_score_ == self.min_score_:
            return 0.5
        return float(np.clip((score - self.min_score_) / (self.max_score_ - self.min_score_), 0.0, 1.0))


def safe_std(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))
