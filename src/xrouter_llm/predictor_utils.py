from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any


def predict_with_optional_task(
    predictor: object,
    prompt: str,
    *,
    model_ids: Sequence[str] | None = None,
    costs: dict[str, float] | None = None,
    latencies: dict[str, float] | None = None,
    task: str | None = None,
) -> object:
    kwargs: dict[str, Any] = {
        "model_ids": model_ids,
        "costs": costs,
        "latencies": latencies,
    }
    if _supports_task_parameter(predictor):
        kwargs["task"] = task
    return predictor.predict(prompt, **kwargs)


def _supports_task_parameter(predictor: object) -> bool:
    try:
        parameters = inspect.signature(predictor.predict).parameters
    except (TypeError, ValueError):
        return False
    if "task" in parameters:
        return True
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
