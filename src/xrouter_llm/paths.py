"""Resolvers for data bundled inside the installed package.

The trained router, the model-profile registry, and the router configs ship as
package data under ``xrouter_llm/resources/`` so a ``pip install`` is enough to
run ``xrouter-llm serve`` with no extra files. Each resolver prefers an existing
path relative to the current working directory (handy in a source checkout)
and falls back to the packaged copy.
"""

from __future__ import annotations

import os
from importlib.resources import as_file, files

__all__ = ["default_model_path", "default_models_dir", "default_routers_dir"]

_RESOURCES = "resources"


def _bundled(*parts: str) -> str:
    resource = files("xrouter_llm").joinpath(_RESOURCES, *parts)
    # Resources live on the real filesystem here (not inside a zip), so the
    # traversable path resolves directly; materialize it to a plain str.
    with as_file(resource) as path:
        return str(path)


def default_model_path() -> str:
    """Path to the bundled trained router artifact (``.joblib``)."""
    return _bundled("models", "irt_router_350k.joblib")


def default_models_dir() -> str:
    """Path to the bundled per-model benchmark-profile registry."""
    local = os.path.join("config", "models")
    return local if os.path.isdir(local) else _bundled("config", "models")


def default_routers_dir() -> str:
    """Path to the bundled named router configs."""
    local = os.path.join("config", "routers")
    return local if os.path.isdir(local) else _bundled("config", "routers")
