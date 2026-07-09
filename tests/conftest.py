"""Shared fixtures for the test suite."""
from __future__ import annotations

import os

import pytest
import sqlalchemy as sa

from xrouter_llm.store import (
    CallRecord,
    CallStore,
    PromptRecord,
    make_engine,
    normalize_db_url,
)

_SAMPLE_RECORD = dict(
    ts=1_000_000.0,
    config="all",
    prompt="hello",
    task=None,
    selected=["model-a"],
    candidates=[{"model_id": "model-a", "mu": 0.8, "sigma": 0.03, "cost": 0.001}],
    expected_quality=0.8,
    cost=0.001,
    latency=0.0,
)


def _env_url(var: str) -> str | None:
    return os.environ.get(var) or None


@pytest.fixture(
    params=["sqlite", "postgres", "mysql"],
    ids=["sqlite", "postgres", "mysql"],
)
def db_url(tmp_path, request):
    if request.param == "sqlite":
        return f"sqlite:///{tmp_path}/calls.db"
    elif request.param == "postgres":
        url = _env_url("TEST_POSTGRES_URL")
        if not url:
            pytest.skip("TEST_POSTGRES_URL not set")
        return url
    elif request.param == "mysql":
        url = _env_url("TEST_MYSQL_URL")
        if not url:
            pytest.skip("TEST_MYSQL_URL not set")
        return url


@pytest.fixture
def store(db_url):
    s = CallStore(db_url)
    yield s
    if not db_url.startswith("sqlite"):
        engine = make_engine(normalize_db_url(db_url))
        with engine.begin() as conn:
            conn.execute(sa.delete(CallRecord))
            conn.execute(sa.delete(PromptRecord))
