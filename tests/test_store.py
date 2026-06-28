"""CallStore integration tests, parameterized over DB backends via conftest.db_url."""
from __future__ import annotations

import sqlalchemy as sa
from xrouter_llm.store import Base, CallStore, make_engine


def test_record_and_recent(store) -> None:
    call_id = store.record(
        ts=1_000_000.0,
        config="all",
        prompt="hello",
        task=None,
        selected=["cheap"],
        candidates=[{"model_id": "cheap", "mu": 0.8, "sigma": 0.03, "cost": 0.001}],
        expected_quality=0.8,
        cost=0.001,
        latency=0.0,
    )
    assert isinstance(call_id, int)

    rows = store.recent(limit=10)
    assert len(rows) == 1
    assert rows[0]["id"] == call_id
    assert rows[0]["selected"] == ["cheap"]
    assert rows[0]["candidates"][0]["model_id"] == "cheap"
    assert rows[0]["config"] == "all"


def test_recent_order_and_limit(store) -> None:
    for i in range(5):
        store.record(
            ts=float(i),
            config="all",
            prompt=f"prompt {i}",
            task=None,
            selected=["model-a"],
            candidates=[],
            expected_quality=0.8,
            cost=0.001,
            latency=0.0,
        )
    rows = store.recent(limit=3)
    assert len(rows) == 3
    # most-recent first
    assert rows[0]["ts"] > rows[1]["ts"]


def test_json_roundtrip(store) -> None:
    candidates = [
        {"model_id": "a", "mu": 0.9, "sigma": 0.02, "cost": 0.005},
        {"model_id": "b", "mu": 0.6, "sigma": 0.05, "cost": 0.001},
    ]
    store.record(
        ts=1.0, config="custom", prompt="p", task="coding",
        selected=["a"], candidates=candidates,
        expected_quality=0.9, cost=0.005, latency=0.0,
    )
    row = store.recent(1)[0]
    assert row["task"] == "coding"
    assert row["candidates"] == candidates
    assert row["selected"] == ["a"]


def test_model_counts(store) -> None:
    for model in ["cheap", "cheap", "strong"]:
        store.record(
            ts=1.0, config="all", prompt="p", task=None,
            selected=[model], candidates=[],
            expected_quality=0.8, cost=0.0, latency=0.0,
        )
    counts = store.model_counts()
    assert counts["cheap"] == 2
    assert counts["strong"] == 1


def _legacy_db(tmp_path):
    """Return a sqlite:// URL for a pre-0002 DB: calls table without feedback column, no alembic_version.

    Uses raw SQL (not Base.metadata.create_all) so the test actually exercises
    the schema-detection path in _stamp_legacy_db_if_needed.
    """
    url = f"sqlite:///{tmp_path}/legacy.db"
    engine = make_engine(url)
    with engine.begin() as conn:
        conn.execute(sa.text("""
            CREATE TABLE calls (
                id INTEGER NOT NULL,
                ts FLOAT NOT NULL,
                config VARCHAR(255) NOT NULL,
                prompt TEXT NOT NULL,
                task VARCHAR(255),
                selected JSON NOT NULL,
                candidates JSON NOT NULL,
                expected_quality FLOAT,
                cost FLOAT,
                latency FLOAT,
                PRIMARY KEY (id)
            )
        """))
    engine.dispose()
    return url


def _legacy_db_empty_version(tmp_path):
    """Return a sqlite:// URL for a DB with old calls table and empty alembic_version."""
    url = f"sqlite:///{tmp_path}/legacy_ev.db"
    engine = make_engine(url)
    with engine.begin() as conn:
        conn.execute(sa.text("""
            CREATE TABLE calls (
                id INTEGER NOT NULL,
                ts FLOAT NOT NULL,
                config VARCHAR(255) NOT NULL,
                prompt TEXT NOT NULL,
                task VARCHAR(255),
                selected JSON NOT NULL,
                candidates JSON NOT NULL,
                expected_quality FLOAT,
                cost FLOAT,
                latency FLOAT,
                PRIMARY KEY (id)
            )
        """))
        conn.execute(sa.text(
            "CREATE TABLE alembic_version "
            "(version_num VARCHAR(32) NOT NULL, "
            "CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num))"
        ))
    engine.dispose()
    return url


def test_legacy_db_no_alembic_version(tmp_path) -> None:
    """CallStore opens a pre-Alembic DB, runs pending migrations (adds feedback column)."""
    url = _legacy_db(tmp_path)
    store = CallStore(url)
    store.record(
        ts=1.0, config="all", prompt="hello", task=None,
        selected=["m"], candidates=[], expected_quality=0.8, cost=0.0, latency=0.0,
    )
    assert store.count() == 1
    # feedback column must exist after migration 0002 runs
    assert store.recent()[0]["feedback"] is None


def test_legacy_db_empty_alembic_version(tmp_path) -> None:
    """CallStore recovers from a DB where alembic_version exists but is empty."""
    url = _legacy_db_empty_version(tmp_path)
    store = CallStore(url)
    store.record(
        ts=1.0, config="all", prompt="hello", task=None,
        selected=["m"], candidates=[], expected_quality=0.8, cost=0.0, latency=0.0,
    )
    assert store.count() == 1


def test_legacy_db_second_open_is_idempotent(tmp_path) -> None:
    """Opening the same legacy DB twice doesn't fail or duplicate version rows."""
    url = _legacy_db(tmp_path)
    CallStore(url)
    store2 = CallStore(url)
    assert store2.count() == 0  # no records, no crash


def test_in_memory_sqlite_works() -> None:
    """In-memory SQLite CallStore can record and query (shared engine path)."""
    store = CallStore("sqlite:///:memory:")
    store.record(
        ts=1.0, config="all", prompt="hello", task=None,
        selected=["m"], candidates=[], expected_quality=0.8, cost=0.0, latency=0.0,
    )
    assert store.count() == 1
    assert store.recent()[0]["prompt"] == "hello"


def test_auto_migrate_false_skips_migration(tmp_path) -> None:
    """auto_migrate=False does not run migrations (table absent → OperationalError on first use)."""
    import pytest
    from sqlalchemy.exc import OperationalError

    url = f"sqlite:///{tmp_path}/nomigrate.db"
    store = CallStore(url, auto_migrate=False)
    with pytest.raises(OperationalError):
        store.count()
