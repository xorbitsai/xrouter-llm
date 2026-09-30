"""Alembic revision checks and SQLite migration regression tests."""
from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

from xrouter_llm.store import CallStore

_MIGRATIONS_DIR = Path(__file__).parent.parent / "src" / "xrouter_llm" / "migrations"


def _script() -> ScriptDirectory:
    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    return ScriptDirectory.from_config(cfg)


def test_single_head() -> None:
    """Multiple heads = branched migrations = merge revision required."""
    heads = _script().get_heads()
    assert len(heads) == 1, (
        f"Multiple migration heads: {heads}. "
        "Run `alembic merge heads -m 'merge'` to fix."
    )


def test_revisions_are_downgrade_able() -> None:
    """Every revision must declare a downgrade path (not None body)."""
    for rev in _script().walk_revisions():
        assert rev.module is not None
        assert hasattr(rev.module, "downgrade"), (
            f"Revision {rev.revision} is missing a downgrade() function"
        )


@pytest.mark.parametrize("entrypoint", ["store", "environment"])
def test_migrations_preserve_percent_in_database_url(
    tmp_path, monkeypatch, entrypoint
) -> None:
    path = tmp_path / "calls%25.db"
    url = sa.URL.create("sqlite", database=str(path)).render_as_string()
    if entrypoint == "store":
        store = CallStore(url)
    else:
        monkeypatch.setenv("DATABASE_URL", url)
        cfg = Config()
        cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
        command.upgrade(cfg, "head")
        store = CallStore(url, auto_migrate=False)

    store.record(
        ts=1.0, config="all", prompt="hello", task=None,
        selected=["m"], candidates=[], expected_quality=0.8, cost=0.0, latency=0.0,
    )
    assert store.recent()[0]["prompt"] == "hello"
    assert path.exists()
    assert list(tmp_path.iterdir()) == [path]
