"""Alembic migration sanity checks — no DB connection required."""
from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

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
