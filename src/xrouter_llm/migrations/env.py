from __future__ import annotations

import os
import sys
from pathlib import Path
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from xrouter_llm.store import Base  # noqa: E402

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)

if "db_url" in config.attributes:
    # programmatic call from run_migrations() — URL is already set
    db_url = config.attributes["db_url"]
else:
    # CLI invocation — honour DATABASE_URL, fall back to alembic.ini value
    db_url = os.environ.get(
        "DATABASE_URL",
        config.get_main_option("sqlalchemy.url", default="sqlite:///artifacts/calls.db"),
    )
config.set_main_option("sqlalchemy.url", db_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    programmatic_engine = config.attributes.get("connection")
    if programmatic_engine is not None:
        with programmatic_engine.connect() as connection:
            context.configure(connection=connection, target_metadata=target_metadata)
            with context.begin_transaction():
                context.run_migrations()
        return

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
