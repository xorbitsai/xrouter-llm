"""SQLAlchemy-backed log of routing decisions."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)
from sqlalchemy.pool import StaticPool

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class Base(DeclarativeBase):
    pass


class PromptRecord(Base):
    """Prompt text stored once per distinct prompt, keyed by content hash."""

    __tablename__ = "prompts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    sha256: Mapped[str] = mapped_column(sa.String(64), nullable=False, unique=True)
    text: Mapped[str] = mapped_column(sa.Text, nullable=False)


class CallRecord(Base):
    __tablename__ = "calls"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ts: Mapped[float] = mapped_column(sa.Float, nullable=False)
    config: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    prompt_id: Mapped[int] = mapped_column(
        sa.ForeignKey("prompts.id"), nullable=False
    )
    task: Mapped[str | None] = mapped_column(sa.String(255), nullable=True)
    selected: Mapped[Any] = mapped_column(sa.JSON, nullable=False)
    candidates: Mapped[Any] = mapped_column(sa.JSON, nullable=False)
    expected_quality: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    cost: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    latency: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    feedback: Mapped[Any] = mapped_column(sa.JSON, nullable=True)
    user_id: Mapped[str | None] = mapped_column(sa.String(255), nullable=True)

    prompt_rec: Mapped[PromptRecord] = relationship(lazy="joined")

    __table_args__ = (
        sa.Index("ix_calls_user_id_id", "user_id", "id"),
        sa.Index("ix_calls_prompt_id", "prompt_id"),
    )


_BASELINE_REVISION = "0001"
# Each entry: (column added by that migration, revision). Keep in ascending order.
# To add a new migration: append ("new_column", "000N") here.
_SCHEMA_CHECKPOINTS: list[tuple[str, str]] = [
    ("feedback", "0002"),
    ("user_id", "0003"),
    ("prompt_id", "0004"),
]


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def run_migrations(engine: Engine) -> None:
    """Run Alembic migrations using an already-created engine.

    Passing the engine (rather than a URL) ensures the same connection pool
    is used for both migrations and the runtime session — critical for
    in-memory SQLite, where each engine is an isolated database.
    """
    db_url = str(engine.url)
    _stamp_legacy_db_if_needed(engine)
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.attributes["db_url"] = db_url          # lets env.py skip DATABASE_URL override
    cfg.attributes["connection"] = engine      # env.py reuses this engine
    alembic_command.upgrade(cfg, "head")


def _stamp_legacy_db_if_needed(engine: Engine) -> None:
    """Ensure alembic_version is correct for pre-Alembic databases.

    Handles two cases:
    - `calls` exists, `alembic_version` missing: DB was created before
      migrations were introduced.
    - `calls` exists, `alembic_version` empty: a previous (buggy) stamp
      created the table but failed to write the revision row.
    """
    with engine.begin() as conn:
        inspector = sa.inspect(conn)
        tables = set(inspector.get_table_names())
        if "calls" not in tables:
            return  # fresh DB — let Alembic create everything
        if "alembic_version" not in tables:
            conn.execute(sa.text(
                "CREATE TABLE alembic_version "
                "(version_num VARCHAR(32) NOT NULL, "
                "CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num))"
            ))
        row = conn.execute(sa.text(
            "SELECT version_num FROM alembic_version LIMIT 1"
        )).fetchone()
        if row is None:
            # Stamp to the highest revision whose schema is already present.
            # A legacy DB without the feedback column must be stamped at 0001
            # so that Alembic will still run 0002 to add the column.
            cols = {c["name"] for c in inspector.get_columns("calls")}
            rev = _BASELINE_REVISION
            for col, checkpoint_rev in _SCHEMA_CHECKPOINTS:
                if col in cols:
                    rev = checkpoint_rev
            conn.execute(
                sa.text("INSERT INTO alembic_version (version_num) VALUES (:rev)"),
                {"rev": rev},
            )


def _enforce_sqlite_fks(engine: Engine) -> Engine:
    """SQLite ships with foreign_keys OFF; without it a dangling
    calls.prompt_id would be silently accepted instead of raising."""

    @sa.event.listens_for(engine, "connect")
    def _fk_pragma(dbapi_conn, _connection_record) -> None:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def make_engine(db_url: str) -> Engine:
    url = sa.engine.make_url(db_url)
    if url.drivername.startswith("sqlite"):
        db_path = url.database
        if db_path and db_path not in (":memory:", ""):
            expanded = str(Path(db_path).expanduser())
            Path(expanded).parent.mkdir(parents=True, exist_ok=True)
            # write expanded path back so SQLite opens the real file, not literal "~"
            url = url.set(database=expanded)
            return _enforce_sqlite_fks(
                create_engine(url, connect_args={"check_same_thread": False})
            )
        # in-memory: StaticPool shares one connection so the DB persists across sessions
        return _enforce_sqlite_fks(
            create_engine(
                url, connect_args={"check_same_thread": False}, poolclass=StaticPool
            )
        )
    return create_engine(db_url, pool_pre_ping=True)


def normalize_db_url(path_or_url: str) -> str:
    """Accept a bare file path or a full SQLAlchemy URL; always return a URL."""
    if "://" not in path_or_url:
        return f"sqlite:///{path_or_url}"
    return path_or_url


class CallStore:
    def __init__(self, db_url: str, *, auto_migrate: bool = True) -> None:
        db_url = normalize_db_url(str(db_url))
        engine = make_engine(db_url)
        if auto_migrate:
            run_migrations(engine)
        self._Session = sessionmaker(bind=engine)

    def record(
        self,
        *,
        ts: float,
        config: str,
        prompt: str,
        task: str | None,
        selected: list[str],
        candidates: list[dict[str, Any]],
        expected_quality: float,
        cost: float,
        latency: float,
        user_id: str | None = None,
    ) -> int:
        try:
            return self._record_once(
                ts=ts, config=config, prompt=prompt, task=task,
                selected=selected, candidates=candidates,
                expected_quality=expected_quality, cost=cost,
                latency=latency, user_id=user_id,
            )
        except IntegrityError:
            # A concurrent delete() GC'd our prompt row between lookup and
            # insert. Reachable on SQLite, where FOR UPDATE is a no-op and
            # the FK violation only surfaces at insert; one retry recreates
            # the prompt.
            return self._record_once(
                ts=ts, config=config, prompt=prompt, task=task,
                selected=selected, candidates=candidates,
                expected_quality=expected_quality, cost=cost,
                latency=latency, user_id=user_id,
            )

    def _record_once(
        self,
        *,
        ts: float,
        config: str,
        prompt: str,
        task: str | None,
        selected: list[str],
        candidates: list[dict[str, Any]],
        expected_quality: float,
        cost: float,
        latency: float,
        user_id: str | None,
    ) -> int:
        with self._Session() as session:
            row = CallRecord(
                ts=ts,
                config=config,
                prompt_id=_get_or_create_prompt(session, prompt),
                task=task,
                selected=selected,
                candidates=candidates,
                expected_quality=expected_quality,
                cost=cost,
                latency=latency,
                user_id=user_id,
            )
            session.add(row)
            session.flush()   # INSERT → DB assigns id; no extra SELECT needed
            call_id = row.id
            session.commit()
            return call_id

    def recent(
        self, limit: int = 50, offset: int = 0, *, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 1000))
        offset = max(0, int(offset))
        stmt = sa.select(CallRecord)
        if user_id is not None:
            stmt = stmt.where(CallRecord.user_id == user_id)
        stmt = stmt.order_by(CallRecord.id.desc()).limit(limit).offset(offset)
        with self._Session() as session:
            rows = session.execute(stmt).scalars().all()
        return [_row_to_dict(r) for r in rows]

    def count(self, *, user_id: str | None = None) -> int:
        stmt = sa.select(sa.func.count(CallRecord.id))
        if user_id is not None:
            stmt = stmt.where(CallRecord.user_id == user_id)
        with self._Session() as session:
            return session.execute(stmt).scalar_one()

    def delete(self, call_id: int, *, owner_user_id: str | None = None) -> bool:
        with self._Session() as session:
            stmt = sa.select(CallRecord).where(CallRecord.id == call_id)
            if owner_user_id is not None:
                stmt = stmt.where(CallRecord.user_id == owner_user_id)
            row = session.scalars(stmt).first()
            if row is None:
                return False
            prompt_id = row.prompt_id
            session.delete(row)
            session.flush()
            # GC the prompt text once no call references it (the log holds
            # user prompts; orphaned text must not outlive its last call).
            # The row lock serializes GC per prompt against concurrent
            # record()/delete() until this transaction commits. No-op on
            # SQLite, whose single-writer transactions already serialize
            # the write paths.
            prompt_row = session.get(PromptRecord, prompt_id, with_for_update=True)
            if prompt_row is not None:
                still_referenced = session.scalar(
                    sa.select(CallRecord.id)
                    .where(CallRecord.prompt_id == prompt_id)
                    .limit(1)
                    # locking read: sees the latest committed refs even under
                    # MySQL REPEATABLE READ, where a plain SELECT would read
                    # the transaction snapshot and miss a concurrent delete.
                    .with_for_update()
                )
                if still_referenced is None:
                    session.delete(prompt_row)
            session.commit()
            return True

    def set_feedback(
        self,
        call_id: int,
        feedback: dict[str, Any] | None,
        *,
        owner_user_id: str | None = None,
    ) -> bool:
        with self._Session() as session:
            stmt = sa.select(CallRecord).where(CallRecord.id == call_id)
            if owner_user_id is not None:
                stmt = stmt.where(CallRecord.user_id == owner_user_id)
            row = session.scalars(stmt).first()
            if row is None:
                return False
            row.feedback = feedback
            session.commit()
            return True

    def model_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self._Session() as session:
            rows = session.execute(
                sa.select(CallRecord.selected),
                execution_options={"yield_per": 200},
            ).scalars()
            for selected in rows:
                if isinstance(selected, list):
                    for model_id in selected:
                        counts[model_id] = counts.get(model_id, 0) + 1
        return counts


def _get_or_create_prompt(session: Session, prompt: str) -> int:
    """Return the id of the deduplicated prompt row, inserting if new.

    Insert-then-recover (savepoint) rather than check-then-insert alone, so a
    concurrent writer inserting the same hash cannot fail this transaction.
    """
    sha = prompt_sha256(prompt)
    # FOR UPDATE holds the prompt row until this transaction commits, so a
    # concurrent delete() cannot GC it before our call row is inserted. It
    # also makes the post-conflict re-read see the winner's committed row
    # under MySQL REPEATABLE READ. No-op on SQLite (see record()).
    lookup = (
        sa.select(PromptRecord.id)
        .where(PromptRecord.sha256 == sha)
        .with_for_update()
    )
    prompt_id = session.scalar(lookup)
    if prompt_id is not None:
        return prompt_id
    try:
        with session.begin_nested():
            row = PromptRecord(sha256=sha, text=prompt)
            session.add(row)
            session.flush()
            return row.id
    except IntegrityError:
        return session.scalar(lookup)


def _row_to_dict(r: CallRecord) -> dict[str, Any]:
    return {
        "id": r.id,
        "ts": r.ts,
        "config": r.config,
        "prompt": r.prompt_rec.text,
        "task": r.task,
        "selected": r.selected,
        "candidates": r.candidates,
        "expected_quality": r.expected_quality,
        "cost": r.cost,
        "latency": r.latency,
        "feedback": r.feedback,
        "user_id": r.user_id,
    }
