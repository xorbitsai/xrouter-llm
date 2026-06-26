"""SQLAlchemy-backed log of routing decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import sqlalchemy as sa
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class Base(DeclarativeBase):
    pass


class CallRecord(Base):
    __tablename__ = "calls"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ts: Mapped[float] = mapped_column(sa.Float, nullable=False)
    config: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    prompt: Mapped[str] = mapped_column(sa.Text, nullable=False)
    task: Mapped[str | None] = mapped_column(sa.String(255), nullable=True)
    selected: Mapped[Any] = mapped_column(sa.JSON, nullable=False)
    candidates: Mapped[Any] = mapped_column(sa.JSON, nullable=False)
    expected_quality: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    cost: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    latency: Mapped[float | None] = mapped_column(sa.Float, nullable=True)


def run_migrations(db_url: str) -> None:
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.attributes["db_url"] = db_url  # lets env.py skip DATABASE_URL override
    alembic_command.upgrade(cfg, "head")


def make_engine(db_url: str) -> Engine:
    if db_url.startswith("sqlite"):
        # Ensure the parent directory exists for file-backed SQLite
        db_path = db_url.split("sqlite:///", 1)[-1]
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        return create_engine(db_url, connect_args={"check_same_thread": False})
    return create_engine(db_url, pool_pre_ping=True)


def normalize_db_url(path_or_url: str) -> str:
    """Accept a bare file path or a full SQLAlchemy URL; always return a URL."""
    if "://" not in path_or_url:
        return f"sqlite:///{path_or_url}"
    return path_or_url


class CallStore:
    def __init__(self, db_url: str) -> None:
        db_url = normalize_db_url(str(db_url))
        run_migrations(db_url)
        self._Session = sessionmaker(bind=make_engine(db_url))

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
    ) -> int:
        with self._Session() as session:
            row = CallRecord(
                ts=ts,
                config=config,
                prompt=prompt,
                task=task,
                selected=selected,
                candidates=candidates,
                expected_quality=expected_quality,
                cost=cost,
                latency=latency,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id

    def recent(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 1000))
        offset = max(0, int(offset))
        with self._Session() as session:
            rows = (
                session.execute(
                    sa.select(CallRecord)
                    .order_by(CallRecord.id.desc())
                    .limit(limit)
                    .offset(offset)
                )
                .scalars()
                .all()
            )
        return [_row_to_dict(r) for r in rows]

    def count(self) -> int:
        with self._Session() as session:
            return session.execute(sa.select(sa.func.count(CallRecord.id))).scalar_one()

    def delete(self, call_id: int) -> bool:
        with self._Session() as session:
            row = session.get(CallRecord, call_id)
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    def model_counts(self) -> dict[str, int]:
        with self._Session() as session:
            all_selected = session.execute(sa.select(CallRecord.selected)).scalars().all()
        counts: dict[str, int] = {}
        for selected in all_selected:
            for model_id in selected:
                counts[model_id] = counts.get(model_id, 0) + 1
        return counts


def _row_to_dict(r: CallRecord) -> dict[str, Any]:
    return {
        "id": r.id,
        "ts": r.ts,
        "config": r.config,
        "prompt": r.prompt,
        "task": r.task,
        "selected": r.selected,
        "candidates": r.candidates,
        "expected_quality": r.expected_quality,
        "cost": r.cost,
        "latency": r.latency,
    }
