"""Deduplicate prompt text into a prompts table

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-08
"""
from __future__ import annotations

import hashlib

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str = "0003"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "prompts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("sha256", name="uq_prompts_sha256"),
    )

    op.add_column("calls", sa.Column("prompt_id", sa.Integer(), nullable=True))

    # Backfill by exact content hash in Python rather than SQL DISTINCT /
    # text-equality joins: under case-insensitive collations (MySQL default)
    # those would merge prompts differing only in casing, silently rewriting
    # the stored text of some calls. Hashing matches the runtime dedup rule.
    conn = op.get_bind()
    calls = conn.execute(sa.text("SELECT id, prompt FROM calls")).fetchall()
    sha_to_id: dict[str, int] = {}
    call_updates: list[dict[str, int]] = []
    for call_id, prompt in calls:
        sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        prompt_id = sha_to_id.get(sha)
        if prompt_id is None:
            conn.execute(
                sa.text("INSERT INTO prompts (sha256, text) VALUES (:sha, :text)"),
                {"sha": sha, "text": prompt},
            )
            prompt_id = conn.execute(
                sa.text("SELECT id FROM prompts WHERE sha256 = :sha"), {"sha": sha}
            ).scalar_one()
            sha_to_id[sha] = prompt_id
        call_updates.append({"cid": call_id, "pid": prompt_id})
    if call_updates:
        conn.execute(
            sa.text("UPDATE calls SET prompt_id = :pid WHERE id = :cid"),
            call_updates,
        )

    # batch mode: SQLite cannot ALTER to NOT NULL / add FK in place
    with op.batch_alter_table("calls") as batch:
        batch.alter_column("prompt_id", existing_type=sa.Integer(), nullable=False)
        batch.create_foreign_key(
            "fk_calls_prompt_id_prompts", "prompts", ["prompt_id"], ["id"]
        )
        batch.drop_column("prompt")
    op.create_index("ix_calls_prompt_id", "calls", ["prompt_id"])


def downgrade() -> None:
    op.drop_index("ix_calls_prompt_id", table_name="calls")
    op.add_column("calls", sa.Column("prompt", sa.Text(), nullable=True))
    conn = op.get_bind()
    conn.execute(sa.text(
        "UPDATE calls SET prompt = "
        "(SELECT p.text FROM prompts p WHERE p.id = calls.prompt_id)"
    ))
    with op.batch_alter_table("calls") as batch:
        batch.alter_column("prompt", existing_type=sa.Text(), nullable=False)
        batch.drop_constraint("fk_calls_prompt_id_prompts", type_="foreignkey")
        batch.drop_column("prompt_id")
    op.drop_table("prompts")
