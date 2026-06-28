"""Add user_id column to calls table

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-28
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str = "0002"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column("calls", sa.Column("user_id", sa.String(255), nullable=True))


def downgrade() -> None:
    op.drop_column("calls", "user_id")
