"""Add feedback column to calls table

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-28
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str = "0001"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column("calls", sa.Column("feedback", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("calls", "feedback")
