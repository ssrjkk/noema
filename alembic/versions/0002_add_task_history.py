"""Add task_history table for tracking task execution records.

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-30 12:01:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "task_history",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("task_id", sa.String(100), nullable=False),
        sa.Column("tenant_id", sa.String(100), nullable=False),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("quality", sa.String(50), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column("model_used", sa.String(200), nullable=True),
        sa.Column("tokens_used", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index(
        "ix_task_history_tenant_started",
        "task_history",
        ["tenant_id", sa.text("started_at DESC")],
    )
    op.create_index(
        "ix_task_history_task_id",
        "task_history",
        ["task_id"],
    )


def downgrade() -> None:
    op.drop_table("task_history")
