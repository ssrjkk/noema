"""Add webhook_registrations table for outgoing webhook configuration.

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-30 12:02:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "webhook_registrations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(100), nullable=False),
        sa.Column("url", sa.String(2048), nullable=False),
        sa.Column("secret", sa.String(512), nullable=True, comment="Encrypted webhook secret"),
        sa.Column("events", JSONB(), nullable=False, server_default="'[]'::jsonb"),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("timeout", sa.Integer(), nullable=False, server_default="10"),
        sa.Column("active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "ix_webhook_registrations_tenant",
        "webhook_registrations",
        ["tenant_id"],
    )


def downgrade() -> None:
    op.drop_table("webhook_registrations")
