"""Create initial tables: memory stores, audit_log, tenant_quotas, feature_flags.

Revision ID: 0001
Revises: None
Create Date: 2026-07-30 12:00:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # --- episodic_memory ---
    op.create_table(
        "episodic_memory",
        sa.Column("id", sa.String(12), primary_key=True),
        sa.Column("timestamp", sa.Float(), nullable=True, index=True),
        sa.Column("task_description", sa.Text(), nullable=True, server_default=""),
        sa.Column("solution_summary", sa.Text(), nullable=True, server_default=""),
        sa.Column("tech_stack", sa.String(500), nullable=True, server_default=""),
        sa.Column("outcome", sa.String(20), nullable=True, server_default="", index=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True, server_default="0.0"),
        sa.Column("error_message", sa.Text(), nullable=True, server_default=""),
        sa.Column("tags", JSONB(), nullable=True, server_default="'[]'::jsonb"),
        sa.Column("context", JSONB(), nullable=True, server_default="'{}'::jsonb"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_episodic_outcome_ts",
        "episodic_memory",
        ["outcome", "timestamp"],
    )

    # --- semantic_memory ---
    op.create_table(
        "semantic_memory",
        sa.Column("id", sa.String(12), primary_key=True),
        sa.Column("topic", sa.String(200), nullable=True, server_default="", index=True),
        sa.Column("fact", sa.Text(), nullable=True, server_default=""),
        sa.Column("confidence", sa.Float(), nullable=True, server_default="0.0"),
        sa.Column("source", sa.String(500), nullable=True, server_default=""),
        sa.Column("use_count", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("last_used", sa.Float(), nullable=True),
        sa.Column("tags", JSONB(), nullable=True, server_default="'[]'::jsonb"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_semantic_topic_confidence",
        "semantic_memory",
        ["topic", "confidence"],
    )

    # --- procedural_memory ---
    op.create_table(
        "procedural_memory",
        sa.Column("id", sa.String(12), primary_key=True),
        sa.Column("procedure_name", sa.String(200), nullable=True, server_default="", index=True),
        sa.Column("steps", JSONB(), nullable=True, server_default="'[]'::jsonb"),
        sa.Column("success_rate", sa.Float(), nullable=True, server_default="1.0"),
        sa.Column("times_applied", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("times_succeeded", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("avg_duration", sa.Float(), nullable=True, server_default="0.0"),
        sa.Column("prerequisites", JSONB(), nullable=True, server_default="'[]'::jsonb"),
        sa.Column("tags", JSONB(), nullable=True, server_default="'[]'::jsonb"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )

    # --- audit_log ---
    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("tenant_id", sa.String(100), nullable=False),
        sa.Column("user_id", sa.String(100), nullable=False),
        sa.Column("task_id", sa.String(100), nullable=True),
        sa.Column("details", JSONB(), nullable=False, server_default="'{}'::jsonb"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "idx_audit_tenant_time",
        "audit_log",
        ["tenant_id", sa.text("timestamp DESC")],
    )
    op.create_index("idx_audit_task", "audit_log", ["task_id"])
    op.create_index("idx_audit_event_type", "audit_log", ["event_type"])

    # --- tenant_quotas ---
    op.create_table(
        "tenant_quotas",
        sa.Column("tenant_id", sa.String(100), primary_key=True),
        sa.Column("monthly_budget", sa.Float(), nullable=False, server_default="100.0"),
        sa.Column("hourly_limit", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("concurrent_limit", sa.Integer(), nullable=False, server_default="5"),
        sa.Column("tasks_run", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tasks_this_hour", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("reset_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    # --- feature_flags ---
    op.create_table(
        "feature_flags",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("flag_key", sa.String(100), nullable=False),
        sa.Column("tenant_id", sa.String(100), nullable=True),
        sa.Column("value", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "ix_feature_flags_key_tenant",
        "feature_flags",
        ["flag_key", "tenant_id"],
        unique=True,
        postgresql_where=sa.text("tenant_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_table("feature_flags")
    op.drop_table("tenant_quotas")
    op.drop_table("audit_log")
    op.drop_table("procedural_memory")
    op.drop_table("semantic_memory")
    op.drop_table("episodic_memory")
