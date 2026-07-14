"""Add local users, revocable sessions, lockout state, and operation audit."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008_auth_rbac_audit"
down_revision: str | None = "0007_scheduler_alerts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("username", sa.String(), nullable=False, unique=True),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("failed_login_attempts", sa.Integer(), nullable=False),
        sa.Column("locked_until", sa.DateTime(timezone=True)),
        sa.Column("last_login_at", sa.DateTime(timezone=True)),
        sa.Column("password_changed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        schema="quantlab",
    )
    op.create_index(
        "idx_users_role_active",
        "users",
        ["role", "active"],
        schema="quantlab",
    )
    op.create_table(
        "auth_sessions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(),
            sa.ForeignKey("quantlab.users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token_hash", sa.String(), nullable=False, unique=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
        sa.Column("ip_hash", sa.String()),
        sa.Column("user_agent", sa.String()),
        schema="quantlab",
    )
    op.create_index(
        "idx_auth_sessions_user_expires",
        "auth_sessions",
        ["user_id", "expires_at"],
        schema="quantlab",
    )
    op.create_table(
        "audit_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("quantlab.users.id", ondelete="SET NULL")),
        sa.Column("username", sa.String(), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("method", sa.String(), nullable=False),
        sa.Column("path", sa.String(), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=False),
        sa.Column("ip_hash", sa.String()),
        sa.Column("user_agent", sa.String()),
        sa.Column("details_json", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        schema="quantlab",
    )
    op.create_index(
        "idx_audit_events_created",
        "audit_events",
        [sa.text("created_at DESC")],
        schema="quantlab",
    )
    op.create_index(
        "idx_audit_events_user_created",
        "audit_events",
        ["user_id", sa.text("created_at DESC")],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index("idx_audit_events_user_created", table_name="audit_events", schema="quantlab")
    op.drop_index("idx_audit_events_created", table_name="audit_events", schema="quantlab")
    op.drop_table("audit_events", schema="quantlab")
    op.drop_index(
        "idx_auth_sessions_user_expires",
        table_name="auth_sessions",
        schema="quantlab",
    )
    op.drop_table("auth_sessions", schema="quantlab")
    op.drop_index("idx_users_role_active", table_name="users", schema="quantlab")
    op.drop_table("users", schema="quantlab")
