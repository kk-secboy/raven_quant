"""Persist component health and data-freshness history."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0013_system_health_history"
down_revision: str | None = "0012_portfolio_reviews"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "system_health_snapshots",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("components_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("summary_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        schema="quantlab",
    )
    op.create_index(
        "idx_system_health_recorded",
        "system_health_snapshots",
        ["recorded_at"],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_system_health_recorded",
        table_name="system_health_snapshots",
        schema="quantlab",
    )
    op.drop_table("system_health_snapshots", schema="quantlab")
