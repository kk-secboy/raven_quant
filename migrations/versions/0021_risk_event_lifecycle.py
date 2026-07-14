"""Add auditable acknowledgement and resolution to trading risk events."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0021_risk_event_lifecycle"
down_revision: str | None = "0020_strategy_allocations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _add_lifecycle_columns(table_name: str) -> None:
    op.add_column(table_name, sa.Column("acknowledged_by", sa.String()), schema="quantlab")
    op.add_column(
        table_name,
        sa.Column("acknowledged_at", sa.DateTime(timezone=True)),
        schema="quantlab",
    )
    op.add_column(table_name, sa.Column("resolved_by", sa.String()), schema="quantlab")
    op.add_column(
        table_name,
        sa.Column("resolved_at", sa.DateTime(timezone=True)),
        schema="quantlab",
    )
    op.add_column(table_name, sa.Column("resolution_reason", sa.Text()), schema="quantlab")


def upgrade() -> None:
    # risk_events already had an acknowledged_at timestamp in 0006.
    op.add_column("risk_events", sa.Column("acknowledged_by", sa.String()), schema="quantlab")
    op.add_column("risk_events", sa.Column("resolved_by", sa.String()), schema="quantlab")
    op.add_column(
        "risk_events",
        sa.Column("resolved_at", sa.DateTime(timezone=True)),
        schema="quantlab",
    )
    op.add_column("risk_events", sa.Column("resolution_reason", sa.Text()), schema="quantlab")
    _add_lifecycle_columns("strategy_allocation_events")


def downgrade() -> None:
    for column in (
        "resolution_reason",
        "resolved_at",
        "resolved_by",
        "acknowledged_at",
        "acknowledged_by",
    ):
        op.drop_column("strategy_allocation_events", column, schema="quantlab")
    for column in ("resolution_reason", "resolved_at", "resolved_by", "acknowledged_by"):
        op.drop_column("risk_events", column, schema="quantlab")
