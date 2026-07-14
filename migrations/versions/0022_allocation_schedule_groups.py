"""Add governed schedule groups and effective schedule suspension state."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0022_allocation_schedule_groups"
down_revision: str | None = "0021_risk_event_lifecycle"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "schedules",
        sa.Column("desired_status", sa.String(), nullable=True),
        schema="quantlab",
    )
    op.add_column(
        "schedules",
        sa.Column("suspension_reason", sa.Text()),
        schema="quantlab",
    )
    op.execute("UPDATE quantlab.schedules SET desired_status = status")
    op.alter_column("schedules", "desired_status", nullable=False, schema="quantlab")

    op.create_table(
        "allocation_schedule_groups",
        sa.Column("allocation_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("timezone", sa.String(), nullable=False),
        sa.Column("run_time", sa.Time(), nullable=False),
        sa.Column("trading_days_only", sa.Boolean(), nullable=False),
        sa.Column("slippage", sa.Float(), nullable=False),
        sa.Column("misfire_grace_seconds", sa.Integer(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["allocation_id"],
            ["quantlab.strategy_allocations.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("allocation_id"),
        schema="quantlab",
    )
    op.create_index(
        "idx_allocation_schedule_groups_status",
        "allocation_schedule_groups",
        ["status"],
        schema="quantlab",
    )
    op.create_table(
        "allocation_schedule_members",
        sa.Column("allocation_id", sa.String(), nullable=False),
        sa.Column("portfolio_id", sa.String(), nullable=False),
        sa.Column("schedule_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["allocation_id"],
            ["quantlab.allocation_schedule_groups.allocation_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["portfolio_id"],
            ["quantlab.paper_portfolios.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["schedule_id"],
            ["quantlab.schedules.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("allocation_id", "portfolio_id"),
        sa.UniqueConstraint("schedule_id", name="uq_allocation_schedule_member_schedule"),
        schema="quantlab",
    )
    op.create_index(
        "idx_allocation_schedule_members_schedule",
        "allocation_schedule_members",
        ["schedule_id"],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_allocation_schedule_members_schedule",
        table_name="allocation_schedule_members",
        schema="quantlab",
    )
    op.drop_table("allocation_schedule_members", schema="quantlab")
    op.drop_index(
        "idx_allocation_schedule_groups_status",
        table_name="allocation_schedule_groups",
        schema="quantlab",
    )
    op.drop_table("allocation_schedule_groups", schema="quantlab")
    op.drop_column("schedules", "suspension_reason", schema="quantlab")
    op.drop_column("schedules", "desired_status", schema="quantlab")
