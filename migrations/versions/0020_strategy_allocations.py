"""Add governed multi-strategy allocations and aggregate risk state."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0020_strategy_allocations"
down_revision: str | None = "0019_data_task_center"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "strategy_allocations",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("allocation_method", sa.String(), nullable=False),
        sa.Column("lookback_days", sa.Integer(), nullable=False),
        sa.Column("target_volatility", sa.Float(), nullable=False),
        sa.Column("max_pairwise_correlation", sa.Float(), nullable=False),
        sa.Column("max_strategy_weight", sa.Float(), nullable=False),
        sa.Column("max_member_drawdown", sa.Float(), nullable=False),
        sa.Column("max_drawdown_reduce", sa.Float(), nullable=False),
        sa.Column("max_drawdown_liquidate", sa.Float(), nullable=False),
        sa.Column("total_capital", sa.Numeric(20, 6), nullable=False),
        sa.Column("cash_reserve", sa.Numeric(20, 6), nullable=False),
        sa.Column("nav", sa.Numeric(20, 6), nullable=False),
        sa.Column("high_water_mark", sa.Numeric(20, 6), nullable=False),
        sa.Column(
            "analysis_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("approved_by", sa.String()),
        sa.Column("approval_reason", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("approved_at", sa.DateTime(timezone=True)),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_strategy_allocations_name"),
        schema="quantlab",
    )
    op.create_index(
        "idx_strategy_allocations_status_updated",
        "strategy_allocations",
        ["status", sa.text("updated_at DESC")],
        schema="quantlab",
    )
    op.create_table(
        "strategy_allocation_members",
        sa.Column("allocation_id", sa.String(), nullable=False),
        sa.Column("strategy_version_id", sa.String(), nullable=False),
        sa.Column("backtest_id", sa.String(), nullable=False),
        sa.Column("portfolio_id", sa.String()),
        sa.Column("target_weight", sa.Float(), nullable=False),
        sa.Column("annualized_volatility", sa.Float(), nullable=False),
        sa.Column("risk_contribution", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["allocation_id"],
            ["quantlab.strategy_allocations.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["strategy_version_id"],
            ["quantlab.strategy_versions.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["backtest_id"],
            ["quantlab.backtest_runs.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["portfolio_id"],
            ["quantlab.paper_portfolios.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("allocation_id", "strategy_version_id"),
        sa.UniqueConstraint("portfolio_id", name="uq_strategy_allocation_member_portfolio"),
        schema="quantlab",
    )
    op.create_index(
        "idx_strategy_allocation_members_portfolio",
        "strategy_allocation_members",
        ["portfolio_id"],
        schema="quantlab",
    )
    op.create_table(
        "strategy_allocation_nav",
        sa.Column("allocation_id", sa.String(), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("nav", sa.Numeric(20, 6), nullable=False),
        sa.Column("daily_return", sa.Float(), nullable=False),
        sa.Column("annualized_volatility", sa.Float(), nullable=False),
        sa.Column("drawdown", sa.Float(), nullable=False),
        sa.Column(
            "member_nav_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "member_weights_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["allocation_id"],
            ["quantlab.strategy_allocations.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("allocation_id", "trade_date"),
        schema="quantlab",
    )
    op.create_index(
        "idx_strategy_allocation_nav_date",
        "strategy_allocation_nav",
        [sa.text("trade_date DESC")],
        schema="quantlab",
    )
    op.create_table(
        "strategy_allocation_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("allocation_id", sa.String(), nullable=False),
        sa.Column("portfolio_id", sa.String()),
        sa.Column("severity", sa.String(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("rule", sa.String(), nullable=False),
        sa.Column("observed", sa.Float()),
        sa.Column("limit_value", sa.Float()),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column(
            "details_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["allocation_id"],
            ["quantlab.strategy_allocations.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["portfolio_id"],
            ["quantlab.paper_portfolios.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        schema="quantlab",
    )
    op.create_index(
        "idx_strategy_allocation_events_created",
        "strategy_allocation_events",
        ["allocation_id", sa.text("created_at DESC")],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_strategy_allocation_events_created",
        table_name="strategy_allocation_events",
        schema="quantlab",
    )
    op.drop_table("strategy_allocation_events", schema="quantlab")
    op.drop_index(
        "idx_strategy_allocation_nav_date",
        table_name="strategy_allocation_nav",
        schema="quantlab",
    )
    op.drop_table("strategy_allocation_nav", schema="quantlab")
    op.drop_index(
        "idx_strategy_allocation_members_portfolio",
        table_name="strategy_allocation_members",
        schema="quantlab",
    )
    op.drop_table("strategy_allocation_members", schema="quantlab")
    op.drop_index(
        "idx_strategy_allocations_status_updated",
        table_name="strategy_allocations",
        schema="quantlab",
    )
    op.drop_table("strategy_allocations", schema="quantlab")
