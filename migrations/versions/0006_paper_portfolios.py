"""Add paper portfolios, order ledger, NAV, and risk events."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006_paper_portfolios"
down_revision: str | None = "0005_strategy_backtests"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "paper_portfolios",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False, unique=True),
        sa.Column(
            "strategy_version_id",
            sa.String(),
            sa.ForeignKey("quantlab.strategy_versions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("base_currency", sa.String(), nullable=False),
        sa.Column("initial_cash", sa.Numeric(20, 6), nullable=False),
        sa.Column("cash", sa.Numeric(20, 6), nullable=False),
        sa.Column("nav", sa.Numeric(20, 6), nullable=False),
        sa.Column("high_water_mark", sa.Numeric(20, 6), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        schema="quantlab",
    )
    op.create_index(
        "idx_paper_portfolios_status_updated",
        "paper_portfolios",
        ["status", sa.text("updated_at DESC")],
        schema="quantlab",
    )
    op.create_table(
        "portfolio_batches",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "portfolio_id",
            sa.String(),
            sa.ForeignKey("quantlab.paper_portfolios.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("job_id", sa.String(), sa.ForeignKey("quantlab.jobs.id", ondelete="SET NULL")),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("trade_date", sa.Date()),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("idempotency_key", sa.String(), nullable=False, unique=True),
        sa.Column("artifact_path", sa.Text()),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("portfolio_id", "as_of_date", name="uq_portfolio_batches_as_of"),
        schema="quantlab",
    )
    op.create_index(
        "idx_portfolio_batches_status_created",
        "portfolio_batches",
        ["status", sa.text("created_at DESC")],
        schema="quantlab",
    )
    op.create_table(
        "paper_orders",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "batch_id",
            sa.String(),
            sa.ForeignKey("quantlab.portfolio_batches.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "portfolio_id",
            sa.String(),
            sa.ForeignKey("quantlab.paper_portfolios.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("instrument", sa.String(), nullable=False),
        sa.Column("side", sa.String(), nullable=False),
        sa.Column("order_type", sa.String(), nullable=False),
        sa.Column("target_weight", sa.Float(), nullable=False),
        sa.Column("requested_quantity", sa.Numeric(20, 6), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("reason", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("batch_id", "instrument", name="uq_paper_orders_batch_instrument"),
        schema="quantlab",
    )
    op.create_index(
        "idx_paper_orders_portfolio_created",
        "paper_orders",
        ["portfolio_id", "created_at"],
        schema="quantlab",
    )
    op.create_table(
        "paper_fills",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "order_id",
            sa.String(),
            sa.ForeignKey("quantlab.paper_orders.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("fill_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("quantity", sa.Numeric(20, 6), nullable=False),
        sa.Column("price", sa.Numeric(20, 6), nullable=False),
        sa.Column("gross_value", sa.Numeric(20, 6), nullable=False),
        sa.Column("fee", sa.Numeric(20, 6), nullable=False),
        sa.Column("slippage", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        schema="quantlab",
    )
    op.create_table(
        "paper_positions",
        sa.Column(
            "portfolio_id",
            sa.String(),
            sa.ForeignKey("quantlab.paper_portfolios.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("instrument", sa.String(), primary_key=True),
        sa.Column("quantity", sa.Numeric(20, 6), nullable=False),
        sa.Column("avg_cost", sa.Numeric(20, 6), nullable=False),
        sa.Column("market_price", sa.Numeric(20, 6), nullable=False),
        sa.Column("market_value", sa.Numeric(20, 6), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("realized_pnl", sa.Numeric(20, 6), nullable=False),
        sa.Column("unrealized_pnl", sa.Numeric(20, 6), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        schema="quantlab",
    )
    op.create_table(
        "portfolio_nav",
        sa.Column(
            "portfolio_id",
            sa.String(),
            sa.ForeignKey("quantlab.paper_portfolios.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("trade_date", sa.Date(), primary_key=True),
        sa.Column("cash", sa.Numeric(20, 6), nullable=False),
        sa.Column("market_value", sa.Numeric(20, 6), nullable=False),
        sa.Column("nav", sa.Numeric(20, 6), nullable=False),
        sa.Column("daily_return", sa.Float(), nullable=False),
        sa.Column("benchmark_return", sa.Float()),
        sa.Column("drawdown", sa.Float(), nullable=False),
        sa.Column("exposure", sa.Float(), nullable=False),
        sa.Column("turnover", sa.Float(), nullable=False),
        sa.Column("fees", sa.Numeric(20, 6), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        schema="quantlab",
    )
    op.create_index(
        "idx_portfolio_nav_trade_date",
        "portfolio_nav",
        [sa.text("trade_date DESC")],
        schema="quantlab",
    )
    op.create_table(
        "risk_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "portfolio_id",
            sa.String(),
            sa.ForeignKey("quantlab.paper_portfolios.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "batch_id",
            sa.String(),
            sa.ForeignKey("quantlab.portfolio_batches.id", ondelete="CASCADE"),
        ),
        sa.Column("severity", sa.String(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("rule", sa.String(), nullable=False),
        sa.Column("observed", sa.Float()),
        sa.Column("limit_value", sa.Float()),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("details_json", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True)),
        schema="quantlab",
    )
    op.create_index(
        "idx_risk_events_portfolio_created",
        "risk_events",
        ["portfolio_id", sa.text("created_at DESC")],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index("idx_risk_events_portfolio_created", table_name="risk_events", schema="quantlab")
    op.drop_table("risk_events", schema="quantlab")
    op.drop_index("idx_portfolio_nav_trade_date", table_name="portfolio_nav", schema="quantlab")
    op.drop_table("portfolio_nav", schema="quantlab")
    op.drop_table("paper_positions", schema="quantlab")
    op.drop_table("paper_fills", schema="quantlab")
    op.drop_index(
        "idx_paper_orders_portfolio_created", table_name="paper_orders", schema="quantlab"
    )
    op.drop_table("paper_orders", schema="quantlab")
    op.drop_index(
        "idx_portfolio_batches_status_created", table_name="portfolio_batches", schema="quantlab"
    )
    op.drop_table("portfolio_batches", schema="quantlab")
    op.drop_index(
        "idx_paper_portfolios_status_updated", table_name="paper_portfolios", schema="quantlab"
    )
    op.drop_table("paper_portfolios", schema="quantlab")
