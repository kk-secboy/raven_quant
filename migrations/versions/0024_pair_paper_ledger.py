"""Add an atomic two-leg paper ledger for governed pair strategies."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0024_pair_paper_ledger"
down_revision: str | None = "0023_pair_strategies"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "pair_paper_portfolios",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False, unique=True),
        sa.Column(
            "strategy_version_id",
            sa.String(),
            sa.ForeignKey("quantlab.strategy_versions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("execution_snapshot", sa.String(), nullable=False),
        sa.Column("minute_dataset", sa.String(), nullable=False),
        sa.Column("shortability_dataset", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("base_currency", sa.String(), nullable=False),
        sa.Column("initial_cash", sa.Numeric(20, 6), nullable=False),
        sa.Column("cash", sa.Numeric(20, 6), nullable=False),
        sa.Column("nav", sa.Numeric(20, 6), nullable=False),
        sa.Column("high_water_mark", sa.Numeric(20, 6), nullable=False),
        sa.Column("position_direction", sa.Integer(), nullable=False),
        sa.Column("quantity_y", sa.BigInteger(), nullable=False),
        sa.Column("quantity_x", sa.BigInteger(), nullable=False),
        sa.Column("entry_nav", sa.Numeric(20, 6)),
        sa.Column("holding_days", sa.Integer(), nullable=False),
        sa.Column("last_signal_date", sa.Date()),
        sa.Column("last_trade_date", sa.Date()),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "position_direction IN (-1, 0, 1)",
            name="ck_pair_paper_position_direction",
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_pair_paper_portfolios_status_updated",
        "pair_paper_portfolios",
        ["status", sa.text("updated_at DESC")],
        schema="quantlab",
    )
    op.create_table(
        "pair_portfolio_batches",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "portfolio_id",
            sa.String(),
            sa.ForeignKey("quantlab.pair_paper_portfolios.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("job_id", sa.String(), sa.ForeignKey("quantlab.jobs.id", ondelete="SET NULL")),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("trade_date", sa.Date()),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("idempotency_key", sa.String(), nullable=False, unique=True),
        sa.Column("starting_state_sha256", sa.String(), nullable=False),
        sa.Column("artifact_path", sa.Text()),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("portfolio_id", "as_of_date", name="uq_pair_batches_as_of"),
        schema="quantlab",
    )
    op.create_index(
        "idx_pair_batches_status_created",
        "pair_portfolio_batches",
        ["status", sa.text("created_at DESC")],
        schema="quantlab",
    )
    op.create_table(
        "pair_paper_orders",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "batch_id",
            sa.String(),
            sa.ForeignKey("quantlab.pair_portfolio_batches.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "portfolio_id",
            sa.String(),
            sa.ForeignKey("quantlab.pair_paper_portfolios.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("leg", sa.String(), nullable=False),
        sa.Column("instrument", sa.String(), nullable=False),
        sa.Column("side", sa.String(), nullable=False),
        sa.Column("requested_quantity", sa.BigInteger(), nullable=False),
        sa.Column("target_quantity", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("reason", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("batch_id", "leg", name="uq_pair_orders_batch_leg"),
        schema="quantlab",
    )
    op.create_index(
        "idx_pair_orders_portfolio_created",
        "pair_paper_orders",
        ["portfolio_id", sa.text("created_at DESC")],
        schema="quantlab",
    )
    op.create_table(
        "pair_paper_fills",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "order_id",
            sa.String(),
            sa.ForeignKey("quantlab.pair_paper_orders.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("fill_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("quantity", sa.BigInteger(), nullable=False),
        sa.Column("price", sa.Numeric(20, 6), nullable=False),
        sa.Column("gross_value", sa.Numeric(20, 6), nullable=False),
        sa.Column("fee", sa.Numeric(20, 6), nullable=False),
        sa.Column("slippage", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        schema="quantlab",
    )
    op.create_table(
        "pair_portfolio_nav",
        sa.Column(
            "portfolio_id",
            sa.String(),
            sa.ForeignKey("quantlab.pair_paper_portfolios.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("trade_date", sa.Date(), primary_key=True),
        sa.Column("cash", sa.Numeric(20, 6), nullable=False),
        sa.Column("long_value", sa.Numeric(20, 6), nullable=False),
        sa.Column("short_value", sa.Numeric(20, 6), nullable=False),
        sa.Column("nav", sa.Numeric(20, 6), nullable=False),
        sa.Column("daily_return", sa.Float(), nullable=False),
        sa.Column("drawdown", sa.Float(), nullable=False),
        sa.Column("gross_exposure", sa.Float(), nullable=False),
        sa.Column("net_exposure", sa.Float(), nullable=False),
        sa.Column("turnover", sa.Float(), nullable=False),
        sa.Column("fees", sa.Numeric(20, 6), nullable=False),
        sa.Column("borrow_cost", sa.Numeric(20, 6), nullable=False),
        sa.Column("zscore", sa.Float(), nullable=False),
        sa.Column("correlation", sa.Float(), nullable=False),
        sa.Column("cointegration_pvalue", sa.Float(), nullable=False),
        sa.Column("position_direction", sa.Integer(), nullable=False),
        sa.Column("quantity_y", sa.BigInteger(), nullable=False),
        sa.Column("quantity_x", sa.BigInteger(), nullable=False),
        sa.Column("price_y", sa.Numeric(20, 6), nullable=False),
        sa.Column("price_x", sa.Numeric(20, 6), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        schema="quantlab",
    )
    op.create_index(
        "idx_pair_nav_trade_date",
        "pair_portfolio_nav",
        [sa.text("trade_date DESC")],
        schema="quantlab",
    )
    op.create_table(
        "pair_portfolio_risk_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "portfolio_id",
            sa.String(),
            sa.ForeignKey("quantlab.pair_paper_portfolios.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "batch_id",
            sa.String(),
            sa.ForeignKey("quantlab.pair_portfolio_batches.id", ondelete="CASCADE"),
        ),
        sa.Column("severity", sa.String(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("rule", sa.String(), nullable=False),
        sa.Column("observed", sa.Float()),
        sa.Column("limit_value", sa.Float()),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("details_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("acknowledged_by", sa.String()),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True)),
        sa.Column("resolved_by", sa.String()),
        sa.Column("resolved_at", sa.DateTime(timezone=True)),
        sa.Column("resolution_reason", sa.Text()),
        schema="quantlab",
    )
    op.create_index(
        "idx_pair_risk_events_portfolio_created",
        "pair_portfolio_risk_events",
        ["portfolio_id", sa.text("created_at DESC")],
        schema="quantlab",
    )
    op.create_table(
        "pair_portfolio_reviews",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "portfolio_id",
            sa.String(),
            sa.ForeignKey("quantlab.pair_paper_portfolios.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "batch_id",
            sa.String(),
            sa.ForeignKey("quantlab.pair_portfolio_batches.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("summary_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        schema="quantlab",
    )
    op.create_index(
        "idx_pair_reviews_portfolio_date",
        "pair_portfolio_reviews",
        ["portfolio_id", sa.text("trade_date DESC")],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_pair_reviews_portfolio_date", table_name="pair_portfolio_reviews", schema="quantlab"
    )
    op.drop_table("pair_portfolio_reviews", schema="quantlab")
    op.drop_index(
        "idx_pair_risk_events_portfolio_created",
        table_name="pair_portfolio_risk_events",
        schema="quantlab",
    )
    op.drop_table("pair_portfolio_risk_events", schema="quantlab")
    op.drop_index("idx_pair_nav_trade_date", table_name="pair_portfolio_nav", schema="quantlab")
    op.drop_table("pair_portfolio_nav", schema="quantlab")
    op.drop_table("pair_paper_fills", schema="quantlab")
    op.drop_index(
        "idx_pair_orders_portfolio_created", table_name="pair_paper_orders", schema="quantlab"
    )
    op.drop_table("pair_paper_orders", schema="quantlab")
    op.drop_index(
        "idx_pair_batches_status_created", table_name="pair_portfolio_batches", schema="quantlab"
    )
    op.drop_table("pair_portfolio_batches", schema="quantlab")
    op.drop_index(
        "idx_pair_paper_portfolios_status_updated",
        table_name="pair_paper_portfolios",
        schema="quantlab",
    )
    op.drop_table("pair_paper_portfolios", schema="quantlab")
