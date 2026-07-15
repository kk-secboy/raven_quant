"""Add recommendation tracking and retire execution-shaped schedules."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0033_recommendation_pipeline"
down_revision: str | None = "0032_job_cancellation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(JSONB(), "postgresql")
    op.add_column(
        "factor_evaluations",
        sa.Column("dataset_identity_sha256", sa.String()),
        schema="quantlab",
    )
    op.create_table(
        "recommendation_portfolios",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False, unique=True),
        sa.Column("strategy_version_id", sa.String(), nullable=False),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("base_currency", sa.String(), nullable=False),
        sa.Column("hypothetical_initial_value", sa.Numeric(20, 6), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["strategy_version_id"], ["quantlab.strategy_versions.id"], ondelete="RESTRICT"
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_recommendation_portfolios_status_updated",
        "recommendation_portfolios",
        ["status", sa.text("updated_at DESC")],
        schema="quantlab",
    )
    op.create_table(
        "recommendation_snapshots",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("portfolio_id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String()),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("effective_date", sa.Date()),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("snapshot_json", json_type),
        sa.Column("cost_model_json", json_type, nullable=False),
        sa.Column("policy_version", sa.String(), nullable=False),
        sa.Column("backtest_engine_version", sa.String(), nullable=False),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("dataset_identity_sha256", sa.String(), nullable=False),
        sa.Column("strategy_version_id", sa.String(), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(
            ["portfolio_id"], ["quantlab.recommendation_portfolios.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["job_id"], ["quantlab.jobs.id"], ondelete="SET NULL"),
        sa.UniqueConstraint("portfolio_id", "as_of_date", name="uq_recommendation_snapshots_as_of"),
        schema="quantlab",
    )
    op.create_index(
        "idx_recommendation_snapshots_portfolio_created",
        "recommendation_snapshots",
        ["portfolio_id", sa.text("created_at DESC")],
        schema="quantlab",
    )
    op.create_table(
        "recommendation_holdings",
        sa.Column("snapshot_id", sa.String(), primary_key=True),
        sa.Column("instrument", sa.String(), primary_key=True),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("previous_weight", sa.Float(), nullable=False),
        sa.Column("weight_change", sa.Float(), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"], ["quantlab.recommendation_snapshots.id"], ondelete="CASCADE"
        ),
        schema="quantlab",
    )
    op.create_table(
        "recommendation_nav",
        sa.Column("portfolio_id", sa.String(), primary_key=True),
        sa.Column("trade_date", sa.Date(), primary_key=True),
        sa.Column("hypothetical_value", sa.Numeric(20, 6), nullable=False),
        sa.Column("daily_return", sa.Float(), nullable=False),
        sa.Column("benchmark_return", sa.Float()),
        sa.Column("drawdown", sa.Float(), nullable=False),
        sa.Column("turnover", sa.Float(), nullable=False),
        sa.Column("estimated_cost", sa.Numeric(20, 6), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["portfolio_id"], ["quantlab.recommendation_portfolios.id"], ondelete="CASCADE"
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_recommendation_nav_trade_date",
        "recommendation_nav",
        [sa.text("trade_date DESC")],
        schema="quantlab",
    )
    op.execute(
        "UPDATE quantlab.schedules SET status='retired', desired_status='retired', "
        "suspension_reason='legacy_execution_path_retired' "
        "WHERE kind IN ('paper_rebalance', 'pair_paper_rebalance', 'broker_reconcile')"
    )


def downgrade() -> None:
    op.drop_index(
        "idx_recommendation_nav_trade_date", table_name="recommendation_nav", schema="quantlab"
    )
    op.drop_table("recommendation_nav", schema="quantlab")
    op.drop_table("recommendation_holdings", schema="quantlab")
    op.drop_index(
        "idx_recommendation_snapshots_portfolio_created",
        table_name="recommendation_snapshots",
        schema="quantlab",
    )
    op.drop_table("recommendation_snapshots", schema="quantlab")
    op.drop_index(
        "idx_recommendation_portfolios_status_updated",
        table_name="recommendation_portfolios",
        schema="quantlab",
    )
    op.drop_table("recommendation_portfolios", schema="quantlab")
    op.drop_column("factor_evaluations", "dataset_identity_sha256", schema="quantlab")
