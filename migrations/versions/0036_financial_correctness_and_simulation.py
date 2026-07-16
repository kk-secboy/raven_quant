"""Add statistical controls and the transactional simulation ledger."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0036_financial_correctness"
down_revision: str | None = "0035_research_policy_v2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(JSONB(), "postgresql")
    for column in (
        sa.Column("experiment_family_id", sa.String()),
        sa.Column("label_horizon_days", sa.Integer()),
        sa.Column("experiment_count", sa.Integer()),
    ):
        op.add_column("factor_candidates", column, schema="quantlab")
    for column in (
        sa.Column("hac_p_value", sa.Float()),
        sa.Column("bh_q_value", sa.Float()),
        sa.Column("statistical_contract_version", sa.String()),
        sa.Column("final_test_key", sa.String()),
        sa.Column("final_test_consumed_at", sa.DateTime(timezone=True)),
    ):
        op.add_column("factor_evaluations", column, schema="quantlab")
    op.create_index(
        "uq_factor_evaluations_final_test_key",
        "factor_evaluations",
        ["final_test_key"],
        unique=True,
        schema="quantlab",
        postgresql_where=sa.text("final_test_key IS NOT NULL"),
    )

    op.create_table(
        "simulation_portfolios",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False, unique=True),
        sa.Column("recommendation_portfolio_id", sa.String(), nullable=False, unique=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("base_currency", sa.String(), nullable=False),
        sa.Column("initial_cash", sa.Numeric(20, 6), nullable=False),
        sa.Column("cash", sa.Numeric(20, 6), nullable=False),
        sa.Column("nav", sa.Numeric(20, 6), nullable=False),
        sa.Column("high_water_mark", sa.Numeric(20, 6), nullable=False),
        sa.Column("execution_algorithm", sa.String(), nullable=False),
        sa.Column("execution_dataset", sa.String(), nullable=False),
        sa.Column("daily_dataset", sa.String(), nullable=False),
        sa.Column("daily_dataset_identity_sha256", sa.String(), nullable=False),
        sa.Column("daily_dataset_lineage_id", sa.String(), nullable=False),
        sa.Column("daily_field_contract_version", sa.String(), nullable=False),
        sa.Column("execution_dataset_identity_sha256", sa.String(), nullable=False),
        sa.Column("execution_dataset_lineage_id", sa.String(), nullable=False),
        sa.Column("execution_field_contract_version", sa.String(), nullable=False),
        sa.Column("execution_engine_version", sa.String(), nullable=False),
        sa.Column("cost_schedule_version", sa.String(), nullable=False),
        sa.Column("execution_policy_json", json_type, nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["recommendation_portfolio_id"],
            ["quantlab.recommendation_portfolios.id"],
            ondelete="RESTRICT",
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_simulation_portfolios_status_updated",
        "simulation_portfolios",
        ["status", sa.text("updated_at DESC")],
        schema="quantlab",
    )
    op.create_table(
        "simulation_batches",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("portfolio_id", sa.String(), nullable=False),
        sa.Column("recommendation_snapshot_id", sa.String(), nullable=False, unique=True),
        sa.Column("signal_date", sa.Date(), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("idempotency_key", sa.String(), nullable=False, unique=True),
        sa.Column("summary_json", json_type),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(
            ["portfolio_id"], ["quantlab.simulation_portfolios.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["recommendation_snapshot_id"],
            ["quantlab.recommendation_snapshots.id"],
            ondelete="RESTRICT",
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_simulation_batches_portfolio_date",
        "simulation_batches",
        ["portfolio_id", sa.text("trade_date DESC")],
        schema="quantlab",
    )
    op.create_table(
        "simulation_orders",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("batch_id", sa.String(), nullable=False),
        sa.Column("instrument", sa.String(), nullable=False),
        sa.Column("side", sa.String(), nullable=False),
        sa.Column("target_weight", sa.Float(), nullable=False),
        sa.Column("requested_quantity", sa.Integer(), nullable=False),
        sa.Column("filled_quantity", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("reject_reason", sa.String()),
        sa.Column("requested_value", sa.Numeric(20, 6), nullable=False),
        sa.Column("filled_value", sa.Numeric(20, 6), nullable=False),
        sa.Column("capacity_fill_ratio", sa.Float(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["batch_id"], ["quantlab.simulation_batches.id"], ondelete="CASCADE"
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_simulation_orders_batch",
        "simulation_orders",
        ["batch_id", "instrument"],
        schema="quantlab",
    )
    op.create_table(
        "simulation_fills",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("order_id", sa.String(), nullable=False),
        sa.Column("batch_id", sa.String(), nullable=False),
        sa.Column("instrument", sa.String(), nullable=False),
        sa.Column("side", sa.String(), nullable=False),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("price", sa.Numeric(20, 8), nullable=False),
        sa.Column("gross_value", sa.Numeric(20, 6), nullable=False),
        sa.Column("fee", sa.Numeric(20, 6), nullable=False),
        sa.Column("cost_breakdown_json", json_type, nullable=False),
        sa.Column("minute_volume", sa.Integer(), nullable=False),
        sa.Column("capacity_quantity", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["order_id"], ["quantlab.simulation_orders.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["batch_id"], ["quantlab.simulation_batches.id"], ondelete="CASCADE"
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_simulation_fills_batch",
        "simulation_fills",
        ["batch_id", "executed_at"],
        schema="quantlab",
    )
    op.create_table(
        "simulation_positions",
        sa.Column("portfolio_id", sa.String(), primary_key=True),
        sa.Column("instrument", sa.String(), primary_key=True),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("available_quantity", sa.Integer(), nullable=False),
        sa.Column("average_cost", sa.Numeric(20, 8), nullable=False),
        sa.Column("last_trade_date", sa.Date()),
        sa.Column("market_price", sa.Numeric(20, 8)),
        sa.Column("market_date", sa.Date()),
        sa.Column("stale", sa.Boolean(), nullable=False),
        sa.Column("market_value", sa.Numeric(20, 6), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["portfolio_id"], ["quantlab.simulation_portfolios.id"], ondelete="CASCADE"
        ),
        schema="quantlab",
    )
    op.create_table(
        "simulation_cash_flows",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("portfolio_id", sa.String(), nullable=False),
        sa.Column("batch_id", sa.String()),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("flow_type", sa.String(), nullable=False),
        sa.Column("amount", sa.Numeric(20, 6), nullable=False),
        sa.Column("balance_after", sa.Numeric(20, 6), nullable=False),
        sa.Column("reference_id", sa.String()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["portfolio_id"], ["quantlab.simulation_portfolios.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["batch_id"], ["quantlab.simulation_batches.id"], ondelete="CASCADE"
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_simulation_cash_flows_portfolio_date",
        "simulation_cash_flows",
        ["portfolio_id", "trade_date"],
        schema="quantlab",
    )
    op.create_table(
        "simulation_nav",
        sa.Column("portfolio_id", sa.String(), primary_key=True),
        sa.Column("trade_date", sa.Date(), primary_key=True),
        sa.Column("cash", sa.Numeric(20, 6), nullable=False),
        sa.Column("market_value", sa.Numeric(20, 6), nullable=False),
        sa.Column("nav", sa.Numeric(20, 6), nullable=False),
        sa.Column("daily_return", sa.Float(), nullable=False),
        sa.Column("drawdown", sa.Float(), nullable=False),
        sa.Column("market_date", sa.Date()),
        sa.Column("has_stale_prices", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("performance_certified", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["portfolio_id"], ["quantlab.simulation_portfolios.id"], ondelete="CASCADE"
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_simulation_nav_trade_date",
        "simulation_nav",
        [sa.text("trade_date DESC")],
        schema="quantlab",
    )
    op.create_table(
        "simulation_events",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("portfolio_id", sa.String(), nullable=False),
        sa.Column("batch_id", sa.String()),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("severity", sa.String(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("instrument", sa.String()),
        sa.Column("reason", sa.String(), nullable=False),
        sa.Column("details_json", json_type, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["portfolio_id"], ["quantlab.simulation_portfolios.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["batch_id"], ["quantlab.simulation_batches.id"], ondelete="CASCADE"
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_simulation_events_portfolio_date",
        "simulation_events",
        ["portfolio_id", sa.text("trade_date DESC")],
        schema="quantlab",
    )

    # Anything created before this contract is historical evidence, never current approval input.
    for table in (
        "factor_evaluations",
        "strategy_versions",
        "backtest_runs",
        "strategy_allocations",
    ):
        op.execute(f"UPDATE quantlab.{table} SET is_legacy=true")
    op.execute("UPDATE quantlab.recommendation_portfolios SET status='retired'")


def downgrade() -> None:
    for index_name, table_name in (
        ("idx_simulation_events_portfolio_date", "simulation_events"),
        ("idx_simulation_nav_trade_date", "simulation_nav"),
        ("idx_simulation_cash_flows_portfolio_date", "simulation_cash_flows"),
        ("idx_simulation_fills_batch", "simulation_fills"),
        ("idx_simulation_orders_batch", "simulation_orders"),
        ("idx_simulation_batches_portfolio_date", "simulation_batches"),
    ):
        op.drop_index(index_name, table_name=table_name, schema="quantlab")
        op.drop_table(table_name, schema="quantlab")
    op.drop_table("simulation_positions", schema="quantlab")
    op.drop_index(
        "idx_simulation_portfolios_status_updated",
        table_name="simulation_portfolios",
        schema="quantlab",
    )
    op.drop_table("simulation_portfolios", schema="quantlab")
    op.drop_index(
        "uq_factor_evaluations_final_test_key",
        table_name="factor_evaluations",
        schema="quantlab",
    )
    for name in (
        "final_test_consumed_at",
        "final_test_key",
        "statistical_contract_version",
        "bh_q_value",
        "hac_p_value",
    ):
        op.drop_column("factor_evaluations", name, schema="quantlab")
    for name in ("experiment_count", "label_horizon_days", "experiment_family_id"):
        op.drop_column("factor_candidates", name, schema="quantlab")
