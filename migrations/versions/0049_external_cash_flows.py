"""External cash flows and the unitized TWR performance curve (4.4/8.3/12.1).

Design draft 12.1: actual deposits/withdrawals are external cash flows — they
must not be booked as investment profit or loss and must not manufacture
return or drawdown. Design draft 4.4: the account TWR chains daily returns
``r_t = (V_t - F_t_close) / (V_{t-1} + F_t_open) - 1`` into the unitized
``investment_wealth`` curve; drawdown and recovery time are computed on that
curve, not on the CNY balance. Both curves coexist: the CNY NAV stays the
ledger reconciliation view.

``simulation_external_flows`` is the governed intake ledger for confirmed
external flows; ``flow_key`` gives a stable idempotency key per portfolio.
The NAV row gains the per-day flow split and the unitized chain state so a
broken chain (undefined base) is visible instead of silently skipped.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0049_external_cash_flows"
down_revision: str | None = "0048_permission_shadow_account"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "simulation_external_flows",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("portfolio_id", sa.String(), nullable=False),
        sa.Column("flow_key", sa.String(), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("timing", sa.String(), nullable=False),
        sa.Column("amount", sa.Numeric(20, 6), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "timing IN ('open', 'close')",
            name="ck_simulation_external_flows_timing",
        ),
        sa.ForeignKeyConstraint(
            ["portfolio_id"], ["quantlab.simulation_portfolios.id"], ondelete="CASCADE"
        ),
        schema="quantlab",
    )
    op.create_index(
        "uq_simulation_external_flows_key",
        "simulation_external_flows",
        ["portfolio_id", "flow_key"],
        unique=True,
        schema="quantlab",
    )
    op.create_index(
        "idx_simulation_external_flows_portfolio_date",
        "simulation_external_flows",
        ["portfolio_id", "trade_date"],
        schema="quantlab",
    )
    op.add_column(
        "simulation_nav",
        sa.Column(
            "external_flow_open",
            sa.Numeric(20, 6),
            nullable=False,
            server_default="0",
        ),
        schema="quantlab",
    )
    op.add_column(
        "simulation_nav",
        sa.Column(
            "external_flow_close",
            sa.Numeric(20, 6),
            nullable=False,
            server_default="0",
        ),
        schema="quantlab",
    )
    op.add_column(
        "simulation_nav",
        sa.Column("twr_daily_return", sa.Float(), nullable=True),
        schema="quantlab",
    )
    op.add_column(
        "simulation_nav",
        sa.Column("investment_wealth", sa.Float(), nullable=True),
        schema="quantlab",
    )
    op.add_column(
        "simulation_nav",
        sa.Column("twr_drawdown", sa.Float(), nullable=True),
        schema="quantlab",
    )
    op.add_column(
        "simulation_nav",
        sa.Column(
            "twr_status",
            sa.String(),
            nullable=False,
            server_default="unavailable_legacy",
        ),
        schema="quantlab",
    )
    op.add_column(
        "simulation_portfolios",
        sa.Column("investment_wealth", sa.Float(), nullable=True),
        schema="quantlab",
    )
    op.add_column(
        "simulation_portfolios",
        sa.Column("twr_high_water_mark", sa.Float(), nullable=True),
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_column("simulation_portfolios", "twr_high_water_mark", schema="quantlab")
    op.drop_column("simulation_portfolios", "investment_wealth", schema="quantlab")
    op.drop_column("simulation_nav", "twr_status", schema="quantlab")
    op.drop_column("simulation_nav", "twr_drawdown", schema="quantlab")
    op.drop_column("simulation_nav", "investment_wealth", schema="quantlab")
    op.drop_column("simulation_nav", "twr_daily_return", schema="quantlab")
    op.drop_column("simulation_nav", "external_flow_close", schema="quantlab")
    op.drop_column("simulation_nav", "external_flow_open", schema="quantlab")
    op.drop_index(
        "idx_simulation_external_flows_portfolio_date",
        table_name="simulation_external_flows",
        schema="quantlab",
    )
    op.drop_index(
        "uq_simulation_external_flows_key",
        table_name="simulation_external_flows",
        schema="quantlab",
    )
    op.drop_table("simulation_external_flows", schema="quantlab")
