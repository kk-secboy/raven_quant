"""Corporate-action position lots, dividend entitlements and receivables."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0038_corporate_actions"
down_revision: str | None = "0037_single_mainline_contract"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "simulation_position_lots",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("portfolio_id", sa.String(), nullable=False),
        sa.Column("instrument", sa.String(), nullable=False),
        sa.Column("lot_key", sa.String(), nullable=False),
        # NULL = acquisition date unknown (legacy lots); dividend tax then uses
        # the conservative top rate.
        sa.Column("acquired_at", sa.Date()),
        sa.Column("sellable_from", sa.Date(), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("cost_basis_total", sa.Numeric(20, 6), nullable=False),
        sa.Column("origin", sa.String(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["portfolio_id"],
            ["quantlab.simulation_portfolios.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "portfolio_id", "instrument", "lot_key", name="uq_simulation_position_lots_key"
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_simulation_position_lots_portfolio",
        "simulation_position_lots",
        ["portfolio_id", "instrument"],
        schema="quantlab",
    )
    op.create_table(
        "simulation_dividend_entitlements",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("portfolio_id", sa.String(), nullable=False),
        sa.Column("instrument", sa.String(), nullable=False),
        sa.Column("lot_key", sa.String(), nullable=False),
        sa.Column("record_date", sa.Date(), nullable=False),
        # cash = pretax cash dividend per share; bonus_par = bonus ratio x par value.
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("income_per_share", sa.Numeric(20, 8), nullable=False),
        sa.Column("untaxed_quantity", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["portfolio_id"],
            ["quantlab.simulation_portfolios.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "portfolio_id",
            "instrument",
            "lot_key",
            "record_date",
            "kind",
            name="uq_simulation_dividend_entitlements_key",
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_simulation_dividend_entitlements_portfolio",
        "simulation_dividend_entitlements",
        ["portfolio_id", "instrument"],
        schema="quantlab",
    )
    op.create_table(
        "simulation_dividend_actions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("portfolio_id", sa.String(), nullable=False),
        sa.Column("instrument", sa.String(), nullable=False),
        sa.Column("ex_date", sa.Date(), nullable=False),
        sa.Column("record_date", sa.Date()),
        sa.Column("pay_date", sa.Date()),
        sa.Column("eligible_quantity", sa.Integer(), nullable=False),
        sa.Column("cash_per_share", sa.Numeric(20, 8), nullable=False, server_default="0"),
        sa.Column("receivable_amount", sa.Numeric(20, 6), nullable=False, server_default="0"),
        sa.Column("bonus_share_ratio", sa.Float(), nullable=False, server_default="0"),
        sa.Column("conversion_ratio", sa.Float(), nullable=False, server_default="0"),
        sa.Column("new_shares", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("div_listdate", sa.Date()),
        # accrued = confirmed at ex-date; paid = reclassified to cash at pay-date.
        sa.Column("status", sa.String(), nullable=False, server_default="accrued"),
        sa.Column("tax_rule_version", sa.String(), nullable=False),
        sa.Column("valuation_uncertain", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("payload_sha256", sa.String(), nullable=False),
        sa.Column("batch_id", sa.String()),
        sa.Column("paid_batch_id", sa.String()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["portfolio_id"],
            ["quantlab.simulation_portfolios.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "portfolio_id", "instrument", "ex_date", name="uq_simulation_dividend_actions_key"
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_simulation_dividend_actions_status",
        "simulation_dividend_actions",
        ["portfolio_id", "status"],
        schema="quantlab",
    )
    op.add_column(
        "simulation_nav",
        sa.Column(
            "corporate_receivables",
            sa.Numeric(20, 6),
            nullable=False,
            server_default="0",
        ),
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_column("simulation_nav", "corporate_receivables", schema="quantlab")
    op.drop_index(
        "idx_simulation_dividend_actions_status",
        table_name="simulation_dividend_actions",
        schema="quantlab",
    )
    op.drop_table("simulation_dividend_actions", schema="quantlab")
    op.drop_index(
        "idx_simulation_dividend_entitlements_portfolio",
        table_name="simulation_dividend_entitlements",
        schema="quantlab",
    )
    op.drop_table("simulation_dividend_entitlements", schema="quantlab")
    op.drop_index(
        "idx_simulation_position_lots_portfolio",
        table_name="simulation_position_lots",
        schema="quantlab",
    )
    op.drop_table("simulation_position_lots", schema="quantlab")
