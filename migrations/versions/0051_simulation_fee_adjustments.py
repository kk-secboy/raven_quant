"""Append-only final simulation fee adjustments (design 5.7/9.4).

The execution fill keeps its originally confirmed estimate.  A final broker
or end-of-day fee confirmation records only ``final_fee - previously
confirmed fee`` as an append-only ledger event.  The portfolio-scoped
``adjustment_key`` makes retries idempotent and prevents a confirmation from
changing cash or NAV twice.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0051_simulation_fee_adjustments"
down_revision: str | None = "0050_corporate_event_types"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "simulation_fee_adjustments",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("portfolio_id", sa.String(), nullable=False),
        sa.Column("fill_id", sa.String(), nullable=False),
        sa.Column("batch_id", sa.String(), nullable=False),
        sa.Column("adjustment_key", sa.String(), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("previously_confirmed_fee", sa.Numeric(20, 6), nullable=False),
        sa.Column("final_fee", sa.Numeric(20, 6), nullable=False),
        sa.Column("adjustment_amount", sa.Numeric(20, 6), nullable=False),
        sa.Column("evidence_sha256", sa.String(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["portfolio_id"],
            ["quantlab.simulation_portfolios.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["fill_id"],
            ["quantlab.simulation_fills.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["batch_id"],
            ["quantlab.simulation_batches.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "portfolio_id",
            "adjustment_key",
            name="uq_simulation_fee_adjustments_key",
        ),
        sa.CheckConstraint(
            "source IN ('end_of_day', 'user_import')",
            name="ck_simulation_fee_adjustments_source",
        ),
        sa.CheckConstraint(
            "final_fee >= 0",
            name="ck_simulation_fee_adjustments_final_fee",
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_simulation_fee_adjustments_fill",
        "simulation_fee_adjustments",
        ["fill_id", "created_at"],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_simulation_fee_adjustments_fill",
        table_name="simulation_fee_adjustments",
        schema="quantlab",
    )
    op.drop_table("simulation_fee_adjustments", schema="quantlab")
