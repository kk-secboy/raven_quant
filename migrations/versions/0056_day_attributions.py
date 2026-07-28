"""Archive immutable simulation day-end attributions."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0056_day_attributions"
down_revision: str | None = "0055_security_reservations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "simulation_day_attributions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("portfolio_id", sa.String(), nullable=False),
        sa.Column("batch_id", sa.String(), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("strategy_json", sa.JSON(), nullable=False),
        sa.Column("industry_json", sa.JSON(), nullable=False),
        sa.Column("asset_json", sa.JSON(), nullable=False),
        sa.Column("cost_json", sa.JSON(), nullable=False),
        sa.Column("execution_json", sa.JSON(), nullable=False),
        sa.Column("coverage_status", sa.String(), nullable=False),
        sa.Column("blocker_reasons_json", sa.JSON(), nullable=False),
        sa.Column("input_sha256", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["portfolio_id"],
            ["quantlab.simulation_portfolios.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["batch_id"],
            ["quantlab.simulation_batches.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "batch_id",
            name="uq_simulation_day_attributions_batch",
        ),
        sa.CheckConstraint(
            "coverage_status IN ('complete', 'partial')",
            name="ck_simulation_day_attributions_coverage",
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_simulation_day_attributions_portfolio_date",
        "simulation_day_attributions",
        ["portfolio_id", "trade_date"],
        schema="quantlab",
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION quantlab.prevent_simulation_attribution_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'simulation day attribution history is append-only';
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_simulation_day_attributions_append_only
        BEFORE UPDATE OR DELETE ON quantlab.simulation_day_attributions
        FOR EACH ROW EXECUTE FUNCTION
        quantlab.prevent_simulation_attribution_mutation()
        """
    )


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS trg_simulation_day_attributions_append_only "
        "ON quantlab.simulation_day_attributions"
    )
    op.execute(
        "DROP FUNCTION IF EXISTS "
        "quantlab.prevent_simulation_attribution_mutation()"
    )
    op.drop_index(
        "idx_simulation_day_attributions_portfolio_date",
        table_name="simulation_day_attributions",
        schema="quantlab",
    )
    op.drop_table("simulation_day_attributions", schema="quantlab")
