"""Freeze sellable securities for persistent simulation orders."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0055_security_reservations"
down_revision: str | None = "0054_simulation_cash_lots"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "simulation_positions",
        sa.Column(
            "frozen_quantity",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        schema="quantlab",
    )
    op.create_check_constraint(
        "ck_simulation_positions_frozen_quantity",
        "simulation_positions",
        "frozen_quantity >= 0 AND frozen_quantity <= available_quantity",
        schema="quantlab",
    )
    op.create_table(
        "simulation_position_reservations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("portfolio_id", sa.String(), nullable=False),
        sa.Column("order_id", sa.String(), nullable=False, unique=True),
        sa.Column("instrument", sa.String(), nullable=False),
        sa.Column("reserved_quantity", sa.Integer(), nullable=False),
        sa.Column("remaining_quantity", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["portfolio_id"],
            ["quantlab.simulation_portfolios.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["order_id"],
            ["quantlab.simulation_orders.id"],
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "reserved_quantity > 0 AND remaining_quantity >= 0 "
            "AND remaining_quantity <= reserved_quantity",
            name="ck_simulation_position_reservations_quantity",
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_simulation_position_reservations_portfolio_instrument",
        "simulation_position_reservations",
        ["portfolio_id", "instrument", "remaining_quantity"],
        schema="quantlab",
    )
    op.create_table(
        "simulation_security_events",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("portfolio_id", sa.String(), nullable=False),
        sa.Column("batch_id", sa.String()),
        sa.Column("order_id", sa.String()),
        sa.Column("event_key", sa.String(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("instrument", sa.String(), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("details_json", sa.JSON(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
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
        sa.ForeignKeyConstraint(
            ["order_id"],
            ["quantlab.simulation_orders.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "portfolio_id",
            "event_key",
            name="uq_simulation_security_events_key",
        ),
        sa.CheckConstraint(
            "event_type IN ('freeze', 'consume', 'release', 'reclassify')",
            name="ck_simulation_security_events_type",
        ),
        sa.CheckConstraint(
            "quantity > 0",
            name="ck_simulation_security_events_quantity",
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_simulation_security_events_portfolio_time",
        "simulation_security_events",
        ["portfolio_id", "occurred_at"],
        schema="quantlab",
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION quantlab.prevent_simulation_security_history_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'simulation security event history is append-only';
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_simulation_security_events_append_only
        BEFORE UPDATE OR DELETE ON quantlab.simulation_security_events
        FOR EACH ROW EXECUTE FUNCTION
        quantlab.prevent_simulation_security_history_mutation()
        """
    )


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS trg_simulation_security_events_append_only "
        "ON quantlab.simulation_security_events"
    )
    op.execute(
        "DROP FUNCTION IF EXISTS "
        "quantlab.prevent_simulation_security_history_mutation()"
    )
    op.drop_index(
        "idx_simulation_security_events_portfolio_time",
        table_name="simulation_security_events",
        schema="quantlab",
    )
    op.drop_table("simulation_security_events", schema="quantlab")
    op.drop_index(
        "idx_simulation_position_reservations_portfolio_instrument",
        table_name="simulation_position_reservations",
        schema="quantlab",
    )
    op.drop_table("simulation_position_reservations", schema="quantlab")
    op.drop_constraint(
        "ck_simulation_positions_frozen_quantity",
        "simulation_positions",
        schema="quantlab",
        type_="check",
    )
    op.drop_column(
        "simulation_positions",
        "frozen_quantity",
        schema="quantlab",
    )
