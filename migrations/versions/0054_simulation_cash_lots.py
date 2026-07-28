"""Cash lots, availability views and append-only reservation events.

Cash lots are the single economic cash asset. ``free_amount`` and
``frozen_amount`` are mutually exclusive classifications, while
``tradable_at`` and ``withdrawable_at`` are permission timestamps. Event
headers are idempotent on a portfolio-scoped key and their allocations are
append-only; mutable reservation rows only project the still-frozen remainder
for a working buy order.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0054_simulation_cash_lots"
down_revision: str | None = "0053_hypothesis_group_governance"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "simulation_cash_lots",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("portfolio_id", sa.String(), nullable=False),
        sa.Column("lot_key", sa.String(), nullable=False),
        sa.Column("source_type", sa.String(), nullable=False),
        sa.Column("source_reference_id", sa.String()),
        sa.Column("free_amount", sa.Numeric(20, 6), nullable=False),
        sa.Column("frozen_amount", sa.Numeric(20, 6), nullable=False),
        sa.Column("tradable_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("withdrawable_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["portfolio_id"],
            ["quantlab.simulation_portfolios.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "portfolio_id",
            "lot_key",
            name="uq_simulation_cash_lots_key",
        ),
        sa.CheckConstraint(
            "free_amount >= 0 AND frozen_amount >= 0",
            name="ck_simulation_cash_lots_nonnegative",
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_simulation_cash_lots_portfolio_availability",
        "simulation_cash_lots",
        ["portfolio_id", "tradable_at", "withdrawable_at"],
        schema="quantlab",
    )

    op.create_table(
        "simulation_cash_events",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("portfolio_id", sa.String(), nullable=False),
        sa.Column("batch_id", sa.String()),
        sa.Column("order_id", sa.String()),
        sa.Column("event_key", sa.String(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("amount", sa.Numeric(20, 6), nullable=False),
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
            name="uq_simulation_cash_events_key",
        ),
        sa.CheckConstraint(
            "event_type IN ('create', 'freeze', 'consume_free', "
            "'consume_frozen', 'release', 'reclassify')",
            name="ck_simulation_cash_events_type",
        ),
        sa.CheckConstraint("amount >= 0", name="ck_simulation_cash_events_amount"),
        schema="quantlab",
    )
    op.create_index(
        "idx_simulation_cash_events_portfolio_time",
        "simulation_cash_events",
        ["portfolio_id", "occurred_at"],
        schema="quantlab",
    )

    op.create_table(
        "simulation_cash_event_allocations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("cash_lot_id", sa.String(), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("amount", sa.Numeric(20, 6), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["event_id"],
            ["quantlab.simulation_cash_events.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["cash_lot_id"],
            ["quantlab.simulation_cash_lots.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "event_id",
            "cash_lot_id",
            "action",
            name="uq_simulation_cash_event_allocations",
        ),
        sa.CheckConstraint(
            "action IN ('create', 'freeze', 'consume_free', "
            "'consume_frozen', 'release', 'reclassify')",
            name="ck_simulation_cash_event_allocations_action",
        ),
        sa.CheckConstraint(
            "amount > 0",
            name="ck_simulation_cash_event_allocations_amount",
        ),
        schema="quantlab",
    )

    op.create_table(
        "simulation_cash_reservations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("portfolio_id", sa.String(), nullable=False),
        sa.Column("order_id", sa.String(), nullable=False),
        sa.Column("cash_lot_id", sa.String(), nullable=False),
        sa.Column("reserved_amount", sa.Numeric(20, 6), nullable=False),
        sa.Column("remaining_amount", sa.Numeric(20, 6), nullable=False),
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
        sa.ForeignKeyConstraint(
            ["cash_lot_id"],
            ["quantlab.simulation_cash_lots.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "order_id",
            "cash_lot_id",
            name="uq_simulation_cash_reservations_order_lot",
        ),
        sa.CheckConstraint(
            "reserved_amount > 0 AND remaining_amount >= 0 "
            "AND remaining_amount <= reserved_amount",
            name="ck_simulation_cash_reservations_amounts",
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_simulation_cash_reservations_order",
        "simulation_cash_reservations",
        ["order_id", "remaining_amount"],
        schema="quantlab",
    )

    # Existing scalar-cash accounts become one auditable legacy lot. This is a
    # classification migration only and therefore does not change NAV.
    op.execute(
        """
        INSERT INTO quantlab.simulation_cash_lots (
            id, portfolio_id, lot_key, source_type, source_reference_id,
            free_amount, frozen_amount, tradable_at, withdrawable_at,
            created_at, updated_at
        )
        SELECT
            'cash-lot-' || id,
            id,
            'migration:0054:' || id,
            'legacy_migration',
            NULL,
            cash,
            0,
            TIMESTAMPTZ '1970-01-01 00:00:00+00',
            TIMESTAMPTZ '1970-01-01 00:00:00+00',
            created_at,
            updated_at
        FROM quantlab.simulation_portfolios
        WHERE cash > 0
        """
    )
    op.execute(
        """
        INSERT INTO quantlab.simulation_cash_events (
            id, portfolio_id, batch_id, order_id, event_key, event_type,
            amount, details_json, occurred_at, created_at
        )
        SELECT
            'cash-event-' || id,
            id,
            NULL,
            NULL,
            'migration:0054:' || id,
            'create',
            cash,
            '{"source":"legacy_scalar_cash_backfill"}'::jsonb,
            created_at,
            created_at
        FROM quantlab.simulation_portfolios
        WHERE cash > 0
        """
    )
    op.execute(
        """
        INSERT INTO quantlab.simulation_cash_event_allocations (
            id, event_id, cash_lot_id, action, amount, created_at
        )
        SELECT
            'cash-allocation-' || id,
            'cash-event-' || id,
            'cash-lot-' || id,
            'create',
            cash,
            created_at
        FROM quantlab.simulation_portfolios
        WHERE cash > 0
        """
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION quantlab.prevent_simulation_cash_history_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'simulation cash event history is append-only';
        END;
        $$
        """
    )
    for table in ("simulation_cash_events", "simulation_cash_event_allocations"):
        op.execute(
            f"""
            CREATE TRIGGER trg_{table}_append_only
            BEFORE UPDATE OR DELETE ON quantlab.{table}
            FOR EACH ROW EXECUTE FUNCTION
            quantlab.prevent_simulation_cash_history_mutation()
            """
        )


def downgrade() -> None:
    for table in ("simulation_cash_events", "simulation_cash_event_allocations"):
        op.execute(
            f"DROP TRIGGER IF EXISTS trg_{table}_append_only ON quantlab.{table}"
        )
    op.execute(
        "DROP FUNCTION IF EXISTS quantlab.prevent_simulation_cash_history_mutation()"
    )
    op.drop_index(
        "idx_simulation_cash_reservations_order",
        table_name="simulation_cash_reservations",
        schema="quantlab",
    )
    op.drop_table("simulation_cash_reservations", schema="quantlab")
    op.drop_table("simulation_cash_event_allocations", schema="quantlab")
    op.drop_index(
        "idx_simulation_cash_events_portfolio_time",
        table_name="simulation_cash_events",
        schema="quantlab",
    )
    op.drop_table("simulation_cash_events", schema="quantlab")
    op.drop_index(
        "idx_simulation_cash_lots_portfolio_availability",
        table_name="simulation_cash_lots",
        schema="quantlab",
    )
    op.drop_table("simulation_cash_lots", schema="quantlab")
