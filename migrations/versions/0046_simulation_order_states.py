"""Persistent simulation order states (design draft 8.1 steps 7-8/9.2/12.4).

Simulation orders stop being same-day terminal rows and gain a persistent
lifecycle: an order plan is committed as ``planned`` (together with its batch,
one transaction), transitions to ``open`` when its execution batch starts, and
ends in a terminal state (``filled`` / ``partial_filled_expired`` /
``rejected`` / ``expired`` / ``cancelled``). A working order whose execution
window (``not_before``/``not_after``) spans multiple days stays ``open``
across batches and accumulates fills; ``cancelled`` only ever releases the
unfilled remainder once (cash moves at fill time in this ledger, so a cancel
is the negation of the unfilled remainder, never a cash refund).

New order fields: ``limit_price`` (nullable price protection),
``not_before``/``not_after`` (nullable execution window), ``target_version``
(final target reference, e.g. a recommendation snapshot or netting plan),
``account_netting_plan_id`` + ``strategy_contributions_json`` (netting-plan
binding and per-instrument strategy attribution), ``plan_op``
(keep/replace/new provenance), ``cancel_reason`` and ``updated_at``.
``portfolio_id`` lets the order book be loaded across batches and is
backfilled from the creation batch. Batches may bind the account netting plan
they execute.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0046_simulation_order_states"
down_revision: str | None = "0045_research_only_pair"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ORDER_STATUSES = (
    "planned",
    "open",
    "filled",
    "partial_filled_expired",
    "rejected",
    "expired",
    "cancelled",
)


def upgrade() -> None:
    op.add_column(
        "simulation_orders",
        sa.Column("portfolio_id", sa.String(), nullable=True),
        schema="quantlab",
    )
    op.create_foreign_key(
        "fk_simulation_orders_portfolio",
        "simulation_orders",
        "simulation_portfolios",
        ["portfolio_id"],
        ["id"],
        source_schema="quantlab",
        referent_schema="quantlab",
        ondelete="CASCADE",
    )
    op.execute(
        """
        UPDATE quantlab.simulation_orders AS o
        SET portfolio_id = b.portfolio_id
        FROM quantlab.simulation_batches AS b
        WHERE o.batch_id = b.id AND o.portfolio_id IS NULL
        """
    )
    op.add_column(
        "simulation_orders",
        sa.Column("limit_price", sa.Numeric(20, 8), nullable=True),
        schema="quantlab",
    )
    op.add_column(
        "simulation_orders",
        sa.Column("not_before", sa.DateTime(timezone=True), nullable=True),
        schema="quantlab",
    )
    op.add_column(
        "simulation_orders",
        sa.Column("not_after", sa.DateTime(timezone=True), nullable=True),
        schema="quantlab",
    )
    op.add_column(
        "simulation_orders",
        sa.Column("target_version", sa.String(), nullable=True),
        schema="quantlab",
    )
    op.add_column(
        "simulation_orders",
        sa.Column("account_netting_plan_id", sa.String(), nullable=True),
        schema="quantlab",
    )
    op.create_foreign_key(
        "fk_simulation_orders_netting_plan",
        "simulation_orders",
        "account_netting_plans",
        ["account_netting_plan_id"],
        ["id"],
        source_schema="quantlab",
        referent_schema="quantlab",
        ondelete="SET NULL",
    )
    op.add_column(
        "simulation_orders",
        sa.Column("strategy_contributions_json", sa.JSON(), nullable=True),
        schema="quantlab",
    )
    op.add_column(
        "simulation_orders",
        sa.Column("plan_op", sa.String(), nullable=True),
        schema="quantlab",
    )
    op.add_column(
        "simulation_orders",
        sa.Column("cancel_reason", sa.String(), nullable=True),
        schema="quantlab",
    )
    op.add_column(
        "simulation_orders",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        schema="quantlab",
    )
    op.create_index(
        "idx_simulation_orders_portfolio_status",
        "simulation_orders",
        ["portfolio_id", "status"],
        schema="quantlab",
    )
    quoted = ", ".join(f"'{status}'" for status in _ORDER_STATUSES)
    op.create_check_constraint(
        "ck_simulation_orders_status",
        "simulation_orders",
        f"status IN ({quoted})",
        schema="quantlab",
    )
    op.add_column(
        "simulation_batches",
        sa.Column("account_netting_plan_id", sa.String(), nullable=True),
        schema="quantlab",
    )
    op.create_foreign_key(
        "fk_simulation_batches_netting_plan",
        "simulation_batches",
        "account_netting_plans",
        ["account_netting_plan_id"],
        ["id"],
        source_schema="quantlab",
        referent_schema="quantlab",
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_simulation_batches_netting_plan",
        "simulation_batches",
        schema="quantlab",
        type_="foreignkey",
    )
    op.drop_column("simulation_batches", "account_netting_plan_id", schema="quantlab")
    op.drop_constraint(
        "ck_simulation_orders_status", "simulation_orders", schema="quantlab", type_="check"
    )
    op.drop_index(
        "idx_simulation_orders_portfolio_status",
        table_name="simulation_orders",
        schema="quantlab",
    )
    op.drop_constraint(
        "fk_simulation_orders_netting_plan",
        "simulation_orders",
        schema="quantlab",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_simulation_orders_portfolio",
        "simulation_orders",
        schema="quantlab",
        type_="foreignkey",
    )
    for column in (
        "updated_at",
        "cancel_reason",
        "plan_op",
        "strategy_contributions_json",
        "account_netting_plan_id",
        "target_version",
        "not_after",
        "not_before",
        "limit_price",
        "portfolio_id",
    ):
        op.drop_column("simulation_orders", column, schema="quantlab")
