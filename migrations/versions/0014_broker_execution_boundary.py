"""Add a sandbox-only broker execution boundary and durable outbox."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0014_broker_execution_boundary"
down_revision: str | None = "0013_system_health_history"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "broker_destinations",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("adapter", sa.String(), nullable=False),
        sa.Column("environment", sa.String(), nullable=False),
        sa.Column("account_ref", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("config_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("activation_requested_by", sa.String()),
        sa.Column("activation_requested_at", sa.DateTime(timezone=True)),
        sa.Column("activated_by", sa.String()),
        sa.Column("activated_at", sa.DateTime(timezone=True)),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
        schema="quantlab",
    )
    op.create_index(
        "idx_broker_destinations_status_updated",
        "broker_destinations",
        ["status", "updated_at"],
        schema="quantlab",
    )
    op.create_table(
        "broker_order_outbox",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("destination_id", sa.String(), nullable=False),
        sa.Column("portfolio_id", sa.String(), nullable=False),
        sa.Column("batch_id", sa.String(), nullable=False),
        sa.Column("source_order_id", sa.String(), nullable=False),
        sa.Column("idempotency_key", sa.String(), nullable=False),
        sa.Column("payload_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("payload_sha256", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("broker_order_id", sa.String()),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("approved_by", sa.String()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("approved_at", sa.DateTime(timezone=True)),
        sa.Column("submitted_at", sa.DateTime(timezone=True)),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_error", sa.Text()),
        sa.ForeignKeyConstraint(
            ["destination_id"], ["quantlab.broker_destinations.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["portfolio_id"], ["quantlab.paper_portfolios.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["batch_id"], ["quantlab.portfolio_batches.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["source_order_id"], ["quantlab.paper_orders.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_key"),
        sa.UniqueConstraint(
            "destination_id",
            "source_order_id",
            name="uq_broker_outbox_destination_source_order",
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_broker_outbox_status_updated",
        "broker_order_outbox",
        ["status", "updated_at"],
        schema="quantlab",
    )
    op.create_table(
        "broker_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("destination_id", sa.String(), nullable=False),
        sa.Column("outbox_id", sa.String()),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("actor", sa.String(), nullable=False),
        sa.Column("details_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["destination_id"], ["quantlab.broker_destinations.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["outbox_id"], ["quantlab.broker_order_outbox.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        schema="quantlab",
    )
    op.create_index(
        "idx_broker_events_destination_created",
        "broker_events",
        ["destination_id", "created_at"],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_broker_events_destination_created",
        table_name="broker_events",
        schema="quantlab",
    )
    op.drop_table("broker_events", schema="quantlab")
    op.drop_index(
        "idx_broker_outbox_status_updated",
        table_name="broker_order_outbox",
        schema="quantlab",
    )
    op.drop_table("broker_order_outbox", schema="quantlab")
    op.drop_index(
        "idx_broker_destinations_status_updated",
        table_name="broker_destinations",
        schema="quantlab",
    )
    op.drop_table("broker_destinations", schema="quantlab")
