"""Persist QMT participation evidence, attempts, and provider callbacks."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0017_qmt_execution_state"
down_revision: str | None = "0016_qmt_gateway_runtime"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "broker_gateway_children",
        sa.Column("filled_quantity", sa.Numeric(20, 6), server_default="0", nullable=False),
        schema="quantlab",
    )
    op.add_column(
        "broker_gateway_children",
        sa.Column("replacement_count", sa.Integer(), server_default="0", nullable=False),
        schema="quantlab",
    )
    op.add_column(
        "broker_gateway_children",
        sa.Column(
            "market_evidence_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        schema="quantlab",
    )
    op.add_column(
        "broker_gateway_children",
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True)),
        schema="quantlab",
    )
    op.create_table(
        "broker_gateway_attempts",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("child_id", sa.String(), nullable=False),
        sa.Column("attempt_no", sa.Integer(), nullable=False),
        sa.Column("client_tag", sa.String(), nullable=False),
        sa.Column("provider_order_id", sa.String()),
        sa.Column("quantity", sa.Numeric(20, 6), nullable=False),
        sa.Column("limit_price", sa.Numeric(20, 6), nullable=False),
        sa.Column("traded_quantity", sa.Numeric(20, 6), server_default="0", nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column(
            "market_evidence_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("submitted_at", sa.DateTime(timezone=True)),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_error", sa.Text()),
        sa.ForeignKeyConstraint(
            ["child_id"], ["quantlab.broker_gateway_children.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("client_tag"),
        sa.UniqueConstraint("child_id", "attempt_no", name="uq_broker_gateway_child_attempt"),
        schema="quantlab",
    )
    op.create_index(
        "idx_broker_gateway_attempts_status_updated",
        "broker_gateway_attempts",
        ["status", "updated_at"],
        schema="quantlab",
    )
    op.create_table(
        "broker_gateway_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("provider_order_id", sa.String()),
        sa.Column("client_tag", sa.String()),
        sa.Column(
            "payload_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        schema="quantlab",
    )
    op.create_index(
        "idx_broker_gateway_events_received",
        "broker_gateway_events",
        [sa.text("received_at DESC")],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_broker_gateway_events_received",
        table_name="broker_gateway_events",
        schema="quantlab",
    )
    op.drop_table("broker_gateway_events", schema="quantlab")
    op.drop_index(
        "idx_broker_gateway_attempts_status_updated",
        table_name="broker_gateway_attempts",
        schema="quantlab",
    )
    op.drop_table("broker_gateway_attempts", schema="quantlab")
    for column in (
        "cancel_requested_at",
        "market_evidence_json",
        "replacement_count",
        "filled_quantity",
    ):
        op.drop_column("broker_gateway_children", column, schema="quantlab")
