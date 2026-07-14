"""Persist provider-gateway parent orders and execution slices."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0016_qmt_gateway_runtime"
down_revision: str | None = "0015_broker_reconciliation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "broker_gateway_parents",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("client_order_id", sa.String(), nullable=False),
        sa.Column("account_ref", sa.String(), nullable=False),
        sa.Column("environment", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("payload_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("payload_sha256", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_error", sa.Text()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("client_order_id"),
        schema="quantlab",
    )
    op.create_index(
        "idx_broker_gateway_parents_status_updated",
        "broker_gateway_parents",
        ["status", sa.text("updated_at DESC")],
        schema="quantlab",
    )
    op.create_table(
        "broker_gateway_children",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("parent_id", sa.String(), nullable=False),
        sa.Column("slice_index", sa.Integer(), nullable=False),
        sa.Column("scheduled_for", sa.DateTime(timezone=True), nullable=False),
        sa.Column("quantity", sa.Numeric(20, 6), nullable=False),
        sa.Column("limit_price", sa.Numeric(20, 6), nullable=False),
        sa.Column("client_tag", sa.String(), nullable=False),
        sa.Column("provider_order_id", sa.String()),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("submitted_at", sa.DateTime(timezone=True)),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_error", sa.Text()),
        sa.ForeignKeyConstraint(
            ["parent_id"], ["quantlab.broker_gateway_parents.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("client_tag"),
        sa.UniqueConstraint("parent_id", "slice_index", name="uq_broker_gateway_parent_slice"),
        schema="quantlab",
    )
    op.create_index(
        "idx_broker_gateway_children_due",
        "broker_gateway_children",
        ["status", "scheduled_for"],
        schema="quantlab",
    )
    op.create_table(
        "broker_gateway_nonces",
        sa.Column("nonce", sa.String(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("nonce"),
        schema="quantlab",
    )
    op.create_index(
        "idx_broker_gateway_nonces_expiry",
        "broker_gateway_nonces",
        ["expires_at"],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_broker_gateway_nonces_expiry",
        table_name="broker_gateway_nonces",
        schema="quantlab",
    )
    op.drop_table("broker_gateway_nonces", schema="quantlab")
    op.drop_index(
        "idx_broker_gateway_children_due",
        table_name="broker_gateway_children",
        schema="quantlab",
    )
    op.drop_table("broker_gateway_children", schema="quantlab")
    op.drop_index(
        "idx_broker_gateway_parents_status_updated",
        table_name="broker_gateway_parents",
        schema="quantlab",
    )
    op.drop_table("broker_gateway_parents", schema="quantlab")
