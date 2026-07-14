"""Bind sandbox destinations to portfolios and persist broker reconciliation."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0015_broker_reconciliation"
down_revision: str | None = "0014_broker_execution_boundary"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "broker_destinations",
        sa.Column("portfolio_id", sa.String(), nullable=True),
        schema="quantlab",
    )
    op.create_foreign_key(
        "fk_broker_destinations_portfolio",
        "broker_destinations",
        "paper_portfolios",
        ["portfolio_id"],
        ["id"],
        source_schema="quantlab",
        referent_schema="quantlab",
        ondelete="RESTRICT",
    )
    op.create_index(
        "uq_broker_destinations_portfolio",
        "broker_destinations",
        ["portfolio_id"],
        unique=True,
        schema="quantlab",
        postgresql_where=sa.text("portfolio_id IS NOT NULL"),
    )
    op.create_table(
        "broker_reconciliations",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("destination_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("broker_as_of", sa.DateTime(timezone=True)),
        sa.Column("expected_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("observed_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("differences_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["destination_id"], ["quantlab.broker_destinations.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        schema="quantlab",
    )
    op.create_index(
        "idx_broker_reconciliations_destination_created",
        "broker_reconciliations",
        ["destination_id", "created_at"],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_broker_reconciliations_destination_created",
        table_name="broker_reconciliations",
        schema="quantlab",
    )
    op.drop_table("broker_reconciliations", schema="quantlab")
    op.drop_index(
        "uq_broker_destinations_portfolio",
        table_name="broker_destinations",
        schema="quantlab",
    )
    op.drop_constraint(
        "fk_broker_destinations_portfolio",
        "broker_destinations",
        schema="quantlab",
        type_="foreignkey",
    )
    op.drop_column("broker_destinations", "portfolio_id", schema="quantlab")
