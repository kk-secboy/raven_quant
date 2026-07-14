"""Persist immutable post-trade portfolio reviews."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0012_portfolio_reviews"
down_revision: str | None = "0011_execution_risk_state"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "portfolio_reviews",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("portfolio_id", sa.String(), nullable=False),
        sa.Column("batch_id", sa.String(), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("summary_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["portfolio_id"],
            ["quantlab.paper_portfolios.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["batch_id"],
            ["quantlab.portfolio_batches.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("batch_id"),
        schema="quantlab",
    )
    op.create_index(
        "idx_portfolio_reviews_portfolio_date",
        "portfolio_reviews",
        ["portfolio_id", "trade_date"],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_portfolio_reviews_portfolio_date",
        table_name="portfolio_reviews",
        schema="quantlab",
    )
    op.drop_table("portfolio_reviews", schema="quantlab")
