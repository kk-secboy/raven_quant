"""Persist point-in-time industry evidence on paper positions."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010_paper_risk_controls"
down_revision: str | None = "0009_strategy_factor_evidence"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "paper_positions",
        sa.Column("industry", sa.String(), nullable=True),
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_column("paper_positions", "industry", schema="quantlab")
