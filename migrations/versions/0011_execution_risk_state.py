"""Persist staged paper-execution risk state."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011_execution_risk_state"
down_revision: str | None = "0010_paper_risk_controls"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "paper_positions",
        sa.Column("take_profit_stage", sa.Integer(), server_default="0", nullable=False),
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_column("paper_positions", "take_profit_stage", schema="quantlab")
