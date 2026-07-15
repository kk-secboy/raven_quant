"""Add cooperative cancellation to durable worker jobs."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0032_job_cancellation"
down_revision: str | None = "0031_research_governance"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "jobs",
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True)),
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_column("jobs", "cancel_requested_at", schema="quantlab")
