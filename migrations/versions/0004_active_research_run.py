"""Prevent overlapping research pipelines of the same kind."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_active_research_run"
down_revision: str | None = "0003_factor_values"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "uq_research_runs_active_kind",
        "research_runs",
        ["kind"],
        unique=True,
        schema="quantlab",
        postgresql_where=sa.text("status IN ('queued', 'running', 'evaluating')"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_research_runs_active_kind",
        table_name="research_runs",
        schema="quantlab",
    )
