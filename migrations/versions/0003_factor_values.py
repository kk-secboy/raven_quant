"""Store durable RD-Agent factor value artifacts."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_factor_values"
down_revision: str | None = "0002_factor_governance"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "factor_candidates",
        sa.Column("values_path", sa.Text()),
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_column("factor_candidates", "values_path", schema="quantlab")
