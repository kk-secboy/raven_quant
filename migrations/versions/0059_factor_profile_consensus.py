"""Persist explicit multi-profile factor admission evidence."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0059_factor_profile_consensus"
down_revision: str | None = "0058_simulation_benchmark"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(JSONB(), "postgresql")
    op.add_column(
        "factor_candidates",
        sa.Column("profile_consensus_json", json_type),
        schema="quantlab",
    )
    op.add_column(
        "factor_candidates",
        sa.Column("profile_consensus_sha256", sa.String()),
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_column("factor_candidates", "profile_consensus_sha256", schema="quantlab")
    op.drop_column("factor_candidates", "profile_consensus_json", schema="quantlab")
