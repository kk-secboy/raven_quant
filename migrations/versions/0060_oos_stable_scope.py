"""Bind final OOS vintages to stable research and dataset-lineage scopes."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0060_oos_stable_scope"
down_revision: str | None = "0059_factor_profile_consensus"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "oos_vintages",
        sa.Column("dataset_lineage_id", sa.String()),
        schema="quantlab",
    )
    op.create_index(
        "idx_oos_vintage_scope_window",
        "oos_vintages",
        ["scope", "test_start", "test_end"],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_oos_vintage_scope_window",
        table_name="oos_vintages",
        schema="quantlab",
    )
    op.drop_column("oos_vintages", "dataset_lineage_id", schema="quantlab")
