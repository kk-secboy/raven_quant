"""Add governed data-lineage rollover evidence to paper ledgers."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0025_data_lineage_rollover"
down_revision: str | None = "0024_pair_paper_ledger"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "paper_portfolios",
        sa.Column("dataset_roll_policy", sa.String(), nullable=False, server_default="pinned"),
        schema="quantlab",
    )
    op.add_column(
        "paper_portfolios",
        sa.Column("dataset_lineage_id", sa.String()),
        schema="quantlab",
    )
    for column in (
        sa.Column("dataset", sa.String()),
        sa.Column("dataset_identity_sha256", sa.String()),
        sa.Column("dataset_lineage_id", sa.String()),
    ):
        op.add_column("portfolio_batches", column, schema="quantlab")

    op.add_column(
        "pair_paper_portfolios",
        sa.Column("dataset_roll_policy", sa.String(), nullable=False, server_default="pinned"),
        schema="quantlab",
    )
    op.add_column(
        "pair_paper_portfolios",
        sa.Column("dataset_lineage_id", sa.String()),
        schema="quantlab",
    )
    op.add_column(
        "pair_paper_portfolios",
        sa.Column("execution_roll_policy", sa.String(), nullable=False, server_default="pinned"),
        schema="quantlab",
    )
    op.add_column(
        "pair_paper_portfolios",
        sa.Column("execution_lineage_id", sa.String()),
        schema="quantlab",
    )
    for column in (
        sa.Column("dataset", sa.String()),
        sa.Column("dataset_identity_sha256", sa.String()),
        sa.Column("dataset_lineage_id", sa.String()),
        sa.Column("execution_snapshot", sa.String()),
        sa.Column("execution_manifest_sha256", sa.String()),
        sa.Column("execution_lineage_id", sa.String()),
    ):
        op.add_column("pair_portfolio_batches", column, schema="quantlab")


def downgrade() -> None:
    for name in (
        "execution_lineage_id",
        "execution_manifest_sha256",
        "execution_snapshot",
        "dataset_lineage_id",
        "dataset_identity_sha256",
        "dataset",
    ):
        op.drop_column("pair_portfolio_batches", name, schema="quantlab")
    for name in (
        "execution_lineage_id",
        "execution_roll_policy",
        "dataset_lineage_id",
        "dataset_roll_policy",
    ):
        op.drop_column("pair_paper_portfolios", name, schema="quantlab")
    for name in ("dataset_lineage_id", "dataset_identity_sha256", "dataset"):
        op.drop_column("portfolio_batches", name, schema="quantlab")
    op.drop_column("paper_portfolios", "dataset_lineage_id", schema="quantlab")
    op.drop_column("paper_portfolios", "dataset_roll_policy", schema="quantlab")
