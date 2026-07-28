"""Bind forward runs to immutable descendant datasets."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0057_forward_rollover"
down_revision: str | None = "0056_day_attributions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "recommendation_portfolios",
        sa.Column(
            "dataset_roll_policy",
            sa.String(),
            nullable=False,
            server_default="pinned",
        ),
        schema="quantlab",
    )
    op.add_column(
        "recommendation_portfolios",
        sa.Column("dataset_lineage_id", sa.String()),
        schema="quantlab",
    )
    op.add_column(
        "recommendation_snapshots",
        sa.Column("dataset_lineage_id", sa.String()),
        schema="quantlab",
    )
    op.add_column(
        "simulation_portfolios",
        sa.Column(
            "daily_roll_policy",
            sa.String(),
            nullable=False,
            server_default="pinned",
        ),
        schema="quantlab",
    )
    op.add_column(
        "simulation_portfolios",
        sa.Column(
            "execution_roll_policy",
            sa.String(),
            nullable=False,
            server_default="pinned",
        ),
        schema="quantlab",
    )
    batch_columns = (
        "daily_dataset",
        "daily_dataset_identity_sha256",
        "daily_dataset_lineage_id",
        "execution_dataset",
        "execution_dataset_identity_sha256",
        "execution_dataset_lineage_id",
        "simulation_semantics_sha256",
    )
    for name in batch_columns:
        op.add_column(
            "simulation_batches",
            sa.Column(name, sa.String()),
            schema="quantlab",
        )
    op.execute(
        """
        UPDATE quantlab.simulation_batches AS batch
        SET daily_dataset = portfolio.daily_dataset,
            daily_dataset_identity_sha256 = portfolio.daily_dataset_identity_sha256,
            daily_dataset_lineage_id = portfolio.daily_dataset_lineage_id,
            execution_dataset = portfolio.execution_dataset,
            execution_dataset_identity_sha256 =
                portfolio.execution_dataset_identity_sha256,
            execution_dataset_lineage_id = portfolio.execution_dataset_lineage_id,
            simulation_semantics_sha256 =
                COALESCE(
                    portfolio.execution_policy_json ->> 'simulation_semantics_sha256',
                    'legacy-unbound-requires-replan'
                )
        FROM quantlab.simulation_portfolios AS portfolio
        WHERE portfolio.id = batch.portfolio_id
        """
    )
    for name in batch_columns:
        op.alter_column(
            "simulation_batches",
            name,
            nullable=False,
            schema="quantlab",
        )


def downgrade() -> None:
    for name in (
        "execution_dataset_lineage_id",
        "execution_dataset_identity_sha256",
        "execution_dataset",
        "simulation_semantics_sha256",
        "daily_dataset_lineage_id",
        "daily_dataset_identity_sha256",
        "daily_dataset",
    ):
        op.drop_column("simulation_batches", name, schema="quantlab")
    op.drop_column(
        "simulation_portfolios", "execution_roll_policy", schema="quantlab"
    )
    op.drop_column("simulation_portfolios", "daily_roll_policy", schema="quantlab")
    op.drop_column(
        "recommendation_snapshots", "dataset_lineage_id", schema="quantlab"
    )
    op.drop_column(
        "recommendation_portfolios", "dataset_lineage_id", schema="quantlab"
    )
    op.drop_column(
        "recommendation_portfolios", "dataset_roll_policy", schema="quantlab"
    )
