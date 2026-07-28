"""Shared hypothesis trial counts and capital caps (design 6.8/6.10).

The group lives on the strategy family, so a new version, frequency or model
wrapper cannot silently reset its experiment history. Allocation members copy
the frozen group evidence used by the decision artifact, making later refresh
and audit independent of mutable request payloads.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0053_hypothesis_group_governance"
down_revision: str | None = "0052_model_artifacts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "strategies",
        sa.Column("economic_hypothesis_group", sa.String(), nullable=True),
        schema="quantlab",
    )
    op.add_column(
        "strategies",
        sa.Column("hypothesis_group_cap", sa.Float(), nullable=True),
        schema="quantlab",
    )
    op.execute(
        "UPDATE quantlab.strategies "
        "SET economic_hypothesis_group = id, hypothesis_group_cap = 0.70"
    )
    op.alter_column(
        "strategies",
        "economic_hypothesis_group",
        nullable=False,
        server_default="legacy-unclassified",
        schema="quantlab",
    )
    op.alter_column(
        "strategies",
        "hypothesis_group_cap",
        nullable=False,
        server_default="0.70",
        schema="quantlab",
    )

    op.add_column(
        "strategy_allocation_members",
        sa.Column("economic_hypothesis_group", sa.String(), nullable=True),
        schema="quantlab",
    )
    op.add_column(
        "strategy_allocation_members",
        sa.Column("hypothesis_group_cap", sa.Float(), nullable=True),
        schema="quantlab",
    )
    op.add_column(
        "strategy_allocation_members",
        sa.Column("shared_experiment_count", sa.Integer(), nullable=True),
        schema="quantlab",
    )
    op.execute(
        "UPDATE quantlab.strategy_allocation_members "
        "SET economic_hypothesis_group = strategy_version_id, "
        "hypothesis_group_cap = member_cap, shared_experiment_count = 1"
    )
    op.alter_column(
        "strategy_allocation_members",
        "economic_hypothesis_group",
        nullable=False,
        server_default="legacy-unclassified",
        schema="quantlab",
    )
    op.alter_column(
        "strategy_allocation_members",
        "hypothesis_group_cap",
        nullable=False,
        server_default="0.70",
        schema="quantlab",
    )
    op.alter_column(
        "strategy_allocation_members",
        "shared_experiment_count",
        nullable=False,
        server_default="1",
        schema="quantlab",
    )


def downgrade() -> None:
    for column in (
        "shared_experiment_count",
        "hypothesis_group_cap",
        "economic_hypothesis_group",
    ):
        op.drop_column("strategy_allocation_members", column, schema="quantlab")
    op.drop_column("strategies", "hypothesis_group_cap", schema="quantlab")
    op.drop_column("strategies", "economic_hypothesis_group", schema="quantlab")
