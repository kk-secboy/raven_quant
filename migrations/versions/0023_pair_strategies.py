"""Add governed statistical-arbitrage pair strategy versions."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0023_pair_strategies"
down_revision: str | None = "0022_allocation_schedule_groups"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "strategy_versions",
        sa.Column(
            "strategy_type",
            sa.String(),
            nullable=False,
            server_default="multifactor",
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_strategy_versions_type_status",
        "strategy_versions",
        ["strategy_type", "status"],
        schema="quantlab",
    )
    op.create_table(
        "strategy_pairs",
        sa.Column(
            "strategy_version_id",
            sa.String(),
            sa.ForeignKey("quantlab.strategy_versions.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("leg_y", sa.String(), nullable=False),
        sa.Column("leg_x", sa.String(), nullable=False),
        sa.Column("asset_class", sa.String(), nullable=False),
        sa.Column("shorting_mode", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("leg_y <> leg_x", name="ck_strategy_pairs_distinct_legs"),
        schema="quantlab",
    )
    op.add_column(
        "backtest_runs",
        sa.Column("execution_dataset", sa.String()),
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_column("backtest_runs", "execution_dataset", schema="quantlab")
    op.drop_table("strategy_pairs", schema="quantlab")
    op.drop_index(
        "idx_strategy_versions_type_status",
        table_name="strategy_versions",
        schema="quantlab",
    )
    op.drop_column("strategy_versions", "strategy_type", schema="quantlab")
