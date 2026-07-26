"""Paper stage and forward evidence gate (design draft 4.5/6.11/9.5).

Lifecycle promotion stage on strategy versions: ``candidate`` (NULL,
pre-gate/legacy) -> ``paper`` (set automatically when the formal hard gate
approves the version) -> ``recommendation_enabled`` (forward evidence gate +
human approval). ``strategy_promotion_stages`` records each isolated forward
paper stage — its own simulation account, contract hash and evidence scope —
so a substantive contract change freezes the old stage read-only and starts
evidence from zero instead of concatenating (design 9.5).
``strategy_forward_gates`` carries the per-version pre-registered forward
gate thresholds; a gate may only be registered/changed before the version
enters paper.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0043_promotion_stages"
down_revision: str | None = "0042_account_netting_plans"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "strategy_versions",
        sa.Column("promotion_stage", sa.String(), nullable=True),
        schema="quantlab",
    )
    op.create_table(
        "strategy_forward_gates",
        sa.Column(
            "strategy_version_id",
            sa.String(),
            sa.ForeignKey("quantlab.strategy_versions.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("min_forward_calendar_days", sa.Integer(), nullable=False),
        sa.Column("min_decision_batches", sa.Integer(), nullable=False),
        sa.Column("min_completed_cycles", sa.Integer(), nullable=False),
        sa.Column("min_data_completeness", sa.Float(), nullable=False),
        sa.Column("min_reconciliation_rate", sa.Float(), nullable=False),
        sa.Column("max_cost_deviation", sa.Float(), nullable=False),
        sa.Column("registered_by", sa.String(), nullable=False),
        sa.Column("registered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        schema="quantlab",
    )
    op.create_table(
        "strategy_promotion_stages",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "strategy_version_id",
            sa.String(),
            sa.ForeignKey("quantlab.strategy_versions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stage_index", sa.Integer(), nullable=False),
        sa.Column(
            "simulation_portfolio_id",
            sa.String(),
            sa.ForeignKey("quantlab.simulation_portfolios.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("source_contract_hash", sa.String(), nullable=True),
        sa.Column("initial_cash", sa.Numeric(20, 6), nullable=True),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("frozen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("freeze_reason", sa.Text(), nullable=True),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.UniqueConstraint(
            "strategy_version_id",
            "stage_index",
            name="uq_strategy_promotion_stages_index",
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_strategy_promotion_stages_version",
        "strategy_promotion_stages",
        ["strategy_version_id", "status"],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_strategy_promotion_stages_version",
        table_name="strategy_promotion_stages",
        schema="quantlab",
    )
    op.drop_table("strategy_promotion_stages", schema="quantlab")
    op.drop_table("strategy_forward_gates", schema="quantlab")
    op.drop_column("strategy_versions", "promotion_stage", schema="quantlab")
