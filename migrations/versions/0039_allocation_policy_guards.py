"""Single-active guards and allocation decision artifacts.

Design draft 6.10/8.1/9.1: at most one active strategy allocation and at most
one active standalone recommendation portfolio (the single account allowed to
send final advice) at any moment; allocation policies only re-solve member
budgets on frozen decision days, recorded as allocation artifacts with
inputs_as_of/valid_until.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0039_allocation_policy_guards"
down_revision: str | None = "0038_corporate_actions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "strategy_allocations",
        sa.Column(
            "decision_frequency",
            sa.String(),
            nullable=False,
            server_default="monthly",
        ),
        schema="quantlab",
    )
    op.create_table(
        "strategy_allocation_artifacts",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("allocation_id", sa.String(), nullable=False),
        sa.Column("decision_date", sa.Date(), nullable=False),
        sa.Column("inputs_as_of", sa.Date(), nullable=False),
        sa.Column("valid_until", sa.Date(), nullable=False),
        sa.Column("member_weights_json", sa.JSON(), nullable=False),
        sa.Column("analysis_json", sa.JSON(), nullable=False),
        sa.Column("artifact_hash", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["allocation_id"],
            ["quantlab.strategy_allocations.id"],
            ondelete="CASCADE",
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_strategy_allocation_artifacts_allocation",
        "strategy_allocation_artifacts",
        ["allocation_id", sa.text("decision_date DESC")],
        schema="quantlab",
    )
    # Recommendation portfolios created via RecommendationStore are standalone
    # sender candidates; portfolios owned by allocation members are structural
    # sub-accounts and never the unique sender.
    op.add_column(
        "recommendation_portfolios",
        sa.Column(
            "recommendation_scope",
            sa.String(),
            nullable=False,
            server_default="standalone",
        ),
        schema="quantlab",
    )
    op.execute(
        """
        UPDATE quantlab.recommendation_portfolios AS portfolio
        SET recommendation_scope = 'allocation_member'
        FROM quantlab.strategy_allocation_members AS member
        WHERE member.recommendation_portfolio_id = portfolio.id
        """
    )
    # Fail-closed uniqueness must not break on historical duplicates: keep the
    # most recently updated active row and demote the rest to paused.
    op.execute(
        """
        UPDATE quantlab.strategy_allocations AS stale
        SET status = 'paused', updated_at = stale.updated_at
        WHERE stale.status = 'active'
          AND stale.id <> (
              SELECT keep.id FROM quantlab.strategy_allocations AS keep
              WHERE keep.status = 'active'
              ORDER BY keep.updated_at DESC, keep.id
              LIMIT 1
          )
        """
    )
    op.execute(
        """
        UPDATE quantlab.recommendation_portfolios AS stale
        SET status = 'paused', updated_at = stale.updated_at
        WHERE stale.status = 'active'
          AND stale.recommendation_scope = 'standalone'
          AND stale.id <> (
              SELECT keep.id FROM quantlab.recommendation_portfolios AS keep
              WHERE keep.status = 'active'
                AND keep.recommendation_scope = 'standalone'
              ORDER BY keep.updated_at DESC, keep.id
              LIMIT 1
          )
        """
    )
    op.create_index(
        "uq_strategy_allocations_single_active",
        "strategy_allocations",
        ["status"],
        unique=True,
        schema="quantlab",
        postgresql_where=sa.text("status = 'active'"),
    )
    op.create_index(
        "uq_recommendation_portfolios_single_active_sender",
        "recommendation_portfolios",
        ["status"],
        unique=True,
        schema="quantlab",
        postgresql_where=sa.text(
            "status = 'active' AND recommendation_scope = 'standalone'"
        ),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_recommendation_portfolios_single_active_sender",
        table_name="recommendation_portfolios",
        schema="quantlab",
    )
    op.drop_index(
        "uq_strategy_allocations_single_active",
        table_name="strategy_allocations",
        schema="quantlab",
    )
    op.drop_column("recommendation_portfolios", "recommendation_scope", schema="quantlab")
    op.drop_index(
        "idx_strategy_allocation_artifacts_allocation",
        table_name="strategy_allocation_artifacts",
        schema="quantlab",
    )
    op.drop_table("strategy_allocation_artifacts", schema="quantlab")
    op.drop_column("strategy_allocations", "decision_frequency", schema="quantlab")
