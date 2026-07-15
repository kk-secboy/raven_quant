"""Add independent factor recomputation evidence and recommendation state."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0035_research_policy_v2"
down_revision: str | None = "0034_legacy_readonly"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(JSONB(), "postgresql")
    op.add_column(
        "jobs",
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        schema="quantlab",
    )
    op.add_column(
        "jobs",
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="1"),
        schema="quantlab",
    )
    op.add_column(
        "jobs", sa.Column("next_attempt_at", sa.DateTime(timezone=True)), schema="quantlab"
    )
    op.alter_column("jobs", "attempts", server_default=None, schema="quantlab")
    op.alter_column("jobs", "max_attempts", server_default=None, schema="quantlab")
    op.drop_index("idx_jobs_status_created", table_name="jobs", schema="quantlab")
    op.create_index(
        "idx_jobs_status_created",
        "jobs",
        ["status", "next_attempt_at", sa.text("created_at DESC")],
        schema="quantlab",
    )
    op.add_column(
        "factor_evaluations", sa.Column("submitted_values_sha256", sa.String()), schema="quantlab"
    )
    op.add_column(
        "factor_evaluations", sa.Column("recomputed_values_sha256", sa.String()), schema="quantlab"
    )
    op.add_column(
        "factor_evaluations", sa.Column("recompute_evidence_json", json_type), schema="quantlab"
    )
    op.add_column(
        "recommendation_holdings", sa.Column("average_cost", sa.Float()), schema="quantlab"
    )
    op.add_column(
        "recommendation_portfolios",
        sa.Column("risk_exposure_override", sa.Float(), nullable=False, server_default="1"),
        schema="quantlab",
    )
    op.alter_column(
        "recommendation_portfolios",
        "risk_exposure_override",
        server_default=None,
        schema="quantlab",
    )
    op.add_column(
        "recommendation_holdings",
        sa.Column("take_profit_stage", sa.Integer(), nullable=False, server_default="0"),
        schema="quantlab",
    )
    op.alter_column(
        "recommendation_holdings", "take_profit_stage", server_default=None, schema="quantlab"
    )
    op.add_column(
        "strategy_allocation_members",
        sa.Column("recommendation_portfolio_id", sa.String()),
        schema="quantlab",
    )
    op.create_foreign_key(
        "fk_strategy_allocation_members_recommendation_portfolio",
        "strategy_allocation_members",
        "recommendation_portfolios",
        ["recommendation_portfolio_id"],
        ["id"],
        source_schema="quantlab",
        referent_schema="quantlab",
        ondelete="RESTRICT",
    )
    op.create_unique_constraint(
        "uq_strategy_allocation_members_recommendation_portfolio",
        "strategy_allocation_members",
        ["recommendation_portfolio_id"],
        schema="quantlab",
    )
    op.add_column(
        "strategy_allocations",
        sa.Column("is_legacy", sa.Boolean(), nullable=False, server_default=sa.true()),
        schema="quantlab",
    )
    op.alter_column(
        "strategy_allocations", "is_legacy", server_default=sa.false(), schema="quantlab"
    )
    op.add_column(
        "strategy_allocation_events",
        sa.Column("recommendation_portfolio_id", sa.String()),
        schema="quantlab",
    )
    op.create_foreign_key(
        "fk_strategy_allocation_events_recommendation_portfolio",
        "strategy_allocation_events",
        "recommendation_portfolios",
        ["recommendation_portfolio_id"],
        ["id"],
        source_schema="quantlab",
        referent_schema="quantlab",
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_strategy_allocation_events_recommendation_portfolio",
        "strategy_allocation_events",
        schema="quantlab",
        type_="foreignkey",
    )
    op.drop_column("strategy_allocation_events", "recommendation_portfolio_id", schema="quantlab")
    op.drop_column("strategy_allocations", "is_legacy", schema="quantlab")
    op.drop_constraint(
        "uq_strategy_allocation_members_recommendation_portfolio",
        "strategy_allocation_members",
        schema="quantlab",
        type_="unique",
    )
    op.drop_constraint(
        "fk_strategy_allocation_members_recommendation_portfolio",
        "strategy_allocation_members",
        schema="quantlab",
        type_="foreignkey",
    )
    op.drop_column(
        "strategy_allocation_members", "recommendation_portfolio_id", schema="quantlab"
    )
    op.drop_column("recommendation_holdings", "take_profit_stage", schema="quantlab")
    op.drop_column("recommendation_holdings", "average_cost", schema="quantlab")
    op.drop_column("recommendation_portfolios", "risk_exposure_override", schema="quantlab")
    op.drop_column("factor_evaluations", "recompute_evidence_json", schema="quantlab")
    op.drop_column("factor_evaluations", "recomputed_values_sha256", schema="quantlab")
    op.drop_column("factor_evaluations", "submitted_values_sha256", schema="quantlab")
    op.drop_index("idx_jobs_status_created", table_name="jobs", schema="quantlab")
    op.create_index(
        "idx_jobs_status_created",
        "jobs",
        ["status", sa.text("created_at DESC")],
        schema="quantlab",
    )
    op.drop_column("jobs", "next_attempt_at", schema="quantlab")
    op.drop_column("jobs", "max_attempts", schema="quantlab")
    op.drop_column("jobs", "attempts", schema="quantlab")
