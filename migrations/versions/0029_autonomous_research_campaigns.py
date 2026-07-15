"""Add durable autonomous research campaigns and audit events."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0029_research_campaigns"
down_revision: str | None = "0028_parameter_experiments"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "research_campaigns",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("stage", sa.String(), nullable=False),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("benchmark", sa.String(), nullable=False),
        sa.Column("universe", sa.String(), nullable=False),
        sa.Column("recipe_id", sa.String(), nullable=False),
        sa.Column("config_json", postgresql.JSONB(), nullable=False),
        sa.Column("state_json", postgresql.JSONB(), nullable=False),
        sa.Column(
            "research_run_id",
            sa.String(),
            sa.ForeignKey("quantlab.research_runs.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "strategy_id",
            sa.String(),
            sa.ForeignKey("quantlab.strategies.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "strategy_version_id",
            sa.String(),
            sa.ForeignKey("quantlab.strategy_versions.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "parameter_experiment_id",
            sa.String(),
            sa.ForeignKey("quantlab.parameter_experiments.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "backtest_id",
            sa.String(),
            sa.ForeignKey("quantlab.backtest_runs.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "paper_portfolio_id",
            sa.String(),
            sa.ForeignKey("quantlab.paper_portfolios.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "paper_schedule_id",
            sa.String(),
            sa.ForeignKey("quantlab.schedules.id", ondelete="SET NULL"),
        ),
        sa.Column("error", sa.Text()),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("next_action_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_research_campaigns_name"),
        schema="quantlab",
    )
    op.create_index(
        "idx_research_campaigns_claim",
        "research_campaigns",
        ["status", "next_action_at", "lease_until"],
        schema="quantlab",
    )
    op.create_table(
        "research_campaign_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column(
            "campaign_id",
            sa.String(),
            sa.ForeignKey("quantlab.research_campaigns.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("actor", sa.String(), nullable=False),
        sa.Column("payload_json", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        schema="quantlab",
    )
    op.create_index(
        "idx_research_campaign_events_created",
        "research_campaign_events",
        ["campaign_id", "created_at"],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_research_campaign_events_created",
        table_name="research_campaign_events",
        schema="quantlab",
    )
    op.drop_table("research_campaign_events", schema="quantlab")
    op.drop_index(
        "idx_research_campaigns_claim",
        table_name="research_campaigns",
        schema="quantlab",
    )
    op.drop_table("research_campaigns", schema="quantlab")
