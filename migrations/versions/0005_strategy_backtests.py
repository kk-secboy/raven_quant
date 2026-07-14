"""Add immutable strategy versions and governed backtest runs."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005_strategy_backtests"
down_revision: str | None = "0004_active_research_run"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "strategies",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        schema="quantlab",
    )
    op.create_table(
        "strategy_versions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "strategy_id",
            sa.String(),
            sa.ForeignKey("quantlab.strategies.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("benchmark", sa.String(), nullable=False),
        sa.Column("universe", sa.String(), nullable=False),
        sa.Column("config_json", postgresql.JSONB(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("approved_by", sa.String()),
        sa.Column("approval_reason", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("approved_at", sa.DateTime(timezone=True)),
        schema="quantlab",
    )
    op.create_index(
        "uq_strategy_versions_number",
        "strategy_versions",
        ["strategy_id", "version"],
        unique=True,
        schema="quantlab",
    )
    op.create_index(
        "uq_strategy_versions_approved",
        "strategy_versions",
        ["strategy_id"],
        unique=True,
        schema="quantlab",
        postgresql_where=sa.text("status = 'approved'"),
    )
    op.create_table(
        "strategy_factors",
        sa.Column(
            "strategy_version_id",
            sa.String(),
            sa.ForeignKey("quantlab.strategy_versions.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "factor_candidate_id",
            sa.String(),
            sa.ForeignKey("quantlab.factor_candidates.id", ondelete="RESTRICT"),
            primary_key=True,
        ),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("direction", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        schema="quantlab",
    )
    op.create_table(
        "backtest_runs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("job_id", sa.String(), sa.ForeignKey("quantlab.jobs.id", ondelete="SET NULL")),
        sa.Column(
            "strategy_version_id",
            sa.String(),
            sa.ForeignKey("quantlab.strategy_versions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("periods_json", postgresql.JSONB(), nullable=False),
        sa.Column("metrics_json", postgresql.JSONB()),
        sa.Column("artifact_path", sa.Text()),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        schema="quantlab",
    )
    op.create_index(
        "idx_backtest_runs_status_created",
        "backtest_runs",
        ["status", sa.text("created_at DESC")],
        schema="quantlab",
    )
    op.create_table(
        "strategy_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "strategy_id",
            sa.String(),
            sa.ForeignKey("quantlab.strategies.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "strategy_version_id",
            sa.String(),
            sa.ForeignKey("quantlab.strategy_versions.id", ondelete="CASCADE"),
        ),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("actor", sa.String(), nullable=False),
        sa.Column("payload_json", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        schema="quantlab",
    )
    op.create_index(
        "idx_strategy_events_strategy_created",
        "strategy_events",
        ["strategy_id", "created_at"],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_strategy_events_strategy_created", table_name="strategy_events", schema="quantlab"
    )
    op.drop_table("strategy_events", schema="quantlab")
    op.drop_index("idx_backtest_runs_status_created", table_name="backtest_runs", schema="quantlab")
    op.drop_table("backtest_runs", schema="quantlab")
    op.drop_table("strategy_factors", schema="quantlab")
    op.drop_index(
        "uq_strategy_versions_approved", table_name="strategy_versions", schema="quantlab"
    )
    op.drop_index("uq_strategy_versions_number", table_name="strategy_versions", schema="quantlab")
    op.drop_table("strategy_versions", schema="quantlab")
    op.drop_table("strategies", schema="quantlab")
