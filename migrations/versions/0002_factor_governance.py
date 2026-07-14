"""Add RD-Agent research runs and governed factor lifecycle."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_factor_governance"
down_revision: str | None = "0001_control_plane"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "research_runs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "job_id",
            sa.String(),
            sa.ForeignKey("quantlab.jobs.id", ondelete="SET NULL"),
        ),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("requested_by", sa.String(), nullable=False),
        sa.Column("budget_json", postgresql.JSONB(), nullable=False),
        sa.Column("config_json", postgresql.JSONB(), nullable=False),
        sa.Column("runtime_json", postgresql.JSONB()),
        sa.Column("artifact_path", sa.Text()),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        schema="quantlab",
    )
    op.create_index(
        "idx_research_runs_status_created",
        "research_runs",
        ["status", sa.text("created_at DESC")],
        schema="quantlab",
    )
    op.create_table(
        "factor_candidates",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "research_run_id",
            sa.String(),
            sa.ForeignKey("quantlab.research_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("formulation", sa.Text()),
        sa.Column("variables_json", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("source_iteration", sa.Integer()),
        sa.Column("code_path", sa.Text()),
        sa.Column("code_sha256", sa.String()),
        sa.Column("rdagent_decision", sa.Boolean()),
        sa.Column("rdagent_feedback", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        schema="quantlab",
    )
    op.create_index(
        "uq_factor_candidate_run_name",
        "factor_candidates",
        ["research_run_id", "name"],
        unique=True,
        schema="quantlab",
    )
    op.create_index(
        "idx_factor_candidates_status_updated",
        "factor_candidates",
        ["status", sa.text("updated_at DESC")],
        schema="quantlab",
    )
    op.create_table(
        "factor_evaluations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "factor_candidate_id",
            sa.String(),
            sa.ForeignKey("quantlab.factor_candidates.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("train_start", sa.Date(), nullable=False),
        sa.Column("train_end", sa.Date(), nullable=False),
        sa.Column("valid_start", sa.Date(), nullable=False),
        sa.Column("valid_end", sa.Date(), nullable=False),
        sa.Column("test_start", sa.Date(), nullable=False),
        sa.Column("test_end", sa.Date(), nullable=False),
        sa.Column("ic", sa.Float()),
        sa.Column("icir", sa.Float()),
        sa.Column("rank_ic", sa.Float()),
        sa.Column("rank_icir", sa.Float()),
        sa.Column("turnover", sa.Float()),
        sa.Column("max_correlation", sa.Float()),
        sa.Column("cost_adjusted_return", sa.Float()),
        sa.Column("metrics_json", postgresql.JSONB(), nullable=False),
        sa.Column("gate_status", sa.String(), nullable=False),
        sa.Column("gate_reasons_json", postgresql.JSONB(), nullable=False),
        sa.Column("evaluator_version", sa.String(), nullable=False),
        sa.Column("artifact_path", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        schema="quantlab",
    )
    op.create_index(
        "idx_factor_evaluations_candidate_created",
        "factor_evaluations",
        ["factor_candidate_id", sa.text("created_at DESC")],
        schema="quantlab",
    )
    op.create_table(
        "research_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "research_run_id",
            sa.String(),
            sa.ForeignKey("quantlab.research_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "factor_candidate_id",
            sa.String(),
            sa.ForeignKey("quantlab.factor_candidates.id", ondelete="CASCADE"),
        ),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("actor", sa.String(), nullable=False),
        sa.Column("payload_json", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        schema="quantlab",
    )
    op.create_index(
        "idx_research_events_run_created",
        "research_events",
        ["research_run_id", "created_at"],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_research_events_run_created", table_name="research_events", schema="quantlab"
    )
    op.drop_table("research_events", schema="quantlab")
    op.drop_index(
        "idx_factor_evaluations_candidate_created",
        table_name="factor_evaluations",
        schema="quantlab",
    )
    op.drop_table("factor_evaluations", schema="quantlab")
    op.drop_index(
        "idx_factor_candidates_status_updated",
        table_name="factor_candidates",
        schema="quantlab",
    )
    op.drop_index(
        "uq_factor_candidate_run_name",
        table_name="factor_candidates",
        schema="quantlab",
    )
    op.drop_table("factor_candidates", schema="quantlab")
    op.drop_index("idx_research_runs_status_created", table_name="research_runs", schema="quantlab")
    op.drop_table("research_runs", schema="quantlab")
