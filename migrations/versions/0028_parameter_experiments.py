"""Add durable parameter experiments and trial evidence."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0028_parameter_experiments"
down_revision: str | None = "0027_web_config_templates"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "parameter_experiments",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column(
            "job_id",
            sa.String(),
            sa.ForeignKey("quantlab.jobs.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "strategy_version_id",
            sa.String(),
            sa.ForeignKey("quantlab.strategy_versions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("periods_json", postgresql.JSONB(), nullable=False),
        sa.Column("parameter_grid_json", postgresql.JSONB(), nullable=False),
        sa.Column("baseline_config_json", postgresql.JSONB(), nullable=False),
        sa.Column("summary_json", postgresql.JSONB()),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.PrimaryKeyConstraint("id"),
        schema="quantlab",
    )
    op.create_index(
        "idx_parameter_experiments_status_created",
        "parameter_experiments",
        ["status", sa.text("created_at DESC")],
        schema="quantlab",
    )
    op.create_table(
        "parameter_experiment_trials",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column(
            "experiment_id",
            sa.String(),
            sa.ForeignKey("quantlab.parameter_experiments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("trial_index", sa.Integer(), nullable=False),
        sa.Column("parameters_json", postgresql.JSONB(), nullable=False),
        sa.Column("config_json", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("score", sa.Float()),
        sa.Column("metrics_json", postgresql.JSONB()),
        sa.Column("warnings_json", postgresql.JSONB()),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "experiment_id", "trial_index", name="uq_parameter_experiment_trial"
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_parameter_experiment_trials_status",
        "parameter_experiment_trials",
        ["experiment_id", "status", "trial_index"],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_parameter_experiment_trials_status",
        table_name="parameter_experiment_trials",
        schema="quantlab",
    )
    op.drop_table("parameter_experiment_trials", schema="quantlab")
    op.drop_index(
        "idx_parameter_experiments_status_created",
        table_name="parameter_experiments",
        schema="quantlab",
    )
    op.drop_table("parameter_experiments", schema="quantlab")
