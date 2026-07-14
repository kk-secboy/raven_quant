"""Add durable schedules, schedule runs, alert delivery, and job idempotency."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007_scheduler_alerts"
down_revision: str | None = "0006_paper_portfolios"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index("uq_jobs_active_kind", table_name="jobs", schema="quantlab")
    op.add_column("jobs", sa.Column("idempotency_key", sa.String()), schema="quantlab")
    op.create_unique_constraint(
        "uq_jobs_idempotency_key",
        "jobs",
        ["idempotency_key"],
        schema="quantlab",
    )
    op.create_table(
        "schedules",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False, unique=True),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("timezone", sa.String(), nullable=False),
        sa.Column("run_time", sa.Time(), nullable=False),
        sa.Column("trading_days_only", sa.Boolean(), nullable=False),
        sa.Column("payload_json", postgresql.JSONB(), nullable=False),
        sa.Column("misfire_grace_seconds", sa.Integer(), nullable=False),
        sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_run_at", sa.DateTime(timezone=True)),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        schema="quantlab",
    )
    op.create_index(
        "idx_schedules_due",
        "schedules",
        ["status", "next_run_at"],
        schema="quantlab",
    )
    op.create_table(
        "schedule_runs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "schedule_id",
            sa.String(),
            sa.ForeignKey("quantlab.schedules.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("job_id", sa.String(), sa.ForeignKey("quantlab.jobs.id", ondelete="SET NULL")),
        sa.Column("scheduled_for", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("dedupe_key", sa.String(), nullable=False, unique=True),
        sa.Column("message", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("schedule_id", "scheduled_for", name="uq_schedule_runs_slot"),
        schema="quantlab",
    )
    op.create_index(
        "idx_schedule_runs_created",
        "schedule_runs",
        [sa.text("created_at DESC")],
        schema="quantlab",
    )
    op.create_table(
        "alerts",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("source_type", sa.String(), nullable=False),
        sa.Column("source_id", sa.String(), nullable=False),
        sa.Column("severity", sa.String(), nullable=False),
        sa.Column("category", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("dedupe_key", sa.String(), nullable=False, unique=True),
        sa.Column("details_json", postgresql.JSONB(), nullable=False),
        sa.Column("delivery_status", sa.String(), nullable=False),
        sa.Column("delivery_attempts", sa.Integer(), nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True)),
        sa.Column("last_delivery_error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("acknowledged_by", sa.String()),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True)),
        sa.Column("resolved_by", sa.String()),
        sa.Column("resolved_at", sa.DateTime(timezone=True)),
        schema="quantlab",
    )
    op.create_index(
        "idx_alerts_status_created",
        "alerts",
        ["status", sa.text("created_at DESC")],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index("idx_alerts_status_created", table_name="alerts", schema="quantlab")
    op.drop_table("alerts", schema="quantlab")
    op.drop_index("idx_schedule_runs_created", table_name="schedule_runs", schema="quantlab")
    op.drop_table("schedule_runs", schema="quantlab")
    op.drop_index("idx_schedules_due", table_name="schedules", schema="quantlab")
    op.drop_table("schedules", schema="quantlab")
    op.drop_constraint("uq_jobs_idempotency_key", "jobs", schema="quantlab", type_="unique")
    op.drop_column("jobs", "idempotency_key", schema="quantlab")
    op.create_index(
        "uq_jobs_active_kind",
        "jobs",
        ["kind"],
        unique=True,
        schema="quantlab",
        postgresql_where=sa.text("status IN ('queued', 'running')"),
    )
