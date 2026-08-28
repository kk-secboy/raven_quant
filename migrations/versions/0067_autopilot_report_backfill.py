"""Add recoverable autopilot cycles and research-report backfill.

Revision ID: 0067_autopilot_report_backfill
Revises: 0066_factor_library_sota
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0067_autopilot_report_backfill"
down_revision: str | None = "0066_factor_library_sota"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON = sa.JSON().with_variant(JSONB(), "postgresql")
SCHEMA = "quantlab"


def upgrade() -> None:
    op.create_table(
        "autopilot_cycles",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("dataset_identity_sha256", sa.String(), nullable=False, unique=True),
        sa.Column("dataset_lineage_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("stage", sa.String(), nullable=False),
        sa.Column("config_revision", sa.Integer(), nullable=False),
        sa.Column("state_json", JSON, nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "status IN ('active', 'blocked', 'succeeded', 'paused')",
            name="ck_autopilot_cycles_status",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_autopilot_cycles_status_updated",
        "autopilot_cycles",
        ["status", sa.text("updated_at DESC")],
        schema=SCHEMA,
    )
    op.create_table(
        "autopilot_branches",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "cycle_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.autopilot_cycles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("scenario", sa.String(), nullable=False),
        sa.Column("scope_key", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column(
            "research_run_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.research_runs.id", ondelete="SET NULL"),
        ),
        sa.Column("job_id", sa.String(), sa.ForeignKey(f"{SCHEMA}.jobs.id", ondelete="SET NULL")),
        sa.Column("details_json", JSON, nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("cycle_id", "scenario", "scope_key", name="uq_autopilot_branch_scope"),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'evaluating', 'succeeded', "
            "'failed', 'blocked', 'skipped')",
            name="ck_autopilot_branches_status",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_autopilot_branches_cycle_status",
        "autopilot_branches",
        ["cycle_id", "status"],
        schema=SCHEMA,
    )
    op.create_table(
        "research_report_backfill_days",
        sa.Column("report_date", sa.Date(), primary_key=True),
        sa.Column("snapshot_name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("selected_count", sa.Integer(), nullable=False),
        sa.Column("published_count", sa.Integer(), nullable=False),
        sa.Column("blocked_count", sa.Integer(), nullable=False),
        sa.Column("bytes_downloaded", sa.BigInteger(), nullable=False),
        sa.Column("job_id", sa.String(), sa.ForeignKey(f"{SCHEMA}.jobs.id", ondelete="SET NULL")),
        sa.Column("last_error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "status IN ('pending', 'queued', 'running', 'succeeded', 'blocked')",
            name="ck_research_report_backfill_status",
        ),
        sa.CheckConstraint(
            "attempts >= 0 AND selected_count >= 0 AND published_count >= 0 "
            "AND blocked_count >= 0 AND bytes_downloaded >= 0",
            name="ck_research_report_backfill_counts",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_research_report_backfill_status_date",
        "research_report_backfill_days",
        ["status", sa.text("report_date DESC")],
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_index(
        "idx_research_report_backfill_status_date",
        table_name="research_report_backfill_days",
        schema=SCHEMA,
    )
    op.drop_table("research_report_backfill_days", schema=SCHEMA)
    op.drop_index(
        "idx_autopilot_branches_cycle_status", table_name="autopilot_branches", schema=SCHEMA
    )
    op.drop_table("autopilot_branches", schema=SCHEMA)
    op.drop_index(
        "idx_autopilot_cycles_status_updated", table_name="autopilot_cycles", schema=SCHEMA
    )
    op.drop_table("autopilot_cycles", schema=SCHEMA)
