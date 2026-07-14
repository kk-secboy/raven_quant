"""Create the PostgreSQL control-plane schema."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_control_plane"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS quantlab")
    op.create_table(
        "work_units",
        sa.Column("unit_key", sa.String(), primary_key=True),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("api_name", sa.String(), nullable=False),
        sa.Column("scope_json", postgresql.JSONB(), nullable=False),
        sa.Column("params_json", postgresql.JSONB(), nullable=False),
        sa.Column("fields_json", postgresql.JSONB(), nullable=False),
        sa.Column("allow_empty", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("next_retry_at", sa.DateTime(timezone=True)),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("output_path", sa.Text()),
        sa.Column("row_count", sa.Integer()),
        sa.Column("sha256", sa.String()),
        sa.Column("last_error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        schema="quantlab",
    )
    op.create_index(
        "idx_work_units_claim",
        "work_units",
        ["status", "next_retry_at", "lease_until", "dataset"],
        schema="quantlab",
    )
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("payload_json", postgresql.JSONB(), nullable=False),
        sa.Column("progress_json", postgresql.JSONB()),
        sa.Column("log_path", sa.Text()),
        sa.Column("exit_code", sa.Integer()),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        schema="quantlab",
    )
    op.create_index(
        "idx_jobs_status_created",
        "jobs",
        ["status", sa.text("created_at DESC")],
        schema="quantlab",
    )
    op.create_index(
        "uq_jobs_active_kind",
        "jobs",
        ["kind"],
        unique=True,
        schema="quantlab",
        postgresql_where=sa.text("status IN ('queued', 'running')"),
    )


def downgrade() -> None:
    op.drop_index("uq_jobs_active_kind", table_name="jobs", schema="quantlab")
    op.drop_index("idx_jobs_status_created", table_name="jobs", schema="quantlab")
    op.drop_table("jobs", schema="quantlab")
    op.drop_index("idx_work_units_claim", table_name="work_units", schema="quantlab")
    op.drop_table("work_units", schema="quantlab")
    op.execute("DROP SCHEMA IF EXISTS quantlab CASCADE")
