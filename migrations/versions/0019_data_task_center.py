"""Add the persistent, dependency-aware market data task catalog."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0019_data_task_center"
down_revision: str | None = "0018_runtime_secrets"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "data_tasks",
        sa.Column("task_key", sa.String(), nullable=False),
        sa.Column("phase", sa.Integer(), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("category", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("implementation_status", sa.String(), nullable=False),
        sa.Column(
            "depends_on_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "config_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("estimated_storage_gb", sa.Integer()),
        sa.Column("job_id", sa.String()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["job_id"], ["quantlab.jobs.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("task_key"),
        schema="quantlab",
    )
    op.create_index(
        "idx_data_tasks_phase_order",
        "data_tasks",
        ["phase", "sort_order"],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_data_tasks_phase_order", table_name="data_tasks", schema="quantlab"
    )
    op.drop_table("data_tasks", schema="quantlab")
