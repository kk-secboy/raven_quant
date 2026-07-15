"""Add durable continuous research programs."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0030_research_programs"
down_revision: str | None = "0029_research_campaigns"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "research_programs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("recipe_id", sa.String(), nullable=False),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("benchmark", sa.String(), nullable=False),
        sa.Column("universe", sa.String(), nullable=False),
        sa.Column("dataset_lineage_id", sa.String(), nullable=False),
        sa.Column("config_json", postgresql.JSONB(), nullable=False),
        sa.Column("min_new_trading_days", sa.Integer(), nullable=False),
        sa.Column("max_active_campaigns", sa.Integer(), nullable=False),
        sa.Column("last_dataset_name", sa.String()),
        sa.Column("last_dataset_identity_sha256", sa.String()),
        sa.Column("last_dataset_end_date", sa.String()),
        sa.Column("last_message", sa.Text()),
        sa.Column("last_checked_at", sa.DateTime(timezone=True)),
        sa.Column("last_triggered_at", sa.DateTime(timezone=True)),
        sa.Column("next_check_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_research_programs_name"),
        schema="quantlab",
    )
    op.create_index(
        "idx_research_programs_claim",
        "research_programs",
        ["status", "next_check_at", "lease_until"],
        schema="quantlab",
    )
    op.create_table(
        "research_program_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column(
            "program_id",
            sa.String(),
            sa.ForeignKey("quantlab.research_programs.id", ondelete="CASCADE"),
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
        "idx_research_program_events_created",
        "research_program_events",
        ["program_id", "created_at"],
        schema="quantlab",
    )
    op.add_column(
        "research_campaigns",
        sa.Column("research_program_id", sa.String()),
        schema="quantlab",
    )
    op.add_column(
        "research_campaigns",
        sa.Column("dataset_identity_sha256", sa.String()),
        schema="quantlab",
    )
    op.create_foreign_key(
        "fk_research_campaign_program",
        "research_campaigns",
        "research_programs",
        ["research_program_id"],
        ["id"],
        source_schema="quantlab",
        referent_schema="quantlab",
        ondelete="SET NULL",
    )
    op.create_unique_constraint(
        "uq_research_campaign_program_dataset",
        "research_campaigns",
        ["research_program_id", "dataset_identity_sha256"],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_research_campaign_program_dataset",
        "research_campaigns",
        schema="quantlab",
        type_="unique",
    )
    op.drop_constraint(
        "fk_research_campaign_program",
        "research_campaigns",
        schema="quantlab",
        type_="foreignkey",
    )
    op.drop_column("research_campaigns", "dataset_identity_sha256", schema="quantlab")
    op.drop_column("research_campaigns", "research_program_id", schema="quantlab")
    op.drop_index(
        "idx_research_program_events_created",
        table_name="research_program_events",
        schema="quantlab",
    )
    op.drop_table("research_program_events", schema="quantlab")
    op.drop_index(
        "idx_research_programs_claim",
        table_name="research_programs",
        schema="quantlab",
    )
    op.drop_table("research_programs", schema="quantlab")
