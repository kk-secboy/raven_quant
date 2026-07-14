"""Add audited Web-managed platform configuration templates."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0027_web_config_templates"
down_revision: str | None = "0026_factor_evidence_integrity"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "platform_configs",
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("value_json", postgresql.JSONB(), nullable=False),
        sa.Column("updated_by", sa.String(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("key"),
        schema="quantlab",
    )
    op.create_table(
        "platform_config_revisions",
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("value_json", postgresql.JSONB(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("updated_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("key", "revision"),
        schema="quantlab",
    )
    op.create_index(
        "idx_platform_config_revisions_created",
        "platform_config_revisions",
        [sa.text("created_at DESC")],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_platform_config_revisions_created",
        table_name="platform_config_revisions",
        schema="quantlab",
    )
    op.drop_table("platform_config_revisions", schema="quantlab")
    op.drop_table("platform_configs", schema="quantlab")
