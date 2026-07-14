"""Add encrypted runtime credentials managed from the control plane."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0018_runtime_secrets"
down_revision: str | None = "0017_qmt_execution_state"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "runtime_secrets",
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("ciphertext", sa.Text(), nullable=False),
        sa.Column(
            "metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_by", sa.String()),
        sa.ForeignKeyConstraint(
            ["updated_by"], ["quantlab.users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("name"),
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_table("runtime_secrets", schema="quantlab")
