"""Add one-time automatic research-asset consumption ledger.

Revision ID: 0062_research_asset_consumptions
Revises: 0061_rdagent_governance
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0062_research_asset_consumptions"
down_revision: str | None = "0061_rdagent_governance"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON = sa.JSON().with_variant(JSONB(), "postgresql")
SCHEMA = "quantlab"


def upgrade() -> None:
    op.create_table(
        "research_asset_consumptions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "asset_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.research_assets.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "research_run_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.research_runs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("scenario", sa.String(), nullable=False),
        sa.Column("selection_mode", sa.String(), nullable=False),
        sa.Column("asset_manifest_sha256", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("reserved_by", sa.String(), nullable=False),
        sa.Column("reserved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("details_json", JSON, nullable=False),
        sa.UniqueConstraint(
            "asset_id", name="uq_research_asset_consumptions_asset"
        ),
        sa.UniqueConstraint(
            "research_run_id",
            "asset_id",
            name="uq_research_asset_consumptions_run_asset",
        ),
        sa.CheckConstraint(
            "selection_mode IN ('automatic')",
            name="ck_research_asset_consumptions_mode",
        ),
        sa.CheckConstraint(
            "status IN ('reserved', 'consumed', 'failed')",
            name="ck_research_asset_consumptions_status",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_research_asset_consumptions_run",
        "research_asset_consumptions",
        ["research_run_id", sa.text("reserved_at DESC")],
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_index(
        "idx_research_asset_consumptions_run",
        table_name="research_asset_consumptions",
        schema=SCHEMA,
    )
    op.drop_table("research_asset_consumptions", schema=SCHEMA)
