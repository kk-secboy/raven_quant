"""Add cross-campaign champion and decay governance."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0031_research_governance"
down_revision: str | None = "0030_research_programs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for column in (
        sa.Column("last_evaluated_campaign_id", sa.String()),
        sa.Column("champion_campaign_id", sa.String()),
        sa.Column("champion_strategy_version_id", sa.String()),
        sa.Column("champion_score", sa.Float()),
        sa.Column("champion_selected_at", sa.DateTime(timezone=True)),
        sa.Column("decay_status", sa.String(), nullable=False, server_default="unavailable"),
        sa.Column("decay_message", sa.Text()),
    ):
        op.add_column("research_programs", column, schema="quantlab")


def downgrade() -> None:
    for name in (
        "decay_message",
        "decay_status",
        "champion_selected_at",
        "champion_score",
        "champion_strategy_version_id",
        "champion_campaign_id",
        "last_evaluated_campaign_id",
    ):
        op.drop_column("research_programs", name, schema="quantlab")
