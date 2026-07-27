"""Platform safe-mode state (design draft 11.3).

Severe data, ledger or version anomalies engage a platform-level safe mode
that stops new recommendations and new simulation orders while preserving
read, reconciliation and recovery paths. The state is a singleton row
(``id = 'platform'``) so every consumer reads one consistent flag.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0047_safe_mode"
down_revision: str | None = "0046_simulation_order_states"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "platform_safe_mode_state",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("active", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("reason", sa.Text(), nullable=False, server_default=""),
        sa.Column("source", sa.String(), nullable=False, server_default=""),
        sa.Column("triggered_by", sa.String(), nullable=False, server_default=""),
        sa.Column("triggered_at", sa.DateTime(timezone=True)),
        sa.Column("details_json", sa.JSON()),
        sa.Column("cleared_by", sa.String()),
        sa.Column("cleared_at", sa.DateTime(timezone=True)),
        sa.Column("clear_reason", sa.Text()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        schema="quantlab",
    )
    op.execute(
        """
        INSERT INTO quantlab.platform_safe_mode_state
            (id, active, reason, source, triggered_by, updated_at)
        VALUES ('platform', false, '', '', '', now())
        """
    )


def downgrade() -> None:
    op.drop_table("platform_safe_mode_state", schema="quantlab")
