"""Two-dimension recommendation account actions (design draft 8.4).

``recommendation_snapshots.account_actions_json`` stores the per-instrument
action x execution-state plan (BUY/SELL/EXIT/HOLD/NO_ACTION x
READY/WAIT/PARTIAL/CANCELLED/EXPIRED/BLOCKED) together with
``projected_position`` and the keep/cancel/replace/new order plan.  The
column is nullable: snapshots without an attached account state keep the
legacy increase/decrease holdings export only.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0044_recommendation_actions"
down_revision: str | None = "0043_promotion_stages"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "recommendation_snapshots",
        sa.Column("account_actions_json", sa.JSON(), nullable=True),
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_column(
        "recommendation_snapshots",
        "account_actions_json",
        schema="quantlab",
    )
