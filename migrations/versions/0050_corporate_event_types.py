"""Non-dividend corporate event ledger (design 5.6 type extension).

Design draft 5.6: splits/reverse splits, rights issues, share exchanges, code
changes, ETF share conversions and fund liquidations are event-driven, and
every corporate action distinguishes the announcement / record / ex /
settlement stages with a unique event key per stage. Announcements only
inform; unsupported types fail closed with their reason recorded.

``simulation_corporate_events`` is the append-only ledger for every
non-dividend corporate event applied to a simulation account (announcement,
name_change, split, reverse_split, code_change, choice_required,
unsupported). ``event_key`` is the idempotency key: an event is applied at
most once per portfolio, replays are no-ops, and the ledger doubles as the
state from which pending holder-choice markers are re-derived on load.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0050_corporate_event_types"
down_revision: str | None = "0049_external_cash_flows"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "simulation_corporate_events",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("portfolio_id", sa.String(), nullable=False),
        sa.Column("event_key", sa.String(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("instrument", sa.String(), nullable=False),
        sa.Column("effective_date", sa.Date(), nullable=False),
        sa.Column("payload_sha256", sa.String(), nullable=False),
        sa.Column("details_json", sa.JSON(), nullable=False),
        sa.Column("batch_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["portfolio_id"], ["quantlab.simulation_portfolios.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint(
            "portfolio_id", "event_key", name="uq_simulation_corporate_events_key"
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_simulation_corporate_events_portfolio_date",
        "simulation_corporate_events",
        ["portfolio_id", "effective_date"],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_simulation_corporate_events_portfolio_date",
        table_name="simulation_corporate_events",
        schema="quantlab",
    )
    op.drop_table("simulation_corporate_events", schema="quantlab")
