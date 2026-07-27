"""Versioned personal market permissions (8.7) and manual/CSV shadow accounts (8.6).

Design draft 8.7: every account binds a versioned personal investment policy —
per exchange/board/risk-warning/ETF-subtype one of buy_sell/sell_only/disabled/
unknown with confirmation source, as_of and validity; new versions may only
tighten unless a relaxation is explicitly confirmed.

Design draft 8.6: users may import real holdings, cash, sellable quantities and
open orders as a manual shadow account; shadow/model/simulation accounts stay
visibly separate and stale shadow state degrades to simulation-only advice.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0048_permission_shadow_account"
down_revision: str | None = "0047_safe_mode"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "market_permission_versions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("scope_type", sa.String(), nullable=False),
        sa.Column("scope_key", sa.String(), nullable=False),
        sa.Column("permission", sa.String(), nullable=False),
        sa.Column("confirmation_source", sa.Text(), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("valid_until", sa.Date(), nullable=True),
        sa.Column("supersedes_id", sa.String(), nullable=True),
        sa.Column(
            "relaxation_confirmed",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        schema="quantlab",
    )
    op.create_index(
        "idx_market_permission_scope",
        "market_permission_versions",
        ["scope_type", "scope_key", sa.text("as_of DESC")],
        schema="quantlab",
    )
    op.create_table(
        "shadow_account_snapshots",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("account_id", sa.String(), nullable=False),
        sa.Column("import_source", sa.String(), nullable=False),
        sa.Column("cash", sa.Numeric(20, 6), nullable=False),
        sa.Column("holdings_json", sa.JSON(), nullable=False),
        sa.Column("open_orders_json", sa.JSON(), nullable=False),
        sa.Column("content_sha256", sa.String(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("imported_by", sa.String(), nullable=False),
        sa.Column("imported_at", sa.DateTime(timezone=True), nullable=False),
        schema="quantlab",
    )
    op.create_index(
        "idx_shadow_account_snapshots_account",
        "shadow_account_snapshots",
        ["account_id", sa.text("imported_at DESC")],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_shadow_account_snapshots_account",
        table_name="shadow_account_snapshots",
        schema="quantlab",
    )
    op.drop_table("shadow_account_snapshots", schema="quantlab")
    op.drop_index(
        "idx_market_permission_scope",
        table_name="market_permission_versions",
        schema="quantlab",
    )
    op.drop_table("market_permission_versions", schema="quantlab")
