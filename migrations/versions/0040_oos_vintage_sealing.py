"""Sealed one-shot consumption ledger for reserved final out-of-sample windows.

Design draft 4.1/12.1: the reserved final test interval is a one-time resource.
Each (research scope, dataset identity, calendar window) triple is an OOS
vintage that is sealed with its candidate set before the first open; once
consumed, no candidate — including new evaluation rows produced by renamed or
newly created campaigns in the same scope — may consume that window again.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0040_oos_vintage_sealing"
down_revision: str | None = "0039_allocation_policy_guards"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "oos_vintages",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("scope", sa.String(), nullable=False),
        sa.Column("dataset_identity", sa.String(), nullable=False),
        sa.Column("test_start", sa.Date(), nullable=False),
        sa.Column("test_end", sa.Date(), nullable=False),
        sa.Column("sealed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("first_opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sealed_candidate_set_json", sa.JSON(), nullable=False),
        sa.Column("sealed_candidate_set_sha256", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "scope",
            "dataset_identity",
            "test_start",
            "test_end",
            name="uq_oos_vintage_window",
        ),
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_table("oos_vintages", schema="quantlab")
