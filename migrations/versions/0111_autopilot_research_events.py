"""Append independently identified manual events to the existing research flow.

Revision ID: 0111_autopilot_research_events
Revises: 0110_strategy_runtime_v39
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0111_autopilot_research_events"
down_revision: str | None = "0110_strategy_runtime_v39"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"


def upgrade() -> None:
    op.add_column(
        "autopilot_cycles",
        sa.Column("research_event_key", sa.String(), nullable=False, server_default="scheduled"),
        schema=SCHEMA,
    )
    op.drop_constraint(
        "uq_autopilot_cycle_dataset_horizon", "autopilot_cycles", schema=SCHEMA, type_="unique"
    )
    op.create_unique_constraint(
        "uq_autopilot_cycle_dataset_horizon_event", "autopilot_cycles",
        ["dataset_identity_sha256", "horizon_profile", "research_event_key"], schema=SCHEMA,
    )
    op.create_index(
        "uq_autopilot_manual_research_event", "autopilot_cycles", ["research_event_key"],
        unique=True, schema=SCHEMA,
        postgresql_where=sa.text("research_event_key <> 'scheduled'"),
    )


def downgrade() -> None:
    # Never discard an event identity or collapse multiple historical activities.
    if op.get_bind().scalar(sa.text(
        "SELECT EXISTS (SELECT 1 FROM quantlab.autopilot_cycles "
        "WHERE research_event_key <> 'scheduled')"
    )):
        raise RuntimeError("cannot downgrade while manual research event history exists")
    op.drop_index(
        "uq_autopilot_manual_research_event", table_name="autopilot_cycles", schema=SCHEMA
    )
    op.drop_constraint(
        "uq_autopilot_cycle_dataset_horizon_event", "autopilot_cycles",
        schema=SCHEMA, type_="unique",
    )
    op.create_unique_constraint(
        "uq_autopilot_cycle_dataset_horizon", "autopilot_cycles",
        ["dataset_identity_sha256", "horizon_profile"], schema=SCHEMA,
    )
    op.drop_column("autopilot_cycles", "research_event_key", schema=SCHEMA)
