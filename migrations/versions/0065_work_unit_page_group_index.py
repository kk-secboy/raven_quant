"""Index durable pagination groups used during checkpoint rehydration.

The production checkpoint contains millions of immutable work units.  A
restart must locate the durable siblings of the live pagination groups without
rescanning the whole ledger once per batch.  ``IF NOT EXISTS`` deliberately
adopts the same index when it was built operationally before this migration was
deployed; the release acceptance check verifies its exact definition.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0065_work_unit_page_group"
down_revision: str | None = "0064_paper_stage_account"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_work_units_dataset_page_group
            ON quantlab.work_units (
                dataset,
                CAST(scope_json ->> 'page_group' AS VARCHAR)
            )
        """
    )


def downgrade() -> None:
    op.execute(
        "DROP INDEX IF EXISTS quantlab.idx_work_units_dataset_page_group"
    )
