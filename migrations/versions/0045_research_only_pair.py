"""Retire legacy approved pair strategy versions (design draft 6.4.3/13).

Product decision 2026-07-22: ``stock_pair_stat_arb`` is research-only — A-share
securities lending is effectively unavailable to retail investors, so a
capitalized pair simulation would produce misleading evidence. The runtime
gates (approval, allocation membership, persistent simulation) now fail closed
against the strategy catalog's ``research_only`` role; this migration retires
any pair strategy versions that were approved before the gates existed and
records an audit event per retired version. No rows are deleted: backtests,
events, simulations and allocations stay readable, while every activation or
refresh path fails closed because the version is no longer ``approved``.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0045_research_only_pair"
down_revision: str | None = "0044_recommendation_actions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_RETIRE_SQL = sa.text(
    """
    UPDATE quantlab.strategy_versions
    SET status = 'retired'
    WHERE strategy_type = 'pair' AND status = 'approved'
    RETURNING id, strategy_id
    """
)

_AUDIT_SQL = sa.text(
    """
    INSERT INTO quantlab.strategy_events
        (strategy_id, strategy_version_id, event_type, actor, payload_json, created_at)
    VALUES (
        :strategy_id,
        :version_id,
        'strategy.pair_retired_research_only',
        'migration-0045',
        CAST(:payload AS JSONB),
        now()
    )
    """
)


def upgrade() -> None:
    connection = op.get_bind()
    retired = connection.execute(_RETIRE_SQL).all()
    for version_id, strategy_id in retired:
        connection.execute(
            _AUDIT_SQL,
            {
                "strategy_id": strategy_id,
                "version_id": version_id,
                "payload": (
                    '{"reason": "research_only: pair strategies are offline '
                    'statistical research (design 6.4.3/13); approved status '
                    'retired by migration 0045"}'
                ),
            },
        )
    # Family rows whose approved pair versions were just retired fall back to
    # retired visibility; nothing is deleted. strategy_type only exists on
    # strategy_versions, so the family update keys off the retired ids.
    for _version_id, strategy_id in retired:
        connection.execute(
            sa.text(
                """
                UPDATE quantlab.strategies
                SET status = 'retired', updated_at = now()
                WHERE id = :strategy_id AND status = 'approved'
                """
            ),
            {"strategy_id": strategy_id},
        )


def downgrade() -> None:
    # The audit events stay as the historical record; the retirement itself is
    # a policy change and is not reversed automatically.
    pass
