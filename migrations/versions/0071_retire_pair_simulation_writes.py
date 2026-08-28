"""Retire every executable pair-simulation write path.

Revision ID: 0071_retire_pair_writes
Revises: 0070_alpha_spending_ledger
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0071_retire_pair_writes"
down_revision: str | None = "0070_alpha_spending_ledger"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Code stops creating/replaying pair ledgers.  This migration closes the
    # race for already-active historical rows at release time while preserving
    # every order, fill, NAV and audit record for read-only inspection.
    op.execute(
        """
        UPDATE quantlab.jobs AS job
        SET status = 'cancelled',
            error = 'pair simulation execution retired; historical ledger is read-only',
            cancel_requested_at = COALESCE(job.cancel_requested_at, now()),
            finished_at = COALESCE(job.finished_at, now()),
            next_attempt_at = NULL
        WHERE (
              job.kind = 'pair_backtest'
              OR (
                job.kind = 'simulation_replay'
                AND EXISTS (
                    SELECT 1
                    FROM quantlab.simulation_batches AS batch
                    JOIN quantlab.simulation_portfolios AS portfolio
                      ON portfolio.id = batch.portfolio_id
                    WHERE batch.id = job.payload_json ->> 'simulation_batch_id'
                      AND portfolio.execution_adapter = 'pair'
                )
              )
          )
          AND job.status IN ('queued', 'running')
        """
    )
    op.execute(
        """
        UPDATE quantlab.backtest_runs AS backtest
        SET status = 'cancelled',
            error = 'pair backtest execution retired; historical artifact is read-only',
            finished_at = COALESCE(backtest.finished_at, now())
        FROM quantlab.strategy_versions AS version
        WHERE version.id = backtest.strategy_version_id
          AND version.strategy_type = 'pair'
          AND backtest.status IN ('queued', 'running')
        """
    )
    op.execute(
        """
        UPDATE quantlab.simulation_batches AS batch
        SET status = 'failed',
            error = 'pair simulation execution retired; historical ledger is read-only',
            finished_at = COALESCE(batch.finished_at, now())
        FROM quantlab.simulation_portfolios AS portfolio
        WHERE portfolio.id = batch.portfolio_id
          AND portfolio.execution_adapter = 'pair'
          AND batch.status IN ('queued', 'running')
        """
    )
    op.execute(
        """
        UPDATE quantlab.simulation_portfolios
        SET status = 'paused', updated_at = now()
        WHERE execution_adapter = 'pair' AND status = 'active'
        """
    )
    op.execute(
        """
        UPDATE quantlab.schedules AS schedule
        SET status = 'retired',
            desired_status = 'retired',
            suspension_reason = 'pair allocation execution retired; history is read-only',
            updated_at = now()
        WHERE schedule.payload_json ->> 'allocation_id' IN (
            SELECT DISTINCT member.allocation_id
            FROM quantlab.strategy_allocation_members AS member
            JOIN quantlab.strategy_versions AS version
              ON version.id = member.strategy_version_id
            WHERE version.strategy_type = 'pair'
        )
          AND schedule.status <> 'retired'
        """
    )
    op.execute(
        """
        UPDATE quantlab.allocation_schedule_groups AS schedule_group
        SET status = 'retired', updated_at = now()
        WHERE schedule_group.allocation_id IN (
            SELECT DISTINCT member.allocation_id
            FROM quantlab.strategy_allocation_members AS member
            JOIN quantlab.strategy_versions AS version
              ON version.id = member.strategy_version_id
            WHERE version.strategy_type = 'pair'
        )
          AND schedule_group.status <> 'retired'
        """
    )
    op.execute(
        """
        UPDATE quantlab.recommendation_portfolios AS portfolio
        SET status = 'paused', risk_exposure_override = 0, updated_at = now()
        WHERE portfolio.id IN (
            SELECT member.recommendation_portfolio_id
            FROM quantlab.strategy_allocation_members AS member
            WHERE member.recommendation_portfolio_id IS NOT NULL
              AND member.allocation_id IN (
                  SELECT DISTINCT pair_member.allocation_id
                  FROM quantlab.strategy_allocation_members AS pair_member
                  JOIN quantlab.strategy_versions AS version
                    ON version.id = pair_member.strategy_version_id
                  WHERE version.strategy_type = 'pair'
              )
        )
          AND portfolio.status = 'active'
        """
    )
    op.execute(
        """
        UPDATE quantlab.simulation_portfolios AS portfolio
        SET status = 'paused', updated_at = now()
        WHERE portfolio.status = 'active'
          AND (
              (
                  portfolio.source_type = 'allocation'
                  AND portfolio.source_id IN (
                      SELECT DISTINCT member.allocation_id
                      FROM quantlab.strategy_allocation_members AS member
                      JOIN quantlab.strategy_versions AS version
                        ON version.id = member.strategy_version_id
                      WHERE version.strategy_type = 'pair'
                  )
              )
              OR (
                  portfolio.source_type = 'recommendation'
                  AND portfolio.source_id IN (
                      SELECT member.recommendation_portfolio_id
                      FROM quantlab.strategy_allocation_members AS member
                      WHERE member.recommendation_portfolio_id IS NOT NULL
                        AND member.allocation_id IN (
                            SELECT DISTINCT pair_member.allocation_id
                            FROM quantlab.strategy_allocation_members AS pair_member
                            JOIN quantlab.strategy_versions AS version
                              ON version.id = pair_member.strategy_version_id
                            WHERE version.strategy_type = 'pair'
                        )
                  )
              )
          )
        """
    )
    op.execute(
        """
        UPDATE quantlab.strategy_allocations AS allocation
        SET status = 'paused', updated_at = now()
        WHERE allocation.status IN ('active', 'risk_reduction_pending', 'liquidation_pending')
          AND EXISTS (
              SELECT 1
              FROM quantlab.strategy_allocation_members AS member
              JOIN quantlab.strategy_versions AS version
                ON version.id = member.strategy_version_id
              WHERE member.allocation_id = allocation.id
                AND version.strategy_type = 'pair'
          )
        """
    )


def downgrade() -> None:
    # Retirement is intentionally one-way. Reconstructing executable work from
    # a historical research ledger would invent authority and duplicate orders.
    pass
