"""Unify strategy contracts, simulation sources, and core/satellite allocations."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0037_single_mainline_contract"
down_revision: str | None = "0036_financial_correctness"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _add_research_contract_columns(table: str) -> None:
    for column in (
        sa.Column("signal_frequency", sa.String(), nullable=False, server_default="day"),
        sa.Column("signal_horizon", sa.String(), nullable=False, server_default="1d"),
        sa.Column("execution_frequency", sa.String(), nullable=False, server_default="5min"),
        sa.Column(
            "execution_contract_hash",
            sa.String(),
            nullable=False,
            server_default="legacy-unversioned",
        ),
        sa.Column("qlib_version", sa.String()),
        sa.Column("qlib_commit", sa.String()),
        sa.Column("rdagent_version", sa.String()),
        sa.Column("rdagent_commit", sa.String()),
    ):
        op.add_column(table, column, schema="quantlab")


def upgrade() -> None:
    json_type = sa.JSON().with_variant(JSONB(), "postgresql")

    _add_research_contract_columns("strategy_versions")
    _add_research_contract_columns("factor_evaluations")
    # Backtests do not own a signal horizon; it is inherited from the immutable version.
    for column in (
        sa.Column("signal_frequency", sa.String(), nullable=False, server_default="day"),
        sa.Column("execution_frequency", sa.String(), nullable=False, server_default="5min"),
        sa.Column(
            "execution_contract_hash",
            sa.String(),
            nullable=False,
            server_default="legacy-unversioned",
        ),
        sa.Column("qlib_version", sa.String()),
        sa.Column("qlib_commit", sa.String()),
        sa.Column("rdagent_version", sa.String()),
        sa.Column("rdagent_commit", sa.String()),
    ):
        op.add_column("backtest_runs", column, schema="quantlab")

    for column in (
        sa.Column("source_type", sa.String(), nullable=True),
        sa.Column("source_id", sa.String(), nullable=True),
        sa.Column("execution_adapter", sa.String(), nullable=False, server_default="long_only"),
        sa.Column("execution_frequency", sa.String(), nullable=False, server_default="5min"),
        sa.Column("execution_contract_hash", sa.String(), nullable=True),
    ):
        op.add_column("simulation_portfolios", column, schema="quantlab")
    op.execute(
        """
        UPDATE quantlab.simulation_portfolios
        SET source_type = 'recommendation',
            source_id = recommendation_portfolio_id,
            status = 'paused',
            execution_contract_hash = 'legacy-unversioned'
        """
    )
    op.alter_column("simulation_portfolios", "source_type", nullable=False, schema="quantlab")
    op.alter_column("simulation_portfolios", "source_id", nullable=False, schema="quantlab")
    op.alter_column(
        "simulation_portfolios", "execution_contract_hash", nullable=False, schema="quantlab"
    )
    op.drop_constraint(
        "simulation_portfolios_recommendation_portfolio_id_key",
        "simulation_portfolios",
        type_="unique",
        schema="quantlab",
    )
    op.alter_column(
        "simulation_portfolios",
        "recommendation_portfolio_id",
        existing_type=sa.String(),
        nullable=True,
        schema="quantlab",
    )
    op.create_index(
        "uq_simulation_portfolios_source_execution",
        "simulation_portfolios",
        ["source_type", "source_id", "execution_dataset"],
        unique=True,
        schema="quantlab",
    )
    op.create_check_constraint(
        "ck_simulation_portfolios_source_type",
        "simulation_portfolios",
        "source_type IN ('recommendation', 'strategy_version', 'allocation')",
        schema="quantlab",
    )
    op.create_check_constraint(
        "ck_simulation_portfolios_adapter",
        "simulation_portfolios",
        "execution_adapter IN ('long_only', 'pair')",
        schema="quantlab",
    )
    op.create_check_constraint(
        "ck_simulation_portfolios_frequency",
        "simulation_portfolios",
        "execution_frequency IN ('1min', '5min')",
        schema="quantlab",
    )

    for column in (
        sa.Column("source_snapshot_id", sa.String()),
        sa.Column("target_payload_json", json_type),
        sa.Column("execution_adapter", sa.String(), nullable=False, server_default="long_only"),
        sa.Column("execution_contract_hash", sa.String(), nullable=True),
        sa.Column("created_by", sa.String(), nullable=False, server_default="legacy-system"),
        sa.Column("signal_at", sa.DateTime(timezone=True)),
        sa.Column("execution_not_before", sa.DateTime(timezone=True)),
    ):
        op.add_column("simulation_batches", column, schema="quantlab")
    op.execute(
        """
        UPDATE quantlab.simulation_batches AS batch
        SET source_snapshot_id = batch.recommendation_snapshot_id,
            execution_contract_hash = portfolio.execution_contract_hash
        FROM quantlab.simulation_portfolios AS portfolio
        WHERE portfolio.id = batch.portfolio_id
        """
    )
    op.alter_column(
        "simulation_batches", "execution_contract_hash", nullable=False, schema="quantlab"
    )
    op.alter_column(
        "simulation_batches",
        "recommendation_snapshot_id",
        existing_type=sa.String(),
        nullable=True,
        schema="quantlab",
    )
    op.drop_constraint(
        "simulation_batches_recommendation_snapshot_id_key",
        "simulation_batches",
        type_="unique",
        schema="quantlab",
    )
    op.create_index(
        "uq_simulation_batches_portfolio_recommendation",
        "simulation_batches",
        ["portfolio_id", "recommendation_snapshot_id"],
        unique=True,
        schema="quantlab",
        postgresql_where=sa.text("recommendation_snapshot_id IS NOT NULL"),
    )
    op.create_check_constraint(
        "ck_simulation_batches_next_bar_time",
        "simulation_batches",
        "(signal_at IS NULL AND execution_not_before IS NULL) OR "
        "(signal_at IS NOT NULL AND execution_not_before IS NOT NULL "
        "AND execution_not_before > signal_at "
        "AND (signal_at AT TIME ZONE 'Asia/Shanghai')::date = signal_date "
        "AND (execution_not_before AT TIME ZONE 'Asia/Shanghai')::date = trade_date)",
        schema="quantlab",
    )

    for table in ("simulation_orders", "simulation_fills", "simulation_positions"):
        for column in (
            sa.Column("atomic_group_id", sa.String()),
            sa.Column("leg_no", sa.Integer()),
            sa.Column("position_side", sa.String(), nullable=False, server_default="long"),
            sa.Column(
                "borrow_cost", sa.Numeric(20, 6), nullable=False, server_default="0"
            ),
        ):
            op.add_column(table, column, schema="quantlab")
        op.create_check_constraint(
            f"ck_{table}_position_side",
            table,
            "position_side IN ('long', 'short')",
            schema="quantlab",
        )

    for column in (
        sa.Column("role", sa.String(), nullable=False, server_default="core"),
        sa.Column("risk_budget", sa.Float(), nullable=False, server_default="1"),
        sa.Column("member_cap", sa.Float(), nullable=False, server_default="0.70"),
    ):
        op.add_column("strategy_allocation_members", column, schema="quantlab")
    op.create_check_constraint(
        "ck_strategy_allocation_members_role",
        "strategy_allocation_members",
        "role IN ('core', 'satellite')",
        schema="quantlab",
    )
    op.create_check_constraint(
        "ck_strategy_allocation_members_risk_budget",
        "strategy_allocation_members",
        "risk_budget > 0 AND risk_budget <= 1",
        schema="quantlab",
    )
    op.create_check_constraint(
        "ck_strategy_allocation_members_cap",
        "strategy_allocation_members",
        "member_cap > 0 AND member_cap <= 0.70",
        schema="quantlab",
    )

    for column in (
        sa.Column(
            "nav_scope",
            sa.String(),
            nullable=False,
            server_default="member_ledger",
        ),
        sa.Column(
            "produced_by",
            sa.String(),
            nullable=False,
            server_default="legacy-system",
        ),
        sa.Column("reviewed_by", sa.String()),
        sa.Column("reviewed_at", sa.DateTime(timezone=True)),
        sa.Column("review_evidence_sha256", sa.String()),
        sa.Column("review_note", sa.Text()),
    ):
        op.add_column("simulation_nav", column, schema="quantlab")
    op.execute(
        """
        UPDATE quantlab.simulation_nav AS nav
        SET nav_scope = 'aggregate_view'
        FROM quantlab.simulation_portfolios AS portfolio
        WHERE portfolio.id = nav.portfolio_id
          AND portfolio.source_type = 'allocation'
        """
    )
    op.create_check_constraint(
        "ck_simulation_nav_scope",
        "simulation_nav",
        "nav_scope IN ('member_ledger', 'aggregate_view')",
        schema="quantlab",
    )
    op.create_check_constraint(
        "ck_simulation_nav_review_complete",
        "simulation_nav",
        """
        (
            reviewed_by IS NULL
            AND reviewed_at IS NULL
            AND review_evidence_sha256 IS NULL
            AND review_note IS NULL
        )
        OR
        (
            reviewed_by IS NOT NULL
            AND reviewed_at IS NOT NULL
            AND review_evidence_sha256 ~ '^[0-9a-f]{64}$'
            AND length(btrim(review_note)) >= 10
            AND reviewed_by <> produced_by
            AND performance_certified IS TRUE
        )
        """,
        schema="quantlab",
    )
    op.execute(
        """
        CREATE FUNCTION quantlab.prevent_simulation_nav_review_mutation()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            IF NEW.nav_scope IS DISTINCT FROM OLD.nav_scope
                OR NEW.produced_by IS DISTINCT FROM OLD.produced_by
            THEN
                RAISE EXCEPTION 'simulation NAV provenance is immutable';
            END IF;
            IF OLD.reviewed_at IS NOT NULL AND (
                NEW.reviewed_by IS DISTINCT FROM OLD.reviewed_by
                OR NEW.reviewed_at IS DISTINCT FROM OLD.reviewed_at
                OR NEW.review_evidence_sha256 IS DISTINCT FROM OLD.review_evidence_sha256
                OR NEW.review_note IS DISTINCT FROM OLD.review_note
                OR NEW.performance_certified IS DISTINCT FROM OLD.performance_certified
            ) THEN
                RAISE EXCEPTION 'simulation NAV review is immutable';
            END IF;
            RETURN NEW;
        END;
        $$;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_simulation_nav_review_immutable
        BEFORE UPDATE ON quantlab.simulation_nav
        FOR EACH ROW
        EXECUTE FUNCTION quantlab.prevent_simulation_nav_review_mutation();
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM quantlab.simulation_batches
                WHERE recommendation_snapshot_id IS NULL
                   OR signal_at IS NOT NULL
                   OR execution_not_before IS NOT NULL
            ) OR EXISTS (
                SELECT 1
                FROM quantlab.simulation_portfolios
                WHERE recommendation_portfolio_id IS NULL
            ) OR EXISTS (
                SELECT recommendation_snapshot_id
                FROM quantlab.simulation_batches
                WHERE recommendation_snapshot_id IS NOT NULL
                GROUP BY recommendation_snapshot_id
                HAVING COUNT(*) > 1
            ) OR EXISTS (
                SELECT recommendation_portfolio_id
                FROM quantlab.simulation_portfolios
                WHERE recommendation_portfolio_id IS NOT NULL
                GROUP BY recommendation_portfolio_id
                HAVING COUNT(*) > 1
            ) OR EXISTS (
                SELECT 1
                FROM quantlab.simulation_nav
                WHERE reviewed_at IS NOT NULL
            ) THEN
                RAISE EXCEPTION
                    '0037 downgrade blocked: governed simulation or NAV review records exist';
            END IF;
        END
        $$;
        """
    )
    op.execute(
        """
        DROP TRIGGER IF EXISTS trg_simulation_nav_review_immutable
        ON quantlab.simulation_nav;
        """
    )
    op.execute(
        """
        DROP FUNCTION IF EXISTS quantlab.prevent_simulation_nav_review_mutation();
        """
    )
    for name in ("ck_simulation_nav_review_complete", "ck_simulation_nav_scope"):
        op.drop_constraint(name, "simulation_nav", type_="check", schema="quantlab")
    for name in (
        "review_note",
        "review_evidence_sha256",
        "reviewed_at",
        "reviewed_by",
        "produced_by",
        "nav_scope",
    ):
        op.drop_column("simulation_nav", name, schema="quantlab")

    for name in (
        "ck_strategy_allocation_members_cap",
        "ck_strategy_allocation_members_risk_budget",
        "ck_strategy_allocation_members_role",
    ):
        op.drop_constraint(name, "strategy_allocation_members", type_="check", schema="quantlab")
    for name in ("member_cap", "risk_budget", "role"):
        op.drop_column("strategy_allocation_members", name, schema="quantlab")

    for table in ("simulation_positions", "simulation_fills", "simulation_orders"):
        op.drop_constraint(
            f"ck_{table}_position_side", table, type_="check", schema="quantlab"
        )
        for name in ("borrow_cost", "position_side", "leg_no", "atomic_group_id"):
            op.drop_column(table, name, schema="quantlab")

    op.drop_index(
        "uq_simulation_batches_portfolio_recommendation",
        table_name="simulation_batches",
        schema="quantlab",
    )
    op.drop_constraint(
        "ck_simulation_batches_next_bar_time",
        "simulation_batches",
        type_="check",
        schema="quantlab",
    )
    op.create_unique_constraint(
        "simulation_batches_recommendation_snapshot_id_key",
        "simulation_batches",
        ["recommendation_snapshot_id"],
        schema="quantlab",
    )
    op.alter_column(
        "simulation_batches",
        "recommendation_snapshot_id",
        existing_type=sa.String(),
        nullable=False,
        schema="quantlab",
    )
    for name in (
        "execution_contract_hash",
        "execution_adapter",
        "target_payload_json",
        "source_snapshot_id",
        "created_by",
        "execution_not_before",
        "signal_at",
    ):
        op.drop_column("simulation_batches", name, schema="quantlab")

    for name in (
        "ck_simulation_portfolios_frequency",
        "ck_simulation_portfolios_adapter",
        "ck_simulation_portfolios_source_type",
    ):
        op.drop_constraint(name, "simulation_portfolios", type_="check", schema="quantlab")
    op.drop_index(
        "uq_simulation_portfolios_source_execution",
        table_name="simulation_portfolios",
        schema="quantlab",
    )
    op.alter_column(
        "simulation_portfolios",
        "recommendation_portfolio_id",
        existing_type=sa.String(),
        nullable=False,
        schema="quantlab",
    )
    op.create_unique_constraint(
        "simulation_portfolios_recommendation_portfolio_id_key",
        "simulation_portfolios",
        ["recommendation_portfolio_id"],
        schema="quantlab",
    )
    for name in (
        "execution_contract_hash",
        "execution_frequency",
        "execution_adapter",
        "source_id",
        "source_type",
    ):
        op.drop_column("simulation_portfolios", name, schema="quantlab")

    for name in (
        "rdagent_commit",
        "rdagent_version",
        "qlib_commit",
        "qlib_version",
        "execution_contract_hash",
        "execution_frequency",
        "signal_frequency",
    ):
        op.drop_column("backtest_runs", name, schema="quantlab")
    for table in ("factor_evaluations", "strategy_versions"):
        for name in (
            "rdagent_commit",
            "rdagent_version",
            "qlib_commit",
            "qlib_version",
            "execution_contract_hash",
            "execution_frequency",
            "signal_horizon",
            "signal_frequency",
        ):
            op.drop_column(table, name, schema="quantlab")
