"""Add explicit strategy horizons, sealed forward criteria and health history.

Revision ID: 0072_strategy_horizons
Revises: 0071_retire_pair_writes
"""

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0072_strategy_horizons"
down_revision: str | None = "0071_retire_pair_writes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
JSON = sa.JSON().with_variant(JSONB(), "postgresql")
HORIZON_VERSION = "research-horizon-v1"
FORWARD_GATE_VERSION = "strategy-forward-gate-v2"
HORIZON_SHA256 = {
    "short_1_5d": "a895312d55f19ffaf35e90c4bd7003af337c94e18b61e27b4ee62a584860e1e9",
    "swing_1_6m": "cfc3d9ea05de009af3b4e283f83e9e840c45a0d3913f3cb938c34882b76b16f0",
    "long_1_3y": "0e9eae6438b44f8a4234a53f55999d4880dd0917772fb74351bd02f770f11879",
    "legacy_ambiguous": "6fb5cd40086f2e46850ba2d6b15e5ca7ec95191f4e42436639c7155cdecf3324",
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _legacy_horizon() -> dict[str, Any]:
    return {
        "horizon_profile": "legacy_ambiguous",
        "label_horizons_sessions": [],
        "decision_interval_sessions": None,
        "review_interval_sessions": None,
        "holding_min_sessions": None,
        "holding_target_sessions": None,
        "holding_max_sessions": None,
        "execution_lag_sessions": None,
        "purge_sessions": None,
        "embargo_sessions": None,
        "sealed_oos_required": False,
        "sealed_oos_sessions": None,
        "contract_version": HORIZON_VERSION,
    }


def _forward_criteria(row: Any) -> dict[str, Any]:
    return {
        "contract_version": FORWARD_GATE_VERSION,
        "horizon_profile": str(row.horizon_profile),
        "horizon_contract_sha256": str(row.horizon_contract_sha256),
        "thresholds": {
            "min_forward_calendar_days": int(row.min_forward_calendar_days),
            "min_forward_trading_days": int(row.min_forward_trading_days),
            "min_decision_batches": int(row.min_decision_batches),
            "min_completed_cycles": int(row.min_completed_cycles),
            "min_closed_round_trips": int(row.min_closed_round_trips),
            "min_review_events": int(row.min_review_events),
            "min_financial_report_reviews": int(row.min_financial_report_reviews),
            "min_data_completeness": float(row.min_data_completeness),
            "min_reconciliation_rate": float(row.min_reconciliation_rate),
            "max_cost_deviation": float(row.max_cost_deviation),
        },
    }


def upgrade() -> None:
    # The governed three-horizon account is deliberately daily-native: D close
    # creates a signal and D+1 open is the earliest execution. Migration 0037
    # predated that contract and restricted simulations to minute datasets.
    op.drop_constraint(
        "ck_simulation_portfolios_frequency",
        "simulation_portfolios",
        schema=SCHEMA,
        type_="check",
    )
    op.create_check_constraint(
        "ck_simulation_portfolios_frequency",
        "simulation_portfolios",
        "execution_frequency IN ('day', '1min', '5min')",
        schema=SCHEMA,
    )

    for column in (
        sa.Column("horizon_profile", sa.String()),
        sa.Column("label_horizons_json", JSON),
        sa.Column("decision_interval_sessions", sa.Integer()),
        sa.Column("review_interval_sessions", sa.Integer()),
        sa.Column("holding_min_sessions", sa.Integer()),
        sa.Column("holding_target_sessions", sa.Integer()),
        sa.Column("holding_max_sessions", sa.Integer()),
        sa.Column("execution_lag_sessions", sa.Integer()),
        sa.Column("purge_sessions", sa.Integer()),
        sa.Column("embargo_sessions", sa.Integer()),
        sa.Column("sealed_oos_required", sa.Boolean()),
        sa.Column("sealed_oos_sessions", sa.Integer()),
        sa.Column("horizon_contract_json", JSON),
        sa.Column("horizon_contract_sha256", sa.String()),
        sa.Column(
            "source_research_artifact_id",
            sa.String(),
            sa.ForeignKey(
                "quantlab.research_run_artifacts.id",
                ondelete="RESTRICT",
            ),
        ),
        sa.Column("strategy_rules_sha256", sa.String()),
    ):
        op.add_column("strategy_versions", column, schema=SCHEMA)

    legacy = _legacy_horizon()
    legacy_sha256 = _canonical_sha256(legacy)
    bind = op.get_bind()
    bind.execute(
        sa.text(
            """
            UPDATE quantlab.strategy_versions
            SET horizon_profile = 'legacy_ambiguous',
                label_horizons_json = CAST(:labels AS jsonb),
                decision_interval_sessions = NULL,
                review_interval_sessions = NULL,
                holding_min_sessions = NULL,
                holding_target_sessions = NULL,
                holding_max_sessions = NULL,
                execution_lag_sessions = NULL,
                purge_sessions = NULL,
                embargo_sessions = NULL,
                sealed_oos_required = false,
                sealed_oos_sessions = NULL,
                horizon_contract_json = CAST(:contract AS jsonb),
                horizon_contract_sha256 = :sha256
            """
        ),
        {
            "labels": json.dumps([], separators=(",", ":")),
            "contract": json.dumps(legacy, ensure_ascii=False, separators=(",", ":")),
            "sha256": legacy_sha256,
        },
    )
    # Pre-0072 recommendation authority has no explicit horizon or sealed
    # forward evidence. Preserve the version and its audit history, but revoke
    # only the authority projection. The migration event lets a schema
    # downgrade restore the exact prior marker without guessing.
    bind.execute(
        sa.text(
            """
            INSERT INTO quantlab.strategy_events (
                strategy_id, strategy_version_id, event_type, actor,
                payload_json, created_at
            )
            SELECT strategy_id,
                   id,
                   'strategy.legacy_authority_revoked_by_0072',
                   'migration:0072',
                   jsonb_build_object(
                       'prior_promotion_stage', promotion_stage,
                       'reason', 'explicit horizon and forward evidence required'
                   ),
                   now()
            FROM quantlab.strategy_versions
            WHERE horizon_profile = 'legacy_ambiguous'
              AND promotion_stage = 'recommendation_enabled'
            """
        )
    )
    bind.execute(
        sa.text(
            """
            UPDATE quantlab.strategy_versions
            SET promotion_stage = 'paper'
            WHERE horizon_profile = 'legacy_ambiguous'
              AND promotion_stage = 'recommendation_enabled'
            """
        )
    )
    for name in (
        "horizon_profile",
        "label_horizons_json",
        "sealed_oos_required",
        "horizon_contract_json",
        "horizon_contract_sha256",
    ):
        op.alter_column("strategy_versions", name, nullable=False, schema=SCHEMA)

    op.create_check_constraint(
        "ck_strategy_versions_horizon_profile",
        "strategy_versions",
        "horizon_profile IN "
        "('short_1_5d', 'swing_1_6m', 'long_1_3y', 'legacy_ambiguous')",
        schema=SCHEMA,
    )
    op.create_check_constraint(
        "ck_strategy_versions_horizon_identity",
        "strategy_versions",
        "(jsonb_typeof(label_horizons_json) = 'array' "
        "AND jsonb_typeof(horizon_contract_json) = 'object' "
        "AND horizon_contract_json ?& ARRAY["
        "'horizon_profile','label_horizons_sessions','decision_interval_sessions',"
        "'review_interval_sessions','holding_min_sessions','holding_target_sessions',"
        "'holding_max_sessions','execution_lag_sessions','purge_sessions',"
        "'embargo_sessions','sealed_oos_required','sealed_oos_sessions',"
        "'contract_version'] "
        "AND (horizon_contract_json - ARRAY["
        "'horizon_profile','label_horizons_sessions','decision_interval_sessions',"
        "'review_interval_sessions','holding_min_sessions','holding_target_sessions',"
        "'holding_max_sessions','execution_lag_sessions','purge_sessions',"
        "'embargo_sessions','sealed_oos_required','sealed_oos_sessions',"
        "'contract_version']) = '{}'::jsonb "
        "AND horizon_contract_json ->> 'horizon_profile' = horizon_profile "
        "AND horizon_contract_json ->> 'contract_version' = 'research-horizon-v1' "
        "AND horizon_contract_json -> 'label_horizons_sessions' = label_horizons_json "
        "AND (horizon_contract_json ->> 'decision_interval_sessions')::integer "
        "    IS NOT DISTINCT FROM decision_interval_sessions "
        "AND (horizon_contract_json ->> 'review_interval_sessions')::integer "
        "    IS NOT DISTINCT FROM review_interval_sessions "
        "AND (horizon_contract_json ->> 'holding_min_sessions')::integer "
        "    IS NOT DISTINCT FROM holding_min_sessions "
        "AND (horizon_contract_json ->> 'holding_target_sessions')::integer "
        "    IS NOT DISTINCT FROM holding_target_sessions "
        "AND (horizon_contract_json ->> 'holding_max_sessions')::integer "
        "    IS NOT DISTINCT FROM holding_max_sessions "
        "AND (horizon_contract_json ->> 'execution_lag_sessions')::integer "
        "    IS NOT DISTINCT FROM execution_lag_sessions "
        "AND (horizon_contract_json ->> 'purge_sessions')::integer "
        "    IS NOT DISTINCT FROM purge_sessions "
        "AND (horizon_contract_json ->> 'embargo_sessions')::integer "
        "    IS NOT DISTINCT FROM embargo_sessions "
        "AND (horizon_contract_json ->> 'sealed_oos_required')::boolean "
        "    IS NOT DISTINCT FROM sealed_oos_required "
        "AND (horizon_contract_json ->> 'sealed_oos_sessions')::integer "
        "    IS NOT DISTINCT FROM sealed_oos_sessions "
        "AND CASE horizon_profile "
        f"WHEN 'short_1_5d' THEN horizon_contract_sha256 = '{HORIZON_SHA256['short_1_5d']}' "
        f"WHEN 'swing_1_6m' THEN horizon_contract_sha256 = '{HORIZON_SHA256['swing_1_6m']}' "
        f"WHEN 'long_1_3y' THEN horizon_contract_sha256 = '{HORIZON_SHA256['long_1_3y']}' "
        "WHEN 'legacy_ambiguous' THEN horizon_contract_sha256 = "
        f"'{HORIZON_SHA256['legacy_ambiguous']}' ELSE false END) IS TRUE",
        schema=SCHEMA,
    )
    op.create_check_constraint(
        "ck_strategy_versions_horizon_values",
        "strategy_versions",
        "((horizon_profile = 'legacy_ambiguous' "
        " AND label_horizons_json = '[]'::jsonb "
        " AND decision_interval_sessions IS NULL "
        " AND review_interval_sessions IS NULL "
        " AND holding_min_sessions IS NULL "
        " AND holding_target_sessions IS NULL "
        " AND holding_max_sessions IS NULL "
        " AND execution_lag_sessions IS NULL "
        " AND purge_sessions IS NULL AND embargo_sessions IS NULL "
        " AND sealed_oos_required = false AND sealed_oos_sessions IS NULL) OR "
        "(horizon_profile = 'short_1_5d' "
        " AND label_horizons_json = '[1,2,3,5]'::jsonb "
        " AND decision_interval_sessions = 1 AND review_interval_sessions = 1 "
        " AND holding_min_sessions = 1 AND holding_target_sessions = 3 "
        " AND holding_max_sessions = 5 AND execution_lag_sessions = 1 "
        " AND purge_sessions = 6 AND embargo_sessions = 6 "
        " AND sealed_oos_required = true AND sealed_oos_sessions = 252) OR "
        "(horizon_profile = 'swing_1_6m' "
        " AND label_horizons_json = '[21,63,126]'::jsonb "
        " AND decision_interval_sessions = 5 AND review_interval_sessions = 5 "
        " AND holding_min_sessions = 21 AND holding_target_sessions = 63 "
        " AND holding_max_sessions = 126 AND execution_lag_sessions = 1 "
        " AND purge_sessions = 127 AND embargo_sessions = 127 "
        " AND sealed_oos_required = true AND sealed_oos_sessions = 504) OR "
        "(horizon_profile = 'long_1_3y' "
        " AND label_horizons_json = '[63,126,252]'::jsonb "
        " AND decision_interval_sessions = 21 AND review_interval_sessions = 21 "
        " AND holding_min_sessions = 252 AND holding_target_sessions = 504 "
        " AND holding_max_sessions = 756 AND execution_lag_sessions = 1 "
        " AND purge_sessions = 253 AND embargo_sessions = 253 "
        " AND sealed_oos_required = true AND sealed_oos_sessions = 756)) IS TRUE",
        schema=SCHEMA,
    )
    op.create_check_constraint(
        "ck_strategy_versions_rule_identity",
        "strategy_versions",
        "((horizon_profile = 'legacy_ambiguous') OR "
        "strategy_rules_sha256 ~ '^[0-9a-f]{64}$') IS TRUE",
        schema=SCHEMA,
    )
    op.create_check_constraint(
        "ck_strategy_versions_legacy_authority",
        "strategy_versions",
        "(horizon_profile <> 'legacy_ambiguous' OR "
        "promotion_stage IS DISTINCT FROM 'recommendation_enabled') IS TRUE",
        schema=SCHEMA,
    )
    # Before this migration, ``approved`` also meant the single active
    # version. The three-horizon lifecycle separates historical approval
    # (paper validation) from recommendation authority, so a paper challenger
    # must coexist with its still-live incumbent until final promotion.
    op.drop_index(
        "uq_strategy_versions_approved",
        table_name="strategy_versions",
        schema=SCHEMA,
    )
    op.create_index(
        "uq_strategy_versions_approved",
        "strategy_versions",
        ["strategy_id"],
        unique=True,
        postgresql_where=sa.text(
            "status = 'approved' AND promotion_stage = 'recommendation_enabled'"
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_strategy_versions_horizon_status",
        "strategy_versions",
        ["horizon_profile", "status"],
        schema=SCHEMA,
    )
    op.create_index(
        "uq_strategy_versions_active_horizon",
        "strategy_versions",
        ["horizon_profile"],
        unique=True,
        postgresql_where=sa.text(
            "status = 'approved' AND promotion_stage = 'recommendation_enabled' "
            "AND horizon_profile <> 'legacy_ambiguous'"
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "uq_strategy_versions_source_research_artifact",
        "strategy_versions",
        ["source_research_artifact_id"],
        unique=True,
        postgresql_where=sa.text("source_research_artifact_id IS NOT NULL"),
        schema=SCHEMA,
    )

    for name in (
        "min_forward_trading_days",
        "min_closed_round_trips",
        "min_review_events",
        "min_financial_report_reviews",
    ):
        op.add_column(
            "strategy_forward_gates",
            sa.Column(name, sa.Integer(), nullable=False, server_default="0"),
            schema=SCHEMA,
        )
        op.alter_column(
            "strategy_forward_gates", name, server_default=None, schema=SCHEMA
        )
    op.add_column(
        "strategy_forward_gates", sa.Column("criteria_json", JSON), schema=SCHEMA
    )
    op.add_column(
        "strategy_forward_gates",
        sa.Column("criteria_sha256", sa.String()),
        schema=SCHEMA,
    )
    gates = sa.table(
        "strategy_forward_gates",
        sa.column("strategy_version_id", sa.String()),
        sa.column("min_forward_calendar_days", sa.Integer()),
        sa.column("min_forward_trading_days", sa.Integer()),
        sa.column("min_decision_batches", sa.Integer()),
        sa.column("min_completed_cycles", sa.Integer()),
        sa.column("min_closed_round_trips", sa.Integer()),
        sa.column("min_review_events", sa.Integer()),
        sa.column("min_financial_report_reviews", sa.Integer()),
        sa.column("min_data_completeness", sa.Float()),
        sa.column("min_reconciliation_rate", sa.Float()),
        sa.column("max_cost_deviation", sa.Float()),
        sa.column("criteria_json", JSON),
        sa.column("criteria_sha256", sa.String()),
        schema=SCHEMA,
    )
    versions = sa.table(
        "strategy_versions",
        sa.column("id", sa.String()),
        sa.column("horizon_profile", sa.String()),
        sa.column("horizon_contract_sha256", sa.String()),
        schema=SCHEMA,
    )
    rows = bind.execute(
        sa.select(
            gates.c.strategy_version_id,
            gates.c.min_forward_calendar_days,
            gates.c.min_forward_trading_days,
            gates.c.min_decision_batches,
            gates.c.min_completed_cycles,
            gates.c.min_closed_round_trips,
            gates.c.min_review_events,
            gates.c.min_financial_report_reviews,
            gates.c.min_data_completeness,
            gates.c.min_reconciliation_rate,
            gates.c.max_cost_deviation,
            versions.c.horizon_profile,
            versions.c.horizon_contract_sha256,
        ).join(versions, versions.c.id == gates.c.strategy_version_id)
    ).all()
    for row in rows:
        criteria = _forward_criteria(row)
        bind.execute(
            gates.update()
            .where(gates.c.strategy_version_id == row.strategy_version_id)
            .values(
                criteria_json=criteria,
                criteria_sha256=_canonical_sha256(criteria),
            )
        )
    op.alter_column(
        "strategy_forward_gates", "criteria_json", nullable=False, schema=SCHEMA
    )
    op.alter_column(
        "strategy_forward_gates", "criteria_sha256", nullable=False, schema=SCHEMA
    )
    op.create_check_constraint(
        "ck_strategy_forward_gate_criteria",
        "strategy_forward_gates",
        "(jsonb_typeof(criteria_json) = 'object' "
        "AND criteria_json ?& ARRAY['contract_version','horizon_profile',"
        "'horizon_contract_sha256','thresholds'] "
        "AND (criteria_json - ARRAY['contract_version','horizon_profile',"
        "'horizon_contract_sha256','thresholds']) = '{}'::jsonb "
        "AND criteria_json ->> 'contract_version' = 'strategy-forward-gate-v2' "
        "AND criteria_json ->> 'horizon_profile' IN "
        "('short_1_5d','swing_1_6m','long_1_3y','legacy_ambiguous') "
        "AND criteria_json ->> 'horizon_contract_sha256' ~ '^[0-9a-f]{64}$' "
        "AND jsonb_typeof(criteria_json -> 'thresholds') = 'object' "
        "AND (criteria_json -> 'thresholds') ?& ARRAY["
        "'min_forward_calendar_days','min_forward_trading_days',"
        "'min_decision_batches','min_completed_cycles','min_closed_round_trips',"
        "'min_review_events','min_financial_report_reviews',"
        "'min_data_completeness','min_reconciliation_rate','max_cost_deviation'] "
        "AND ((criteria_json -> 'thresholds') - ARRAY["
        "'min_forward_calendar_days','min_forward_trading_days',"
        "'min_decision_batches','min_completed_cycles','min_closed_round_trips',"
        "'min_review_events','min_financial_report_reviews',"
        "'min_data_completeness','min_reconciliation_rate','max_cost_deviation']) "
        "= '{}'::jsonb "
        "AND (criteria_json -> 'thresholds' ->> 'min_forward_calendar_days')::integer "
        "IS NOT DISTINCT FROM min_forward_calendar_days "
        "AND (criteria_json -> 'thresholds' ->> 'min_forward_trading_days')::integer "
        "IS NOT DISTINCT FROM min_forward_trading_days "
        "AND (criteria_json -> 'thresholds' ->> 'min_decision_batches')::integer "
        "IS NOT DISTINCT FROM min_decision_batches "
        "AND (criteria_json -> 'thresholds' ->> 'min_completed_cycles')::integer "
        "IS NOT DISTINCT FROM min_completed_cycles "
        "AND (criteria_json -> 'thresholds' ->> 'min_closed_round_trips')::integer "
        "IS NOT DISTINCT FROM min_closed_round_trips "
        "AND (criteria_json -> 'thresholds' ->> 'min_review_events')::integer "
        "IS NOT DISTINCT FROM min_review_events "
        "AND (criteria_json -> 'thresholds' ->> 'min_financial_report_reviews')::integer "
        "IS NOT DISTINCT FROM min_financial_report_reviews "
        "AND (criteria_json -> 'thresholds' ->> 'min_data_completeness')::double precision "
        "IS NOT DISTINCT FROM min_data_completeness "
        "AND (criteria_json -> 'thresholds' ->> 'min_reconciliation_rate')::double precision "
        "IS NOT DISTINCT FROM min_reconciliation_rate "
        "AND (criteria_json -> 'thresholds' ->> 'max_cost_deviation')::double precision "
        "IS NOT DISTINCT FROM max_cost_deviation "
        "AND min_forward_calendar_days >= 0 "
        "AND min_forward_trading_days >= 0 AND min_closed_round_trips >= 0 "
        "AND min_decision_batches >= 0 AND min_completed_cycles >= 0 "
        "AND min_review_events >= 0 "
        "AND min_financial_report_reviews >= 0 "
        "AND min_data_completeness >= 0 AND min_data_completeness <= 1 "
        "AND min_reconciliation_rate >= 0 AND min_reconciliation_rate <= 1 "
        "AND max_cost_deviation >= 0 "
        "AND criteria_sha256 ~ '^[0-9a-f]{64}$') IS TRUE",
        schema=SCHEMA,
    )

    op.create_table(
        "investor_simulation_profiles",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("profile_key", sa.String(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column(
            "supersedes_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.investor_simulation_profiles.id"),
        ),
        # Deliberately no server default: first-run capital is a required
        # investor decision, not a platform assumption.
        sa.Column("initial_capital", sa.Numeric(20, 6), nullable=False),
        sa.Column("risk_profile", sa.String(), nullable=False),
        sa.Column("min_cash_weight", sa.Float(), nullable=False),
        sa.Column("max_gross_exposure", sa.Float(), nullable=False),
        sa.Column("market_permissions_json", JSON, nullable=False),
        sa.Column("content_sha256", sa.String(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_by", sa.String(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "profile_key", "version", name="uq_investor_simulation_profile_version"
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'active', 'retired')",
            name="ck_investor_simulation_profile_status",
        ),
        sa.CheckConstraint(
            "initial_capital > 0 AND min_cash_weight >= 0 AND min_cash_weight < 1 "
            "AND max_gross_exposure > 0 AND max_gross_exposure <= 1 "
            "AND min_cash_weight + max_gross_exposure <= 1",
            name="ck_investor_simulation_profile_risk_values",
        ),
        sa.CheckConstraint(
            "length(trim(profile_key)) > 0 AND length(trim(risk_profile)) > 0 "
            "AND jsonb_typeof(market_permissions_json) = 'object' "
            "AND content_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_investor_simulation_profile_identity",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "uq_investor_simulation_profile_active",
        "investor_simulation_profiles",
        ["profile_key"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_investor_simulation_profiles_updated",
        "investor_simulation_profiles",
        [sa.text("updated_at DESC")],
        schema=SCHEMA,
    )

    op.create_table(
        "strategy_health_snapshots",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "strategy_version_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.strategy_versions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("horizon_profile", sa.String(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("health_status", sa.String(), nullable=False),
        sa.Column("criteria_json", JSON, nullable=False),
        sa.Column("criteria_sha256", sa.String(), nullable=False),
        sa.Column("evidence_json", JSON, nullable=False),
        sa.Column("evidence_sha256", sa.String(), nullable=False),
        sa.Column("snapshot_sha256", sa.String(), nullable=False),
        sa.Column("recorded_by", sa.String(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "strategy_version_id",
            "as_of",
            "snapshot_sha256",
            name="uq_strategy_health_snapshot_identity",
        ),
        sa.CheckConstraint(
            "horizon_profile IN "
            "('short_1_5d', 'swing_1_6m', 'long_1_3y', 'legacy_ambiguous')",
            name="ck_strategy_health_horizon_profile",
        ),
        sa.CheckConstraint(
            "health_status IN "
            "('healthy', 'watch', 'restricted', 'suspended', 'retired')",
            name="ck_strategy_health_status",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(criteria_json) = 'object' "
            "AND jsonb_typeof(evidence_json) = 'object' "
            "AND criteria_sha256 ~ '^[0-9a-f]{64}$' "
            "AND evidence_sha256 ~ '^[0-9a-f]{64}$' "
            "AND snapshot_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_strategy_health_seals",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_strategy_health_version_as_of",
        "strategy_health_snapshots",
        ["strategy_version_id", sa.text("as_of DESC")],
        schema=SCHEMA,
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION quantlab.reject_strategy_health_snapshot_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'strategy health snapshots are append-only';
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_strategy_health_snapshots_append_only
        BEFORE UPDATE OR DELETE ON quantlab.strategy_health_snapshots
        FOR EACH ROW EXECUTE FUNCTION quantlab.reject_strategy_health_snapshot_mutation()
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_strategy_health_snapshots_no_truncate
        BEFORE TRUNCATE ON quantlab.strategy_health_snapshots
        FOR EACH STATEMENT EXECUTE FUNCTION quantlab.reject_strategy_health_snapshot_mutation()
        """
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_simulation_portfolios_frequency",
        "simulation_portfolios",
        schema=SCHEMA,
        type_="check",
    )
    op.create_check_constraint(
        "ck_simulation_portfolios_frequency",
        "simulation_portfolios",
        "execution_frequency IN ('1min', '5min')",
        schema=SCHEMA,
    )

    op.execute(
        "DROP TRIGGER IF EXISTS trg_strategy_health_snapshots_no_truncate "
        "ON quantlab.strategy_health_snapshots"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS trg_strategy_health_snapshots_append_only "
        "ON quantlab.strategy_health_snapshots"
    )
    op.execute(
        "DROP FUNCTION IF EXISTS quantlab.reject_strategy_health_snapshot_mutation()"
    )
    op.drop_index(
        "idx_strategy_health_version_as_of",
        table_name="strategy_health_snapshots",
        schema=SCHEMA,
    )
    op.drop_table("strategy_health_snapshots", schema=SCHEMA)

    op.drop_index(
        "idx_investor_simulation_profiles_updated",
        table_name="investor_simulation_profiles",
        schema=SCHEMA,
    )
    op.drop_index(
        "uq_investor_simulation_profile_active",
        table_name="investor_simulation_profiles",
        schema=SCHEMA,
    )
    op.drop_table("investor_simulation_profiles", schema=SCHEMA)

    op.drop_constraint(
        "ck_strategy_forward_gate_criteria",
        "strategy_forward_gates",
        schema=SCHEMA,
        type_="check",
    )
    op.drop_column("strategy_forward_gates", "criteria_sha256", schema=SCHEMA)
    op.drop_column("strategy_forward_gates", "criteria_json", schema=SCHEMA)
    for column in (
        "min_financial_report_reviews",
        "min_review_events",
        "min_closed_round_trips",
        "min_forward_trading_days",
    ):
        op.drop_column("strategy_forward_gates", column, schema=SCHEMA)

    op.drop_index(
        "uq_strategy_versions_source_research_artifact",
        table_name="strategy_versions",
        schema=SCHEMA,
    )
    op.drop_index(
        "uq_strategy_versions_active_horizon",
        table_name="strategy_versions",
        schema=SCHEMA,
    )
    op.drop_index(
        "idx_strategy_versions_horizon_status",
        table_name="strategy_versions",
        schema=SCHEMA,
    )
    op.drop_index(
        "uq_strategy_versions_approved",
        table_name="strategy_versions",
        schema=SCHEMA,
    )
    # The pre-0072 schema can represent only one approved version per family.
    # Never silently retire challengers during schema rollback: fail closed so
    # an operator can make and audit an explicit lifecycle decision first.
    op.execute(
        sa.text(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM quantlab.strategy_versions
                    WHERE status = 'approved'
                    GROUP BY strategy_id
                    HAVING count(*) > 1
                ) THEN
                    RAISE EXCEPTION
                        '0072 downgrade blocked: multiple approved versions need a decision';
                END IF;
            END;
            $$
            """
        )
    )
    op.create_index(
        "uq_strategy_versions_approved",
        "strategy_versions",
        ["strategy_id"],
        unique=True,
        postgresql_where=sa.text("status = 'approved'"),
        schema=SCHEMA,
    )
    for constraint in (
        "ck_strategy_versions_legacy_authority",
        "ck_strategy_versions_rule_identity",
        "ck_strategy_versions_horizon_values",
        "ck_strategy_versions_horizon_identity",
        "ck_strategy_versions_horizon_profile",
    ):
        op.drop_constraint(
            constraint,
            "strategy_versions",
            schema=SCHEMA,
            type_="check",
        )
    op.execute(
        sa.text(
            """
            UPDATE quantlab.strategy_versions AS version
            SET promotion_stage = event.payload_json ->> 'prior_promotion_stage'
            FROM (
                SELECT DISTINCT ON (strategy_version_id)
                       strategy_version_id, payload_json
                FROM quantlab.strategy_events
                WHERE event_type = 'strategy.legacy_authority_revoked_by_0072'
                  AND actor = 'migration:0072'
                ORDER BY strategy_version_id, created_at DESC, id DESC
            ) AS event
            WHERE version.id = event.strategy_version_id
            """
        )
    )
    for column in (
        "strategy_rules_sha256",
        "source_research_artifact_id",
        "horizon_contract_sha256",
        "horizon_contract_json",
        "sealed_oos_sessions",
        "sealed_oos_required",
        "embargo_sessions",
        "purge_sessions",
        "execution_lag_sessions",
        "holding_max_sessions",
        "holding_target_sessions",
        "holding_min_sessions",
        "review_interval_sessions",
        "decision_interval_sessions",
        "label_horizons_json",
        "horizon_profile",
    ):
        op.drop_column("strategy_versions", column, schema=SCHEMA)
