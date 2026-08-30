from datetime import UTC, datetime

import pytest
from alembic import command
from sqlalchemy import insert, inspect, text
from sqlalchemy.pool import NullPool

from quant_data.database import (
    audit_events,
    open_database,
    transparent_baseline_pre_result_repairs,
)
from quant_platform.db_cli import alembic_config


@pytest.mark.no_database
def test_one_shot_database_mode_does_not_retain_idle_connections(monkeypatch) -> None:
    monkeypatch.setenv("QUANTLAB_DATABASE_DISABLE_POOL", "1")
    engine = open_database("postgresql+psycopg://user:password@127.0.0.1/example")
    try:
        assert isinstance(engine.pool, NullPool)
    finally:
        engine.dispose()


def test_work_units_page_group_index_matches_runtime_lookup(database_url: str) -> None:
    engine = open_database(database_url)
    with engine.connect() as connection:
        definition = connection.execute(
            text(
                "SELECT pg_get_indexdef(to_regclass("
                "'quantlab.idx_work_units_dataset_page_group'))"
            )
        ).scalar_one()

    assert "ON quantlab.work_units USING btree" in definition
    assert "dataset" in definition
    assert "scope_json ->> 'page_group'::text" in definition
    assert "character varying" in definition


def test_explicit_migration_url_overrides_host_database_environment(
    database_url: str, monkeypatch
) -> None:
    from alembic import command

    from quant_platform.db_cli import alembic_config

    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://invalid:invalid@127.0.0.1:1/must_not_be_used",
    )
    command.current(alembic_config(database_url))


def test_database_is_at_versioned_control_plane_schema(database_url: str) -> None:
    engine = open_database(database_url)
    inspector = inspect(engine)
    assert set(inspector.get_table_names(schema="quantlab")) >= {
        "alembic_version",
        "account_netting_plans",
        "strategy_forward_gates",
        "strategy_promotion_stages",
        "strategy_health_snapshots",
        "investor_simulation_profiles",
        "jobs",
        "research_runs",
        "factor_candidates",
        "factor_evaluations",
        "factor_definitions",
        "factor_library_versions",
        "factor_library_members",
        "factor_similarity_edges",
        "factor_definition_similarity_edges",
        "research_sota_versions",
        "research_sota_members",
        "oos_vintages",
        "transparent_baseline_pre_result_repairs",
        "autopilot_cycles",
        "autopilot_branches",
        "research_tournaments",
        "research_tournament_trials",
        "model_ensemble_candidates",
        "model_ensemble_evaluations",
        "capital_oos_alpha_families",
        "capital_oos_alpha_batches",
        "capital_oos_legacy_attempts",
        "research_report_backfill_days",
        "research_assets",
        "research_asset_consumptions",
        "research_run_artifacts",
        "model_candidates",
        "model_evaluations",
        "quant_bundle_candidates",
        "quant_bundle_evaluations",
        "candidate_asset_links",
        "recommendation_portfolios",
        "recommendation_snapshots",
        "recommendation_holdings",
        "recommendation_nav",
        "simulation_portfolios",
        "simulation_batches",
        "simulation_orders",
        "simulation_fills",
        "simulation_positions",
        "simulation_position_reservations",
        "simulation_security_events",
        "simulation_day_attributions",
        "simulation_cash_flows",
        "simulation_cash_lots",
        "simulation_cash_events",
        "simulation_cash_event_allocations",
        "simulation_cash_reservations",
        "simulation_nav",
        "simulation_events",
        "research_events",
        "strategies",
        "strategy_versions",
        "model_artifacts",
        "strategy_factors",
        "strategy_pairs",
        "pair_paper_portfolios",
        "pair_portfolio_batches",
        "pair_paper_orders",
        "pair_paper_fills",
        "pair_portfolio_nav",
        "pair_portfolio_risk_events",
        "pair_portfolio_reviews",
        "backtest_runs",
        "parameter_experiments",
        "parameter_experiment_trials",
        "research_campaigns",
        "research_campaign_events",
        "research_programs",
        "research_program_events",
        "broker_destinations",
        "broker_order_outbox",
        "broker_events",
        "broker_reconciliations",
        "broker_gateway_parents",
        "broker_gateway_children",
        "broker_gateway_attempts",
        "broker_gateway_events",
        "broker_gateway_nonces",
        "strategy_events",
        "paper_portfolios",
        "portfolio_batches",
        "paper_orders",
        "paper_fills",
        "paper_positions",
        "portfolio_nav",
        "risk_events",
        "portfolio_reviews",
        "strategy_allocations",
        "strategy_allocation_members",
        "strategy_allocation_nav",
        "strategy_allocation_events",
        "system_health_snapshots",
        "schedules",
        "schedule_runs",
        "allocation_schedule_groups",
        "allocation_schedule_members",
        "alerts",
        "users",
        "auth_sessions",
        "audit_events",
        "work_units",
        "data_tasks",
        "platform_configs",
        "platform_config_revisions",
        "market_permission_versions",
        "shadow_account_snapshots",
        "simulation_external_flows",
        "simulation_corporate_events",
        "simulation_fee_adjustments",
    }
    with engine.connect() as connection:
        revision = connection.execute(
            text("SELECT version_num FROM quantlab.alembic_version")
        ).scalar_one()
    assert revision == "0080_baseline_v13_seal"
    assert {"horizon_profile", "primary_label_policy_sha256"} <= {
        column["name"]
        for column in inspector.get_columns("autopilot_cycles", schema="quantlab")
    }
    assert any(
        constraint.get("name") == "uq_autopilot_cycle_dataset_horizon"
        and constraint.get("column_names")
        == ["dataset_identity_sha256", "horizon_profile"]
        for constraint in inspector.get_unique_constraints(
            "autopilot_cycles", schema="quantlab"
        )
    )
    assert "ck_autopilot_cycles_horizon" in {
        constraint["name"]
        for constraint in inspector.get_check_constraints(
            "autopilot_cycles", schema="quantlab"
        )
    }
    assert "ck_strategy_versions_v12_runtime_identity" in {
        constraint["name"]
        for constraint in inspector.get_check_constraints(
            "strategy_versions", schema="quantlab"
        )
    }
    assert "ck_strategy_versions_v13_runtime_identity" in {
        constraint["name"]
        for constraint in inspector.get_check_constraints(
            "strategy_versions", schema="quantlab"
        )
    }
    assert "capital_oos_alpha_batch_id" in {
        column["name"]
        for column in inspector.get_columns("oos_vintages", schema="quantlab")
    }


    assert {
        "economic_hypothesis_group",
        "hypothesis_group_cap",
    } <= {
        column["name"]
        for column in inspector.get_columns("strategies", schema="quantlab")
    }
    assert {
        "economic_hypothesis_group",
        "hypothesis_group_cap",
        "shared_experiment_count",
    } <= {
        column["name"]
        for column in inspector.get_columns(
            "strategy_allocation_members", schema="quantlab"
        )
    }
    assert {
        "strategy_version_id",
        "artifact_key",
        "strategy_spec_sha256",
        "model_recipe_sha256",
        "dataset_identity_sha256",
        "execution_environment_sha256",
        "artifact_sha256",
        "predictions_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_format",
        "model_data_contract_sha256",
        "training_kind",
        "training_evidence_json",
        "training_evidence_sha256",
        "valid_until",
    } <= {
        column["name"]
        for column in inspector.get_columns("model_artifacts", schema="quantlab")
    }
    assert {
        "portfolio_id",
        "fill_id",
        "adjustment_key",
        "previously_confirmed_fee",
        "final_fee",
        "adjustment_amount",
        "evidence_sha256",
    } <= {
        column["name"]
        for column in inspector.get_columns(
            "simulation_fee_adjustments", schema="quantlab"
        )
    }
    assert {
        "portfolio_id",
        "lot_key",
        "source_type",
        "free_amount",
        "frozen_amount",
        "tradable_at",
        "withdrawable_at",
    } <= {
        column["name"]
        for column in inspector.get_columns(
            "simulation_cash_lots", schema="quantlab"
        )
    }
    assert {
        "portfolio_id",
        "event_key",
        "event_type",
        "amount",
        "occurred_at",
    } <= {
        column["name"]
        for column in inspector.get_columns(
            "simulation_cash_events", schema="quantlab"
        )
    }
    assert {
        "portfolio_id",
        "order_id",
        "cash_lot_id",
        "reserved_amount",
        "remaining_amount",
    } <= {
        column["name"]
        for column in inspector.get_columns(
            "simulation_cash_reservations", schema="quantlab"
        )
    }
    assert {"frozen_quantity"} <= {
        column["name"]
        for column in inspector.get_columns(
            "simulation_positions", schema="quantlab"
        )
    }
    assert {
        "portfolio_id",
        "order_id",
        "instrument",
        "reserved_quantity",
        "remaining_quantity",
    } <= {
        column["name"]
        for column in inspector.get_columns(
            "simulation_position_reservations", schema="quantlab"
        )
    }
    assert {
        "portfolio_id",
        "order_id",
        "event_key",
        "event_type",
        "instrument",
        "quantity",
        "occurred_at",
    } <= {
        column["name"]
        for column in inspector.get_columns(
            "simulation_security_events", schema="quantlab"
        )
    }
    assert {
        "portfolio_id",
        "batch_id",
        "trade_date",
        "strategy_json",
        "industry_json",
        "asset_json",
        "cost_json",
        "execution_json",
        "coverage_status",
        "input_sha256",
    } <= {
        column["name"]
        for column in inspector.get_columns(
            "simulation_day_attributions", schema="quantlab"
        )
    }
    assert {
        "portfolio_id",
        "event_key",
        "event_type",
        "instrument",
        "effective_date",
        "payload_sha256",
        "details_json",
    } <= {
        column["name"]
        for column in inspector.get_columns(
            "simulation_corporate_events", schema="quantlab"
        )
    }
    assert {
        "portfolio_id",
        "flow_key",
        "trade_date",
        "timing",
        "amount",
        "created_by",
    } <= {
        column["name"]
        for column in inspector.get_columns("simulation_external_flows", schema="quantlab")
    }
    assert {
        "external_flow_open",
        "external_flow_close",
        "twr_daily_return",
        "investment_wealth",
        "twr_drawdown",
        "twr_status",
    } <= {
        column["name"]
        for column in inspector.get_columns("simulation_nav", schema="quantlab")
    }
    assert {
        "scope_type",
        "scope_key",
        "permission",
        "confirmation_source",
        "as_of",
        "valid_until",
        "relaxation_confirmed",
    } <= {
        column["name"]
        for column in inspector.get_columns("market_permission_versions", schema="quantlab")
    }
    assert {
        "account_id",
        "import_source",
        "cash",
        "holdings_json",
        "open_orders_json",
        "content_sha256",
        "imported_by",
        "imported_at",
    } <= {
        column["name"]
        for column in inspector.get_columns("shadow_account_snapshots", schema="quantlab")
    }
    order_columns = {
        column["name"]
        for column in inspector.get_columns("simulation_orders", schema="quantlab")
    }
    assert {
        "portfolio_id",
        "limit_price",
        "not_before",
        "not_after",
        "target_version",
        "account_netting_plan_id",
        "strategy_contributions_json",
        "plan_op",
        "cancel_reason",
        "updated_at",
    } <= order_columns
    batch_columns = {
        column["name"]
        for column in inspector.get_columns("simulation_batches", schema="quantlab")
    }
    assert {"account_netting_plan_id"} <= batch_columns
    assert {"promotion_stage"} <= {
        column["name"]
        for column in inspector.get_columns("strategy_versions", schema="quantlab")
    }
    assert {
        "min_forward_calendar_days",
        "min_forward_trading_days",
        "min_decision_batches",
        "min_completed_cycles",
        "min_closed_round_trips",
        "min_review_events",
        "min_financial_report_reviews",
        "min_data_completeness",
        "min_reconciliation_rate",
        "max_cost_deviation",
        "criteria_json",
        "criteria_sha256",
    } <= {
        column["name"]
        for column in inspector.get_columns("strategy_forward_gates", schema="quantlab")
    }
    assert {
        "strategy_version_id",
        "stage_index",
        "simulation_portfolio_id",
        "status",
        "source_contract_hash",
        "opened_at",
        "frozen_at",
        "freeze_reason",
    } <= {
        column["name"]
        for column in inspector.get_columns("strategy_promotion_stages", schema="quantlab")
    }
    assert {"liability_per_share"} <= {
        column["name"]
        for column in inspector.get_columns(
            "simulation_dividend_entitlements", schema="quantlab"
        )
    }
    assert {"tax_liability_amount"} <= {
        column["name"]
        for column in inspector.get_columns("simulation_dividend_actions", schema="quantlab")
    }
    assert {"corporate_tax_liabilities"} <= {
        column["name"]
        for column in inspector.get_columns("simulation_nav", schema="quantlab")
    }
    assert {
        "plan_key",
        "account_id",
        "allocation_artifact_id",
        "decision_date",
        "inputs_as_of",
        "policy_version",
        "execution_policy",
        "tranche_index",
        "plan_hash",
        "plan_json",
    } <= {
        column["name"]
        for column in inspector.get_columns("account_netting_plans", schema="quantlab")
    }
    assert {
        "scope",
        "dataset_identity",
        "dataset_lineage_id",
        "test_start",
        "test_end",
        "sealed_at",
        "first_opened_at",
        "consumed_at",
        "sealed_candidate_set_json",
        "sealed_candidate_set_sha256",
    } <= {
        column["name"]
        for column in inspector.get_columns("oos_vintages", schema="quantlab")
    }
    assert {"research_program_id", "dataset_identity_sha256"} <= {
        column["name"]
        for column in inspector.get_columns("research_campaigns", schema="quantlab")
    }
    assert {
        "experiment_family_id",
        "label_horizon_days",
        "experiment_count",
        "values_sha256",
        "profile_consensus_json",
        "profile_consensus_sha256",
        "promoted_evaluation_id",
        "promotion_evidence_sha256",
        "promoted_by",
        "promoted_at",
    } <= {
        column["name"]
        for column in inspector.get_columns("factor_candidates", schema="quantlab")
    }
    assert {
        "artifact_sha256",
        "candidate_code_sha256",
        "candidate_values_sha256",
        "metrics_sha256",
        "policy_json",
        "policy_sha256",
        "evidence_sha256",
        "dataset_identity_sha256",
        "is_legacy",
        "submitted_values_sha256",
        "recomputed_values_sha256",
        "recompute_evidence_json",
        "hac_p_value",
        "bh_q_value",
        "statistical_contract_version",
        "final_test_key",
        "final_test_consumed_at",
        "signal_frequency",
        "signal_horizon",
        "execution_frequency",
        "execution_contract_hash",
        "qlib_version",
        "qlib_commit",
        "rdagent_version",
        "rdagent_commit",
    } <= {
        column["name"]
        for column in inspector.get_columns("factor_evaluations", schema="quantlab")
    }
    assert {"dataset_roll_policy", "dataset_lineage_id"} <= {
        column["name"] for column in inspector.get_columns("paper_portfolios", schema="quantlab")
    }
    assert {
        "asset_key",
        "content_sha256",
        "manifest_json",
        "manifest_sha256",
        "status",
    } <= {
        column["name"]
        for column in inspector.get_columns("research_assets", schema="quantlab")
    }
    assert {
        "asset_id",
        "research_run_id",
        "scenario",
        "selection_mode",
        "asset_manifest_sha256",
        "status",
        "reserved_by",
        "reserved_at",
        "completed_at",
    } <= {
        column["name"]
        for column in inspector.get_columns(
            "research_asset_consumptions", schema="quantlab"
        )
    }
    assert {
        "research_run_id",
        "content_sha256",
        "manifest_sha256",
        "capital_eligible",
    } <= {
        column["name"]
        for column in inspector.get_columns("research_run_artifacts", schema="quantlab")
    }
    assert {
        "code_artifact_id",
        "manifest_sha256",
        "feature_set_definition_sha256",
        "pre_final_end",
        "final_oos_start",
        "final_oos_end",
        "admission_evidence_sha256",
        "capital_eligible",
    } <= {
        column["name"]
        for column in inspector.get_columns("model_candidates", schema="quantlab")
    }
    assert {
        "evidence_role",
        "profile_id",
        "seed",
        "final_oos_start",
        "final_oos_end",
        "evidence_sha256",
    } <= {
        column["name"]
        for column in inspector.get_columns("model_evaluations", schema="quantlab")
    }
    assert {
        "model_candidate_id",
        "model_ensemble_candidate_id",
        "factor_candidate_ids_json",
        "bundle_manifest_sha256",
        "feature_set_definition_sha256",
        "pre_final_end",
        "final_oos_start",
        "final_oos_end",
        "ablation_evidence_sha256",
        "capital_eligible",
    } <= {
        column["name"]
        for column in inspector.get_columns("quant_bundle_candidates", schema="quantlab")
    }
    assert {"ablation", "profile_id", "seed", "evidence_sha256"} <= {
        column["name"]
        for column in inspector.get_columns("quant_bundle_evaluations", schema="quantlab")
    }
    assert {"dataset_roll_policy", "dataset_lineage_id"} <= {
        column["name"]
        for column in inspector.get_columns(
            "recommendation_portfolios", schema="quantlab"
        )
    }
    assert {"dataset", "dataset_identity_sha256", "dataset_lineage_id"} <= {
        column["name"] for column in inspector.get_columns("portfolio_batches", schema="quantlab")
    }
    assert {
        "dataset_roll_policy",
        "dataset_lineage_id",
        "execution_roll_policy",
        "execution_lineage_id",
    } <= {
        column["name"]
        for column in inspector.get_columns("pair_paper_portfolios", schema="quantlab")
    }
    assert {
        "dataset",
        "dataset_identity_sha256",
        "dataset_lineage_id",
        "execution_snapshot",
        "execution_manifest_sha256",
        "execution_lineage_id",
    } <= {
        column["name"]
        for column in inspector.get_columns("pair_portfolio_batches", schema="quantlab")
    }
    strategy_version_columns = {
        column["name"]
        for column in inspector.get_columns("strategy_versions", schema="quantlab")
    }
    assert "strategy_type" in strategy_version_columns
    assert "is_legacy" in strategy_version_columns
    assert {
        "signal_frequency",
        "signal_horizon",
        "execution_frequency",
        "execution_contract_hash",
        "qlib_version",
        "qlib_commit",
        "rdagent_version",
        "rdagent_commit",
        "horizon_profile",
        "label_horizons_json",
        "decision_interval_sessions",
        "review_interval_sessions",
        "holding_min_sessions",
        "holding_target_sessions",
        "holding_max_sessions",
        "execution_lag_sessions",
        "purge_sessions",
        "embargo_sessions",
        "sealed_oos_required",
        "sealed_oos_sessions",
        "horizon_contract_json",
        "horizon_contract_sha256",
        "source_research_artifact_id",
        "strategy_rules_sha256",
    } <= strategy_version_columns
    strategy_version_indexes = {
        item["name"]: item
        for item in inspector.get_indexes("strategy_versions", schema="quantlab")
    }
    active_horizon_index = strategy_version_indexes[
        "uq_strategy_versions_active_horizon"
    ]
    assert active_horizon_index["unique"] is True
    assert "recommendation_enabled" in str(
        active_horizon_index.get("dialect_options") or {}
    )
    assert "legacy_ambiguous" in str(
        active_horizon_index.get("dialect_options") or {}
    )
    approved_family_index = strategy_version_indexes[
        "uq_strategy_versions_approved"
    ]
    assert approved_family_index["unique"] is True
    approved_family_predicate = str(
        approved_family_index.get("dialect_options") or {}
    )
    assert "recommendation_enabled" in approved_family_predicate
    assert "promotion_stage" in approved_family_predicate
    source_artifact_index = strategy_version_indexes[
        "uq_strategy_versions_source_research_artifact"
    ]
    assert source_artifact_index["unique"] is True
    assert "source_research_artifact_id" in str(
        source_artifact_index.get("dialect_options") or {}
    )
    assert {
        "strategy_version_id",
        "horizon_profile",
        "as_of",
        "health_status",
        "criteria_json",
        "criteria_sha256",
        "evidence_json",
        "evidence_sha256",
        "snapshot_sha256",
        "recorded_by",
        "recorded_at",
    } <= {
        column["name"]
        for column in inspector.get_columns(
            "strategy_health_snapshots", schema="quantlab"
        )
    }
    investor_columns = {
        column["name"]: column
        for column in inspector.get_columns(
            "investor_simulation_profiles", schema="quantlab"
        )
    }
    assert {
        "id",
        "profile_key",
        "version",
        "status",
        "supersedes_id",
        "initial_capital",
        "risk_profile",
        "min_cash_weight",
        "max_gross_exposure",
        "market_permissions_json",
        "content_sha256",
        "created_by",
        "created_at",
        "updated_by",
        "updated_at",
    } <= set(investor_columns)
    assert investor_columns["initial_capital"]["default"] is None
    assert {
        "execution_dataset",
        "is_legacy",
        "signal_frequency",
        "execution_frequency",
        "execution_contract_hash",
        "qlib_version",
        "qlib_commit",
        "rdagent_version",
        "rdagent_commit",
    } <= {
        column["name"] for column in inspector.get_columns("backtest_runs", schema="quantlab")
    }
    for table_name in (
        "paper_portfolios",
        "paper_orders",
        "paper_fills",
        "pair_paper_portfolios",
        "pair_paper_orders",
        "pair_paper_fills",
    ):
        assert "is_legacy" in {
            column["name"] for column in inspector.get_columns(table_name, schema="quantlab")
        }
    schedule_columns = {
        column["name"] for column in inspector.get_columns("schedules", schema="quantlab")
    }
    assert {"desired_status", "suspension_reason"} <= schedule_columns
    assert "industry" in {
        column["name"] for column in inspector.get_columns("paper_positions", schema="quantlab")
    }
    assert "take_profit_stage" in {
        column["name"] for column in inspector.get_columns("paper_positions", schema="quantlab")
    }
    risk_columns = {
        column["name"] for column in inspector.get_columns("risk_events", schema="quantlab")
    }
    allocation_event_columns = {
        column["name"]
        for column in inspector.get_columns("strategy_allocation_events", schema="quantlab")
    }
    assert {"acknowledged_by", "resolved_by", "resolved_at", "resolution_reason"} <= risk_columns
    assert {
        "acknowledged_by",
        "acknowledged_at",
        "resolved_by",
        "resolved_at",
        "resolution_reason",
    } <= allocation_event_columns
    assert "is_legacy" in {
        column["name"]
        for column in inspector.get_columns("strategy_allocations", schema="quantlab")
    }
    assert {"role", "risk_budget", "member_cap"} <= {
        column["name"]
        for column in inspector.get_columns("strategy_allocation_members", schema="quantlab")
    }
    assert {
        "source_type",
        "source_id",
        "promotion_stage_id",
        "execution_adapter",
        "execution_frequency",
        "execution_contract_hash",
        "benchmark",
        "daily_roll_policy",
        "execution_roll_policy",
    } <= {
        column["name"]
        for column in inspector.get_columns("simulation_portfolios", schema="quantlab")
    }
    simulation_portfolio_indexes = {
        item["name"]: item
        for item in inspector.get_indexes("simulation_portfolios", schema="quantlab")
    }
    assert simulation_portfolio_indexes[
        "uq_simulation_portfolios_source_execution"
    ]["unique"]
    assert simulation_portfolio_indexes[
        "uq_simulation_portfolios_promotion_stage"
    ]["unique"]
    simulation_portfolio_fks = inspector.get_foreign_keys(
        "simulation_portfolios", schema="quantlab"
    )
    assert any(
        item.get("constrained_columns") == ["promotion_stage_id"]
        and item.get("referred_table") == "strategy_promotion_stages"
        for item in simulation_portfolio_fks
    )
    assert {
        "source_snapshot_id",
        "target_payload_json",
        "execution_adapter",
        "execution_contract_hash",
        "created_by",
        "signal_at",
        "execution_not_before",
        "daily_dataset",
        "daily_dataset_identity_sha256",
        "daily_dataset_lineage_id",
        "execution_dataset",
        "execution_dataset_identity_sha256",
        "execution_dataset_lineage_id",
        "simulation_semantics_sha256",
    } <= {
        column["name"]
        for column in inspector.get_columns("simulation_batches", schema="quantlab")
    }
    batch_checks = {
        item["name"]: str(item.get("sqltext") or "")
        for item in inspector.get_check_constraints(
            "simulation_batches", schema="quantlab"
        )
    }
    next_bar_check = batch_checks["ck_simulation_batches_next_bar_time"]
    assert "execution_not_before > signal_at" in next_bar_check
    assert "Asia/Shanghai" in next_bar_check
    batch_indexes = {
        item["name"]: item
        for item in inspector.get_indexes("simulation_batches", schema="quantlab")
    }
    recommendation_index = batch_indexes[
        "uq_simulation_batches_portfolio_recommendation"
    ]
    assert recommendation_index["unique"] is True
    assert recommendation_index["column_names"] == [
        "portfolio_id",
        "recommendation_snapshot_id",
    ]
    assert "recommendation_snapshot_id IS NOT NULL" in str(
        recommendation_index.get("dialect_options") or {}
    )
    assert {
        "nav_scope",
        "produced_by",
        "reviewed_by",
        "reviewed_at",
        "review_evidence_sha256",
        "review_note",
        "benchmark_close",
        "benchmark_return",
        "benchmark_wealth",
    } <= {
        column["name"]
        for column in inspector.get_columns("simulation_nav", schema="quantlab")
    }
    for table_name in ("simulation_orders", "simulation_fills", "simulation_positions"):
        assert {"atomic_group_id", "leg_no", "position_side", "borrow_cost"} <= {
            column["name"]
            for column in inspector.get_columns(table_name, schema="quantlab")
        }


def test_0045_retires_legacy_approved_pair_versions(database_url: str) -> None:
    """Seed a pre-gate approved pair version, replay 0045, expect retirement."""

    import uuid
    from dataclasses import asdict

    from alembic import command
    from sqlalchemy import select, update

    from quant_data.database import strategies, strategy_events, strategy_versions
    from quant_platform.db_cli import alembic_config
    from quant_platform.pair_trading import PairTradingConfig
    from quant_platform.strategy_store import StrategyStore

    store = StrategyStore(database_url)
    created = store.create_pair(
        name=f"legacy-approved-pair-{uuid.uuid4().hex}",
        description="pre-gate approved pair version retired by migration 0045",
        leg_y="SH510300",
        leg_x="SZ159919",
        asset_class="etf",
        shorting_mode="margin_borrow",
        config=asdict(PairTradingConfig()),
        actor="legacy-researcher",
    )
    version = created["versions"][0]
    engine = open_database(database_url)
    with engine.begin() as connection:
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version["id"])
            .values(status="approved")
        )
        connection.execute(
            update(strategies)
            .where(strategies.c.id == version["strategy_id"])
            .values(status="approved")
        )

    config = alembic_config(database_url)
    command.stamp(config, "0044_recommendation_actions")
    # Replay only 0045's data migration: later migrations are DDL and cannot
    # re-run on the already-migrated schema; restore the head stamp after.
    command.upgrade(config, "0045_research_only_pair")
    command.stamp(config, "head")

    with engine.connect() as connection:
        retired = connection.execute(
            select(strategy_versions.c.status).where(
                strategy_versions.c.id == version["id"]
            )
        ).scalar_one()
        family = connection.execute(
            select(strategies.c.status).where(strategies.c.id == version["strategy_id"])
        ).scalar_one()
        audit = connection.execute(
            select(strategy_events.c.event_type, strategy_events.c.actor).where(
                strategy_events.c.strategy_version_id == version["id"],
                strategy_events.c.event_type == "strategy.pair_retired_research_only",
            )
        ).first()
    assert retired == "retired"
    assert family == "retired"
    assert audit is not None and audit[1] == "migration-0045"


def test_same_lineage_repair_constraint_is_limited_to_exact_v2_v3_v4_v5(
    database_url: str,
) -> None:
    engine = open_database(database_url)
    with engine.connect() as connection:
        definition = connection.scalar(
            text(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conname = 'ck_transparent_baseline_repair_distinct_target' "
                "AND conrelid = "
                "'quantlab.transparent_baseline_pre_result_repairs'::regclass"
            )
        )

    # PostgreSQL may render text-column comparisons with explicit ``::text``
    # casts, so assert the governed operands instead of its cosmetic SQL form.
    assert "source_dataset_lineage_id" in definition
    assert "target_dataset_lineage_id" in definition
    assert "transparent-baseline-pre-result-repair-v2" in definition
    assert "v7-to-v8-optimizer-applicability" in definition
    assert "transparent-baseline-pre-result-repair-v3" in definition
    assert "v8-to-v9-canonical-lf-packaging" in definition
    assert "transparent-baseline-pre-result-repair-v4" in definition
    assert "v9-to-v10-runtime-contract-alignment" in definition
    assert "transparent-baseline-runtime-contract-alignment-v1" in definition
    assert "3e3bc56a2daf7e35206e94c8d36a4c9ffb2a9fd6" in definition
    assert "0f868f2f3beaf5cff5db461f11c411ab3ef960c620c3f2f902ac385fc79eff4d" in (
        definition
    )
    assert "b2b501cf1b59201b8732bacfa787ab38e41867993234aa2f9f605d9b163e7c68" in (
        definition
    )
    assert "eea7ca8854acbed0370ead8d97fdfdb128059f41d8a5995a5050b7ec9e9f3a34" in (
        definition
    )
    assert "6a988b654c9cd5ac9caca183d0efc83ce2f337f88b4fe33c8ac32c355b8fb86c" in (
        definition
    )
    assert "fa1090deaa66ca77a045c1a872f7b6451043e116e717954533217908135c584e" in (
        definition
    )
    assert "1ce281eb2e0141922215b9966f1ba91e09d073019a2e6e4b6a4614049b769e44" in (
        definition
    )
    assert "256bbfd579865e7bc1442f1f64003d655241abecb320e5d27b440dd635224d57" in (
        definition
    )
    assert "transparent-baseline-pre-result-repair-v5" in definition
    assert "v10-to-v11-runtime-input-scope" in definition
    assert "transparent-baseline-runtime-input-scope-v1" in definition
    assert "bb4d1139f848f0e39b82d13f7c4d58ff7659d8d2" in definition
    assert "9bc6a3017c2c718497dd53395da43c6d7392ceefdd0a268587c48d77813bcd65" in (
        definition
    )
    assert "4771fc24680dcdca18fbcf73c887f604aad70d5f586c25116102f75d51d6e042" in (
        definition
    )
    assert "6e169632f9f322db93856f0e7ade3c9436b310e3509806f68d0b47db837e1b6d" in (
        definition
    )
    assert "64fa634b4e774279356c5655e70c741890ed9b208ccf75df757a378c0f56a432" in (
        definition
    )
    assert "687ad83efd9d734a238bd6b52b3f5e670cec1e5d165ec2b25cabcef724feb7cf" in (
        definition
    )
    assert "c2727672b33fed580841721551c45df1a11e864dd899cb6d473c220821da0dc0" in (
        definition
    )
    for target_change_code in (
        "missing-5d-extension-evidence-per-instrument-new-entry-rejection",
        "missing-trend-evidence-holding-continuity-new-entry-rejection",
        "shared-governed-style-exposure-snapshot-backtest-recommendation",
        "topk-benchmark-weight-non-consumption",
    ):
        assert target_change_code in definition


def test_downgrade_rejects_append_only_same_lineage_v5_atomically(
    database_url: str,
) -> None:
    engine = open_database(database_url)
    source_ids = [
        "00f1b9171d3d4ec3a04ade9ddec7d061",
        "04e84f759929477a9cae3de4ba1749fe",
        "4aa9972029d34793848f2c5a3e4ddb43",
    ]
    verification = {
        "receipt_contract_version": "transparent-baseline-pre-result-repair-v5",
        "repair_generation": "v10-to-v11-runtime-input-scope",
        "source_release_commit": "bb4d1139f848f0e39b82d13f7c4d58ff7659d8d2",
        "source_batch_sha256": (
            "9bc6a3017c2c718497dd53395da43c6d7392ceefdd0a268587c48d77813bcd65"
        ),
        "source_dataset_identity_sha256": (
            "4771fc24680dcdca18fbcf73c887f604aad70d5f586c25116102f75d51d6e042"
        ),
        "source_dataset_lineage_id": (
            "6e169632f9f322db93856f0e7ade3c9436b310e3509806f68d0b47db837e1b6d"
        ),
        "source_runner_sha256": (
            "0f868f2f3beaf5cff5db461f11c411ab3ef960c620c3f2f902ac385fc79eff4d"
        ),
        "target_runner_sha256": (
            "64fa634b4e774279356c5655e70c741890ed9b208ccf75df757a378c0f56a432"
        ),
        "source_runtime_bundle_sha256": (
            "eea7ca8854acbed0370ead8d97fdfdb128059f41d8a5995a5050b7ec9e9f3a34"
        ),
        "target_runtime_bundle_sha256": (
            "687ad83efd9d734a238bd6b52b3f5e670cec1e5d165ec2b25cabcef724feb7cf"
        ),
        "runtime_contract_version": "transparent-baseline-runtime-input-scope-v1",
        "source_artifact_inventories_sha256": (
            "c2727672b33fed580841721551c45df1a11e864dd899cb6d473c220821da0dc0"
        ),
        "target_change_codes": [
            "missing-5d-extension-evidence-per-instrument-new-entry-rejection",
            "missing-trend-evidence-holding-continuity-new-entry-rejection",
            "shared-governed-style-exposure-snapshot-backtest-recommendation",
            "topk-benchmark-weight-non-consumption",
        ],
    }
    with engine.begin() as connection:
        audit_id = connection.execute(
            insert(audit_events)
            .values(
                user_id=None,
                username="system:migration-test",
                action="transparent_baseline_pre_result_repair_registered",
                method="INTERNAL",
                path="transparent-baseline/pre-result-repair",
                status_code=201,
                ip_hash=None,
                user_agent="pytest",
                details_json={},
                created_at=datetime.now(UTC),
            )
            .returning(audit_events.c.id)
        ).scalar_one()
        connection.execute(
            insert(transparent_baseline_pre_result_repairs).values(
                receipt_sha256="a" * 64,
                source_audit_event_id=audit_id,
                source_batch_sha256=verification["source_batch_sha256"],
                target_batch_sha256="b" * 64,
                source_dataset_lineage_id=verification[
                    "source_dataset_lineage_id"
                ],
                target_dataset_lineage_id=verification[
                    "source_dataset_lineage_id"
                ],
                target_recipe_version=(
                    "qlib-rdagent-single-mainline-2026-08-30-v11"
                ),
                source_backtest_ids_json=source_ids,
                target_strategy_version_ids_json=["1" * 32, "2" * 32, "3" * 32],
                verification_json=verification,
                created_at=datetime.now(UTC),
            )
        )

    with pytest.raises(RuntimeError, match="runtime input-scope repair evidence"):
        command.downgrade(
            alembic_config(database_url), "0076_baseline_runtime_repair"
        )

    with engine.connect() as connection:
        assert connection.scalar(
            text("SELECT version_num FROM quantlab.alembic_version")
        ) == "0080_baseline_v13_seal"


def test_downgrade_rejects_append_only_same_lineage_v4_atomically(
    database_url: str,
) -> None:
    engine = open_database(database_url)
    source_ids = sorted(
        [
            "ef927367c58143448ebeaab2eceab5f1",
            "f795119754ea41c2960a710e2626bd19",
            "e05b237e364c4811a97ff5e2b49fc68c",
        ]
    )
    verification = {
        "receipt_contract_version": "transparent-baseline-pre-result-repair-v4",
        "repair_generation": "v9-to-v10-runtime-contract-alignment",
        "source_release_commit": "3e3bc56a2daf7e35206e94c8d36a4c9ffb2a9fd6",
        "source_runner_sha256": (
            "256bbfd579865e7bc1442f1f64003d655241abecb320e5d27b440dd635224d57"
        ),
        "target_runner_sha256": (
            "0f868f2f3beaf5cff5db461f11c411ab3ef960c620c3f2f902ac385fc79eff4d"
        ),
        "source_runtime_bundle_sha256": (
            "b2b501cf1b59201b8732bacfa787ab38e41867993234aa2f9f605d9b163e7c68"
        ),
        "target_runtime_bundle_sha256": (
            "eea7ca8854acbed0370ead8d97fdfdb128059f41d8a5995a5050b7ec9e9f3a34"
        ),
        "runtime_contract_version": (
            "transparent-baseline-runtime-contract-alignment-v1"
        ),
        "source_artifact_inventories_sha256": (
            "6a988b654c9cd5ac9caca183d0efc83ce2f337f88b4fe33c8ac32c355b8fb86c"
        ),
    }
    with engine.begin() as connection:
        audit_id = connection.execute(
            insert(audit_events)
            .values(
                user_id=None,
                username="system:migration-test",
                action="transparent_baseline_pre_result_repair_registered",
                method="INTERNAL",
                path="transparent-baseline/pre-result-repair",
                status_code=201,
                ip_hash=None,
                user_agent="pytest",
                details_json={},
                created_at=datetime.now(UTC),
            )
            .returning(audit_events.c.id)
        ).scalar_one()
        connection.execute(
            insert(transparent_baseline_pre_result_repairs).values(
                receipt_sha256="f" * 64,
                source_audit_event_id=audit_id,
                source_batch_sha256="a" * 64,
                target_batch_sha256="b" * 64,
                source_dataset_lineage_id="c" * 64,
                target_dataset_lineage_id="c" * 64,
                target_recipe_version=(
                    "qlib-rdagent-single-mainline-2026-08-30-v10"
                ),
                source_backtest_ids_json=source_ids,
                target_strategy_version_ids_json=["1" * 32, "2" * 32, "3" * 32],
                verification_json=verification,
                created_at=datetime.now(UTC),
            )
        )

    with pytest.raises(RuntimeError, match="runtime-alignment repair evidence"):
        command.downgrade(alembic_config(database_url), "0075_baseline_lf_repair")

    with engine.connect() as connection:
        assert connection.scalar(
            text("SELECT version_num FROM quantlab.alembic_version")
        ) == "0080_baseline_v13_seal"


def test_downgrade_rejects_append_only_same_lineage_v3_atomically(
    database_url: str,
) -> None:
    engine = open_database(database_url)
    source_ids = sorted(
        [
            "1ca979f22a0d4e2981e6e5c5e478f583",
            "7b4386b52cf24d6d913dae3b232fbe7f",
            "0c5f5a0b66284a66ba68225afd22b623",
        ]
    )
    verification = {
        "receipt_contract_version": "transparent-baseline-pre-result-repair-v3",
        "repair_generation": "v8-to-v9-canonical-lf-packaging",
        "source_release_commit": "413024ff5971115d4eb6a33872ebcafe9619cc9e",
        "source_runner_expected_sha256": (
            "fa1090deaa66ca77a045c1a872f7b6451043e116e717954533217908135c584e"
        ),
        "source_runner_observed_sha256": (
            "1ce281eb2e0141922215b9966f1ba91e09d073019a2e6e4b6a4614049b769e44"
        ),
        "target_runner_sha256": (
            "256bbfd579865e7bc1442f1f64003d655241abecb320e5d27b440dd635224d57"
        ),
        "packaging_contract_version": "git-archive-canonical-lf-v1",
    }
    with engine.begin() as connection:
        audit_id = connection.execute(
            insert(audit_events)
            .values(
                user_id=None,
                username="system:migration-test",
                action="transparent_baseline_pre_result_repair_registered",
                method="INTERNAL",
                path="transparent-baseline/pre-result-repair",
                status_code=201,
                ip_hash=None,
                user_agent="pytest",
                details_json={},
                created_at=datetime.now(UTC),
            )
            .returning(audit_events.c.id)
        ).scalar_one()
        connection.execute(
            insert(transparent_baseline_pre_result_repairs).values(
                receipt_sha256="e" * 64,
                source_audit_event_id=audit_id,
                source_batch_sha256="a" * 64,
                target_batch_sha256="b" * 64,
                source_dataset_lineage_id="c" * 64,
                target_dataset_lineage_id="c" * 64,
                target_recipe_version="qlib-rdagent-single-mainline-2026-08-30-v9",
                source_backtest_ids_json=source_ids,
                target_strategy_version_ids_json=["1" * 32, "2" * 32, "3" * 32],
                verification_json=verification,
                created_at=datetime.now(UTC),
            )
        )

    with pytest.raises(RuntimeError, match="canonical-LF v3"):
        command.downgrade(alembic_config(database_url), "0074_baseline_repair_chain")

    with engine.connect() as connection:
        assert connection.scalar(
            text("SELECT version_num FROM quantlab.alembic_version")
        ) == "0080_baseline_v13_seal"


def test_downgrade_rejects_append_only_same_lineage_v2_atomically(
    database_url: str,
) -> None:
    engine = open_database(database_url)
    source_ids = sorted(
        [
            "f51d7fa2f4fd463e97fd5f6990b3721c",
            "8090c21aa11546bd9d59f732975afc25",
            "9c8a75ac646f452e8a5666bacd708936",
        ]
    )
    verification = {
        "receipt_contract_version": "transparent-baseline-pre-result-repair-v2",
        "repair_generation": "v7-to-v8-optimizer-applicability",
        "target_runner_sha256": (
            "fa1090deaa66ca77a045c1a872f7b6451043e116e717954533217908135c584e"
        ),
    }
    with engine.begin() as connection:
        audit_id = connection.execute(
            insert(audit_events)
            .values(
                user_id=None,
                username="system:migration-test",
                action="transparent_baseline_pre_result_repair_registered",
                method="INTERNAL",
                path="transparent-baseline/pre-result-repair",
                status_code=201,
                ip_hash=None,
                user_agent="pytest",
                details_json={},
                created_at=datetime.now(UTC),
            )
            .returning(audit_events.c.id)
        ).scalar_one()
        connection.execute(
            insert(transparent_baseline_pre_result_repairs).values(
                receipt_sha256="d" * 64,
                source_audit_event_id=audit_id,
                source_batch_sha256="a" * 64,
                target_batch_sha256="b" * 64,
                source_dataset_lineage_id="c" * 64,
                target_dataset_lineage_id="c" * 64,
                target_recipe_version="qlib-rdagent-single-mainline-2026-08-30-v8",
                source_backtest_ids_json=source_ids,
                target_strategy_version_ids_json=["1" * 32, "2" * 32, "3" * 32],
                verification_json=verification,
                created_at=datetime.now(UTC),
            )
        )

    with pytest.raises(RuntimeError, match="append-only same-lineage v2"):
        command.downgrade(
            alembic_config(database_url), "0073_baseline_pre_result_repair"
        )

    with engine.connect() as connection:
        assert connection.scalar(
            text("SELECT version_num FROM quantlab.alembic_version")
        ) == "0080_baseline_v13_seal"
