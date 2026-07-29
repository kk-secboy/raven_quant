from sqlalchemy import inspect, text

from quant_data.database import open_database


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
        "jobs",
        "research_runs",
        "factor_candidates",
        "factor_evaluations",
        "oos_vintages",
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
    assert revision == "0058_simulation_benchmark"
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
        "artifact_sha256",
        "predictions_sha256",
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
        "min_decision_batches",
        "min_completed_cycles",
        "min_data_completeness",
        "min_reconciliation_rate",
        "max_cost_deviation",
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
    } <= strategy_version_columns
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
